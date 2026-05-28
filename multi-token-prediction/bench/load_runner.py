"""Concurrent benchmark client for vLLM gateway.

Drives the public endpoint, measures TTFT, end-to-end latency, throughput,
and samples GPU memory during the run. Results persisted to JSON + Prometheus
text in metrics/runs/<timestamp>/.

No averaging hides anything: per-request raw values stored. Aggregates use
exact percentiles via numpy.percentile.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.gpu_probe import GPUSnapshot, snapshot
from bench.mfu import GEMMA_4_E2B_N_ACTIVE, compute_mbu, compute_mfu

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


PROMPT_SETS: dict[str, list[str]] = {
    "generic": [
        "Explain the concept of model FLOP utilization in transformer inference.",
        "Write a Python function that performs binary search on a sorted list.",
        "Summarize the key innovations of the original Transformer paper.",
        "What are the trade-offs between Q4_K_M and Q5_K_M GGUF quantization?",
        "Describe how speculative decoding accelerates LLM inference.",
        "Compare and contrast vLLM PagedAttention with FlashAttention.",
        "Outline a production deployment plan for a 7B parameter LLM on a single A100.",
        "Walk through how RoPE positional embeddings differ from learned positions.",
    ],
    # Code-heavy prompts. Hypothesis: code output has higher token-level
    # predictability (boilerplate, indentation, common identifiers), so
    # drafter argmax should match target argmax more often than on prose.
    # If acceptance climbs vs the generic set, MTP may flip net-positive.
    "code": [
        "Write a complete Python function `two_sum(nums, target)` returning indices of two numbers that add to target. Include docstring and a test in __main__.",
        "Implement merge sort in Python. Provide the full function with a recursive merge and test on [3,1,4,1,5,9,2,6,5,3].",
        "Write a Python function `is_balanced(s)` that returns True if parentheses, brackets, and braces are balanced. Include three test cases.",
        "Write a Python class `LRUCache(capacity)` with `get(key)` and `put(key, value)` in O(1). Use OrderedDict and include a usage example.",
        "Write a Python function `quicksort(arr)` with the Lomuto partition. Include a docstring with complexity analysis and a test on a 10-element list.",
        "Implement Dijkstra's shortest path in Python using heapq. Function signature: `dijkstra(graph, start)`. Show usage on a 5-node weighted graph.",
        "Write a Python function `flatten(lst)` that flattens an arbitrarily nested list. Include three test cases of varying depth.",
        "Implement a binary tree in Python with `insert`, `inorder`, and `search` methods. Show insertion of [5,3,7,1,4,6,8] and an inorder traversal.",
    ],
    # Structured-output prompts. Hypothesis: JSON / template completion is
    # the most predictable regime for a small drafter (delimiters, field
    # names recur), so acceptance should be highest here.
    "structured": [
        "Return a JSON object with fields name (string), age (int), email (string), and a list of three hobbies (strings). Use the values: Alice, 30, alice@example.com, hobbies of your choice.",
        "Return a JSON array of three book objects, each with title, author, year, and isbn. Pick well-known books.",
        "Return a YAML document describing a Kubernetes Deployment for an nginx container with 3 replicas, named 'web', port 80.",
        "Return a JSON object representing a HTTP 200 response with headers (Content-Type, Cache-Control, X-Request-ID) and a body field with a short JSON-encoded payload.",
        "Return a JSON object describing a user record with id (uuid), created_at (ISO8601), profile (nested object with first_name, last_name, country), and roles (array of strings).",
        "Return a JSON array of five GeoJSON Point features with random-looking but valid lat/lon coordinates and a 'name' property each.",
        "Return a TOML config for a Rust crate with package name 'mytool', version 0.2.1, edition 2021, and three dependencies (serde, tokio, anyhow) with versions.",
        "Return a JSON object that represents an OpenAPI 3.1 path entry for GET /users/{id} with a 200 response containing a User schema reference.",
    ],
}

# Backwards-compat alias for any external import.
PROMPTS = PROMPT_SETS["generic"]


@dataclass
class RequestRecord:
    idx: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_latency_ms: float
    decode_tokens_per_sec: float
    status: int
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    error: str | None = None


@dataclass
class RunResult:
    started_at: float
    finished_at: float
    config: dict[str, Any]
    gpu_before: dict[str, Any]
    gpu_peak: dict[str, Any]
    gpu_after: dict[str, Any]
    n_requests: int
    n_success: int
    requests: list[RequestRecord] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)


class GpuSampler(threading.Thread):
    """Polls NVML in background thread, records peak memory used."""

    def __init__(self, interval_s: float = 0.25, gpu_index: int = 0) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self.peak: GPUSnapshot | None = None
        self.samples: list[GPUSnapshot] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                s = snapshot(self.gpu_index)
                self.samples.append(s)
                if self.peak is None or s.memory_used_bytes > self.peak.memory_used_bytes:
                    self.peak = s
            except Exception as exc:
                logger.warning("nvml sample failed: %s", exc)
            self._stop_event.wait(self.interval_s)


async def stream_one(
    client: httpx.AsyncClient,
    idx: int,
    prompt: str,
    max_tokens: int,
    api_key: str,
    model: str,
) -> RequestRecord:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }
    headers = {"x-api-key": api_key, "content-type": "application/json"}

    start = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    accepted_tokens = 0
    proposed_tokens = 0
    last_tok_time = start
    decode_durations: list[float] = []
    status = 0
    err: str | None = None

    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload, headers=headers
        ) as r:
            status = r.status_code
            if r.status_code != 200:
                err = (await r.aread()).decode("utf-8", "replace")[:500]
                return RequestRecord(idx, 0, 0, 0.0, 0.0, 0.0, status, err)
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = now - start
                        else:
                            decode_durations.append(now - last_tok_time)
                        last_tok_time = now
                        completion_tokens += 1
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                    spec = usage.get("speculative_decoding") or {}
                    accepted_tokens = int(spec.get("accepted_tokens", accepted_tokens))
                    proposed_tokens = int(spec.get("proposed_tokens", proposed_tokens))
    except Exception as exc:
        err = repr(exc)
        return RequestRecord(idx, prompt_tokens, completion_tokens, 0.0, 0.0, 0.0, status or 0, err)

    e2e = time.perf_counter() - start
    decode_tps = (completion_tokens - 1) / sum(decode_durations) if decode_durations else 0.0
    return RequestRecord(
        idx=idx,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=(ttft or 0.0) * 1000.0,
        e2e_latency_ms=e2e * 1000.0,
        decode_tokens_per_sec=decode_tps,
        status=status,
        accepted_tokens=accepted_tokens,
        proposed_tokens=proposed_tokens,
        error=err,
    )


async def run_load(
    base_url: str,
    api_key: str,
    model: str,
    n_requests: int,
    concurrency: int,
    max_tokens: int,
    prompts: list[str],
) -> tuple[list[RequestRecord], float]:
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits, verify=False) as client:
        async def task(i: int) -> RequestRecord:
            async with sem:
                prompt = prompts[i % len(prompts)]
                return await stream_one(client, i, prompt, max_tokens, api_key, model)

        wall_start = time.perf_counter()
        records = await asyncio.gather(*(task(i) for i in range(n_requests)))
        wall_elapsed = time.perf_counter() - wall_start
    return records, wall_elapsed


def aggregate(
    records: list[RequestRecord],
    wall_s: float,
    gpu_peak: GPUSnapshot,
    n_active_params: int,
    param_bytes: int,
) -> dict[str, Any]:
    successes = [r for r in records if r.status == 200 and r.completion_tokens > 0]
    if not successes:
        return {"n_success": 0}
    ttft = np.array([r.ttft_ms for r in successes])
    e2e = np.array([r.e2e_latency_ms for r in successes])
    decode_tps = np.array([r.decode_tokens_per_sec for r in successes])
    total_completion = sum(r.completion_tokens for r in successes)
    total_prompt = sum(r.prompt_tokens for r in successes)
    total_accepted = sum(r.accepted_tokens for r in successes)
    total_proposed = sum(r.proposed_tokens for r in successes)
    overall_acceptance = (total_accepted / total_proposed) if total_proposed else 0.0
    per_req_accept = np.array(
        [r.accepted_tokens / r.proposed_tokens for r in successes if r.proposed_tokens]
    )
    system_throughput = total_completion / wall_s if wall_s > 0 else 0.0
    mfu = compute_mfu(
        tokens_per_sec=system_throughput,
        peak_tflops=gpu_peak.peak_fp16_tflops,
        n_active_params=n_active_params,
    )
    # MBU uses TPOT (steady-state per-output-token latency).
    # TPOT_ms = (e2e - TTFT) / (completion_tokens - 1), per-request, then mean.
    tpot_per_req: list[float] = []
    for r in successes:
        if r.completion_tokens > 1 and r.e2e_latency_ms > r.ttft_ms:
            tpot_per_req.append(
                (r.e2e_latency_ms - r.ttft_ms) / 1000.0 / (r.completion_tokens - 1)
            )
    mean_tpot = float(np.mean(tpot_per_req)) if tpot_per_req else 0.0
    mbu = compute_mbu(
        tpot_seconds=mean_tpot,
        peak_hbm_gbps=gpu_peak.peak_hbm_gbps,
        param_bytes=param_bytes,
        kv_cache_bytes=0,
    )
    return {
        "n_success": len(successes),
        "wall_seconds": wall_s,
        "total_completion_tokens": total_completion,
        "total_prompt_tokens": total_prompt,
        "system_throughput_tokens_per_sec": system_throughput,
        "ttft_ms": {
            "mean": float(ttft.mean()),
            "p50": float(np.percentile(ttft, 50)),
            "p90": float(np.percentile(ttft, 90)),
            "p99": float(np.percentile(ttft, 99)),
            "min": float(ttft.min()),
            "max": float(ttft.max()),
        },
        "e2e_latency_ms": {
            "mean": float(e2e.mean()),
            "p50": float(np.percentile(e2e, 50)),
            "p90": float(np.percentile(e2e, 90)),
            "p99": float(np.percentile(e2e, 99)),
            "min": float(e2e.min()),
            "max": float(e2e.max()),
        },
        "per_request_decode_tokens_per_sec": {
            "mean": float(decode_tps.mean()),
            "p50": float(np.percentile(decode_tps, 50)),
            "p90": float(np.percentile(decode_tps, 90)),
            "stdev": float(decode_tps.std(ddof=0)),
        },
        "vram_peak_used_bytes": gpu_peak.memory_used_bytes,
        "vram_peak_used_gib": gpu_peak.memory_used_bytes / (1024 ** 3),
        "vram_total_bytes": gpu_peak.memory_total_bytes,
        "vram_total_gib": gpu_peak.memory_total_bytes / (1024 ** 3),
        "mfu": {
            "achieved_tflops": mfu.achieved_tflops,
            "peak_tflops": mfu.peak_tflops,
            "mfu_fraction": mfu.mfu,
            "flops_per_token": mfu.flops_per_token,
            "n_active_params": mfu.n_active_params,
            "formula": mfu.formula,
        },
        "mbu": {
            "achieved_gbps": mbu.achieved_gbps,
            "peak_hbm_gbps": mbu.peak_gbps,
            "mbu_fraction": mbu.mbu,
            "bytes_per_token": mbu.bytes_per_token,
            "tpot_seconds": mbu.tpot_seconds,
            "param_bytes": param_bytes,
            "formula": mbu.formula,
        },
        "speculative_decoding": {
            "total_accepted_tokens": total_accepted,
            "total_proposed_tokens": total_proposed,
            "overall_acceptance_rate": overall_acceptance,
            "per_request_acceptance_rate": {
                "mean": float(per_req_accept.mean()) if per_req_accept.size else 0.0,
                "p50": float(np.percentile(per_req_accept, 50)) if per_req_accept.size else 0.0,
                "p90": float(np.percentile(per_req_accept, 90)) if per_req_accept.size else 0.0,
                "min": float(per_req_accept.min()) if per_req_accept.size else 0.0,
                "max": float(per_req_accept.max()) if per_req_accept.size else 0.0,
            },
        },
    }


def write_prometheus(out_dir: Path, agg: dict[str, Any], config: dict[str, Any]) -> None:
    lines: list[str] = []
    label = ",".join(f'{k}="{v}"' for k, v in config.items() if isinstance(v, (int, float, str, bool)))
    def _emit(metric: str, value: float, help_text: str) -> None:
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric}{{{label}}} {value}")
    if "system_throughput_tokens_per_sec" in agg:
        _emit("bench_throughput_tokens_per_second", agg["system_throughput_tokens_per_sec"], "System throughput")
        _emit("bench_ttft_ms_p50", agg["ttft_ms"]["p50"], "TTFT p50 ms")
        _emit("bench_ttft_ms_p99", agg["ttft_ms"]["p99"], "TTFT p99 ms")
        _emit("bench_e2e_latency_ms_p50", agg["e2e_latency_ms"]["p50"], "E2E latency p50 ms")
        _emit("bench_e2e_latency_ms_p99", agg["e2e_latency_ms"]["p99"], "E2E latency p99 ms")
        _emit("bench_vram_peak_used_bytes", agg["vram_peak_used_bytes"], "Peak VRAM used (bytes)")
        _emit("bench_mfu_fraction", agg["mfu"]["mfu_fraction"], "Model FLOP Utilization fraction")
        _emit("bench_achieved_tflops", agg["mfu"]["achieved_tflops"], "Achieved TFLOPS")
        _emit("bench_peak_tflops", agg["mfu"]["peak_tflops"], "Peak TFLOPS (FP16 tensor)")
        spec = agg.get("speculative_decoding", {})
        if spec:
            _emit("bench_acceptance_rate_overall", spec["overall_acceptance_rate"], "Overall MTP acceptance rate")
            _emit("bench_accepted_tokens_total", spec["total_accepted_tokens"], "Total accepted speculative tokens")
            _emit("bench_proposed_tokens_total", spec["total_proposed_tokens"], "Total proposed speculative tokens")
        mbu_block = agg.get("mbu", {})
        if mbu_block:
            _emit("bench_mbu_fraction", mbu_block["mbu_fraction"], "Memory Bandwidth Utilization fraction")
            _emit("bench_achieved_hbm_gbps", mbu_block["achieved_gbps"], "Achieved HBM bandwidth (GB/s)")
            _emit("bench_peak_hbm_gbps", mbu_block["peak_hbm_gbps"], "Peak HBM bandwidth (GB/s)")
    (out_dir / "metrics.prom").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8443 or https://xxx.modal.run")
    p.add_argument("--api-key", default=os.getenv("MODEL_API_KEY"))
    p.add_argument("--model", default="gemma-4-E2B-it")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--n-active-params",
        type=int,
        default=GEMMA_4_E2B_N_ACTIVE,
        help="Effective compute params for E2B (excludes PLE lookups)",
    )
    p.add_argument(
        "--param-bytes",
        type=int,
        default=10_246_621_918,
        help="Bytes streamed from HBM per decode step. Default = exact BF16 "
        "size of google/gemma-4-E2B-it/model.safetensors (HF API).",
    )
    p.add_argument("--label", default="run")
    p.add_argument("--metrics-dir", default=os.getenv("METRICS_DIR", "metrics/runs"))
    p.add_argument(
        "--prompt-set",
        choices=sorted(PROMPT_SETS.keys()),
        default="generic",
        help="Prompt rotation: generic (prose), code (code-heavy), structured (JSON/YAML/TOML).",
    )
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("MODEL_API_KEY env var or --api-key required")

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.metrics_dir) / f"{ts}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler(interval_s=0.25)
    gpu_before = snapshot(0)
    sampler.start()
    try:
        records, wall_s = asyncio.run(run_load(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            n_requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            prompts=PROMPT_SETS[args.prompt_set],
        ))
    finally:
        sampler.stop()
        sampler.join(timeout=5.0)
    gpu_after = snapshot(0)
    gpu_peak = sampler.peak or gpu_after

    config = {
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "label": args.label,
        "n_active_params": args.n_active_params,
        "param_bytes": args.param_bytes,
        "prompt_set": args.prompt_set,
    }
    agg = aggregate(records, wall_s, gpu_peak, args.n_active_params, args.param_bytes)
    result = RunResult(
        started_at=time.time() - wall_s,
        finished_at=time.time(),
        config=config,
        gpu_before=asdict(gpu_before),
        gpu_peak=asdict(gpu_peak),
        gpu_after=asdict(gpu_after),
        n_requests=args.requests,
        n_success=agg.get("n_success", 0),
        requests=records,
        aggregate=agg,
    )

    out_dir.joinpath("result.json").write_text(json.dumps(asdict(result), indent=2))
    write_prometheus(out_dir, agg, config)
    logger.info("results: %s", out_dir)
    print(json.dumps(agg, indent=2))
    return 0 if agg.get("n_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
