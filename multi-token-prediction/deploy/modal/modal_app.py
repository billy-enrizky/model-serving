"""Modal deployment of the MTP Gemma server on H100.

Exposes server.api:app as a Modal ASGI web endpoint.

Layout:
  - 1x H100 per container, scale-to-zero
  - HF weights cached on a persistent volume mounted at /cache/hf
  - HF_TOKEN and MODEL_API_KEY come from Modal secrets
  - Engine has a global lock; we keep concurrency=1 per container
    and let Modal scale by adding containers, not threads.

Deploy:
    modal deploy deploy/modal/modal_app.py

Serve (ephemeral preview):
    modal serve deploy/modal/modal_app.py
"""

from __future__ import annotations

import os

import modal

# Per-deploy overrides. Set MTP_GPU and MTP_NUM_ASSISTANT in the shell before
# `modal deploy ...` to target a specific accelerator and toggle MTP on/off
# without editing this file. Defaults are H100 + N=4.
GPU_TYPE = os.environ.get("MTP_GPU", "H100")
NUM_ASSISTANT_TOKENS = os.environ.get("MTP_NUM_ASSISTANT", "4")

# Distinct app per GPU so deploys do not stomp on each other's container pool.
# H100 keeps the legacy app name for URL stability; other GPUs get a suffix.
_BASE_APP_NAME = "mtp-gemma-server"
_APP_SUFFIX = "" if GPU_TYPE == "H100" else f"-{GPU_TYPE.lower().replace('-', '').replace('!', '').replace('+', 'plus')}"
APP_NAME = f"{_BASE_APP_NAME}{_APP_SUFFIX}"

REPO_ROOT_LOCAL = "/Users/billy/Documents/model-serving/multi-token-prediction"
REPO_ROOT_IMAGE = "/app"

HF_CACHE_DIR = "/cache/hf"

# Per-GPU torch wheel selection. cu126 wheels target up to Hopper (sm_90); they
# do not contain Blackwell (sm_100) kernels. For B200/B300 we install the
# cu128 wheels of torch 2.9.1, which include sm_100. transformers >=5.8.0 is
# required for Gemma 4 + the `assistant_model=` heuristic schedule.
def _is_blackwell(gpu: str) -> bool:
    g = gpu.upper()
    return g.startswith("B200") or g.startswith("B300")


if _is_blackwell(GPU_TYPE):
    _TORCH = "torch==2.9.1"
    _TV = "torchvision==0.24.1"
    _TA = "torchaudio==2.9.1"
    _CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
else:
    _TORCH = "torch==2.7.0"
    _TV = "torchvision==0.22.0"
    _TA = "torchaudio==2.7.0"
    _CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(_TORCH, _TV, _TA, extra_index_url=_CUDA_INDEX)
    .pip_install(
        "transformers>=5.8.0",
        "accelerate>=1.2",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        "tokenizers>=0.21",
        "pillow>=11",
        "fastapi>=0.118",
        "uvicorn[standard]>=0.34",
        "httpx>=0.28.1",
        "pydantic>=2.12",
        "pydantic-settings>=2.7.1",
        "prometheus-client>=0.21.1",
        "nvidia-ml-py>=12.560.30",
        "huggingface-hub>=0.27.1",
    )
    .env(
        {
            # Models
            "TARGET_MODEL": "google/gemma-4-E2B-it",
            "ASSISTANT_MODEL": "google/gemma-4-E2B-it-assistant",
            "NUM_ASSISTANT_TOKENS": NUM_ASSISTANT_TOKENS,
            "NUM_ASSISTANT_TOKENS_SCHEDULE": "heuristic",
            "SERVED_MODEL_NAME": "gemma-4-E2B-it",
            # Cache on the mounted volume (persists across cold starts)
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "TRANSFORMERS_CACHE": HF_CACHE_DIR,
            # Allocator tuning
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Logging
            "LOG_LEVEL": "INFO",
            # Make /app importable so `from server.api import app` works
            "PYTHONPATH": REPO_ROOT_IMAGE,
        }
    )
    # Ship the project source. server/, configs/ etc.
    .add_local_dir(
        local_path=REPO_ROOT_LOCAL,
        remote_path=REPO_ROOT_IMAGE,
        # Skip large local artifacts: bench results, venv, caches, modal state
        # (the deploy log races with the mount upload otherwise).
        ignore=[
            ".venv/**",
            ".venv",
            "metrics/runs/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "logs/**",
            ".git/**",
            ".DS_Store",
            "deploy/modal/.state/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)

hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

TIMEOUT_SECONDS = 60 * 30  # generations are short but cold-load + first compile can be slow
SCALEDOWN_SECONDS = 60 * 5  # idle window before container shuts down


@app.function(
    gpu=GPU_TYPE,
    image=image,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("mtp-api-key"),
    ],
    volumes={HF_CACHE_DIR: hf_volume},
    timeout=TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_SECONDS,
    min_containers=0,
    max_containers=4,
    # Engine is single-threaded behind a global lock. One inflight request per
    # container; Modal scales horizontally by spawning more containers.
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Return the FastAPI app from server.api.

    Imports happen here (not at module top) so that env vars from Modal
    secrets are present before server.api / server.mtp_engine read them.
    """
    import sys

    if REPO_ROOT_IMAGE not in sys.path:
        sys.path.insert(0, REPO_ROOT_IMAGE)

    from server.api import app as fastapi  # noqa: WPS433

    return fastapi


bench_results_volume = modal.Volume.from_name(
    "mtp-bench-results", create_if_missing=True
)


@app.function(
    gpu=GPU_TYPE,
    image=image,
    secrets=[modal.Secret.from_name("mtp-api-key")],
    volumes={"/results": bench_results_volume},
    timeout=TIMEOUT_SECONDS,
)
def bench_run(
    base_url: str,
    label: str,
    requests: int = 16,
    concurrency: int = 1,
    max_tokens: int = 128,
    prompt_set: str = "generic",
) -> dict:
    """Run bench/load_runner against an external base_url.

    Runs on a Modal H100 so NVML reports H100 (sm_90, HBM3) values for the MFU
    and peak-bandwidth math. The bench client hits the deployed server over
    HTTPS, so this function's GPU does no inference, only NVML probing.
    """
    import asyncio
    import json
    import logging
    import os
    import sys
    import time
    from pathlib import Path

    if REPO_ROOT_IMAGE not in sys.path:
        sys.path.insert(0, REPO_ROOT_IMAGE)

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bench_run")

    from bench.gpu_probe import snapshot
    from bench.load_runner import (
        GpuSampler,
        PROMPT_SETS,
        aggregate,
        run_load,
        write_prometheus,
    )
    from bench.mfu import GEMMA_4_E2B_N_ACTIVE

    api_key = os.environ["MODEL_API_KEY"]
    n_active = GEMMA_4_E2B_N_ACTIVE
    param_bytes = 10_246_621_918  # bf16 google/gemma-4-E2B-it model.safetensors

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path("/results") / f"{ts}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler(interval_s=0.25)
    gpu_before = snapshot(0)
    sampler.start()
    try:
        records, wall_s = asyncio.run(
            run_load(
                base_url=base_url,
                api_key=api_key,
                model="gemma-4-E2B-it",
                n_requests=requests,
                concurrency=concurrency,
                max_tokens=max_tokens,
                prompts=PROMPT_SETS[prompt_set],
            )
        )
    finally:
        sampler.stop()
        sampler.join(timeout=5.0)
    gpu_after = snapshot(0)
    gpu_peak = sampler.peak or gpu_after

    from dataclasses import asdict

    config = {
        "base_url": base_url,
        "model": "gemma-4-E2B-it",
        "requests": requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "label": label,
        "n_active_params": n_active,
        "param_bytes": param_bytes,
        "prompt_set": prompt_set,
    }
    agg = aggregate(records, wall_s, gpu_peak, n_active, param_bytes)

    result = {
        "started_at": time.time() - wall_s,
        "finished_at": time.time(),
        "config": config,
        "gpu_before": asdict(gpu_before),
        "gpu_peak": asdict(gpu_peak),
        "gpu_after": asdict(gpu_after),
        "n_requests": len(records),
        "n_success": sum(1 for r in records if r.status == 200),
        "requests": [asdict(r) for r in records],
        "aggregate": agg,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    write_prometheus(out_dir, agg, config)
    bench_results_volume.commit()

    log.info("Bench done. Wrote %s", out_dir)
    log.info("Throughput: %.2f tok/s", agg.get("system_throughput_tokens_per_sec", 0.0))
    log.info(
        "Acceptance: %.2f%%",
        100.0 * agg.get("speculative_decoding", {}).get("overall_acceptance_rate", 0.0),
    )
    return {"out_dir": str(out_dir), "aggregate": agg, "config": config}


@app.function(
    gpu=GPU_TYPE,
    image=image,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={HF_CACHE_DIR: hf_volume},
    timeout=TIMEOUT_SECONDS,
)
def warm_weights():
    """One-shot job: pre-download target + drafter into the volume.

    Run with:
        modal run deploy/modal/modal_app.py::warm_weights
    """
    import logging
    import os

    from huggingface_hub import snapshot_download

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("warm_weights")

    token = os.environ["HF_TOKEN"]
    for repo_id in (
        os.environ["TARGET_MODEL"],
        os.environ["ASSISTANT_MODEL"],
    ):
        log.info("Downloading %s ...", repo_id)
        snapshot_download(repo_id=repo_id, token=token, cache_dir=HF_CACHE_DIR)
        log.info("Done %s", repo_id)

    hf_volume.commit()
    log.info("Volume committed.")
