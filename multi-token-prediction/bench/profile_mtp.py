
Writes a Chrome trace + a top-K kernel/op summary to stdout. Bypasses the
HTTP server: loads the same models the engine loads, runs generate() with
assistant_model directly, captures CUDA events.

Usage: python -m bench.profile_mtp
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    target_id = os.environ.get("TARGET_MODEL", "google/gemma-4-E2B-it")
    assistant_id = os.environ.get("ASSISTANT_MODEL", "google/gemma-4-E2B-it-assistant")
    n = int(os.environ.get("NUM_ASSISTANT_TOKENS", "4"))
    schedule = os.environ.get("NUM_ASSISTANT_TOKENS_SCHEDULE", "heuristic")
    out_dir = Path(os.environ.get("PROFILE_DIR", "metrics/profile"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading target %s", target_id)
    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=torch.bfloat16).to("cuda").eval()
    logger.info("loading assistant %s", assistant_id)
    asst = AutoModelForCausalLM.from_pretrained(assistant_id, torch_dtype=torch.bfloat16).to("cuda").eval()
    asst.generation_config.num_assistant_tokens = n
    asst.generation_config.num_assistant_tokens_schedule = schedule

    prompt = "Write a Python function that performs binary search on a sorted list."
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    # Warmup.
    logger.info("warmup")
    use_assistant = n > 0
    gen_kwargs = dict(max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    if use_assistant:
        gen_kwargs["assistant_model"] = asst
    with torch.inference_mode():
        for _ in range(2):
            target.generate(**inputs, **gen_kwargs)

    # Profile.
    logger.info("profiling N=%d schedule=%s", n, schedule)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            with record_function("generate_full"):
                t0 = time.time()
                gen_kwargs2 = dict(max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
                if use_assistant:
                    gen_kwargs2["assistant_model"] = asst
                target.generate(**inputs, **gen_kwargs2)
                t1 = time.time()
    logger.info("generate wall = %.2fs", t1 - t0)

    # Write Chrome trace.
    trace_path = out_dir / f"mtp_n{n}_chrome_trace.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info("chrome trace: %s", trace_path)

    # Top-K CUDA-time and self-CPU-time tables.
    print("\n=== Top 30 ops by CUDA total time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("\n=== Top 30 ops by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    # Quick attribution: anything whose name contains 'assistant' or 'draft'
    # we count as drafter; everything else is target. (Heuristic.)
    drafter_us = 0
    target_us = 0
    other_us = 0
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        name = evt.key.lower()
        if "draft" in name or "assistant" in name or "candidate" in name:
            drafter_us += cuda_us
        elif "generate_full" in name:
            other_us += cuda_us
        else:
            target_us += cuda_us
    total = drafter_us + target_us
    if total > 0:
        print(f"\n=== Heuristic split (by op name) ===")
        print(f"drafter-tagged CUDA us = {drafter_us:,} ({drafter_us/total*100:.2f}%)")
        print(f"other (target+kernels) CUDA us = {target_us:,} ({target_us/total*100:.2f}%)")
        print(f"NOTE: most ops are not name-tagged; this split is approximate.")


if __name__ == "__main__":
    main()
