"""Smoke test: load engine + 1 generation. Used during deployment validation only."""
from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("MODEL_API_KEY", "smoke")

from server.mtp_engine import MTPEngine

print("loading...", flush=True)
t0 = time.time()
engine = MTPEngine()
print(f"loaded in {time.time() - t0:.1f}s", flush=True)
print(f"VRAM after load (MiB): {torch.cuda.memory_allocated(0) / 1024 / 1024:.1f}", flush=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
t0 = time.time()
text, stats = engine.generate(messages, max_new_tokens=64, temperature=1.0, top_p=0.95, top_k=64)
print(f"gen time: {time.time() - t0:.2f}s", flush=True)
print(f"text: {text[:300]!r}", flush=True)
print(f"stats: {stats}", flush=True)
