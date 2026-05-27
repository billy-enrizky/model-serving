"""Compute-throughput utilization metrics for LLM inference.

Two metrics, both reported:

1. **MFU** (Model FLOP Utilization) -- Kaplan et al. 2020 forward-pass form.
   Compute-bound regime metric. Decode at batch=1 is memory-bound, so MFU
   typically reads <1% on a single GPU; the absolute number is not the
   takeaway, the comparison across configurations is.

2. **MBU** (Memory Bandwidth Utilization) -- Databricks formulation, the
   correct primary metric for autoregressive decode TPOT.

References:
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models",
  https://arxiv.org/abs/2001.08361 (forward FLOPs ~ 2N per token, with the
  attention correction `2*L*n_ctx*d_attn` becoming significant once
  `n_ctx > 12 * d_model`).
- Chowdhery et al. 2022, "PaLM" Sec 5: defines MFU.
- Databricks, "LLM Inference Performance Engineering: Best Practices":
  defines MBU = (param_bytes + KV_cache_bytes) / TPOT / peak_HBM_BW.

For Gemma 4 E2B-it: Google publishes 1.91B as the effective active count
(PLE lookups excluded). See the model card and Google docs.
"""

from __future__ import annotations

from dataclasses import dataclass


# Gemma 4 / Gemma 3n E2B published effective active params.
# Source: Google Gemma docs ("effective memory load of just under 2 billion
# (1.91B) parameters") and gemma-3n / gemma-4 E2B model cards.
GEMMA_4_E2B_N_ACTIVE = 1_910_000_000


@dataclass
class MFUResult:
    achieved_tflops: float
    peak_tflops: float
    mfu: float
    flops_per_token: float
    tokens_per_sec: float
    n_active_params: int
    formula: str


@dataclass
class MBUResult:
    achieved_gbps: float
    peak_gbps: float
    mbu: float
    bytes_per_token: float
    tpot_seconds: float
    formula: str


def compute_mfu(
    tokens_per_sec: float,
    peak_tflops: float,
    n_active_params: int,
) -> MFUResult:
    """Compute MFU using the Kaplan 2N forward-pass form.

    Reported as the dense matmul lower bound. Add the attention correction
    `2*L*n_ctx*d_attn` if you also want the attention-inclusive figure
    (Gemma 4 E2B has L=35, d_attn varies between sliding 256 and global 512;
    attention is negligible for the prompts in this benchmark).
    """
    flops_per_token = 2.0 * float(n_active_params)
    achieved_flops = flops_per_token * tokens_per_sec
    achieved_tflops = achieved_flops / 1e12
    mfu = achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0
    formula = (
        f"FLOPs/token = 2 * N_active = 2 * {n_active_params} = {flops_per_token:.0f} "
        "(Kaplan 2020 forward-pass dense lower bound); "
        "achieved = FLOPs/token * tokens/sec; MFU = achieved / peak"
    )
    return MFUResult(
        achieved_tflops=achieved_tflops,
        peak_tflops=peak_tflops,
        mfu=mfu,
        flops_per_token=flops_per_token,
        tokens_per_sec=tokens_per_sec,
        n_active_params=n_active_params,
        formula=formula,
    )


def compute_mbu(
    tpot_seconds: float,
    peak_hbm_gbps: float,
    param_bytes: int,
    kv_cache_bytes: int = 0,
) -> MBUResult:
    """Compute MBU = (param + KV bytes) / TPOT / peak_HBM_BW.

    `tpot_seconds` is time per output token (steady-state decode latency,
    not TTFT). For batch=1 autoregressive decode, this is the right primary
    metric: every step streams the entire weight set + the KV cache slice
    through HBM exactly once.

    Reference: Databricks, "LLM Inference Performance Engineering: Best
    Practices".
    """
    bytes_per_token = float(param_bytes + kv_cache_bytes)
    achieved_bps = bytes_per_token / tpot_seconds if tpot_seconds > 0 else 0.0
    achieved_gbps = achieved_bps / 1e9
    mbu = achieved_gbps / peak_hbm_gbps if peak_hbm_gbps > 0 else 0.0
    formula = (
        f"bytes/token = param_bytes + kv_cache_bytes = {param_bytes} + {kv_cache_bytes} "
        f"= {int(bytes_per_token)}; achieved = bytes/token / TPOT; MBU = achieved / peak_HBM_BW"
    )
    return MBUResult(
        achieved_gbps=achieved_gbps,
        peak_gbps=peak_hbm_gbps,
        mbu=mbu,
        bytes_per_token=bytes_per_token,
        tpot_seconds=tpot_seconds,
        formula=formula,
    )
