"""Model FLOP Utilization for Gemma 4 E2B inference.

MFU = achieved_flops / peak_flops.

For decoder-only inference (forward only), achieved FLOPs per token:
    F_per_token = 2 * N_active_params + extra_attention_term

Where 2*N is the dominant matmul cost (FP16 ops). For Gemma 4 with PLE,
"effective" params (~2.3B for E2B) drive compute, while total params (~5.1B)
include lookups not multiplied per token.

Reference formula: Chinchilla / OpenAI scaling: C = 6 N_eff D for training,
2 N_eff per token for forward-only inference. Source:
Hoffmann et al. 2022 (Chinchilla), Kaplan et al. 2020 (Scaling Laws).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MFUResult:
    achieved_tflops: float
    peak_tflops: float
    mfu: float
    flops_per_token: float
    tokens_per_sec: float
    n_active_params: int
    formula: str


def compute_mfu(
    tokens_per_sec: float,
    peak_tflops: float,
    n_active_params: int,
    n_total_params: int | None = None,
) -> MFUResult:
    """Compute MFU exact (no approximation) given measured tokens/sec."""
    flops_per_token = 2.0 * float(n_active_params)
    achieved_flops = flops_per_token * tokens_per_sec
    achieved_tflops = achieved_flops / 1e12
    mfu = achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0
    formula = (
        f"FLOPs/token = 2 * N_active = 2 * {n_active_params} = {flops_per_token:.0f}; "
        f"achieved = FLOPs/token * tokens/sec; MFU = achieved / peak"
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
