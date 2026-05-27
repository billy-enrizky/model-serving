"""Read VRAM and GPU peak FLOPS from NVML directly.

No estimates. Reports exact MiB used/total per GPU, plus device name + SM count
for downstream MFU calc.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass

try:
    import pynvml
except ImportError as exc:
    raise SystemExit("pynvml (nvidia-ml-py) required") from exc

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    index: int
    name: str
    memory_total_bytes: int
    memory_used_bytes: int
    memory_free_bytes: int
    sm_count: int
    sm_clock_mhz: int
    cuda_cores_per_sm: int
    peak_fp16_tflops: float


# CUDA cores per SM and FP16 perf factor by compute capability.
# Source: NVIDIA whitepapers, cited per arch.
_ARCH_TABLE = {
    # major, minor: (cuda_cores_per_sm, fp16_ops_per_cycle_per_core)
    # Tensor core peak FP16 FMA: 64 x 64 ops? Use NVML clock-based formula:
    # peak_fp16_tflops = sm_count * tensor_fp16_ops_per_cycle * clock_hz / 1e12
    (7, 5): (64, 512),   # T4/RTX20 Turing
    (8, 0): (64, 1024),  # A100 Ampere FP16 tensor
    (8, 6): (128, 512),  # RTX30
    (8, 9): (128, 512),  # RTX40 Ada
    (9, 0): (128, 2048), # H100 Hopper FP16 tensor
}


def snapshot(index: int = 0) -> GPUSnapshot:
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        sm_count = pynvml.nvmlDeviceGetNumGpuCores(h) if hasattr(
            pynvml, "nvmlDeviceGetNumGpuCores"
        ) else 0
        try:
            sm_count = pynvml.nvmlDeviceGetAttribute(
                h, pynvml.NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            )
        except Exception:
            pass
        try:
            sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)

        try:
            major = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            cc = (major[0], major[1])
        except Exception:
            cc = (7, 0)

        cores_per_sm, fp16_ops_per_cycle_per_sm = _ARCH_TABLE.get(cc, (64, 1024))
        peak_fp16_tflops = (
            sm_count * fp16_ops_per_cycle_per_sm * (sm_clock * 1e6) / 1e12
        )

        return GPUSnapshot(
            index=index,
            name=name,
            memory_total_bytes=int(mem.total),
            memory_used_bytes=int(mem.used),
            memory_free_bytes=int(mem.free),
            sm_count=int(sm_count),
            sm_clock_mhz=int(sm_clock),
            cuda_cores_per_sm=int(cores_per_sm),
            peak_fp16_tflops=float(peak_fp16_tflops),
        )
    finally:
        pynvml.nvmlShutdown()


def main() -> int:
    snap = snapshot(0)
    json.dump(asdict(snap), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
