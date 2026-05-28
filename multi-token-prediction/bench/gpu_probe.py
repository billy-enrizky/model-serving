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
    mem_bus_width_bits: int
    mem_clock_mhz: int
    peak_hbm_gbps: float


# CUDA cores per SM and FP16 perf factor by compute capability.
# Source: NVIDIA whitepapers, cited per arch.
_ARCH_TABLE = {
    # major, minor: (cuda_cores_per_sm, fp16_ops_per_cycle_per_core)
    # Tensor core peak FP16 FMA: 64 x 64 ops? Use NVML clock-based formula:
    # peak_fp16_tflops = sm_count * tensor_fp16_ops_per_cycle * clock_hz / 1e12
    (7, 5): (64, 512),   # T4/RTX20 Turing
    (8, 0): (64, 1024),  # A100 Ampere FP16 tensor
    (8, 6): (128, 512),  # RTX30 / A10 / A40
    (8, 9): (128, 512),  # RTX40 Ada / L4
    (9, 0): (128, 2048), # H100 Hopper FP16 tensor
    # Blackwell (B100/B200/B300) sm_100/sm_120: 128 CUDA cores/SM. FP16 tensor
    # ops/cycle/SM is approximate; per public Blackwell datasheet B200 hits
    # ~2250 TFLOPS dense FP16 across 148 SMs at ~1.85 GHz boost, so
    # ops/cycle/SM ~8200. Round to 8192 for power-of-two parity with prior
    # entries. Resulting peak_fp16_tflops is a lower-bound estimate; absolute
    # MFU should be treated as approximate on Blackwell. Throughput/TPOT/
    # MBU/acceptance are unaffected by this constant.
    (10, 0): (128, 8192),
    (12, 0): (128, 8192),
}


def _resolve_sm_count(handle, total_cuda_cores: int, cores_per_sm: int) -> int:
    """Return the count of streaming multiprocessors.

    nvmlDeviceGetNumGpuCores returns total CUDA cores (SMs * cores_per_sm),
    not SM count. NVML only exposes the proper SM attribute on driver R535+.
    Fall back to (total_cores / cores_per_sm) when the attribute is missing.
    """
    attr_name = "NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"
    if hasattr(pynvml, attr_name):
        try:
            return int(
                pynvml.nvmlDeviceGetAttribute(handle, getattr(pynvml, attr_name))
            )
        except Exception:
            pass
    if cores_per_sm > 0 and total_cuda_cores > 0:
        return total_cuda_cores // cores_per_sm
    return 0


def snapshot(index: int = 0) -> GPUSnapshot:
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)

        try:
            major = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            cc = (major[0], major[1])
        except Exception:
            cc = (7, 0)
        cores_per_sm, fp16_ops_per_cycle_per_sm = _ARCH_TABLE.get(cc, (64, 1024))

        try:
            total_cuda_cores = int(pynvml.nvmlDeviceGetNumGpuCores(h))
        except Exception:
            total_cuda_cores = 0
        sm_count = _resolve_sm_count(h, total_cuda_cores, cores_per_sm)

        try:
            sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)

        peak_fp16_tflops = (
            sm_count * fp16_ops_per_cycle_per_sm * (sm_clock * 1e6) / 1e12
        )

        try:
            mem_bus_width = int(pynvml.nvmlDeviceGetMemoryBusWidth(h))
        except Exception:
            mem_bus_width = 0
        try:
            mem_clock = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
        except Exception:
            try:
                mem_clock = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            except Exception:
                mem_clock = 0
        # HBM2 effective rate = 2 transfers/cycle (DDR). Peak BW (bytes/sec) =
        # bus_width_bits/8 * mem_clock_hz * 2. Reported as GB/s (decimal, /1e9).
        peak_hbm_gbps = (
            mem_bus_width / 8.0 * (mem_clock * 1e6) * 2.0 / 1e9
            if mem_bus_width and mem_clock
            else 0.0
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
            mem_bus_width_bits=int(mem_bus_width),
            mem_clock_mhz=int(mem_clock),
            peak_hbm_gbps=float(peak_hbm_gbps),
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
