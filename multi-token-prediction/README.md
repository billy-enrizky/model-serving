# multi-token-prediction

Benchmark study of **Gemma 4 E2B-it Multi-Token Prediction (MTP)** at batch=1,
using the official Google reference path: Hugging Face `transformers` with
the `assistant_model=` kwarg, mirroring the model card and Google MTP docs
exactly, compared against vLLM 0.21.0 spec-decode on the same GPUs.

> The drafter proposes N tokens generated autoregressively; the target
> model verifies all N tokens in **one** forward pass; drafted tokens with
> high probabilities are accepted, low probabilities are rejected.
> Source: [ai.google.dev/gemma/docs/mtp/mtp](https://ai.google.dev/gemma/docs/mtp/mtp)

The implementation does not reinvent the speculative-decoding loop. It calls
`target_model.generate(..., assistant_model=assistant_model, ...)` with the
documented heuristic schedule (`num_assistant_tokens=4`,
`num_assistant_tokens_schedule="heuristic"`), and exposes the result behind
an OpenAI-compatible HTTP API gated by an API key. All benchmarks run on
Modal across four NVIDIA GPUs (A10, A100-80GB, H100, B200).

## Table of contents

- [Executive summary](#executive-summary)
- [Glossary](#glossary)
  - [Speculative decoding / MTP](#speculative-decoding--mtp)
  - [Latency / throughput metrics](#latency--throughput-metrics)
  - [Utilization (incl. MFU / MBU formulas)](#utilization-incl-mfu--mbu-formulas)
  - [NVIDIA hardware / CUDA](#nvidia-hardware--cuda)
  - [Number formats](#number-formats)
  - [Inference-engine internals](#inference-engine-internals)
- [Why vLLM 0.21.0 specifically](#why-vllm-0210-specifically)
- [Minimum GPU to run vLLM 0.21.0 + Gemma 4 MTP](#minimum-gpu-to-run-vllm-0210--gemma-4-mtp)
- [Architecture](#architecture)
- [Components](#components)
- [Models (verified against HF API)](#models-verified-against-hf-api)
- [Hardware (verified)](#hardware-verified)
- [Quickstart](#quickstart)
- [API](#api)
- [Metrics (benchmark results)](#metrics-benchmark-results)
  - [Three-regime summary (all GPUs, all prompt sets)](#three-regime-summary-all-gpus-all-prompt-sets)
  - [Generic prompts](#generic-prompts)
  - [Code prompts](#code-prompts)
  - [Structured prompts](#structured-prompts)
- [Methodology and caveats](#methodology-and-caveats)
- [Benchmark](#benchmark)
- [Sources](#sources)

## Executive summary

**Bottom line: at batch=1, MTP is not a blanket speedup, and run-to-run
variance is large enough that most per-cell ratios are not trustworthy as
point estimates. Two things are robust across n=3 re-benching: acceptance is
fixed by the model+prompt (not the hardware), and the vLLM engine beats the
transformers reference path by 9-27x with or without MTP. The MTP/baseline
ratio itself is a clean win on only a few cells; on most it straddles 1.0
once you measure its spread.**

All cells were re-benched at **n=3** (3 independent cold runs, constant
gamma=4 on both engines). Numbers are mean +/- sd over the 3 runs. Four
findings, ordered by how well they survive that spread:

1. **Acceptance is a model+prompt property, not a hardware property (robust).**
   Across 8 (engine,GPU) cells per regime, acceptance lands in a tight band:
   generic **34.6-37.3%** (2.7 pp), code **49.8-52.6%** (2.9 pp), structured
   **52.5-58.0%** (5.5 pp). Hardware does not move acceptance; the prompt does.

2. **vLLM's core engine beats the transformers reference path 9-27x, MTP on or
   off (robust).** vLLM mtp / transformers mtp ranges from 8.9x (B200 generic)
   to 26.8x (A100-80GB structured) across every cell. PagedAttention +
   FlashAttention + continuous batching is the dominant value prop; MTP is a
   secondary lever.

3. **MTP is a clean win only on a few cells; most straddle 1.0 (the n=3
   correction).** Of 12 vLLM GPU x regime cells, only **4** keep their entire
   3-run range above 1.0: A10 (all three regimes) and A100-80GB code. Across
   both engines, **7 of 24** cells stay entirely above 1.0 (4 vLLM + 3
   transformers: A10/B200/H100 structured); the other 17 cross 1.0 or sit
   below, so their mean is not distinguishable from breakeven at n=3. The
   robust signal is not which cell wins; it is that acceptance is prompt-fixed
   and the vLLM engine dominates the transformers path 9-27x regardless of MTP.

4. **The one durable MTP-win regime is the A10 (sm_86, mid-tier).** A10 is the
   only GPU whose vLLM ratio stays above 1.0 in all three regimes
   (generic 1.38x, code 2.06x, structured 1.79x): its slower per-decode step
   leaves the most slack for the drafter to hide in. On the faster datacenter
   GPUs the drafter cost and the acceptance lift roughly cancel, and cold-cache
   variance then dominates the sign of the ratio.

### Headline: three-regime MTP/baseline ratio (vLLM, constant gamma=4, n=3)

Mean +/- sd over 3 cold runs. "(crosses 1.0)" = the 3-run range spans both
sides of breakeven, i.e. inconclusive at n=3.

| GPU | generic | code | structured |
|-----|--------:|-----:|-----------:|
| A10       | **1.38x +/- 0.06** | **2.06x +/- 0.41** | **1.79x +/- 0.29** |
| A100-80GB | 1.06x +/- 0.14 (crosses 1.0) | **1.53x +/- 0.25** | 1.07x +/- 0.07 (crosses 1.0) |
| B200      | 1.41x +/- 0.42 (crosses 1.0) | 1.18x +/- 0.29 (crosses 1.0) | 1.22x +/- 0.37 (crosses 1.0) |
| H100      | 1.05x +/- 0.27 (crosses 1.0) | 1.28x +/- 0.43 (crosses 1.0) | 1.37x +/- 0.59 (crosses 1.0) |

Bold = 3-run range stays entirely above 1.0 (robust MTP win). Full transformers
companion table and provenance in
[Three-regime summary](#three-regime-summary-all-gpus-all-prompt-sets).

## Glossary

Every abbreviation used in the rest of this README is defined here. Read
this section first if any term below is unfamiliar.

### Speculative decoding / MTP

- **MTP** = Multi-Token Prediction. Gemma 4's branding for speculative
  decoding (the drafter is shipped as part of the model release).
- **Speculative decoding** = inference acceleration technique. A small
  drafter model proposes N tokens; the large target model verifies all N
  in one forward pass; matching tokens accepted, mismatches rejected.
  Reference: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192).
- **Drafter / assistant model** = the small model that proposes tokens.
  Here: `google/gemma-4-E2B-it-assistant` (78M params).
- **Target model** = the large model that verifies. Here:
  `google/gemma-4-E2B-it` (5.1B params).
- **gamma (γ)** = number of tokens drafter proposes per step before
  target verifies. In transformers this is `num_assistant_tokens`; in
  vLLM, `num_speculative_tokens`. Both default to 4 in this repo.
- **N** = same as gamma. Used in `NUM_ASSISTANT_TOKENS=N` env var.
- **Acceptance rate** = `accepted_tokens / proposed_tokens`. Fraction of
  drafter proposals that pass the target's verification step.
- **Heuristic schedule** = transformers' adaptive gamma adjuster (raises
  N when drafter wins, lowers N when drafter loses). Set via
  `num_assistant_tokens_schedule="heuristic"`.
- **Constant schedule** = fixed gamma every step
  (`num_assistant_tokens_schedule="constant"`). Used here for parity
  with vLLM, which has no adapter.
- **DFlash** = the other Gemma 4 speculative path in vLLM (uses a fused
  drafter+target kernel). Not used by this repo.
- **EAGLE** = a different speculative-decoding family (extra
  feature-extracting head, self-speculation). vLLM ships EAGLE kernels; if
  the kernel symbols (`eagle_prepare_next_token_padded_kernel`) appear in
  baseline logs, a spec-decode path got accidentally enabled.
- **What this repo actually runs:** *draft-model* speculative decoding, not
  EAGLE/self-speculation and not vLLM's `method:"mtp"`. vLLM is launched with
  `--speculative-config '{"model":"google/gemma-4-E2B-it-assistant","num_speculative_tokens":4}'`
  (a separate `-assistant` draft model, no `"method"` key); transformers uses
  `generate(assistant_model=...)`. "MTP" here is Gemma's branding for this
  drafter-based setup, distinct from vLLM's `method:mtp`/EAGLE proposers (which
  use a trained head on the target and the `eagle_*` kernels above).

### Latency / throughput metrics

- **TTFT** = Time To First Token. Wall-clock from request submit to
  first SSE token byte. Captures prefill + queue.
- **TPOT** = Time Per Output Token. Steady-state per-token decode cost,
  computed as `(e2e_latency - TTFT) / (completion_tokens - 1)`. The
  `- 1` is deliberate: TTFT already covers the first token (it includes
  prefill + queue), so `e2e - TTFT` is the time spent producing the
  *remaining* `completion_tokens - 1` tokens. Dividing by
  `completion_tokens - 1` isolates the steady-state per-token cost from
  the one-time prefill cost. Undefined when `completion_tokens == 1`.
- **e2e latency** = end-to-end per-request latency (TTFT + TPOT × tokens).
- **System throughput** = `total_completion_tokens / wall_clock`.
  Aggregate across all requests in a run.
- **SSE** = Server-Sent Events. The streaming HTTP transport
  OpenAI-compatible APIs use for token streaming.
- **c=1** = concurrency 1 (one in-flight request at a time).
- **p50 / p99** = percentiles. `p50` = 50th percentile = median (half of
  requests faster, half slower). `p99` = 99th percentile (only 1% of
  requests slower). Reported in milliseconds (ms). `TTFT p50 = 349 ms`
  reads as "median time to first token was 349 ms".
- **ms** = milliseconds (10^-3 seconds).
- **mean** = arithmetic average (sum / count). Differs from p50 (median)
  when distribution is skewed; both are reported where meaningful.

### Utilization (incl. MFU / MBU formulas)

- **FLOP** = Floating-point Operation. One scalar multiply-add counts as
  2 FLOPs (1 mul + 1 add). Kaplan 2020 uses `FLOPs/token = 2 * N_active`
  for forward-pass dense transformers (the `2` is the mul+add pair, not
  a fudge factor).
- **FLOPS** (capital S) = FLOPs per Second. Rate, not a count. Common
  source of confusion: lowercase "flops" = count of operations,
  uppercase "FLOPS" = ops/sec. This README uses "FLOPs" for counts and
  "FLOPS" / "TFLOPS" for rates.
- **TFLOPS** = TeraFLOPS = 10^12 FLOPS. NVIDIA datasheets quote peak
  tensor-core throughput in TFLOPS at a given precision (FP16, BF16,
  FP8), read live via NVML.
- **GB/s** = Gigabytes per Second. HBM bandwidth unit. Decimal (10^9),
  not binary, in GPU datasheets.
- **FLOP/byte (arithmetic intensity)** = `peak_TFLOPS * 1e12 /
  peak_HBM_bytes_per_sec`. How many FLOPs the GPU can do per byte read
  from HBM. Workloads with intensity above this ratio are compute-bound;
  below, memory-bound. LLM decode at batch=1 = ~2 FLOP/byte (one
  multiply-add per weight read), far below any modern GPU's intensity,
  so always memory-bound.
- **MFU** = Model FLOP Utilization. `achieved_TFLOPS / peak_TFLOPS`.
  Measures how much of the GPU's compute the workload actually uses.
- **MBU** = Memory Bandwidth Utilization.
  `achieved_GB/s / peak_HBM_GB/s`. Measures how much HBM bandwidth the
  workload uses. Decode at batch=1 is bandwidth-bound, so MBU is the
  primary number.
- **N_active** = number of active parameters per forward pass that
  participate in matmuls (the only ops Kaplan 2020's
  `FLOPs/token = 2 * N_active` accounts for). Gemma 4 E2B has
  **5,123,178,051 total parameters** on disk (`model.safetensors`, HF
  API), but only **~1.91B to 2.3B** are "active" per token (this repo
  uses 1.91B; Google's HF model card cites 2.3B "effective": see
  PLE entry below for why both numbers exist). The rest live in
  Per-Layer Embedding (PLE) tables: gather ops, zero FLOPs, excluded
  from `N_active`.
- **PLE** = Per-Layer Embedding. Gemma 4 E2B/E4B (the small dense
  variants) parameter-efficiency trick. Each decoder layer gets its
  own learned embedding vector per token-id, stored in big lookup
  tables. At inference, the layer reads `PLE[layer_idx][token_id]`
  with a **gather** op (memory read, no multiply-adds) and adds the
  vector into the hidden state. Inflates total param count without
  inflating per-token FLOPs. The "E" in "E2B" / "E4B" stands for
  **effective**.
- **PLE is NOT MoE.** Both reduce active-vs-total ratio, opposite
  reasons:

  | | MoE (e.g. Gemma 4 26B-A4B) | PLE (Gemma 4 E2B / E4B) |
  |---|---|---|
  | What's stored | N expert FFN weight matrices | Per-token embedding vectors per layer |
  | Op type | Matmul (gated FFN) | Gather (lookup by token id) |
  | FLOPs/token | Nonzero (top-k experts × FFN) | Zero (memory read only) |
  | Routed by | Learned router over hidden state | Token id (deterministic index) |
  | Why "active" < total | Top-k of N experts selected | Only one row per layer touched |
  | Adds | Compute capacity (sparsely activated) | Memory capacity (deterministically indexed) |

  In Gemma 4: **E2B / E4B / 31B = dense + PLE; 26B-A4B = MoE** (8 of
  128 experts active + 1 shared). This repo uses E2B-it (dense + PLE,
  not MoE). Source:
  [HF model card](https://huggingface.co/google/gemma-4-E2B-it).
- **HBM** = High-Bandwidth Memory. The stacked DRAM on data-center
  GPUs (HBM2/HBM2e/HBM3/HBM3e). Read peak via NVML.
- **NVML** = NVIDIA Management Library. Userspace API for live GPU
  telemetry (clock, mem, power). Accessed via `nvidia-ml-py`.

#### MFU and MBU formulas (no approximation)

Two utilization metrics are reported. For autoregressive decode at batch=1
(this benchmark) the GPU is memory-bandwidth bound, so MBU is the primary
metric and MFU is included as a lower-bound dense-FLOPs reference.

**MFU (Kaplan 2020 forward-pass dense form)**

```
FLOPs/token = 2 * N_active                    (Kaplan 2020)
achieved_TFLOPS = FLOPs/token * tokens/sec / 1e12
MFU = achieved_TFLOPS / peak_TFLOPS
```

- `N_active = 1.91e9` for Gemma 4 E2B (Google's published "effective"
  count; PLE lookups are excluded because they are gathers, not matmuls).
- `peak_TFLOPS` (FP16 tensor cores) is read live via NVML:
  `peak = sm_count * fp16_ops_per_cycle_per_sm * sm_clock_hz / 1e12`.

**MBU (Databricks formulation)**

```
bytes/token = param_bytes + kv_cache_bytes
achieved_GBps = bytes/token / TPOT / 1e9
MBU = achieved_GBps / peak_HBM_GBps
```

- `param_bytes` defaults to the exact BF16 size of
  `google/gemma-4-E2B-it/model.safetensors` (10,246,621,918 B, HF API).
- `peak_HBM_GBps` is read live via NVML
  (`bus_width / 8 * mem_clock_hz * 2 / 1e9`).
- `TPOT` is the steady-state time per output token, computed per request as
  `(e2e - TTFT) / (completion_tokens - 1)` and averaged.

References: Kaplan et al. 2020 ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361));
Chowdhery et al. 2022 PaLM ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311));
[Databricks LLM inference performance](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices).

### NVIDIA hardware / CUDA

- **`sm_XY`** = NVIDIA *compilation target* string. Two digits encode
  the **compute capability major.minor** of the GPU's ISA generation
  (NOT a count of anything). Heads-up on the naming collision: NVML
  also reports a field called `sm_count` (the number of physical
  Streaming Multiprocessors on the chip; A10=72, A100=108, H100=132,
  B200=148). `sm_count` is hardware quantity (more SMs = more parallel
  CUDA cores); `sm_XY` is ISA version (which kernel binaries the chip
  can run). Different numbers, same prefix. Source:
  `nvmlDeviceGetAttribute(NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)`.

  Mapping used in this repo:
  `sm_75` = 7.5 (Turing, T4 / RTX 20-series),
  `sm_80` = 8.0 (Ampere, A100), `sm_86` = 8.6 (Ampere, A10 / RTX 30-series),
  `sm_89` = 8.9 (Ada, L4 / RTX 4090), `sm_90` = 9.0 (Hopper, H100),
  `sm_100` = 10.0 (Blackwell, B200), `sm_120` = 12.0 (next-gen Blackwell).
  Code compiled for `sm_X` does NOT run on a GPU with capability `< X`
  unless a matching `+PTX` fallback is embedded.
- **PTX** = Parallel Thread Execution. NVIDIA's *virtual* ISA, an
  intermediate assembly the driver JIT-compiles to real GPU code at
  load time. Forward-compatible (one PTX blob targets newer GPUs), not
  backward-compatible (PTX built for a newer `sm_XY` cannot run on an
  older one).
- **SASS** = native GPU machine code (per-architecture binary
  instructions) that PTX is translated to. Cannot be JITed across
  generations; if SASS for your `sm_XY` is missing and no PTX fallback
  is embedded, no kernel runs.
- **ISA** = Instruction Set Architecture.
- **JIT** = Just-In-Time compilation. PTX → SASS at driver load time.
- **`TORCH_CUDA_ARCH_LIST`** = PyTorch build-time env var listing the
  compute capabilities to compile kernels for. Determines which `sm_XY`
  targets the resulting torch wheel (and any extension built against
  it, including vLLM's `_C.so`) can launch on.
- **GEMM** = General Matrix Multiply. The dominant CUDA op in LLM
  inference (`aten::mm` in torch).
- **`LD_LIBRARY_PATH`** = Linux dynamic-linker env var; colon-separated
  list of directories the loader searches for shared libraries
  (`.so` files) BEFORE the system default paths. Used to point
  CUDA-built binaries at forward-compat libs when the host driver is
  older than what the binaries expect.
- **CUDA forward-compatibility package** (`cuda-compat-XX-Y`) = NVIDIA
  shipped userspace CUDA driver libs newer than the host kernel-mode
  driver. Lets a CUDA-N+1 binary load on a CUDA-N system. Does NOT
  add missing SASS for older `sm_XY`; only fixes driver-ABI version
  skew. See [NVIDIA forward-compat docs](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

### Number formats

- **FP16** = Floating Point, 16-bit. IEEE 754 half-precision float
  (1 sign / 5 exponent / 10 mantissa bits, total 16 bits).
- **BF16** = Brain Float, 16-bit (Google Brain's format). 1 / 8 / 7
  bits. Same exponent range as FP32 (avoids overflow during training),
  less mantissa precision than FP16. Native on Ampere+ tensor cores.
- **FP32** = Floating Point, 32-bit. IEEE 754 single-precision float
  (1 / 8 / 23, total 32 bits).
- **FP8** = Floating Point, 8-bit. Two variants: E4M3 (1 / 4 / 3) or
  E5M2 (1 / 5 / 2). Native on Hopper+ tensor cores.

### Inference-engine internals

- **PagedAttention** = vLLM's KV-cache allocator. Splits KV cache into
  fixed-size pages so memory fragments do not wedge the scheduler.
- **FlashAttention** = fused-attention kernel; computes softmax(QK^T)V
  without materializing the full attention matrix, cutting HBM traffic.
- **KV cache** = stored Key/Value tensors from prior tokens. Decode at
  step `t` reads KV from steps `0..t-1`.
- **Batched verify** = vLLM's spec-decode optimization that runs target
  verification across N proposed tokens in one fused pass. Transformers
  reference path does not have this.
- **Verify pass** = the target-model forward that checks the drafter's
  N proposed tokens.
- **EOS** = end-of-sequence token. Greedy decoding stops on EOS.

## Why vLLM 0.21.0 specifically

0.21.0 is the first vLLM release that ships **Gemma 4 MTP** (the
speculative-decoding path used by this repo). The MTP integration
landed in
[PR #41745](https://github.com/vllm-project/vllm/pull/41745)
("[Spec Decode] Add Gemma4 MTP speculative decoding support",
merged 2026-05-06) and was first cut in v0.21.0 (released
2026-05-15).

Base Gemma 4 model support predates MTP. The original architecture
PR is
[#38826](https://github.com/vllm-project/vllm/pull/38826)
("feat(models): implement Google Gemma 4 architecture support",
merged 2026-04-02), first shipped in v0.20.0 (2026-04-27). So if
you only need vanilla single-token Gemma 4 decode, v0.20.0+ works;
the `--speculative-config '{"method":"mtp",...}'` path requires
v0.21.0+.

Note on DFlash (the other Gemma 4 speculative path in vLLM, not used
by this repo). Verified against the upstream git history (`git tag
--contains <sha>` on `vllm-project/vllm`):

- DFlash was introduced in
  [PR #36847](https://github.com/vllm-project/vllm/pull/36847)
  ("[Feat][Spec Decode] DFlash", merged 2026-03-30). First contained
  in **v0.20.0** (cut 2026-04-27).
- The FP8 KV-cache fix
  [PR #42692](https://github.com/vllm-project/vllm/pull/42692)
  (commit `0fe7550`) merged 2026-05-15 14:29 UTC, **~10 hours after
  v0.21.0 was tagged** (`ad7125a`, 2026-05-15 04:28 UTC). It is NOT
  in v0.21.0; first release containing it is **v0.22.0rc1** (and
  v0.22.0 when cut).
- The lookahead-slot allocation fix
  [PR #43733](https://github.com/vllm-project/vllm/pull/43733)
  (merged 2026-05-27, well after v0.21.0) will also ship in v0.22.0.
- Several Gemma4-specific DFlash fixes (KV-cache page-size alignment,
  batched-verification rejected-slot masking) are still in flight as
  of 2026-05-28
  ([#40391](https://github.com/vllm-project/vllm/pull/40391),
  [#41703](https://github.com/vllm-project/vllm/pull/41703)) and are
  unmerged.

If your workload uses Gemma 4 + DFlash + FP8 KV cache, v0.21.0 is not
sufficient: track v0.22.0+. If your workload uses Gemma 4 + MTP at
batch=1 (the path this repo benchmarks), v0.21.0 is the floor.

Every reference to "vLLM" in the rest of this README means v0.21.0+.

## Minimum GPU to run vLLM 0.21.0 + Gemma 4 MTP

vLLM 0.21.0 publishes only `+cu129` and `+cu130` wheels. Their bundled
torch builds compile for `['sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90',
'sm_100', 'sm_120']`, so compute capability below **sm_75 (Turing)** is not
supported: the wheels load but every kernel launch raises `no kernel image
is available for execution on the device`. Driver upgrade does not fix it;
the binaries simply lack the older SASS.

**Floor for vLLM Gemma 4 MTP**: compute capability **>= 7.5 (Turing, T4)**
and **>= 16 GiB VRAM** at BF16 with `max_model_len=4096`.

| Tier | Example GPU | sm_ | VRAM | vLLM 0.21.0 Gemma 4 MTP |
|------|-------------|-----|------|-------------------------|
| Minimum | T4 / RTX 2080 Ti | sm_75 | 16 / 11 GiB | Works at BF16 with `max_model_len <= 4096`; T4 is the cheapest cloud option |
| Recommended | A10 / L4 | sm_86 / sm_89 | 24 GiB | Headroom for longer context and MTP draft cache |
| Recommended | RTX 4090 | sm_89 | 24 GiB | Best single-card consumer option |
| Production | A100 40/80GB | sm_80 | 40 / 80 GiB | Full MTP + batching + PagedAttention |
| Production | H100 / H200 | sm_90 | 80 / 141 GiB | Highest acceptance throughput; MTP scales with batch |

Driver requirement for the cu129/cu130 wheels: NVIDIA R535+ baseline,
plus `cuda-compat-12-9` or `cuda-compat-13-0` if the system driver is
older. See [NVIDIA forward-compatibility docs](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

Launch flag (verified in vLLM PR #41745):

```bash
vllm serve google/gemma-4-E2B-it \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-E2B-it-assistant","num_speculative_tokens":2}' \
  --dtype bfloat16 --max-model-len 4096
```

Pascal (sm_60) and earlier are excluded: PyTorch wheels for torch >= 2.6
drop them entirely.

## Architecture

```mermaid
graph LR
    Deploy[deploy scripts] -->|modal deploy| Server[Server GPU<br/>transformers or vLLM]
    Deploy -->|modal run| Bench[Bench GPU<br/>load_runner + NVML]
    Bench -->|HTTPS .modal.run| Server
    Bench --> Vol[(bench-results volume)]
    Vol -->|modal volume get| Results[(metrics/runs/)]
```

Two same-type GPU containers: the server GPU runs inference; the bench GPU
runs `load_runner` and only NVML-probes its own GPU for the MFU/bandwidth math
(it does no inference, `deploy/modal/modal_app.py:194`). The deploy scripts
(`run_const.sh`, `run_ab.sh`, `vllm_run_ab.sh`) deploy a per-GPU app, warm it,
launch `bench_run` on a matching Modal GPU, then pull the run from the
`mtp-bench-results` volume into `metrics/runs/`.

## Components

| Path | Role |
|------|------|
| `server/mtp_engine.py` | Loads target + drafter, calls `generate(assistant_model=...)` per the Gemma 4 reference. Exposes `generate()` and `stream_generate()`. Tracks accept/propose counters. |
| `server/api.py` | OpenAI-compatible FastAPI service: `/v1/chat/completions` (stream + non-stream), `/v1/models`, `/healthz`, `/metrics`. `X-API-Key` and Bearer auth. |
| `bench/load_runner.py` | Concurrent SSE benchmark; persists JSON + Prometheus per run. Captures acceptance rate. |
| `bench/gpu_probe.py` | Live VRAM and peak FP16 TFLOPS via NVML. |
| `bench/mfu.py` | Exact MFU = `2 * N_active * tokens/sec / peak_TFLOPS`. |
| `deploy/modal/modal_app.py` | Modal app: GPU container, ASGI mount, persistent volume for weights. Transformers MTP engine. |
| `deploy/modal/vllm_modal_app.py` | Modal app: vLLM v0.21.0 serving stack with `num_speculative_tokens` env propagation. |
| `deploy/modal/run_ab.sh` | One-shot A/B sweep: deploys MTP, warms, benches, redeploys baseline, benches, restores MTP. |
| `deploy/modal/run_const.sh` | Same flow with `MTP_SCHEDULE=constant`, `N=4` for apples-to-apples vLLM parity. |
| `deploy/modal/vllm_run_ab.sh` | vLLM A/B: `num_speculative_tokens=4` vs no-spec-config baseline. |

## Models (verified against HF API)

| Model | Params | BF16 size |
|-------|--------|-----------|
| `google/gemma-4-E2B-it` | 5,123,178,051 | 10,246,621,918 B (9.5429 GiB) |
| `google/gemma-4-E2B-it-assistant` (drafter) | 77,993,476 | 157,565,344 B (0.1467 GiB) |

Effective compute params per forward pass: **1.91B** (Google's published
"effective" E2B count; Per-Layer Embeddings inflate the total without
contributing to per-token math because they are gathers, not matmuls).

> **Not MoE, PLE.** The 5.12B / 1.91B gap looks like Mixture-of-Experts
> sparsity but is not. E2B is a **dense** transformer with Per-Layer
> Embedding lookup tables. PLE adds memory capacity that is
> deterministically indexed by token id (gather op, zero FLOPs); MoE
> adds compute capacity that a learned router selects (top-k experts,
> nonzero FLOPs). The Gemma 4 family ships both: E2B / E4B / 31B are
> dense + PLE; 26B-A4B is the actual MoE variant (8 of 128 experts +
> 1 shared, "A4B" = active 4B). See **PLE** entry in
> [Glossary](#glossary) for the side-by-side comparison.

Note on the 1.91B vs 2.3B effective-param discrepancy: the HF
[model card for `gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
cites **2.3B effective** for E2B (and 4.5B for E4B); this repo's
`bench/mfu.py` uses **1.91B**. Both numbers come from Google but at
different rounding / accounting boundaries (token-embedding sharing,
norm params). Discrepancy flagged for follow-up; MFU figures in this
README use 1.91B for now and would shift down ~17% if recomputed at
2.3B (`MFU_new = MFU_old * 1.91 / 2.3`).

## Hardware (verified)

All four GPUs are accessed through Modal's GPU containers. Each deploy pins
a GPU class (`"A10"`, `"A100-80GB"`, `"H100"`, `"B200"`) and mounts a
persistent volume for the Gemma weights so cold starts pull from volume, not
HF. Compute/bandwidth specs are read live via NVML on each container.

| GPU | Arch | sm_ | sm_count | mem_bus_bits | peak FP16 TFLOPS | peak HBM GB/s | VRAM (GiB) |
|-----|------|----:|---------:|-------------:|-----------------:|--------------:|-----------:|
| A10 | Ampere | 86 | 72 | 384 | 62.48 | 600.10 | 24 |
| A100-80GB PCIe | Ampere | 80 | 108 | 5120 | 155.93 | 1935.36 | 80 |
| H100 80GB HBM3 | Hopper | 90 | 132 | 5120 | 535.27 | 3352.32 | 80 |
| B200 | Blackwell | 100 | 148 | 7680 | 2382.40 | 7672.32 | 180 |

`N_active = 1.91B`; `param_bytes = 10,246,621,918 B` (BF16
`model.safetensors` of `google/gemma-4-E2B-it`, HF API);
`kv_cache_bytes = 0` (lower bound, all GPUs). The transformers app and the
vLLM app (pinned to `vllm/vllm-openai:v0.21.0`) are both built into the Modal
app specs (`deploy/modal/modal_app.py`, `deploy/modal/vllm_modal_app.py`).

> **B200 MFU/MBU caveat.** `bench/gpu_probe.py::_ARCH_TABLE` uses 8192 FP16
> ops/cycle/SM for sm_100, giving a peak of ~2382 TFLOPS; this constant is
> approximate (no definitive Blackwell ops-per-cycle figure at write time).
> B200 MFU/MBU numbers inherit that approximation; throughput, TPOT, latency,
> and acceptance are independent of it.

## Quickstart

### Local prep

```bash
cd multi-token-prediction
cp .env.example .env
# fill HF_TOKEN, MODEL_API_KEY (any opaque string)
uv sync --extra bench
```

### Modal account bootstrap (one time)

```bash
# Idempotent: token, secrets, and persistent volume for weights.
bash deploy/modal/setup_modal.sh
```

This sets up the Modal CLI auth, registers the `model-api-key` and
`hf-token` secrets, and creates the `gemma-models` persistent volume.

### Pre-download weights into the Modal volume (one time per model)

```bash
bash deploy/modal/warm_weights.sh
```

Pulls `google/gemma-4-E2B-it` (target) and
`google/gemma-4-E2B-it-assistant` (drafter) into `gemma-models` so
cold starts no longer pay HF download time.

### Run the full A/B sweep on a target GPU

```bash
# Transformers MTP A/B (default heuristic schedule):
GPU=h100 bash deploy/modal/run_ab.sh

# Transformers MTP, schedule=constant, N=4 (apples-to-apples vLLM parity):
GPU=h100 bash deploy/modal/run_const.sh

# vLLM A/B (num_speculative_tokens=4 vs no-spec-config baseline):
GPU=h100 bash deploy/modal/vllm_run_ab.sh
```

Each script deploys the MTP variant, warms the container, runs
`bench/load_runner.py`, redeploys the baseline variant, benches that,
then restores the MTP variant. Results land in
`metrics/runs/<ts>_{tx_mtp,tx_mtp_const,vllm_mtp,baseline_n0}_<gpu>_<prompt-set>_c<concurrency>/`.

### Smoke test a single deployment

```bash
GPU=h100 bash deploy/modal/smoke_test.sh        # transformers MTP
GPU=h100 bash deploy/modal/vllm_smoke_test.sh   # vLLM stack
```

The script deploys the app, prints the Modal HTTPS URL, hits
`/healthz`, and tears the deployment down.

## API

OpenAI-compatible. Auth via `X-API-Key` header or `Authorization: Bearer <key>`.

```bash
curl https://<random>.modal.run/v1/chat/completions \
  -H "X-API-Key: ${MODEL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E2B-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short joke about saving RAM."}
    ],
    "max_tokens": 256,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64
  }'
```

The `usage` block in the response includes a `speculative_decoding` field
with `accepted_tokens`, `proposed_tokens`, and `acceptance_rate`.

## Metrics (benchmark results)

- **Prometheus**: `GET /metrics`
  - `mtp_request_latency_seconds` histogram
  - `mtp_ttft_seconds` histogram
  - `mtp_decode_tokens_per_second` histogram
  - `mtp_acceptance_rate` histogram
  - `mtp_accepted_tokens_total`, `mtp_proposed_tokens_total` counters
  - `mtp_completion_tokens_total` counter
  - `mtp_vram_used_bytes{device="cuda:N"}` gauge
- **Persistent benchmark output**: `metrics/runs/<timestamp>_<label>/result.json` and `metrics.prom`.

All A/B runs below: 16 requests, c=1, max_tokens=128, temperature=0.0
(greedy), 8-prompt rotation per regime, on Modal (A10, A100-80GB, H100,
B200). `NUM_ASSISTANT_TOKENS=0` disables MTP entirely (engine omits
`assistant_model=` from `target.generate(...)`); `=4` enables Gemma 4 MTP.
For each prompt set, four comparisons per GPU:

- **(a) transformers mtp vs baseline** , does MTP pay off on the reference path
- **(b) vllm mtp vs baseline** , does MTP pay off on vLLM
- **(c) vllm mtp vs transformers mtp** , engine gap with MTP on both
- **(d) vllm baseline vs transformers baseline** , engine gap with MTP off

GPU compute/bandwidth specs are in [Hardware (verified)](#hardware-verified).
Cold-start (idx=0) tax and n=3 spread are in
[Methodology and caveats](#methodology-and-caveats). Every results table below
is **n=3** (mean +/- sd over 3 cold runs), regenerated from the run dirs by
`bench/rebuild_readme_tables.py`.

For external context, Google's published Gemma 4 MTP speedups
([blog post](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/),
caption: "up to, depending on tasks, batch_size=1, gamma=4"):

| Variant | Hardware | Published speedup |
|---------|----------|-------------------|
| Gemma 4 E2B | Samsung S26 mobile GPU | up to 1.8x |
| Gemma 4 E4B | Samsung S26 mobile GPU | up to 2.2x |
| Gemma 4 E2B | Pixel TPU | up to 2.8x |
| Gemma 4 E4B | Pixel TPU | up to 3.1x |
| Gemma 4 31B | Apple M4 | up to 2.5x |
| Gemma 4 26B | NVIDIA A100 | up to 1.5x |
| Gemma 4 31B | NVIDIA A100 | up to 3.0x |

The lowest published speedup is 1.5x (A100 + 26B). The "up to" caveat is
load-bearing: each number is the best speedup over the workload set Google
evaluated, not a uniform floor.

### Three-regime summary (all GPUs, all prompt sets)

The single cross-cutting answer table. MTP/baseline warm-tps ratio at
constant gamma=4, drop-idx=0 (warm), **n=3** (mean +/- sd over 3 cold runs,
ratio paired per run-index).

**vLLM:**

| GPU | generic | code | structured |
|-----|--------:|-----:|-----------:|
| A10       | **1.38x +/- 0.06** | **2.06x +/- 0.41** | **1.79x +/- 0.29** |
| A100-80GB | 1.06x +/- 0.14 (crosses 1.0) | **1.53x +/- 0.25** | 1.07x +/- 0.07 (crosses 1.0) |
| B200      | 1.41x +/- 0.42 (crosses 1.0) | 1.18x +/- 0.29 (crosses 1.0) | 1.22x +/- 0.37 (crosses 1.0) |
| H100      | 1.05x +/- 0.27 (crosses 1.0) | 1.28x +/- 0.43 (crosses 1.0) | 1.37x +/- 0.59 (crosses 1.0) |

**Transformers reference path:**

| GPU | generic | code | structured |
|-----|--------:|-----:|-----------:|
| A10       | **0.78x +/- 0.08** | 0.90x +/- 0.07 (crosses 1.0) | **1.16x +/- 0.11** |
| A100-80GB | 0.93x +/- 0.26 (crosses 1.0) | 1.01x +/- 0.19 (crosses 1.0) | 0.97x +/- 0.12 (crosses 1.0) |
| B200      | 1.24x +/- 0.23 (crosses 1.0) | 1.20x +/- 0.30 (crosses 1.0) | **1.40x +/- 0.26** |
| H100      | 0.85x +/- 0.24 (crosses 1.0) | **0.83x +/- 0.10** | **1.04x +/- 0.01** |

Bold = 3-run range entirely on one side of 1.0 (robust). "(crosses 1.0)" =
the 3 runs straddle breakeven; inconclusive. Of 24 Modal ratio cells, 9 are
robust (7 above 1.0, 2 below); the other 15 cross.

Provenance: the 2026-06-04 n=3 constant-gamma=4 re-bench (`*_c1_r{1,2,3}` dirs,
aggregated by `bench/summarize_n3.py`); each ratio divides mtp warm-tps by
baseline warm-tps **paired by run index**, then takes the mean and sd of the 3
paired ratios. Raw: `metrics/n3_aggregate.json`.

**Reading the table:** the robustly MTP-positive cells are A10 across all three
regimes (vLLM) and A100-80GB code (vLLM), plus A10/B200/H100 structured
(transformers). The robustly-negative cells are A10 generic and H100 code
(transformers). The other 15 cells' 3-run ranges cross 1.0, so at n=3 they are
indistinguishable from breakeven; their means should not be read as wins or
losses. The robust signal is not which cell wins; it is that acceptance is
prompt-fixed and the vLLM engine dominates the transformers path 9-27x.

### Generic prompts

8 generic prose prompts.

All tables in this subsection are the **n=3** re-bench (mean +/- sd over
r1/r2/r3, warm-only system throughput).

**(a) transformers mtp vs baseline (5-GPU headline, n=3 warm tok/s).** Ratio is
the mean +/- sd of the 3 paired (per-run) ratios; bold = 3-run range entirely
on one side of 1.0.

| GPU | Arch | sm_ | HBM (GB/s) | Baseline (tok/s) | MTP n=4 (tok/s) | MTP/Base | Acceptance |
|-----|------|-----|-----------:|-----------------:|-----------------:|---------:|-----------:|
| NVIDIA A10 | Ampere | 8.6 | 600 | 11.67 +/- 0.11 | 9.14 +/- 0.89 | **0.78x +/- 0.08** | 35.7% |
| NVIDIA A100 80GB PCIe | Ampere | 8.0 | 1935 | 9.84 +/- 2.53 | 8.46 +/- 0.19 | 0.93x +/- 0.26 (crosses 1.0) | 35.5% |
| NVIDIA B200 | Blackwell | 10.0 | 7672 | 17.87 +/- 4.20 | 21.21 +/- 0.98 | 1.24x +/- 0.23 (crosses 1.0) | 35.2% |
| NVIDIA H100 80GB HBM3 | Hopper | 9.0 | 3352 | 16.12 +/- 2.39 | 13.16 +/- 1.51 | 0.85x +/- 0.24 (crosses 1.0) | 34.6% |

On the transformers path only A10 stays robustly below 1.0 across all 3 runs;
A100/B200/H100 generic all cross 1.0 and are breakeven-indistinguishable.
Acceptance lands in a tight 34.6-35.7% band across all four GPUs: a model
property, not a hardware property.

**(b) vllm mtp vs baseline + (c) vllm mtp vs transformers mtp (cross-engine
headline, n=3 warm system tok/s).** Throughput columns are the n=3 mean;
ratios are mean +/- sd of the paired per-run ratios.

| GPU | transformers_mtp | vllm_mtp | vllm_baseline | (b) vllm_mtp / vllm_baseline | (c) vllm_mtp / transformers_mtp | acceptance band |
|-----|---:|---:|---:|---:|---:|---:|
| A10 | 9.14 | 100.43 | 72.97 | **1.38x +/- 0.06** | 11.0x | 37.3% |
| A100-80GB | 8.46 | 142.73 | 136.76 | 1.06x +/- 0.14 (crosses 1.0) | 16.9x | 36.5% |
| B200 | 21.21 | 189.62 | 143.52 | 1.41x +/- 0.42 (crosses 1.0) | 8.9x | 37.1% |
| H100 | 13.16 | 166.44 | 164.96 | 1.05x +/- 0.27 (crosses 1.0) | 12.6x | 37.2% |

(b) At n=3 vLLM generic MTP/baseline is robust only on A10 (1.38x); A100-80GB
(1.06x), B200 (1.41x), and H100 (1.05x) all cross 1.0. (c) vLLM MTP is
8.9-16.9x faster than transformers MTP on every GPU.

**(d) vllm baseline vs transformers baseline (MTP off both), n=3 warm tok/s.**
vllm_baseline (72.97 / 136.76 / 143.52 / 164.96 on A10 / A100 / B200 / H100)
vs transformers_baseline generic (11.67 / 9.84 / 17.87 / 16.12): that is
**6.3-13.9x**, confirming the vLLM core engine is the dominant value prop
independent of MTP. (The code regime below has an explicit (d) side-by-side.)

vLLM emits 2048 completion tokens (16 reqs x 128, EOS not hit early);
transformers emits ~770 (rejection-sampler interaction with EOS truncates
greedy decoding). System-throughput comparisons divide by wall so remain
valid; per-request e2e latency is the fairer cross-engine view.

**(b) vllm mtp vs baseline, per-prompt (generic), n=3.** Per-prompt token/s
ratio (vllm_mtp / vllm_baseline); idx=0 = the lone cold request, idx=0..7 warm
are the n=3 mean of each prompt's warm repeats. idx=0 is a cold-start outlier:

| GPU | idx=0 (cold) | idx=0 warm | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H100 | **0.44** | 0.98 | 1.04 | 1.03 | 1.06 | 1.03 | 0.90 | 1.05 | 0.99 |
| A100-80GB | **0.71** | 1.06 | 1.06 | 1.04 | 1.08 | 1.04 | 0.92 | 1.15 | 1.04 |
| A10 | **0.90** | 1.39 | 1.49 | 1.35 | 1.42 | 1.32 | 1.26 | 1.48 | 1.35 |
| B200 | 1.17 | 1.29 | 1.38 | 1.35 | 1.35 | 1.26 | 1.28 | 1.37 | 1.29 |

Aggregates (mean ratio across cold idx=0 + 8 warm prompts vs warm only):

| GPU | All (incl cold) mean | Drop idx=0 cold mean | idx warm spread |
|-----|---:|---:|:---|
| H100 | 0.95 | **1.01** | 0.90 to 1.06 |
| A100-80GB | 1.01 | **1.05** | 0.92 to 1.15 |
| A10 | 1.33 | **1.38** | 1.26 to 1.49 |
| B200 | 1.30 | **1.32** | 1.26 to 1.38 |

**Once the cold idx=0 request is excluded, vllm_mtp ties or wins on every GPU.**
A100 lands at 1.05x (warm), H100 at 1.01x (breakeven), A10 a clean 1.38x win,
B200 1.32x. The cold idx=0 cost (vLLM spec-decode kernel JIT on first call) is
the asymmetric drag pulling the headline ratios toward breakeven; it is a
harness artifact, see [Methodology and caveats](#methodology-and-caveats).

**(c) engine-vs-engine per-prompt (transformers_mtp N=4 vs vllm_mtp N=4),
n=3.** Ratio = vllm_mtp tps / transformers_mtp tps, per prompt (idx=0 cold,
idx=0..7 warm n=3 mean).

| GPU | idx=0 (cold) | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|
| H100 | 6.6 | 13.5 | 11.7 | 14.3 | 13.4 | 10.9 | 15.8 | 10.6 |
| A100-80GB | 8.1 | 18.0 | 15.4 | 19.4 | 16.4 | 14.8 | 21.7 | 15.2 |
| A10 | 8.0 | 12.3 | 10.1 | 12.3 | 10.1 | 9.9 | 13.4 | 10.1 |
| B200 | 7.1 | 9.6 | 8.6 | 10.1 | 8.4 | 8.1 | 11.1 | 8.1 |

idx=0 carries cold-start drag on both engines, so the cold ratio is lower than
warm on every GPU. Warm, vLLM is 8.1-11.1x faster on B200 and 9.9-21.7x on
every other GPU.

**(a) per-prompt acceptance (transformers_mtp, generic), n=3 mean.**

| GPU | idx=0 | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 | mean | spread (pp) |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A10 | 33.0% | 44.6% | 33.2% | 38.9% | 34.7% | 27.9% | 40.4% | 36.4% | 36.1% | 16.7 |
| A100-80GB | 32.0% | 47.4% | 33.3% | 36.2% | 37.6% | 28.8% | 39.5% | 32.6% | 35.9% | 18.6 |
| B200 | 33.0% | 50.9% | 34.4% | 40.4% | 32.4% | 23.6% | 39.5% | 34.3% | 36.1% | 27.3 |
| H100 | 33.0% | 43.2% | 26.3% | 41.6% | 32.9% | 29.7% | 39.5% | 34.9% | 35.1% | 16.9 |

Across rows (per-GPU mean): 35.1-36.1%, a 1.0 pp band over four GPUs spanning
Ampere to Blackwell. Down columns (per-prompt across GPUs): each prompt picks
its own band near-independent of GPU (idx=1 "binary search" 43-51%
everywhere; idx=5 "PagedAttn vs FlashAttn" 24-30% everywhere). Per-prompt
variance is the dominant signal (16.7-27.3 pp within a single GPU): code-heavy
text is more predictable token-to-token than high-entropy prose, so the
drafter's argmax matches the target's more often.

**Hole:** vLLM per-prompt acceptance is not available in any regime. vLLM's
OpenAI server emits only process-cumulative `/metrics`
(`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`),
not per-response `usage.speculative_decoding`. vLLM acceptance bands are
bench-wall aggregates only.

### Code prompts

8 leetcode-style code prompts (two_sum, merge_sort, is_balanced, LRUCache,
quicksort, Dijkstra, flatten, binary_tree). Constant N=4 both engines.

**(a) transformers mtp vs baseline + (b) vllm mtp vs baseline (headline, code
vs generic), n=3 warm tok/s.**

| GPU | tx_mtp generic | tx_mtp code | tx code/gen | vllm_mtp generic | vllm_mtp code | vllm code/gen | tx accept gen | tx accept code |
|-----|---------------------:|------------------:|------------:|-----------------:|--------------:|--------------:|--------------:|---------------:|
| A10 | 9.14 | 9.73 | 1.06x | 100.43 | 136.04 | 1.35x | 35.7% | 50.5% |
| A100-80GB | 8.46 | 8.38 | 0.99x | 142.73 | 203.26 | 1.42x | 35.5% | 51.0% |
| B200 | 21.21 | 19.37 | 0.91x | 189.62 | 190.83 | 1.01x | 35.2% | 50.7% |
| H100 | 13.16 | 13.46 | 1.02x | 166.44 | 189.64 | 1.14x | 34.6% | 50.0% |

vLLM acceptance on code (from `/metrics` deltas): A10 52.5%, A100-80GB 52.6%,
B200 52.3%, H100 52.4%, a tight 52.3-52.6% band, ~2 pp above transformers.

**(d) vllm baseline vs transformers baseline (MTP off both, code vs generic),
n=3 warm tok/s.**

| GPU | tx_baseline generic | tx_baseline code | tx code/gen | vllm_baseline generic | vllm_baseline code | vllm code/gen |
|-----|------------------------------:|---------------------------:|----------------------:|----------------------:|-------------------:|--------------:|
| A10 | 11.67 | 10.85 | 0.93x | 72.97 | 66.62 | 0.91x |
| A100-80GB | 9.84 | 8.55 | 0.87x | 136.76 | 133.88 | 0.98x |
| B200 | 17.87 | 16.57 | 0.93x | 143.52 | 167.52 | 1.17x |
| H100 | 16.12 | 16.02 | 0.99x | 164.96 | 157.22 | 0.95x |

The (d) engine gap on code (vllm_baseline / tx_baseline, pure engine, no spec-decode):

| GPU | vllm_baseline code (tok/s) | tx_baseline code (tok/s) | vllm / tx |
|-----------|---------------------------:|-------------------------:|----------:|
| A10 | 66.62 | 10.85 | **6.1x** |
| A100-80GB | 133.88 | 8.55 | **15.7x** |
| B200 | 167.52 | 16.57 | **10.1x** |
| H100 | 157.22 | 16.02 | **9.8x** |

**(a)+(b) MTP/baseline ratio, code vs generic , the flip table (n=3 mean +/-
sd of paired per-run ratios).**

| GPU | tx mtp/baseline gen | tx mtp/baseline code | vllm mtp/baseline gen | vllm mtp/baseline code |
|-----|-----------------------------:|-------------------------------:|----------------------:|-----------------------:|
| A10 | **0.78x +/- 0.08** | 0.90x +/- 0.07 | **1.38x +/- 0.06** | **2.06x +/- 0.41** |
| A100-80GB | 0.93x +/- 0.26 | 1.01x +/- 0.19 | 1.06x +/- 0.14 | **1.53x +/- 0.25** |
| B200 | 1.24x +/- 0.23 | 1.20x +/- 0.30 | 1.41x +/- 0.42 | 1.18x +/- 0.29 |
| H100 | 0.85x +/- 0.24 | **0.83x +/- 0.10** | 1.05x +/- 0.27 | 1.28x +/- 0.43 |

**The robust code result is the vLLM A10 and A100-80GB code wins.** vLLM A10
code is 2.06x +/- 0.41 (range [1.54-2.52]) and vLLM A100-80GB code is 1.53x
+/- 0.25 (range [1.25-1.86]); both stay above 1.0 across all 3 runs. The
directional story holds, code acceptance (~50-52%) is higher than generic
(~35-37%) and lifts mid-tier GPUs, but only A100-80GB among the datacenter
cells crosses breakeven robustly. vLLM B200 code (1.18x +/- 0.29, range
[0.91-1.59]) and H100 code (1.28x +/- 0.43, range [0.96-1.90]) both cross 1.0
and are inconclusive at n=3. On the transformers path no code cell is a robust
win; H100 code is a robust regression (0.83x +/- 0.10).

**(a) per-prompt acceptance (transformers_mtp, code), n=3 mean.** Prompt index
0-7: two_sum, merge_sort, is_balanced, LRUCache, quicksort, Dijkstra,
flatten, binary_tree.

| GPU | idx=0 | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 | mean | spread (pp) |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A10 | 56.3% | 54.4% | 48.9% | 48.8% | 48.7% | 48.0% | 50.6% | 48.3% | 50.5% | 8.3 |
| A100-80GB | 57.2% | 54.4% | 49.4% | 47.7% | 49.1% | 51.5% | 50.6% | 48.3% | 51.0% | 9.5 |
| B200 | 53.1% | 54.4% | 47.7% | 48.8% | 49.1% | 51.5% | 52.4% | 48.3% | 50.7% | 6.7 |
| H100 | 55.1% | 54.4% | 47.7% | 43.0% | 49.1% | 49.7% | 50.6% | 50.0% | 50.0% | 12.1 |

Per-GPU mean 50.0-51.0% (1.0 pp band, hardware-portable). Per-prompt spread
6.7-9.5 pp , much narrower than generic's 16.7-27.3 pp: code prompts
compress per-prompt variance (boilerplate identifiers + recurring syntactic
structure apply more uniformly than the prose set's predictable/high-entropy
mix).

**(c) engine-vs-engine per-prompt (transformers_mtp N=4 vs vllm_mtp N=4,
code), n=3.** Ratio = vllm_mtp tps / transformers_mtp tps (idx=0 cold,
idx=0..7 warm n=3 mean).

| GPU | idx=0 (cold) | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 |
|-----|-------------:|------:|------:|------:|------:|------:|------:|------:|
| H100 | 11.7 | 16.1 | 15.6 | 14.2 | 13.8 | 12.7 | 13.4 | 13.3 |
| A100-80GB | 7.7 | 25.5 | 26.9 | 23.0 | 24.3 | 23.9 | 22.2 | 22.6 |
| A10 | 9.8 | 16.0 | 15.6 | 13.9 | 13.8 | 12.3 | 13.2 | 13.3 |
| B200 | 6.6 | 11.0 | 10.5 | 9.7 | 10.3 | 9.0 | 9.6 | 9.5 |

Warm-prompt mean: H100 14.3x, A100-80GB 24.2x, A10 14.2x, B200 10.0x. The
vLLM-vs-transformers gap is wider on code than generic (10-24x warm, no GPU
below 10x), consistent with vLLM extracting more per-token throughput when
each accepted MTP step carries more useful tokens.

**(c) per-prompt e2e latency (ms): transformers_mtp vs vllm_mtp, code, n=3
mean.**

| GPU | transformers idx=0 (cold, ms) | vllm idx=0 (cold, ms) | transformers warm mean (ms) | vllm warm mean (ms) |
|-----|--------------:|----------------:|---------------------:|-----------------------:|
| A10 | 4079 | 1619 | 4089 | 964 |
| A100-80GB | 4561 | 2267 | 4729 | 644 |
| B200 | 2283 | 1830 | 2121 | 689 |
| H100 | 3105 | 1029 | 3204 | 714 |

idx=0 on vLLM is 1.5-3.5x slower than warm steady-state (cold spec-decode
kernel JIT); transformers shows almost no cold penalty (same monkey-patched
assisted_decoding loop every request). See
[Methodology and caveats](#methodology-and-caveats).

### Structured prompts

8 JSON / YAML / TOML / OpenAPI / GeoJSON skeleton prompts
(`bench/load_runner.py:64-73`). Constant N=4 transformers,
`num_speculative_tokens=4` vLLM. Hypothesis: structured output has the most
predictable token boundaries (delimiters, field names, schema patterns), so
the drafter's argmax matches the target's argmax most often.

**(a) transformers mtp vs baseline + (b) vllm mtp vs baseline (headline,
structured warm tps, idx=0 cold cohort excluded).**

| GPU | (a) tx_const | tx_baseline | (a) tx mtp/baseline | (b) vllm_mtp | vllm_baseline | (b) vllm mtp/baseline | tx accept | vllm accept |
|-----|---------:|------------:|----------------:|---------:|--------------:|------------------:|----------:|------------:|
| A10       | 7.22  | 6.28  | **1.16x +/- 0.11** | 111.55 | 64.15  | **1.79x +/- 0.29** | 55.3% | 57.1%  |
| A100-80GB | 5.83  | 6.11  | 0.97x +/- 0.12 | 156.37 | 147.27 | 1.07x +/- 0.07 | 54.7% | 58.0%  |
| B200      | 11.65 | 8.32  | **1.40x +/- 0.26** | 182.74 | 157.66 | 1.22x +/- 0.37 | 52.5% | 57.0%  |
| H100      | 9.32  | 8.98  | 1.04x +/- 0.01 | 211.05 | 156.56 | 1.37x +/- 0.59 | 52.8% | 57.2%  |

tps columns are the mean of 3 warm runs (idx 1..15, cold idx=0 excluded);
ratio columns are the mean +/- sd of the 3 paired per-run ratios. Bold = 3-run
range entirely on one side of 1.0. The unbolded vLLM cells (A100 1.07, B200
1.22, H100 1.37) all have ranges crossing 1.0 ([0.97-1.12], [0.77-1.67],
[0.72-2.14]).

**Findings (n=3):**

1. **Structured lifts vLLM acceptance to 57-58% (robust).** Up from ~35-37%
   generic and ~50-52% code, in a 5.5 pp band across all four GPUs. Survives
   n=3 cleanly; the load-bearing structured result.
2. **A10 + structured is the only robust structured vLLM win (1.79x +/- 0.29,
   range [1.42-2.13]).** A100-80GB structured is 1.07x +/- 0.07 (touches 0.97),
   i.e. breakeven, not a win.
3. **H100 + structured is inconclusive, not a win.** The n=3 mean is 1.37x, but
   the three runs are [2.14, 1.25, 0.72] (sd 0.59), the widest spread in the
   matrix. Run-to-run cold-cache variance swamps the MTP effect: a single draw
   can land from 2x win to 0.7x regression. Do not cite H100 structured as
   breakeven or win; cite the [0.72-2.14] range.
4. **Transformers structured: only A10 (1.16x) and B200 (1.40x) hold; A100 and
   H100 are breakeven.** A10 1.16x +/- 0.11 and B200 1.40x +/- 0.26 stay above
   1.0; A100 (0.97x +/- 0.12) and H100 (1.04x +/- 0.01) sit at breakeven.

**(c) vllm mtp vs transformers mtp (structured, n=3 means).** Engine gap with MTP on:

| GPU | vllm_mtp (tok/s) | tx_mtp (tok/s) | vllm / tx |
|-----------|-----------------:|---------------:|----------:|
| A10 | 111.55 | 7.22 | **15.5x** |
| A100-80GB | 156.37 | 5.83 | **26.8x** |
| B200 | 182.74 | 11.65 | **15.7x** |
| H100 | 211.05 | 9.32 | **22.6x** |

**(d) vllm baseline vs transformers baseline (structured, n=3 means).** Engine gap with MTP off:

| GPU | vllm_base (tok/s) | tx_base (tok/s) | vllm / tx |
|-----------|------------------:|----------------:|----------:|
| A10 | 64.15 | 6.28 | **10.2x** |
| A100-80GB | 147.27 | 6.11 | **24.1x** |
| B200 | 157.66 | 8.32 | **18.9x** |
| H100 | 156.56 | 8.98 | **17.4x** |

The engine gap (10-27x, MTP on or off) is the robust structured result.

**(a) per-prompt acceptance (transformers_mtp, structured), n=3 mean.** idx 0-7:
JSON object, JSON book array, K8s YAML, HTTP-200 JSON, user-record JSON,
GeoJSON points, TOML crate config, OpenAPI path. Near GPU-portable (greedy
decode + fixed prompts):

| GPU | idx=0 | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 | mean | spread (pp) |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A10 | 94.2% | 44.0% | 62.8% | 57.1% | 74.2% | 41.7% | 48.5% | 56.4% | 59.9% | 52.5 |
| A100-80GB | 94.2% | 44.0% | 62.8% | 54.6% | 74.2% | 37.8% | 52.1% | 56.4% | 59.5% | 56.4 |
| B200 | 94.2% | 44.0% | 62.8% | 54.6% | 56.4% | 37.8% | 48.5% | 58.6% | 57.1% | 56.4 |
| H100 | 94.2% | 44.0% | 62.8% | 54.6% | 56.4% | 37.8% | 52.1% | 56.4% | 57.3% | 56.4 |

idx=0 (a fully-specified JSON object with the field values handed to the model)
accepts 94.2%: short, near-deterministic output the drafter nails. idx=5
(GeoJSON with "random-looking" coordinates) is the lowest at 37.8%. The
per-prompt spread (52.5-56.4 pp) is the widest of the three regimes.

**(a) per-prompt system tok/s (transformers_mtp, structured), n=3 mean.**

| GPU | idx=0 | idx=1 | idx=2 | idx=3 | idx=4 | idx=5 | idx=6 | idx=7 |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|
| A10 | 8.2 | 6.0 | 8.6 | 6.0 | 5.7 | 5.9 | 8.5 | 9.8 |
| A100-80GB | 7.8 | 4.8 | 7.4 | 4.9 | 4.8 | 4.6 | 6.3 | 8.1 |
| B200 | 12.8 | 9.8 | 14.2 | 9.7 | 8.1 | 9.5 | 14.1 | 16.6 |
| H100 | 11.0 | 8.1 | 11.8 | 7.9 | 6.6 | 7.6 | 10.4 | 13.0 |

**Documented holes (structured).** vLLM per-prompt acceptance is unavailable in
every regime (process-cumulative `/metrics` only). Per-prompt MFU/MBU/TPOT
detail is not tabulated.

## Methodology and caveats

Cross-cutting methodology and known data-quality issues, referenced by the
per-regime sections above so result tables stay clean.

**All cells are n=3.** On 2026-06-04 every cell (4 GPUs x 3 regimes x
{transformers,vLLM} x {mtp,baseline}) was re-benched as **3 independent cold
runs** at constant gamma=4, via `deploy/modal/run_n3.sh` (labels
`*_c1_r{1,2,3}`), aggregated by `bench/summarize_n3.py` into mean +/- sd and a
paired-per-run ratio. 144/144 runs succeeded, 0 failures.

**What n=3 changed.** Run-to-run warm-throughput spread is large (typically
10-40% of the mean), dominated by cold-cache / cold-kernel variance on the
serverless containers. As a result **most per-cell MTP/baseline ratios straddle
1.0 across their 3 runs** and are inconclusive: of 24 ratio cells (12 per
engine), only 9 keep their full 3-run range on one side of 1.0: 7 entirely
above (vLLM A10 x3, vLLM A100-80GB code, transformers A10 structured,
transformers B200 structured, transformers H100 structured) and 2 entirely
below (transformers A10 generic 0.78x, transformers H100 code 0.83x). The
remaining 15 cross 1.0 and are breakeven-indistinguishable at n=3. Two findings
DO survive cleanly: acceptance is prompt-fixed (generic 34.6-37.3%, code
49.8-52.6%, structured 52.5-58.0%), and the vLLM engine beats the transformers
path 9-27x with MTP on or off. Report cells as mean +/- sd; treat
"(crosses 1.0)" cells as breakeven-indistinguishable. Raw aggregate:
`metrics/n3_aggregate.json`. Every results table (per-cell, per-prompt,
cold-start) is regenerated from the run dirs by
`bench/rebuild_readme_tables.py`.

**Warm-only and cold-start tax (idx=0 outlier).** Each run warms three short
requests before measurement so cold-start container init is not in the wall
clock; baseline runs are additionally verified to emit `proposed_tokens=0`
for three independent requests before bench launch. Even so, vLLM cold-starts
pay a one-time draft-model speculative-decode kernel JIT compile that the
baseline path does not, and the harness's first *timed* prompt hits it.
`warm_only` aggregates (and every three-regime ratio) exclude idx=0.

The tax is measured on the **system-throughput basis**: cold tok/s =
idx=0 `completion_tokens / (e2e_latency_ms/1000)`, warm tok/s = token-weighted
`sum(completion)/sum(e2e)` over idx=1..15. Both are n=3 mean +/- sd. An earlier
version read the per-request `decode_tokens_per_sec` field, which excludes TTFT
and is length-biased, and so reported warm as *faster* than cold; that was an
artifact of the field. On the system basis the cold request is uniformly the
slow one. The spec-decode (vllm_mtp) cells pay the largest tax:

| vllm_mtp cell | cold idx=0 tps (n=3) | warm tps (n=3) | cold/warm |
|---------------|---------------------:|---------------:|----------:|
| A10 generic       | 63.9 +/- 14.2 | 100.4 +/- 3.4  | 0.64x |
| A100-80GB generic | 50.1 +/- 20.6 | 142.7 +/- 7.0  | 0.35x |
| B200 generic      | 140.1 +/- 38.4 | 189.6 +/- 22.0 | 0.74x |
| H100 generic      | 68.6 +/- 12.3 | 166.4 +/- 22.7 | 0.41x |
| A100-80GB structured | 64.1 +/- 36.2 | 156.4 +/- 5.5 | 0.41x |

The tax is asymmetric: the spec-decode (draft-model) deploy pays the extra
first-request hit from drafter init + spec-decode kernel JIT on top of the
plain warmup, while the no-spec baseline pays only the ordinary cold-start
(weight load + CUDA-graph capture), so vllm_baseline cells sit at cold/warm
0.82-0.98x on generic vs vllm_mtp's 0.35-0.74x. Harness fix tracked.

**B200 utilization constant.** `bench/gpu_probe.py::_ARCH_TABLE` for sm_100
uses 8192 FP16 ops/cycle/SM (~2382 TFLOPS peak), an approximate Blackwell
constant. B200 MFU/MBU numbers inherit it; throughput, TPOT, latency, and
acceptance are independent of it.

## Benchmark

```bash
uv run python -m bench.load_runner \
  --base-url https://<endpoint>.modal.run \
  --api-key "${MODEL_API_KEY}" \
  --requests 64 --concurrency 4 --max-tokens 256 \
  --label mtp_on
```

A/B against single-token decoding: set `NUM_ASSISTANT_TOKENS=0` in `.env`
and restart, then rerun bench with `--label mtp_off` (the engine still
loads the drafter but proposes zero tokens).

Reproduce per regime on Modal (transformers + vLLM A/B):

```bash
# Transformers MTP constant N=4 (PROMPT_SET in {generic,code,structured}):
PROMPT_SET=code bash deploy/modal/run_const.sh H100
PROMPT_SET=code bash deploy/modal/run_const.sh A100-80GB
PROMPT_SET=code bash deploy/modal/run_const.sh A10
PROMPT_SET=code bash deploy/modal/run_const.sh B200

# Transformers baseline (N=0, MTP off):
for GPU in H100 A100-80GB A10 B200; do
  PROMPT_SET=code MODES=baseline bash deploy/modal/run_ab.sh "$GPU"
done

# vLLM A/B (mtp + baseline):
for GPU in H100 A100-80GB A10 B200; do
  PROMPT_SET=code bash deploy/modal/vllm_run_ab.sh "$GPU"
done
```

`run_const.sh`, `run_ab.sh`, and `vllm_run_ab.sh` honor
`PROMPT_SET=<generic|code|structured>`; the label gets a `_<prompt_set>`
suffix automatically (omitted for generic so existing labels do not break).
`run_ab.sh` accepts `MODES="mtp baseline"` (default) or a single arm;
`vllm_run_ab.sh` accepts the same via `VLLM_MODES`.

Modal note: free workspace caps web-functions at 8 deployed apps. Parallel
A/B sweeps across 4 GPUs x 2 engines x 2 modes hit the cap; `modal app stop
-y <name>` between cells frees a slot (the deploy scripts already do this for
the apps they own). The B200 image branch installs `torch==2.9.1+cu128`
(sm_100 kernels); other GPUs use `torch==2.7.0+cu126`.

## Sources

- [Gemma 4 E2B-it model card](https://huggingface.co/google/gemma-4-E2B-it)
- [Gemma 4 E2B-it-assistant (MTP drafter) model card](https://huggingface.co/google/gemma-4-E2B-it-assistant)
- [Google Gemma 4 MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp)
- [Multi-Token Prediction blog post](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [Speculative decoding paper (Leviathan et al. 2023)](https://arxiv.org/abs/2211.17192)
