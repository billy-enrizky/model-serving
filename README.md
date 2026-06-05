# model-serving

Modal-deployed serving + benchmarks for inference acceleration methods on
small language models. Each method lives in its own subdirectory with a
self-contained Modal app, A/B sweep scripts, and result artifacts.

## Methods

- [`multi-token-prediction/`](multi-token-prediction/) , Gemma 4 E2B-it +
  drafter, transformers MTP engine and vLLM v0.21.0 across four NVIDIA GPUs
  (A10, A100-80GB, H100, B200); three prompt regimes (generic, code,
  structured). Headline result: MTP/baseline ratio is **regime-dependent**:
  it depends on `(engine, GPU, prompt-set)`.

https://github.com/user-attachments/assets/4c07e70d-1295-435e-81df-cea1e6cd74eb


Future methods (e.g. `dflash/`) follow the same shape: subdir +
Modal app + A/B + measured artifacts.

## Quickstart

```bash
git clone https://github.com/billy-enrizky/model-serving.git
cd model-serving/multi-token-prediction
cp .env.example .env
# fill HF_TOKEN, MODEL_API_KEY (any opaque string)
uv sync --extra bench

# One-time Modal bootstrap (token, secrets, gemma-models volume):
bash deploy/modal/setup_modal.sh
bash deploy/modal/warm_weights.sh

# Run a sweep on a target GPU:
GPU=h100 bash deploy/modal/run_const.sh
GPU=h100 bash deploy/modal/vllm_run_ab.sh
```

Results land in `multi-token-prediction/metrics/runs/`. Per-method
README documents which cells back which numbers.

## What this repo is

- A reproducible bench stack on Modal. Every cell in every results
  table traces back to a `metrics/runs/<ts>_<label>/result.json` file
  that this repo ships.
- A reference implementation of speculative decoding A/B comparisons
  across two engines and four GPU classes.

## What this repo is not

- A general-purpose serving framework. The Modal apps are pinned to
  the Gemma 4 E2B-it target and its drafter; bring-your-own-model
  requires editing `deploy/modal/*.py`.
- A claim that speculative decoding is universally good. The headline
  finding is the opposite: same model + same gamma, MTP wins on some
  GPU x engine x prompt combinations and loses on others.
