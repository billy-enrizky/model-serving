# Modal H100 Deployment

Serves `server.api:app` (Gemma 4 E2B-it + MTP drafter) as a Modal ASGI web
endpoint on a single H100, scale-to-zero. Weights are cached on a persistent
Modal volume so cold starts skip the 9.7 GiB HuggingFace download.

## Files

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal app: image, secrets, volume, `fastapi_app`, `warm_weights` |
| `setup_modal.sh` | Idempotent: sets Modal token, creates `hf-token` + `mtp-api-key` secrets, creates `hf-cache` volume |
| `warm_weights.sh` | Pre-downloads target + drafter into the volume (one-shot) |
| `deploy.sh` | `modal deploy ...`, captures URL into `.state/url` |
| `smoke_test.sh` | Hits `/healthz`, `/v1/models`, `/v1/chat/completions` |
| `.state/` | Generated, gitignored: holds `url` and `api_key` for the smoke test |

## First-time setup

```bash
# 1) Install modal CLI into the project venv
uv pip install modal

# 2) Put credentials in .env at repo root:
#    MODAL_TOKEN_ID=ak-...
#    MODAL_TOKEN_SECRET=as-...
#    HF_TOKEN=hf_...
#    (MODEL_API_KEY optional; auto-generated if absent)

# 3) Bootstrap Modal account (token, 2 secrets, volume)
bash deploy/modal/setup_modal.sh

# 4) Pre-download weights into the hf-cache volume
bash deploy/modal/warm_weights.sh

# 5) Deploy
bash deploy/modal/deploy.sh

# 6) Verify
bash deploy/modal/smoke_test.sh
```

## Re-deploy after code change

```bash
bash deploy/modal/deploy.sh
```

`add_local_dir` re-uploads server source on every deploy. Image layers are cached.

## Configuration knobs

Tune in `modal_app.py`:

- `GPU_TYPE` (default `H100`). For cheaper testing: `A10G`, `L4`.
- `min_containers=0` (scale-to-zero) vs `1` (always warm).
- `scaledown_window` (sec) idle window before container shuts down.
- `max_containers` cap on horizontal scale.
- `NUM_ASSISTANT_TOKENS` env baked into image; rebuild to change. Set to `0`
  for the no-MTP A/B baseline (engine omits `assistant_model=`).

## Why concurrency=1 per container

`MTPEngine.generate()` runs behind a single global `Lock`, and the
acceptance-counter monkey-patch on
`AssistedCandidateGenerator.update_candidate_strategy` is process-global.
We let Modal scale by adding containers, not by adding threads inside one.

## Endpoints

```
GET  /healthz                  no auth
GET  /metrics                  no auth (Prometheus)
GET  /v1/models                X-API-Key
POST /v1/chat/completions      X-API-Key  (stream + non-stream)
```

Auth: `X-API-Key: <MODEL_API_KEY>` or `Authorization: Bearer <MODEL_API_KEY>`.

`POST /v1/chat/completions` returns
`usage.speculative_decoding={accepted_tokens, proposed_tokens, acceptance_rate}`
in non-streaming responses, so the bench harness can compute acceptance per
request.
