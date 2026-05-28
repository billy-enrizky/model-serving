"""Modal deployment of vLLM serving Gemma 4 E2B with MTP.

Per-GPU app, parametrized by MTP_GPU + VLLM_MODE env.

VLLM_MODE=mtp      -> --speculative-config with N=4
VLLM_MODE=baseline -> no spec config

Image: vllm/vllm-openai:v0.21.0. v0.21.0 is the first vLLM release
containing the Gemma 4 MTP path (PR #41745, merged 2026-05-06; v0.21.0
tagged 2026-05-15). v0.22.0 was not yet released as of 2026-05-28; the
"latest" tag points at v0.21.0 today. Pin explicitly for reproducibility.
"""

from __future__ import annotations
import os
import modal

GPU_TYPE = os.environ.get("MTP_GPU", "H100")
VLLM_MODE = os.environ.get("VLLM_MODE", "mtp")  # mtp | baseline
VLLM_VERSION = os.environ.get("VLLM_VERSION", "v0.21.0")

# App name uniqueness: vllm-gemma-<gpu>-<mode>
_GPU_SUFFIX = GPU_TYPE.lower().replace("-", "").replace("!", "").replace("+", "plus")
APP_NAME = f"vllm-gemma-{_GPU_SUFFIX}-{VLLM_MODE}"

# vLLM official image. Add Python so Modal can install its agent.
image = (
    modal.Image.from_registry(
        f"vllm/vllm-openai:{VLLM_VERSION}",
        add_python="3.12",
    )
    .entrypoint([])
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "VLLM_DO_NOT_TRACK": "1",
            # Propagate deploy-time mode into the container; otherwise
            # serve() at cold-start reads os.environ inside the container
            # (which has no VLLM_MODE) and defaults to "mtp".
            "VLLM_MODE": VLLM_MODE,
            "MTP_GPU": GPU_TYPE,
        }
    )
)

app = modal.App(APP_NAME, image=image)
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

TIMEOUT_SECONDS = 60 * 30
SCALEDOWN_SECONDS = 60 * 5


@app.function(
    gpu=GPU_TYPE,
    image=image,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("mtp-api-key"),
    ],
    volumes={"/cache/hf": hf_volume},
    timeout=TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_SECONDS,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    import os
    import subprocess

    cmd = [
        "vllm", "serve", "google/gemma-4-E2B-it",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--api-key", os.environ["MODEL_API_KEY"],
        "--served-model-name", "gemma-4-E2B-it",
    ]
    if VLLM_MODE == "mtp":
        cmd += [
            "--speculative-config",
            '{"model":"google/gemma-4-E2B-it-assistant","num_speculative_tokens":4}',
        ]
    subprocess.Popen(cmd)
