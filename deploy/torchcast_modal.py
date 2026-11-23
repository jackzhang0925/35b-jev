"""Serve Torchcast Decision 31B on Modal (A100-80GB) behind a small gateway.

Layout inside the container:
    127.0.0.1:8011   torchcast_decision.server (bundled, required -- vLLM/llama.cpp
                     apply a causal mask and give different probabilities)
    0.0.0.0:8012     deploy/torchcast_gateway.py (FastAPI) <- this is what Modal exposes

Why the gateway: the deployed web_server answered /health instantly but hung on
POST /v1/systemone, while the same POST to loopback inside the container returned
in 0.10 s (see deploy/torchcast_diag.py). So the model was never the problem.

    modal deploy deploy/torchcast_modal.py
    modal app stop -y torchcast-decision        # stop paying for the GPU

GPU: the model needs >=80 GB (62 GB of bf16 weights). A100-80GB schedules on this
account; H100 also works. FA3 is Hopper-only, so A100 falls back to SDPA -- fine
at the measured 84-96 ms.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

import modal

app = modal.App("torchcast-decision")

REVISION = "v1.0.0"
MODEL_DIR = "/weights/torchcast-decision-31b"
UPSTREAM_PORT = 8011
GATEWAY_PORT = 8012

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "transformers==5.17.0",
        "kernels==0.17.2",
        "accelerate==1.15.0",
        "safetensors",
        "tokenizers",
        "jinja2",
        "numpy",
        "huggingface_hub[cli]",
        "fastapi",
        "uvicorn",
    )
    .add_local_file(
        "deploy/torchcast_gateway.py", "/root/torchcast_gateway.py", copy=True
    )
)

weights = modal.Volume.from_name("torchcast-weights", create_if_missing=True)


def _ensure_weights() -> None:
    if os.path.exists(os.path.join(MODEL_DIR, "decision_config.json")):
        print("weights already on the volume", flush=True)
        return
    print("downloading weights (~62 GB, one time)", flush=True)
    subprocess.run(
        ["hf", "download", "torchcast-ai/torchcast-decision-31b",
         "--revision", REVISION, "--local-dir", MODEL_DIR],
        check=True,
    )
    weights.commit()


def _wait_for(url: str, timeout_s: int, label: str) -> bool:
    started = time.time()
    while time.time() - started < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                print(f"{label} up after {time.time() - started:.0f}s: "
                      f"{resp.read().decode()[:200]}", flush=True)
                return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    print(f"{label} never came up", flush=True)
    return False


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/weights": weights},
    timeout=3600,
    scaledown_window=600,
)
@modal.web_server(port=GATEWAY_PORT, startup_timeout=1800)
def serve() -> None:
    _ensure_weights()
    env = {**os.environ, "TC_GRAPHS": "0", "HF_HUB_OFFLINE": "1"}

    subprocess.Popen(
        [sys.executable, "-m", "torchcast_decision.server",
         "--model", ".", "--host", "127.0.0.1", "--port", str(UPSTREAM_PORT)],
        cwd=MODEL_DIR, env=env,
    )
    if not _wait_for(f"http://127.0.0.1:{UPSTREAM_PORT}/health", 600, "torchcast server"):
        return

    # warm the model once so the first user request does not pay the eager-path cost
    try:
        body = (b'{"state":"warm up","questions":{"q":{"type":"noul",'
                b'"instructions":"Is this text English?"}}}')
        req = urllib.request.Request(
            f"http://127.0.0.1:{UPSTREAM_PORT}/v1/systemone", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"warm-up ok: {resp.read().decode()[:200]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warm-up failed (continuing): {exc}", flush=True)

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "torchcast_gateway:app",
         "--host", "0.0.0.0", "--port", str(GATEWAY_PORT), "--log-level", "warning"],
        cwd="/root", env=env,
    )
    print(f"gateway on :{GATEWAY_PORT} -> upstream :{UPSTREAM_PORT}", flush=True)


@app.local_entrypoint()
def main() -> None:
    print("modal deploy deploy/torchcast_modal.py")
