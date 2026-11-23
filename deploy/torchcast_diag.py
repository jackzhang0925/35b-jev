"""Diagnose the Torchcast server *inside* the container, bypassing Modal's proxy.

The deployed web_server answers /health instantly but hangs on /v1/systemone.
That could be (a) the server never finishing the forward pass, (b) something in
Modal's HTTP proxy, or (c) the request shape. This runs the server on loopback
inside the container and talks to it directly, so the two are distinguishable.

    modal run deploy/torchcast_diag.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

import modal

app = modal.App("torchcast-diag")

MODEL_DIR = "/weights/torchcast-decision-31b"
REVISION = "v1.0.0"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        "transformers==5.17.0", "kernels==0.17.2", "accelerate==1.15.0",
        "safetensors", "tokenizers", "jinja2", "numpy", "huggingface_hub[cli]",
    )
)

weights = modal.Volume.from_name("torchcast-weights", create_if_missing=True)

BODY = {
    "state": "A customer reports two charges for one order.",
    "questions": {
        "decision": {
            "type": "choice",
            "instructions": "Choose the support team.",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Product malfunctions",
                "other": "Other requests",
            },
        }
    },
}


def _ensure_weights() -> None:
    if os.path.exists(os.path.join(MODEL_DIR, "decision_config.json")):
        return
    subprocess.run(
        ["hf", "download", "torchcast-ai/torchcast-decision-31b",
         "--revision", REVISION, "--local-dir", MODEL_DIR],
        check=True,
    )
    weights.commit()


@app.function(image=image, gpu="A100-80GB", volumes={"/weights": weights}, timeout=1800)
def diag() -> str:
    _ensure_weights()
    out = []
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    out.append("gpu: " + gpu.stdout.strip())

    env = {**os.environ, "TC_GRAPHS": "0", "HF_HUB_OFFLINE": "1"}
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "torchcast_decision.server",
         "--model", ".", "--host", "127.0.0.1", "--port", "8011"],
        cwd=MODEL_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    healthy = False
    for _ in range(180):
        if proc.poll() is not None:
            out.append("server exited early: " + (proc.stdout.read()[-2000:] if proc.stdout else ""))
            return "\n".join(out)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8011/health", timeout=5) as r:
                out.append(f"healthy after {time.time() - t0:.1f}s: {r.read().decode()[:300]}")
                healthy = True
                break
        except Exception:
            time.sleep(2)
    if not healthy:
        out.append("never became healthy")
        proc.kill()
        return "\n".join(out)

    for label, timeout in (("first request", 600), ("second request", 120)):
        t1 = time.time()
        req = urllib.request.Request(
            "http://127.0.0.1:8011/v1/systemone",
            data=json.dumps(BODY).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode()
            out.append(f"{label}: OK in {time.time() - t1:.2f}s -> {body[:400]}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"{label}: FAILED after {time.time() - t1:.1f}s: {type(exc).__name__}: {exc}")
            break

    proc.kill()
    return "\n".join(out)


@app.local_entrypoint()
def main() -> None:
    print(diag.remote())
