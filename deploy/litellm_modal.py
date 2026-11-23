"""Deploy a LiteLLM proxy on Modal that gives Codex CLI the /v1/responses
endpoint Tinker does not have.

Why: Codex CLI dropped `wire_api = "chat"` (openai/codex#7782) and now always
speaks the Responses API, while Tinker only serves /chat/completions. LiteLLM
bridges /v1/responses -> /v1/chat/completions for providers that have no native
Responses endpoint (`openai_like/`), so Codex -> LiteLLM -> Tinker works.

    modal secret create tinker-secrets TINKER_API_KEY=... LITELLM_MASTER_KEY=...
    modal deploy deploy/litellm_modal.py

The printed URL is what goes into config/codex_tinker_litellm.toml as base_url
(append /v1) and into the agent env as LITELLM_API_KEY.
"""
from __future__ import annotations

import os
import subprocess

import modal

app = modal.App("qwen9b-litellm")

CONFIG = """
model_list:
  - model_name: qwen9b
    litellm_params:
      # custom_openai/ is a chat-completions-only provider, so LiteLLM bridges
      # /v1/responses -> /v1/chat/completions instead of forwarding it upstream.
      model: custom_openai/Qwen/Qwen3.5-9B
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      # Tinker takes a boolean for this model; strings are rejected with HTTP 400
      drop_params: true
    model_info:
      # 'mode: chat' is what makes LiteLLM bridge /v1/responses -> /v1/chat/completions
      mode: chat

  - model_name: oss120b
    litellm_params:
      model: custom_openai/openai/gpt-oss-120b
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      drop_params: true
    model_info:
      mode: chat
  - model_name: qwen35b
    litellm_params:
      model: custom_openai/Qwen/Qwen3.5-35B-A3B
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      drop_params: true
    model_info:
      mode: chat

  - model_name: qwen36
    litellm_params:
      model: custom_openai/Qwen/Qwen3.6-35B-A3B
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      drop_params: true
    model_info:
      mode: chat

  - model_name: qwen36rl
    litellm_params:
      model: custom_openai/Qwen/Qwen3.6-35B-A3B;tinker://c970ccd4-f1c3-55e9-b565-e178feb3837b:train:0/sampler_weights/jev-final-1795297281
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      drop_params: true
    model_info:
      mode: chat
  - model_name: qwen38
    litellm_params:
      model: custom_openai/Qwen/Qwen3.8-27B
      api_base: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
      api_key: os.environ/TINKER_API_KEY
      drop_params: true
    model_info:
      mode: chat

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  disable_spend_logs: true

litellm_settings:
  drop_params: true
  num_retries: 2
  request_timeout: 600
"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("litellm[proxy]>=1.63.8")
    .run_commands("mkdir -p /root/litellm")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tinker-secrets")],
    timeout=3600,
    scaledown_window=1800,
)
@modal.web_server(port=4000, startup_timeout=600)
def serve() -> None:
    with open("/root/litellm/config.yaml", "w") as f:
        f.write(CONFIG)
    subprocess.Popen(
        [
            "litellm",
            "--config", "/root/litellm/config.yaml",
            "--host", "0.0.0.0",
            "--port", "4000",
            "--num_workers", "1",
        ],
        env={**os.environ},
    )


@app.local_entrypoint()
def main() -> None:
    print("run: modal deploy deploy/litellm_modal.py")


