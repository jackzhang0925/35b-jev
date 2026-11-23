"""Harbor task  <->  Tinker RL environment bridge.

Why this file exists
--------------------
Harbor ships `harbor.rl` in the cookbook, but the installed Harbor 0.26.0 has no
`rl` subpackage, so the bridge is ours. It is also where the one architectural
fact of this project lives:

    RL updates need the *sampled tokens and their logprobs*, so training
    rollouts cannot go through Codex CLI (a separate process that talks HTTP).
    They run through a local agent loop that calls Tinker's SamplingClient.
    Codex CLI is used for *evaluation* -- that is the number we report.

That leaves a train/eval mismatch, which is measured, not hand-waved: the same
subset is scored with both harnesses (see eval/run_eval.py and loop.py --internal).

Shape
-----
    env = HarborTask(task_dir)          # one container per task (Modal sandbox by default)
    obs = env.reset()                   # -> instruction text
    for turn in range(max_turns):
        prompt_tokens = render(obs, tools)
        seq = sample(training_client, prompt_tokens)   # tokens + logprobs
        action = parse_tool_call(seq)
        obs = env.exec(action.command)
    reward = env.grade()                # verifier: 0.0 / 1.0

The container backend lives in `envs.py` and is chosen by `RL_ENV=modal|docker`
(default `modal`, because Docker is not installed on the rollout host). `HarborTask`
keeps its old four-method surface and now delegates to that `TaskEnv`; nothing above
it -- `rollout`, `loop.py`, `train_probe.py` -- changed.

Everything marked TODO needs the Tinker key or a Harbor sandbox to verify; the
API calls are pinned to the Tinker docs, not guessed.
"""
from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION = re.compile(r"<function=([A-Za-z0-9_.\-]+)>")
_PARAM = re.compile(r"<parameter=([A-Za-z0-9_\-]+)>(.*?)(?:</parameter>|(?=</function>))", re.S)

try:  # rl/loop.py and rl/train_probe.py put rl/ on sys.path and import this as `bridge`
    from envs import TaskEnv, make_env
except ImportError:  # imported as `rl.bridge`
    from .envs import TaskEnv, make_env

MAX_TURNS = int(os.getenv("RL_MAX_TURNS", "40"))  # 12 is plenty for polyglot

# The tool surface is deliberately the smallest set that Codex also exposes, so a
# policy learned here transfers to the eval harness. Extend only if Codex does.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the task container and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


@dataclass
class HarborTask:
    """One Terminal-Bench task, driven through a `TaskEnv` (Modal sandbox by default).

    The docker run/exec that used to live here is now `envs.LocalDockerTask`; the default
    backend is `envs.ModalTask`, selected with `RL_ENV=modal|docker` (see `envs.make_env`).
    The public surface (`reset`/`exec`/`grade`/`close`, plus `cfg`/`name`/`image`) is
    unchanged, so `rollout`, `loop.py` and `train_probe.py` did not have to change.
    """

    task_dir: Path
    container: str | None = None  # docker container id when RL_ENV=docker, else None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        with (self.task_dir / "task.toml").open("rb") as f:
            self.cfg = tomllib.load(f)
        self.name = self.cfg["task"]["name"]
        # data/train/aider-polyglot/* have no docker_image: they ship a Dockerfile instead,
        # which envs.ModalTask builds remotely (docker only ever saw registry images).
        self.image = (self.cfg.get("environment") or {}).get("docker_image")
        self.env: TaskEnv = make_env(self.task_dir)
        self.backend = type(self.env).__name__

    # -- lifecycle: delegate to the TaskEnv backend ------------------------
    def reset(self) -> str:
        self.history = []
        instruction = self.env.reset()
        self.container = getattr(self.env, "container", None)
        return instruction

    def exec(self, command: str) -> str:
        return self.env.exec(command)

    def grade(self) -> float:
        """Run the task's own verifier; 1.0 iff it passes (envs.verdict)."""
        return self.env.grade()

    def close(self) -> None:
        self.env.close()
        self.container = None


# ---------------------------------------------------------------------------
# Rollout: sample -> act -> record. Returns Datums ready for forward_backward.
# ---------------------------------------------------------------------------
async def rollout(
    task_dir: Path,
    sampling_client,
    tokenizer,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> tuple[list, float]:
    import tinker

    env = HarborTask(task_dir)
    instruction = env.reset()
    messages = [
        {"role": "system", "content": "You are a shell agent. Call shell() to act. Stop when the task is done."},
        {"role": "user", "content": instruction},
    ]
    datums, reward = [], 0.0
    try:
        for _ in range(MAX_TURNS):
            prompt_text = tokenizer.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=True, tokenize=False
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            result = await sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature),
            )
            seq = result.sequences[0]
            reply = tokenizer.decode(seq.tokens, skip_special_tokens=True)

            # TODO(parser): use the model's real tool-call format once FT-0 shows
            # what Qwen3.5-9B actually emits; a text fallback is intentional.
            command = _extract_shell(reply)
            if command is None:
                break
            observation = env.exec(command)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"<observation>\n{observation}\n</observation>"},
            ]
            n_prompt = len(prompt_tokens) - 1
            full = prompt_tokens + seq.tokens
            datums.append(
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full[1:],
                        "logprobs": [0.0] * n_prompt + list(seq.logprobs),
                        "advantages": [0.0] * n_prompt + [0.0] * len(seq.tokens),
                    },
                )
            )
        reward = env.grade()
    finally:
        env.close()

    # GRPO: the advantage is filled in by the caller once the group's rewards are known.
    for d in datums:
        d.loss_fn_inputs["_reward"] = reward
    return datums, reward


def parse_tool_call(reply: str) -> dict | None:
    """Parse the tool-call syntax Qwen3.5 actually emits.

    Observed live (rl/debug_rollout.py, 2026-11-20): the model writes

        <tool_call>
        <function=shell>
        <parameter=command>
        cat transpose.py
        </parameter>
        </function>
        </tool_call>

    Sampling raw tokens from Tinker gives us this text. In the OpenAI-compatible
    path the server parses it into `tool_calls`; when we sample ourselves we have
    to do it here, otherwise every rollout ends at turn 0 with reward 0.
    """
    if "<function=" not in reply:
        return None
    for block in _TOOL_BLOCK.findall(reply) or [reply]:
        name = _FUNCTION.search(block)
        if not name:
            continue
        args = {k: v.strip() for k, v in _PARAM.findall(block)}
        return {"name": name.group(1), "arguments": args}
    return None


def _extract_shell(reply: str) -> str | None:
    """Pull a command out of the model's tool call, a JSON call, or a fenced block."""
    reply = reply.strip()
    call = parse_tool_call(reply)
    if call and isinstance(call["arguments"].get("command"), str):
        return call["arguments"]["command"]
    if reply.startswith("{"):
        try:
            call = json.loads(reply)
            args = call.get("arguments") or call.get("function", {}).get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                return args["command"]
        except (json.JSONDecodeError, AttributeError):
            pass
    if "```" in reply:
        block = reply.split("```")[1]
        return block.split("\n", 1)[1].strip() if "\n" in block else None
    return None

