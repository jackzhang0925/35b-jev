"""Group-relative RL loop: N parallel rollouts of one task -> Datums -> one step.

Rollout shape is bridge.rollout's (sample -> parse tool call -> exec -> grade), but the
env factory, tokenizer and sampler are injected, so `python rl/loop.py --dry-run` runs
this exact code path with a fake container and a fake policy: no Tinker, no Docker, no
key. Advantage is group-relative (GRPO): reward - mean(rewards of the group). A group
with no reward spread carries no signal, so it is dropped, not stepped on with zeros.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge import MAX_TURNS, TOOLS, _extract_shell  # noqa: E402  (same directory)

SYSTEM = "You are a shell agent. Call shell() to act. Stop when the task is done."


@dataclass
class Turn:
    """One sampled step: the context it saw and what it emitted."""

    prompt: list[int]
    tokens: list[int]
    logprobs: list[float]


# -------------------------------------------------------------------------- rollout
async def rollout_once(
    env, sampler, tokenizer, temperature: float = 1.0, max_tokens: int = 4096
) -> tuple[list[Turn], float]:
    """One episode: reset -> (sample | exec) x turns -> grade. Mirrors bridge.rollout.

    Env calls run in a thread so `asyncio.gather` over a group is genuinely concurrent.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": await asyncio.to_thread(env.reset)},
    ]
    turns: list[Turn] = []
    try:
        for _ in range(MAX_TURNS):
            prompt_text = tokenizer.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                # Rollouts run with thinking OFF: measured live, the 9B spends
                # 2-3k tokens of reasoning before its first tool call, which makes
                # an episode cost ~50x more and still often truncates. Evaluation
                # (Codex) keeps thinking on, so this is a deliberate train/eval
                # difference -- set RL_THINK=1 to match eval instead.
                enable_thinking=os.getenv("RL_THINK") == "1",
            )
            prompt = list(tokenizer.encode(prompt_text, add_special_tokens=False))
            tokens, logprobs = await sampler.sample(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
            reply = tokenizer.decode(tokens, skip_special_tokens=True)
            command = _extract_shell(reply)
            if command is None:
                break
            observation = await asyncio.to_thread(env.exec, command)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"<observation>\n{observation}\n</observation>"},
            ]
            turns.append(Turn(prompt=prompt, tokens=list(tokens), logprobs=list(logprobs)))
        reward = float(await asyncio.to_thread(env.grade))
    finally:
        await asyncio.to_thread(env.close)
    return turns, reward


def advantages(rewards: list[float]) -> list[float] | None:
    """reward - mean(group); None when the group has no spread."""
    if len(rewards) < 2 or max(rewards) == min(rewards):
        return None
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def build_datums(turns: list[Turn], advantage: float) -> list[dict]:
    """One Datum per turn, in bridge.rollout's layout: predict token i+1 from i.

    Positions that predict prompt tokens are masked (logprob 0, advantage 0); only
    sampled tokens get the group advantage, so a rollout's turns share one scalar.
    """
    out = []
    for turn in turns:
        full = list(turn.prompt) + list(turn.tokens)
        if len(full) < 2:
            continue
        n_prompt, sampled = max(len(turn.prompt) - 1, 0), len(turn.tokens)
        logprobs = (list(turn.logprobs) + [0.0] * sampled)[:sampled]
        out.append({  # fmt: skip
            "model_input": full[:-1],
            "target_tokens": full[1:],
            "logprobs": [0.0] * n_prompt + logprobs,
            "advantages": [0.0] * n_prompt + [advantage] * sampled,
        })
    return out


class SamplingAdapter:
    """Tinker SamplingClient -> `await sample(tokens) -> (tokens, logprobs)`."""

    def __init__(self, sampling_client) -> None:
        self.client = sampling_client

    async def sample(self, tokens, *, temperature: float = 1.0, max_tokens: int = 4096):
        import tinker

        result = self.client.sample(
            prompt=tinker.ModelInput.from_ints(list(tokens)),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature),
        )
        if hasattr(result, "__await__"):  # tinker ships both sync and async clients
            result = await result
        seq = result.sequences[0]
        return list(seq.tokens), list(seq.logprobs)


def to_tinker_datums(datums: list[dict]) -> list:
    """Plain dicts -> tinker.Datum. Lazy import keeps --dry-run free of Tinker."""
    import tinker

    return [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints(d["model_input"]),
            loss_fn_inputs={
                "target_tokens": d["target_tokens"],
                "logprobs": d["logprobs"],
                "advantages": d["advantages"],
            },
        )
        for d in datums
    ]


async def run_batch(
    tasks: list[str], sampler, tokenizer, make_env, group=4, temperature=1.0, max_tokens=4096
) -> tuple[list[dict], list[float], int]:
    """One batch: `group` parallel rollouts per task -> (datums, rewards, skipped)."""
    datums: list[dict] = []
    rewards: list[float] = []
    skipped = 0
    for task in tasks:
        rollouts = await asyncio.gather(*(
            rollout_once(make_env(task), sampler, tokenizer, temperature, max_tokens)
            for _ in range(group)
        ))
        group_rewards = [reward for _, reward in rollouts]
        advs = advantages(group_rewards)
        rewards += group_rewards
        if advs is None:
            skipped += 1
            continue
        for (turns, _), advantage in zip(rollouts, advs):
            datums += build_datums(turns, advantage)
    return datums, rewards, skipped


# ---------------------------------------------------------------------------- fakes
STOP = 0


class FakeTokenizer:
    """Deterministic ids in, a canned shell reply out."""
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)[-512:]
    def encode(self, text, add_special_tokens=False):
        return [(ord(ch) % 251) + 1 for ch in text]
    def decode(self, tokens, skip_special_tokens=True):
        return "all done." if not tokens or tokens[0] == STOP else "```sh\nls /tmp\n```"


class FakeSampler:
    """Stands in for a Tinker SamplingClient; the fake policy gives up on long history."""
    async def sample(self, tokens, *, temperature: float = 1.0, max_tokens: int = 4096):
        if len(tokens) > 500:  # deterministic stop; the canned history window is 512 ids
            return [STOP], [0.0]
        ids = [100 + i % 50 for i in range(24)]
        return ids, [-(0.10 + 0.01 * i) for i in range(len(ids))]


class FakeEnv:
    """No container: text in, text out, and a grade that alternates so groups spread."""
    def __init__(self, index: int = 0) -> None:
        self.index = index
    def reset(self) -> str:
        return "Fake task: find the flag under /tmp and write it to /app/answer.txt."
    def exec(self, command: str) -> str:
        return f"$ {command}\n(no output)"
    def grade(self) -> float:
        return 1.0 if self.index % 2 == 0 else 0.0
    def close(self) -> None:
        pass


def fake_envs():
    """Env factory that numbers instances, so 2 of every 4 rollouts pass."""
    counter = itertools.count()
    return lambda task: FakeEnv(next(counter))


def harbor_envs(root: Path):
    """Env factory for real rollouts: one Harbor/Docker task per container."""
    from bridge import HarborTask

    return lambda task: HarborTask(root / task.split("/")[-1])


if __name__ == "__main__":
    if "--dry-run" not in sys.argv:
        raise SystemExit("only the --dry-run self-check lives here; real training is rl/train_probe.py")
    datums, rewards, skipped = asyncio.run(run_batch(
        ["terminal-bench/fake-a", "terminal-bench/fake-b"], FakeSampler(), FakeTokenizer(), fake_envs()
    ))
    first = datums[0]
    print(f"dry run ok: datums={len(datums)} rewards={rewards} skipped_groups={skipped}")
    print(f"  datum[0]: tokens={len(first['model_input'])} "
          f"advantages={sorted({round(a, 2) for a in first['advantages']})}")

