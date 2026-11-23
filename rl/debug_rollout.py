"""Debug one real rollout: what does the 9B actually emit, and does it parse?

    python rl/debug_rollout.py --task data/train/aider-polyglot/polyglot_python_transpose

Prints, per turn: the model's raw text (truncated), the extracted command, the
observation, and the final reward. Costs a few cents.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402
import envs  # noqa: E402

MAX_TURNS = 6
SYSTEM = (
    "You are a shell agent working in a Linux container. "
    'Call the shell tool with {"command": "..."} as a structured function call. '
    "Never write a tool call as text. Work in small steps and stop when the "
    "task's own tests pass."
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="data/train/aider-polyglot/polyglot_python_transpose")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--no-think", action="store_true", help="disable Qwen thinking in the rollout")
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    model = "Qwen/Qwen3.5-9B"
    service = tinker.ServiceClient()
    training = await service.create_lora_training_client(base_model=model, rank=16)
    sampling = training.save_weights_and_get_sampling_client()
    sampling = await sampling if asyncio.iscoroutine(sampling) else sampling
    if hasattr(sampling, "result"):
        sampling = sampling.result()
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"tokenizer: {type(tokenizer).__name__}")

    env = envs.make_env(Path(args.task))
    instruction = env.reset()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    print(f"\n=== instruction (first 300) ===\n{instruction[:300]}")

    for turn in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(
            messages, tools=bridge.TOOLS, add_generation_prompt=True, tokenize=False,
            enable_thinking=not args.no_think,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        result = await sampling.sample(
            prompt=tinker.ModelInput.from_ints(tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.7),
        )
        seq = result.sequences[0]
        reply = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        command = bridge._extract_shell(reply)
        print(f"\n--- turn {turn} ---")
        print(f"raw   : {reply[:400]!r}")
        print(f"parsed: {command!r}")
        if command is None:
            break
        obs = env.exec(command)
        print(f"obs   : {obs[:300]!r}")
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"<observation>\n{obs}\n</observation>"},
        ]

    print(f"\nreward: {env.grade()}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

