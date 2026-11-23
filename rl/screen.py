"""Screen tasks for RL variance: how often does the base model solve each one?

GRPO learns nothing from a group where every rollout scores the same, so before
spending on training we measure per-task pass rate and keep only the tasks whose
rate lands in a learnable band.

    python rl/screen.py --tasks eval/train16_ext.txt --group 4 --out rl/screen.json

Each task gets `group` rollouts with the same loop settings used for training
(no-think, capped turns). Prints a table and the selected band.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop  # noqa: E402

LO, HI = 0.25, 0.75


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--data", default="data/train/aider-polyglot")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--ckpt", default=None, help="tinker:// checkpoint; omit for the base model")
    ap.add_argument("--out", default="rl/screen.json")
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    service = tinker.ServiceClient()
    if args.ckpt:
        sampling = service.create_sampling_client(model_path=args.ckpt)
    else:
        sampling = service.create_sampling_client(base_model=args.model)
    if asyncio.iscoroutine(sampling):
        sampling = await sampling
    if hasattr(sampling, "result"):
        sampling = sampling.result()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    make_env = loop.harbor_envs(Path(args.data))
    tasks = [ln.strip() for ln in Path(args.tasks).read_text(encoding="utf8").splitlines() if ln.strip()]

    rows = []
    for task in tasks:
        started = time.time()
        _, rewards, _ = await loop.run_batch(
            [task], loop.SamplingAdapter(sampling), tokenizer, make_env,
            group=args.group, temperature=0.9, max_tokens=args.max_tokens,
        )
        rate = sum(rewards) / len(rewards) if rewards else 0.0
        rows.append({"task": task, "rate": rate, "rewards": rewards,
                     "seconds": round(time.time() - started, 1)})
        print(f"{task:<40} {sum(rewards):.0f}/{len(rewards)}  {rate:.2f}  {rows[-1]['seconds']}s", flush=True)

    keep = [r["task"] for r in rows if LO <= r["rate"] <= HI]
    print(f"\nlearnable band [{LO}, {HI}]: {len(keep)}/{len(rows)} tasks")
    for t in keep:
        print("  ", t)
    Path(args.out).write_text(json.dumps({"band": [LO, HI], "rows": rows, "keep": keep}, indent=2) + "\n", encoding="utf8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
