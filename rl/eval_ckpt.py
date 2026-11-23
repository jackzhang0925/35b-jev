"""Score a Tinker checkpoint (or the base model) on a fixed task list.

Same rollout loop as training (same prompt, same tool schema, thinking off), so
the only difference between the two runs is the weights.

    python rl/eval_ckpt.py --tasks eval/holdout8_ext.txt --label base
    python rl/eval_ckpt.py --tasks eval/holdout8_ext.txt --label rl \
        --ckpt tinker://29e22cdf-.../sampler_weights/probe-12x4-...

Writes one line per task and a mean, and appends a JSON line to rl/eval_log.jsonl.
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


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--data", default="data/train/aider-polyglot")
    ap.add_argument("--ckpt", default=None, help="tinker:// checkpoint; omit for the base model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-tokens", type=int, default=1200)
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
        _, rewards, _ = await loop.run_batch(
            [task], loop.SamplingAdapter(sampling), tokenizer, make_env,
            group=1, temperature=0.7, max_tokens=args.max_tokens,
        )
        reward = rewards[0] if rewards else 0.0
        rows.append({"task": task, "reward": reward})
        print(f"{task:<40} {reward:.1f}", flush=True)

    mean = sum(r["reward"] for r in rows) / len(rows) if rows else 0.0
    print(f"\n{args.label}: {sum(r['reward'] for r in rows):.0f}/{len(rows)} = {mean:.3f}")
    with (Path(__file__).parent / "eval_log.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": args.label,
            "ckpt": args.ckpt,
            "tasks": args.tasks,
            "n": len(rows),
            "mean": mean,
            "rows": rows,
        }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
