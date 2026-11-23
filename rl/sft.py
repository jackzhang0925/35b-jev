"""Rejection-sampling fine-tuning (RFT): SFT on trajectories that actually passed.

Why this exists: GRPO at our scale has almost no signal (measured: 510 datums over
4 batches moved held-out pass@1 by 0). RFT is the cheap warm start that usually
does move a small model: sample the base policy, keep only the rollouts whose
verifier said 1.0, and train cross-entropy on the sampled tokens.

    python rl/sft.py --data data/train/aider-polyglot --tasks eval/band_train6.txt \
        --rollouts-per-task 6 --epochs 2 --out rl/sft_ckpt.json

Prints how many trajectories were kept and the SFT loss per epoch, then saves a
sampler checkpoint whose path feeds rl/train_probe.py --init.
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


def to_sft_datums(turns):
    import tinker

    datums = []
    for turn in turns:
        full = list(turn.prompt) + list(turn.tokens)
        if len(full) < 2:
            continue
        n_prompt = max(len(turn.prompt) - 1, 0)
        sampled = len(turn.tokens)
        weights = [0.0] * n_prompt + [1.0] * sampled
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(full[:-1]),
                loss_fn_inputs={
                    "target_tokens": full[1:],
                    "weights": weights[: len(full) - 1],
                },
            )
        )
    return datums


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train/aider-polyglot")
    ap.add_argument("--tasks", default="eval/band_train6.txt")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rollouts-per-task", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="rl/sft_ckpt.json")
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    service = tinker.ServiceClient()
    training = await service.create_lora_training_client(base_model=args.model, rank=args.rank)
    sampler = training.save_weights_and_get_sampling_client()
    if asyncio.iscoroutine(sampler):
        sampler = await sampler
    if hasattr(sampler, "result"):
        sampler = sampler.result()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    make_env = loop.harbor_envs(Path(args.data))
    tasks = [t.strip() for t in Path(args.tasks).read_text(encoding="utf8").splitlines() if t.strip()]

    print(f"collecting {len(tasks)} tasks x {args.rollouts_per_task} rollouts", flush=True)
    kept, datums, tried = [], [], 0
    for task in tasks:
        for i in range(args.rollouts_per_task):
            tried += 1
            turns, reward = await loop.rollout_once(
                make_env(task), loop.SamplingAdapter(sampler), tokenizer,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            if reward >= 1.0:
                kept.append(task)
                datums += to_sft_datums(turns)
                print(f"  keep {task} #{i} ({len(turns)} turns)", flush=True)
        print(f"{task:<40} kept so far {len(kept)}/{tried}", flush=True)

    print(f"\nrollouts={tried} kept={len(kept)} datums={len(datums)}")
    if not datums:
        print("no successful trajectories: nothing to train on", file=sys.stderr)
        return 2

    for epoch in range(1, args.epochs + 1):
        result = training.forward_backward(datums, "cross_entropy")
        if hasattr(result, "result"):
            result = result.result()
        step = training.optim_step(tinker.AdamParams(learning_rate=args.lr))
        if hasattr(step, "result"):
            step.result()
        try:
            loss = float(sum(result.loss_fn_outputs[i]["loss"].item() for i in range(len(datums))) / len(datums))
        except Exception:
            loss = float("nan")
        print(f"epoch {epoch}/{args.epochs}  loss {loss:.4f}", flush=True)

    saved = training.save_weights_for_sampler(name=f"sft-{int(time.time())}")
    if hasattr(saved, "result"):
        saved = saved.result()
    Path(args.out).write_text(
        json.dumps({"ckpt": saved.path, "kept": len(kept), "datums": len(datums),
                    "tasks": args.tasks, "epochs": args.epochs}, indent=2) + "\n",
        encoding="utf8",
    )
    print(f"\ncheckpoint: {saved.path}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
