"""Probe: a few GRPO batches over eval/train16.txt against Tinker (LoRA).

    python rl/train_probe.py --train eval/train16.txt --holdout eval/holdout6.txt \
        --batches 10 --group 4 --lr 5e-5 --model Qwen/Qwen3.5-9B [--dry-run]

Per batch: `group` rollouts per task -> grade -> Datums with group-relative
advantages (flat groups skipped) -> forward_backward + optim_step. At the end it saves
weights for the sampler and prints the checkpoint path. --dry-run swaps in loop.Fake*
plus a recording training client: no Tinker, no Docker, no key, no spend.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop  # noqa: E402

DATA = Path("data/tb2/terminal-bench-2")  # loop.harbor_envs resolves task names here


class FakeTrainingClient:
    """Echoes the real call shape so a dry run still exercises forward/backward/step."""
    def __init__(self) -> None:
        self.steps: list[int] = []
    def forward_backward(self, datums, loss_fn):
        self.steps.append(len(datums))
        return SimpleNamespace(result=lambda: None)
    def optim_step(self, params):
        return SimpleNamespace(result=lambda: None)
    def save_weights_for_sampler(self, name):
        return SimpleNamespace(path=f"tinker://dry-run/sampler_weights/{name}")


async def real_stack(model: str, rank: int):
    """Training client, sampling client, tokenizer. Costs money; needs TINKER_API_KEY."""
    import tinker
    from transformers import AutoTokenizer

    service = tinker.ServiceClient()  # reads TINKER_API_KEY from the environment
    # these two are documented as async; awaiting a plain object is harmless
    training = service.create_lora_training_client(base_model=model, rank=rank)
    if asyncio.iscoroutine(training):
        training = await training
    sampling = training.save_weights_and_get_sampling_client()
    if asyncio.iscoroutine(sampling):
        sampling = await sampling
    if hasattr(sampling, "result"):  # APIFuture
        sampling = sampling.result()
    tokenizer = AutoTokenizer.from_pretrained(model)
    return training, loop.SamplingAdapter(sampling), tokenizer


def read_tasks(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


async def run(args) -> int:
    train, holdout = read_tasks(args.train), read_tasks(args.holdout)
    overlap = set(train) & set(holdout)
    if overlap:
        print(f"train and holdout overlap: {sorted(overlap)}", file=sys.stderr)
        return 2
    if args.dry_run:
        training, sampler, tokenizer = FakeTrainingClient(), loop.FakeSampler(), loop.FakeTokenizer()
        make_env = loop.fake_envs()
    else:
        training, sampler, tokenizer = await real_stack(args.model, args.rank)
        make_env = loop.harbor_envs(Path(args.data))

    pool = list(train)
    random.Random(args.seed).shuffle(pool)
    print(
        f"{'DRY RUN' if args.dry_run else 'REAL RUN'}: model={args.model} tasks={len(pool)} "
        f"holdout={len(holdout)} batches={args.batches} group={args.group} lr={args.lr:g}"
    )

    rewards: list[float] = []
    datums_total = skipped_total = 0
    for batch in range(1, args.batches + 1):
        start = (batch - 1) * args.tasks_per_batch
        batch_tasks = [pool[(start + i) % len(pool)] for i in range(args.tasks_per_batch)]
        datums, batch_rewards, skipped = await loop.run_batch(
            batch_tasks, sampler, tokenizer, make_env,
            group=args.group, temperature=args.temperature, max_tokens=args.max_tokens,
        )
        rewards += batch_rewards
        datums_total += len(datums)
        skipped_total += skipped
        if datums:  # a batch whose groups were all flat has nothing to learn from
            if args.dry_run:
                training.forward_backward(datums, "importance_sampling").result()
                training.optim_step({"learning_rate": args.lr}).result()
            else:
                import tinker
                training.forward_backward(loop.to_tinker_datums(datums), "importance_sampling").result()
                training.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
                # the step moved the weights; the sampler must follow them for the next batch
                refreshed = training.save_weights_and_get_sampling_client()
                if hasattr(refreshed, "result"):
                    refreshed = refreshed.result()
                sampler = loop.SamplingAdapter(refreshed)
        if batch % args.eval_every == 0 or batch == args.batches:
            print(
                f"batch {batch:>3}/{args.batches}  train_reward {mean(batch_rewards):.2f}  "
                f"running {mean(rewards):.2f}  datums {len(datums):>4}  skipped_groups {skipped}",
                flush=True,
            )
        # checkpoint every N batches: a stuck batch must not cost us the whole run
        if not args.dry_run and args.save_every and batch % args.save_every == 0:
            snap = training.save_weights_for_sampler(name=f"probe-snap-{batch:02d}")
            if hasattr(snap, "result"):
                snap = snap.result()
            print(f"  snapshot: {snap.path}", flush=True)

    saved = training.save_weights_for_sampler(
        name=f"probe-{args.batches}x{args.group}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if hasattr(saved, "result"):
        saved = saved.result()
    checkpoint = saved
    print(f"\nrollouts={len(rewards)} datums={datums_total} skipped_groups={skipped_total}")
    print(f"checkpoint: {checkpoint.path}")
    print("(fake path)" if args.dry_run else f"model: {args.model};{checkpoint.path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/tb2/terminal-bench-2", help="root holding the task directories")
    ap.add_argument("--train", default="eval/train16.txt")
    ap.add_argument("--holdout", default="eval/holdout6.txt", help="never trained on")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--group", type=int, default=4, help="rollouts per task")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tasks-per-batch", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=2, help="save a sampler checkpoint every N batches (0 = only at the end)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fake sampler/env; no Tinker, no Docker")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())


