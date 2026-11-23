"""Option A: RL on Qwen3.6-35B-A3B with a Jev-shaped reward.

Why a shaped reward: the verifier only returns 0/1, and on these tasks most
groups are all-zero (measured on the 9B: 50% of groups). The decision model gives
a per-rollout P(solves), which is non-degenerate on failures, so

    shaped = pass + alpha * (p_solves - 0.5)

gives gradient information to trajectories that were nearly right. alpha is kept
small (default 0.3) so the verifier stays the dominant term; the shaping is
logged per rollout so its contribution can be audited.

    python rl/train_jev.py --model Qwen/Qwen3.6-35B-A3B --batches 20 --group 4 \
        --tasks-per-batch 2 --alpha 0.3 --out rl/jev_rl_ckpt.json

Dry run (no Tinker, no sandbox, no gateway):
    python rl/train_jev.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop  # noqa: E402

GATEWAY = "https://jackzhang0356--torchcast-decision-serve.us-east.modal.run/v1/systemone"
MAX_STATE = 5000


def render_for_judge(turns, instruction: str, tokenizer=None) -> str:
    """Compact view of a rollout for the decision model (16k prompt budget).

    loop.Turn stores tokens, not text, so the tokenizer is required -- without it
    the state would be empty and every score meaningless (the dry run hid this).
    """
    parts = []
    for turn in turns:
        tokens = turn.get("tokens") if isinstance(turn, dict) else getattr(turn, "tokens", None)
        if tokens and tokenizer is not None:
            parts.append(tokenizer.decode(list(tokens), skip_special_tokens=True))
    tail = "\n".join(parts)[-MAX_STATE:]
    return f"TASK:\n{instruction[:1500]}\n\nAGENT ACTIONS (tail):\n{tail}"


def jev_p(state: str, timeout: int = 120) -> float | None:
    body = json.dumps({
        "state": state,
        "questions": {"solves": {"type": "noul", "instructions": (
            "The agent's actions fully solve the task in `TASK`: required files exist with "
            "the required content and the task's own tests would pass.")}},
    }).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return float(json.loads(resp.read().decode())["answers"]["solves"]["noul"])
    except Exception:  # noqa: BLE001
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train/aider-polyglot")
    ap.add_argument("--tasks", default="eval/rl_train_35b.txt")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--tasks-per-batch", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--rollout-timeout", type=float, default=300.0, help="seconds per rollout before it counts as a failure")
    ap.add_argument("--out", default="rl/jev_rl_ckpt.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = [t.strip() for t in Path(args.tasks).read_text(encoding="utf8").splitlines() if t.strip()]
    print(f"{'DRY RUN' if args.dry_run else 'REAL'}: model={args.model} tasks={len(tasks)} "
          f"batches={args.batches} group={args.group} alpha={args.alpha}", flush=True)

    if args.dry_run:
        print("dry run: reward shaping + datum routing only")
        fake_turns = [{"reply": "cat > f.py <<'EOF'\npass\nEOF"} for _ in range(3)]
        print(" state chars:", len(render_for_judge(fake_turns, "do the thing")))
        shaped = [0.0 + args.alpha * (0.8 - 0.5), 1.0 + args.alpha * (0.9 - 0.5)]
        print(" shaped rewards:", [round(x, 3) for x in shaped])
        print(" advantages:", loop.advantages(shaped))
        return 0

    import tinker
    from transformers import AutoTokenizer

    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=args.model, rank=args.rank)
    if asyncio.iscoroutine(training):
        training = await training
    sampler = training.save_weights_and_get_sampling_client()
    if asyncio.iscoroutine(sampler):
        sampler = await sampler
    if hasattr(sampler, "result"):
        sampler = sampler.result()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    make_env = loop.harbor_envs(Path(args.data))
    # keep the sampler as an adapter for its whole life; wrapping it again after a
    # refresh is what crashed batch 2 with "unexpected keyword argument 'prompt'"
    sampler = loop.SamplingAdapter(sampler)

    pool = list(tasks)
    audit, datums_total, rollouts_total = [], 0, 0
    for batch in range(1, args.batches + 1):
        picked = [pool[(batch * args.tasks_per_batch + i) % len(pool)]
                  for i in range(args.tasks_per_batch)]
        shaped_rewards, batch_datums = [], []
        for task in picked:
            env_factory = make_env
            instr = (Path(args.data) / task / "instruction.md").read_text(encoding="utf8", errors="ignore")

            async def one_rollout():
                # a hung sandbox command must not stall the whole run: every run so
                # far stalled after 4-6 batches with no snapshot saved
                try:
                    return await asyncio.wait_for(
                        loop.rollout_once(env_factory(task), sampler, tokenizer,
                                          temperature=args.temperature,
                                          max_tokens=args.max_tokens),
                        timeout=args.rollout_timeout,
                    )
                except asyncio.TimeoutError:
                    return [], 0.0

            jobs = [one_rollout() for _ in range(args.group)]
            results = await asyncio.gather(*jobs)
            rewards, ps = [], []
            for turns, reward in results:
                if not turns:
                    ps.append(None)
                    rewards.append(0.0)
                    continue
                p = jev_p(render_for_judge(turns, instr, tokenizer))
                ps.append(p)
                shaped = reward + args.alpha * ((p if p is not None else 0.5) - 0.5)
                rewards.append(shaped)
            advs = loop.advantages([r for r in rewards])
            if advs is None:
                audit.append({"batch": batch, "task": task, "skipped": True,
                              "raw": [r for r in rewards], "p": ps})
                continue
            for (turns, raw), adv, p in zip(results, advs, ps):
                batch_datums += loop.build_datums(turns, adv)
                shaped_rewards.append(raw)
            rollouts_total += len(results)
            audit.append({"batch": batch, "task": task, "skipped": False,
                          "shaped": [round(r, 3) for r in rewards],
                          "p": [round(p, 3) if p is not None else None for p in ps]})
        if batch_datums:
            training.forward_backward(loop.to_tinker_datums(batch_datums),
                                      "importance_sampling").result()
            training.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
            refreshed = training.save_weights_and_get_sampling_client()
            if hasattr(refreshed, "result"):
                refreshed = refreshed.result()
            sampler = loop.SamplingAdapter(refreshed)
            datums_total += len(batch_datums)
        print(f"batch {batch:>3}/{args.batches} rollouts {rollouts_total:>5} datums {datums_total:>5} "
              f"mean_shaped {sum(shaped_rewards)/max(1,len(shaped_rewards)):.3f}", flush=True)
        if args.save_every and batch % args.save_every == 0:
            snap = training.save_weights_for_sampler(name=f"jev-{batch:03d}")
            if hasattr(snap, "result"):
                snap = snap.result()
            print(f"  snapshot: {snap.path}", flush=True)

    saved = training.save_weights_for_sampler(name=f"jev-final-{int(time.time())}")
    if hasattr(saved, "result"):
        saved = saved.result()
    Path(args.out).write_text(json.dumps({
        "ckpt": saved.path, "model": args.model, "batches": args.batches,
        "group": args.group, "alpha": args.alpha, "rollouts": rollouts_total,
        "datums": datums_total, "audit": audit,
    }, indent=2) + "\n", encoding="utf8")
    print(f"\nrollouts={rollouts_total} datums={datums_total}")
    print(f"checkpoint: {saved.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
