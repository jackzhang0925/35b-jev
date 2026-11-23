"""Paired comparison of two runs on the same task list.

Because both models attempted the same tasks, the task-level difficulty cancels:
what matters is the per-task difference, and the binomial noise of each task's
rate. This reports the paired delta with a task-level bootstrap interval and the
McNemar-style discordant counts.

    python eval/paired_compare.py --a jobs/<9B job> --b jobs/<35B job> [--k-a 2] [--k-b 1]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402


def rates(job: Path) -> dict[str, list[float]]:
    per_task: dict[str, list[float]] = defaultdict(list)
    for rec in run_eval.collect(job):
        per_task[rec["task"]].append(rec["reward"])
    return dict(per_task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference job")
    ap.add_argument("--b", required=True, help="comparison job")
    ap.add_argument("--repeats", type=int, default=4000)
    args = ap.parse_args()

    a, b = rates(Path(args.a)), rates(Path(args.b))
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared tasks", file=sys.stderr)
        return 1

    diffs = []
    only_a = only_b = both = neither = 0
    for t in shared:
        ra = sum(a[t]) / len(a[t])
        rb = sum(b[t]) / len(b[t])
        diffs.append(rb - ra)
        a_pass, b_pass = ra > 0, rb > 0
        if a_pass and b_pass:
            both += 1
        elif b_pass:
            only_b += 1
        elif a_pass:
            only_a += 1
        else:
            neither += 1

    n = len(shared)
    delta = sum(diffs) / n
    rng = random.Random(0)
    boots = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.repeats)
    )
    lo, hi = boots[int(0.025 * args.repeats)], boots[int(0.975 * args.repeats)]

    pass_a = sum(sum(a[t]) / len(a[t]) for t in shared) / n
    pass_b = sum(sum(b[t]) / len(b[t]) for t in shared) / n

    print(f"shared tasks     : {n}")
    print(f"reference (A)    : {pass_a:.3f}")
    print(f"comparison (B)   : {pass_b:.3f}")
    print(f"paired delta     : {delta:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"discordant       : B-only {only_b}   A-only {only_a}   both {both}   neither {neither}")
    verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant"
    print(f"verdict          : {verdict}")
    print(f"note             : A has {len(a)} tasks, B has {len(b)}; only shared tasks compared")

    Path("docs/paired_compare.json").parent.mkdir(exist_ok=True)
    Path("docs/paired_compare.json").write_text(json.dumps({
        "a": args.a, "b": args.b, "n": n,
        "pass_a": pass_a, "pass_b": pass_b, "delta": delta, "ci95": [lo, hi],
        "b_only": only_b, "a_only": only_a, "both": both, "neither": neither,
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
