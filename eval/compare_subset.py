"""Paired comparison restricted to a task list (for a subset measured on both sides).

    python eval/compare_subset.py --a jobs/<base> --b jobs/<rl> --tasks eval/ceiling10.txt
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
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--repeats", type=int, default=4000)
    args = ap.parse_args()

    wanted = [t.strip().split("/")[-1] for t in Path(args.tasks).read_text(encoding="utf8").splitlines() if t.strip()]
    a, b = rates(Path(args.a)), rates(Path(args.b))
    shared = [t for t in wanted if t in a and t in b]

    diffs = []
    for t in shared:
        diffs.append(sum(b[t]) / len(b[t]) - sum(a[t]) / len(a[t]))
    n = len(shared)
    if n == 0:
        print("no shared tasks", file=sys.stderr)
        return 1

    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.repeats))
    delta = sum(diffs) / n
    lo, hi = boots[int(0.025 * args.repeats)], boots[int(0.975 * args.repeats)]

    pa = sum(sum(a[t]) / len(a[t]) for t in shared) / n
    pb = sum(sum(b[t]) / len(b[t]) for t in shared) / n
    print(f"tasks {n}: A {pa:.3f}  B {pb:.3f}  delta {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    for t in shared:
        print(f"  {t:<34} A={sum(a[t])/len(a[t]):.2f}  B={sum(b[t])/len(b[t]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
