"""How large must an effect be before this evaluation can see it?

The question this answers: with N tasks x k attempts, what is the 95% interval
around measured pass@1, and what is the smallest real improvement the setup can
detect? It uses a task-level (cluster) bootstrap, because attempts on the same
task are not independent -- a task the model cannot do contributes 0 no matter
how many times we retry it.

    python eval/noise_band.py --job jobs/<id> [--k 2] [--repeats 4000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402


def cluster_bootstrap(per_task: list[list[float]], repeats: int, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(per_task)
    means = []
    for _ in range(repeats):
        sample = [per_task[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sum(t) / len(t) for t in sample) / n)
    means.sort()
    return means[int(0.025 * repeats)], means[int(0.975 * repeats)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=4000)
    args = ap.parse_args()

    records = run_eval.collect(Path(args.job))
    if not records:
        print(f"no trials in {args.job}", file=sys.stderr)
        return 1
    per_task: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        per_task[rec["task"]].append(rec["reward"])

    groups = list(per_task.values())
    n = len(groups)
    k = len(groups[0])
    pass1 = sum(sum(g) / len(g) for g in groups) / n
    lo, hi = cluster_bootstrap(groups, args.repeats)
    attempt_rate = sum(sum(g) for g in groups) / sum(len(g) for g in groups)

    solved = sum(1 for g in groups if sum(g) == len(g))
    never = sum(1 for g in groups if sum(g) == 0)
    some = n - solved - never

    # Comparing two configs on the SAME tasks (paired): the task-level difficulty
    # cancels, so the noise is the binomial noise of each task's rate. With k
    # attempts the per-task variance is p(1-p)/k and the difference of two
    # independent configs has 2p(1-p)/k. Tasks that are always 0 or always 1
    # contribute nothing -- which is why a bimodal task set is so weak.
    variance_sum = sum((sum(g) / len(g)) * (1 - sum(g) / len(g)) / k for g in groups)
    se_paired = (2 * variance_sum) ** 0.5 / n
    mde_paired = 2.8 * se_paired
    se_unpaired = (hi - lo) / 3.92

    print(f"job            : {args.job}")
    print(f"tasks          : {n}   attempts/task: {k}   trials: {sum(len(g) for g in groups)}")
    print(f"pass@1         : {pass1:.3f}   (attempt-level {attempt_rate:.3f})")
    print(f"95% interval   : [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")
    print(f"task split     : always-pass {solved}   mixed {some}   always-fail {never}")
    print(f"SE(level)      : {se_unpaired:.3f}")
    print(f"SE(paired diff): {se_paired:.3f}")
    print(f"MDE (paired)   : {mde_paired:.3f}   <- two configs on the SAME tasks, 80% power")
    print()
    print("interpretation:")
    if mde_paired > 0.10:
        print(f"  this evaluation cannot see a real improvement smaller than {mde_paired:.0%}.")
        print("  a +5-8pt post-training gain would be invisible here; do not spend on training first.")
    else:
        print(f"  it can resolve about {mde_paired:.0%}; a +10pt effect would be visible.")

    Path("docs/noise_band.json").parent.mkdir(exist_ok=True)
    Path("docs/noise_band.json").write_text(json.dumps({
        "job": args.job, "tasks": n, "attempts": k,
        "pass1": pass1, "ci95": [lo, hi], "se_level": se_unpaired,
        "se_paired": se_paired, "mde_paired": mde_paired,
        "always_pass": solved, "mixed": some, "always_fail": never,
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
