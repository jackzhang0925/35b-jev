"""Autoresearch over selection rules -- the part that costs nothing.

We already have 37 tasks x 3 candidates, each with a real verifier reward and a
decision-model score. That is enough to search hundreds of selection policies
offline instead of paying for new rollouts.

Policies are functions (candidates) -> chosen candidate. Each is scored by the
fraction of tasks where the chosen candidate actually passed. Because the policy
is picked on the same data it is scored on, the winner is optimistic; the script
reports a leave-one-task-out estimate next to the in-sample number so the gap is
visible.

    python eval/policy_search.py --scores docs/jev_scores_rerank.json docs/jev_scores_hard.json docs/jev_scores_rest.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def policies():
    out = {"argmax": lambda c: max(c, key=lambda r: r["score"]),
           "first": lambda c: c[0],
           "last": lambda c: c[-1]}

    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        out[f"thresh_{t:.2f}_fallback_first"] = (
            lambda c, t=t: (max(c, key=lambda r: r["score"])
                            if max(c, key=lambda r: r["score"])["score"] >= t else c[0]))
        out[f"thresh_{t:.2f}_fallback_none"] = (
            lambda c, t=t: (max(c, key=lambda r: r["score"])
                            if max(c, key=lambda r: r["score"])["score"] >= t else None))

    for m in [0.05, 0.10, 0.20, 0.30]:
        out[f"margin_{m:.2f}"] = (
            lambda c, m=m: (sorted(c, key=lambda r: -r["score"])[0]
                            if len(c) > 1 and sorted(c, key=lambda r: -r["score"])[0]["score"]
                            - sorted(c, key=lambda r: -r["score"])[1]["score"] >= m
                            else c[0]))

    out["argmax_then_first_if_low"] = lambda c: (
        max(c, key=lambda r: r["score"]) if max(c, key=lambda r: r["score"])["score"] >= 0.9 else c[0])
    return out


def score_policy(name, fn, tasks, loo: bool) -> tuple[float, float]:
    if not loo:
        hits = 0
        for cands in tasks:
            pick = fn(cands)
            hits += bool(pick and pick["reward"] >= 1)
        return hits / len(tasks), float("nan")

    # leave-one-task-out: the rule is fixed (no parameters fitted), so this only
    # measures task-sampling variability, not policy overfitting
    accs = []
    for i in range(len(tasks)):
        rest = tasks[:i] + tasks[i + 1:]
        hits = sum(bool(fn(c) and fn(c)["reward"] >= 1) for c in rest)
        accs.append(hits / len(rest))
    return st.mean(accs), st.pstdev(accs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+",
                    default=["docs/jev_scores_rerank.json", "docs/jev_scores_hard.json",
                             "docs/jev_scores_rest.json"])
    ap.add_argument("--repeats", type=int, default=2000)
    args = ap.parse_args()

    rows = []
    for path in args.scores:
        rows += json.loads(Path(path).read_text(encoding="utf8"))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    # a policy needs >1 candidate to be interesting; keep tasks with >=2
    tasks = [sorted(v, key=lambda r: r["trial"]) for v in by_task.values() if len(v) >= 2]
    print(f"tasks {len(tasks)}   candidates {sum(len(t) for t in tasks)}")

    results = []
    for name, fn in policies().items():
        acc, sd = score_policy(name, fn, tasks, loo=True)
        results.append((acc, sd, name))
    results.sort(reverse=True)

    print(f"\n{'policy':<32} {'acc':>6} {'loo_sd':>8}")
    for acc, sd, name in results:
        print(f"{name:<32} {acc:>6.3f} {sd:>8.3f}")

    best = results[0]
    print(f"\nbest: {best[2]}  acc {best[0]:.3f}")
    print("note: all policies use the same 37 tasks, so differences below ~0.05 are noise;")
    print("      the ranking is a screening tool, not a significance test.")

    Path("docs/policy_search.json").write_text(json.dumps(
        [{"policy": n, "acc": a, "loo_sd": s} for a, s, n in results], indent=2) + "\n",
        encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
