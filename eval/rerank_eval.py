"""Best-of-k reranking: does picking by the decision model's score beat not picking?

Input: a job with k trials per task (all graded) plus the decision-model score for
every trial (from eval/jev_score.py --out docs/jev_scores_rerank.json).

    python eval/jev_score.py --job jobs/<rerank job> --out docs/jev_scores_rerank.json
    python eval/rerank_eval.py --scores docs/jev_scores_rerank.json

Reports, per task set:
  * pass@1      -- what one attempt gives you on average
  * pass@k      -- did ANY candidate pass
  * selected    -- did the argmax-scored candidate pass
  * first       -- did the first candidate pass (control for ordering)
and a task-level bootstrap interval on (selected - pass@1).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", default=["docs/jev_scores_rerank.json"])
    ap.add_argument("--repeats", type=int, default=4000)
    args = ap.parse_args()

    rows = []
    for path in args.scores:
        rows += json.loads(Path(path).read_text(encoding="utf8"))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    per_task = []
    for task, cands in sorted(by_task.items()):
        cands = sorted(cands, key=lambda r: r["trial"])
        n = len(cands)
        pass1 = st.mean(c["reward"] for c in cands)
        passk = 1.0 if any(c["reward"] >= 1 for c in cands) else 0.0
        best = max(cands, key=lambda r: r["score"])
        first = cands[0]["reward"]
        per_task.append({"task": task, "n": n, "pass1": pass1, "passk": passk,
                         "selected": float(best["reward"] >= 1),
                         "first": float(first >= 1),
                         "picked_score": best["score"]})

    n = len(per_task)
    mean = {k: st.mean(t[k] for t in per_task) for k in ("pass1", "passk", "selected", "first")}
    diffs = [t["selected"] - t["pass1"] for t in per_task]
    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.repeats))
    lo, hi = boots[int(0.025 * args.repeats)], boots[int(0.975 * args.repeats)]

    print(f"tasks            : {n}   candidates/task: {per_task[0]['n']}")
    print(f"pass@1 (mean)    : {mean['pass1']:.3f}")
    print(f"pass@k (any)     : {mean['passk']:.3f}   <- ceiling for any picker")
    print(f"selected (Jev)   : {mean['selected']:.3f}")
    print(f"first candidate  : {mean['first']:.3f}   <- control")
    print(f"delta vs pass@1  : {mean['selected'] - mean['pass1']:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print()
    for t in per_task:
        print(f"  {t['task']:<32} cands={t['n']} pass1={t['pass1']:.2f} any={t['passk']:.0f} "
              f"selected={t['selected']:.0f} first={t['first']:.0f} p={t['picked_score']:.3f}")
    print()
    if lo > 0:
        print("verdict: reranking beats pass@1 with a 95% interval clear of zero.")
    elif mean["selected"] > mean["pass1"]:
        print("verdict: reranking points in the right direction but is not significant at this n.")
    else:
        print("verdict: no gain from reranking on this sample.")

    Path("docs/rerank_eval.json").write_text(json.dumps({
        "tasks": n, **{k: mean[k] for k in mean},
        "delta": mean["selected"] - mean["pass1"], "ci95": [lo, hi],
        "per_task": per_task,
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
