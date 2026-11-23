"""Pick the acceptance threshold for best-of-k selection, from data already paid for.

Policy under test (on k candidates per task):
    score = decision model's P(solves | candidate)
    if max(score) >= t: pick argmax
    else:               fall back to the FIRST candidate
The fallback matters because the model is known to over-score plausible-looking
failures (observed: a failing candidate scored 0.980). Selection accuracy is
measured per task against the real verifier reward.

    python eval/calibrate_threshold.py --scores docs/jev_scores_rerank.json docs/jev_scores_hard.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", default=["docs/jev_scores_rerank.json", "docs/jev_scores_hard.json"])
    args = ap.parse_args()

    rows = []
    for path in args.scores:
        rows += json.loads(Path(path).read_text(encoding="utf8"))

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks = [sorted(v, key=lambda r: r["trial"]) for v in by_task.values()]

    print(f"tasks {len(tasks)}   candidates {sum(len(t) for t in tasks)}")
    print(f"\n{'t':>5}  {'selected':>8}  {'picked_score_mix':>10}")
    results = []
    for i in range(50, 101, 5):
        t = i / 100
        hits = 0
        for cands in tasks:
            best = max(cands, key=lambda r: r["score"])
            pick = best if best["score"] >= t else cands[0]
            hits += pick["reward"] >= 1
        acc = hits / len(tasks)
        results.append((t, acc))
        print(f"{t:>5.2f}  {acc:>8.3f}")

    best_t, best_acc = max(results, key=lambda x: x[1])
    argmax_acc = sum(max(c, key=lambda r: r["score"])["reward"] >= 1 for c in tasks) / len(tasks)
    first_acc = sum(c[0]["reward"] >= 1 for c in tasks) / len(tasks)
    print(f"\nargmax (t=0)     : {argmax_acc:.3f}")
    print(f"first candidate  : {first_acc:.3f}")
    print(f"best threshold   : t={best_t:.2f} -> {best_acc:.3f}")

    Path("docs/threshold_sweep.json").write_text(json.dumps({
        "tasks": len(tasks), "sweep": results, "argmax": argmax_acc,
        "first": first_acc, "best_t": best_t, "best_acc": best_acc,
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
