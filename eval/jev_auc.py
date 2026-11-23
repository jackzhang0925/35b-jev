"""AUC + selection metrics for decision-model scores against real pass/fail.

    python eval/jev_auc.py docs/jev_scores.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def auc(pairs: list[tuple[float, float]]) -> float:
    """Rank-based AUC: P(score of a pass > score of a failure), ties = 0.5."""
    pos = [s for s, y in pairs if y >= 1]
    neg = [s for s, y in pairs if y < 1]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def main() -> int:
    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    pairs = [(r["score"], r["reward"]) for r in rows]
    a = auc(pairs)
    pos = [s for s, y in pairs if y >= 1]
    neg = [s for s, y in pairs if y < 1]

    # threshold sweep: precision/recall of "accept if p >= t"
    best = (0.0, 0.0, 0.0)
    for t in [i / 100 for i in range(0, 101)]:
        tp = sum(1 for s, y in pairs if s >= t and y >= 1)
        fp = sum(1 for s, y in pairs if s >= t and y < 1)
        fn = sum(1 for s, y in pairs if s < t and y >= 1)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, t, prec, rec)

    print(f"trials      : {len(pairs)}   passes {len(pos)}   failures {len(neg)}")
    if pos:
        print(f"p_solve|pass: mean {sum(pos)/len(pos):.3f}  min {min(pos):.3f}  max {max(pos):.3f}")
    if neg:
        print(f"p_solve|fail: mean {sum(neg)/len(neg):.3f}  min {min(neg):.3f}  max {max(neg):.3f}")
    print(f"AUC         : {a:.3f}")
    print(f"best F1     : {best[0]:.3f} at threshold {best[1]:.2f} (precision {best[2]:.2f}, recall {best[3]:.2f})")
    print()
    if a != a:
        print("cannot compute: need at least one pass and one failure")
    elif a >= 0.80:
        print("verdict: the decision model separates passes from failures well.")
        print("         best-of-k reranking is worth the inference cost -- run it.")
    elif a >= 0.65:
        print("verdict: weak but real separation. Reranking may help a little;")
        print("         test on tasks with >=2 passing candidates before trusting it.")
    else:
        print("verdict: no usable separation. Reranking with this signal cannot beat")
        print("         picking at random; do not spend on best-of-k.")

    Path("docs/jev_auc.json").write_text(json.dumps({
        "n": len(pairs), "passes": len(pos), "failures": len(neg), "auc": a,
        "best_f1": best[0], "best_threshold": best[1],
        "mean_pass": sum(pos) / len(pos) if pos else None,
        "mean_fail": sum(neg) / len(neg) if neg else None,
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
