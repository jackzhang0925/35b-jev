"""Did the Jev shaping actually carry information?

The trainer stored, per batch/task, the decision model's P(solves) and the shaped
reward, where

    shaped = pass + alpha * (p - 0.5)      =>      pass = shaped - alpha * (p - 0.5)

so both the raw outcome and the judge score can be recovered. If p does not
separate passes from failures here, the RL gain cannot be attributed to the
shaping -- it would be plain GRPO on the verifier reward.

    python rl/analyze_shaping.py --ckpt rl/jev_rl_ckpt.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def auc(pairs: list[tuple[float, float]]) -> float | None:
    pos = [s for s, y in pairs if y >= 1]
    neg = [s for s, y in pairs if y < 1]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rl/jev_rl_ckpt.json")
    args = ap.parse_args()

    data = json.loads(Path(args.ckpt).read_text(encoding="utf8"))
    alpha = data.get("alpha", 0.3)
    pairs, rows = [], []
    for entry in data["audit"]:
        if entry.get("skipped"):
            continue
        for shaped, p in zip(entry.get("shaped", []), entry.get("p", [])):
            if p is None:
                continue
            passed = shaped - alpha * (p - 0.5)
            pairs.append((p, 1.0 if passed >= 0.5 else 0.0))
            rows.append((entry["batch"], entry["task"], round(p, 3), round(passed, 2)))

    if not pairs:
        print("no shaped rollouts with a judge score")
        return 1

    pos = [s for s, y in pairs if y >= 1]
    neg = [s for s, y in pairs if y < 1]
    a = auc(pairs)
    print(f"shaped rollouts : {len(pairs)}   passes {len(pos)}   failures {len(neg)}")
    if pos:
        print(f"p_solves | pass : mean {sum(pos)/len(pos):.3f}")
    if neg:
        print(f"p_solves | fail : mean {sum(neg)/len(neg):.3f}")
    print(f"AUC             : {a if a is None else round(a, 3)}")
    print(f"pass rate       : {len(pos)/len(pairs):.3f}")
    print()
    if a is None:
        print("cannot compute AUC: need at least one pass and one failure")
    elif a >= 0.75:
        print("verdict: the shaping signal is informative on-policy; the gain can be")
        print("         attributed partly to it, not only to the verifier reward.")
    elif a >= 0.6:
        print("verdict: weak signal. The gain is mostly plain GRPO on the verifier.")
    else:
        print("verdict: the judge score does not separate passes here; shaping added")
        print("         noise. Report the gain as GRPO-on-verifier, not 'Jev-shaped'.")

    Path("docs/shaping_analysis.json").parent.mkdir(exist_ok=True)
    Path("docs/shaping_analysis.json").write_text(json.dumps({
        "rollouts": len(pairs), "passes": len(pos), "failures": len(neg),
        "mean_p_pass": sum(pos) / len(pos) if pos else None,
        "mean_p_fail": sum(neg) / len(neg) if neg else None,
        "auc": a, "pass_rate": len(pos) / len(pairs),
    }, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
