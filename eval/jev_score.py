"""Ask the Torchcast decision model to score submissions, then measure whether the
score can separate passes from failures (AUC).

This is the free prerequisite for a reranker experiment: if a "this submission
satisfies the task" probability cannot rank real passes above real failures on
data we already paid for, no amount of best-of-k will help.

    python eval/jev_score.py --job jobs/2026-11-21__00-01-40 --out docs/jev_scores.json
    python eval/jev_auc.py docs/jev_scores.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402

GATEWAY = "https://jackzhang0356--torchcast-decision-serve.us-east.modal.run/v1/systemone"
MAX_STATE = 6000


def compact_submission(trajectory: Path, max_chars: int = MAX_STATE) -> str:
    """Instruction + what the agent finally did, small enough for a 16k prompt."""
    data = json.loads(trajectory.read_text(encoding="utf8"))
    steps = data.get("steps", [])
    instruction = next((s.get("message") or "" for s in steps if s.get("source") == "user"), "")
    parts = []
    for step in steps:
        if step.get("source") != "agent":
            continue
        for call in step.get("tool_calls") or []:
            args = call.get("arguments") or {}
            name = call.get("function_name") or call.get("name") or "tool"
            text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            parts.append(f"[{name}] {text}")
        msg = (step.get("message") or "").strip()
        if msg:
            parts.append(f"[say] {msg}")
    tail = "\n".join(parts)[-max_chars:]
    return f"TASK:\n{instruction[:2000]}\n\nAGENT ACTIONS (tail):\n{tail}"


def ask(state: str) -> dict:
    body = json.dumps({
        "state": state,
        "questions": {
            "solves": {
                "type": "noul",
                "instructions": (
                    "The agent's actions in `AGENT ACTIONS` fully solve the task described in "
                    "`TASK`: the required files exist with the required content and the task's "
                    "own tests would pass."
                ),
            }
        },
    }).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", default="docs/jev_scores.json")
    args = ap.parse_args()

    job = Path(args.job)
    rewards = {r["task"] + "__" + r["task"]: r for r in []}  # unused placeholder
    graded = {}
    for trial in sorted(job.glob("*/result.json")):
        rec = json.loads(trial.read_text(encoding="utf8"))
        # verifier_result can be explicitly null on errored trials, so `or {}`
        rewards = (rec.get("verifier_result") or {}).get("rewards") or {}
        graded[trial.parent.name] = (
            rec.get("task_name", "").split("/")[-1],
            rewards.get("reward", 0.0),
        )

    rows = []
    for trial_dir in sorted(job.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name not in graded:
            continue
        traj = trial_dir / "agent" / "trajectory.json"
        if not traj.exists():
            continue
        task, reward = graded[trial_dir.name]
        state = compact_submission(traj)
        try:
            resp = ask(state)
            score = float(resp["answers"]["solves"]["noul"])
        except Exception as exc:  # noqa: BLE001
            print(f"{task:<34} ERROR {exc}", flush=True)
            continue
        rows.append({"task": task, "trial": trial_dir.name, "reward": reward,
                     "score": score, "state_chars": len(state)})
        print(f"{task:<34} reward={reward:.0f}  p_solve={score:.3f}  ({len(state)} chars)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    print(f"\nwrote {args.out}  ({len(rows)} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
