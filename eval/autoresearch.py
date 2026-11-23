"""Autoresearch driver: run a queue of harness experiments, keep the winner.

The loop is deliberately boring, in the spirit of karpathy/autoresearch:
one variable per experiment, a fixed task set, a fixed budget, a results ledger,
and a keep/discard rule that is decided before the run.

    python eval/autoresearch.py --queue config/queue.json --budget-usd 25
    python eval/autoresearch.py --queue config/queue.json --dry-run

Each entry is one experiment:
    {"tag": "...", "model": "qwen36", "config": "config/codex_tinker_36b.toml",
     "prompt": "config/codex_prompt.j2", "tasks": "eval/ceiling10.txt",
     "k": 1, "concurrency": 5, "est_usd": 9.0}

The driver does not invent numbers: it shells out to eval/run_eval.py, then reads
the row that run appended to results.tsv. If the run fails, the row is missing and
the experiment is marked failed rather than scored 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
RESULTS = Path("results.tsv")


def last_row() -> dict | None:
    if not RESULTS.exists():
        return None
    rows = list(csv.DictReader(RESULTS.open(encoding="utf8"), delimiter="\t"))
    return rows[-1] if rows else None


def spent() -> float:
    """Sum of the usd column of results.tsv (the ledger is the budget)."""
    if not RESULTS.exists():
        return 0.0
    total = 0.0
    for row in csv.DictReader(RESULTS.open(encoding="utf8"), delimiter="\t"):
        try:
            total += float(row.get("usd") or 0)
        except ValueError:
            pass
    return total


def run_one(exp: dict) -> dict:
    cmd = [
        PY, "-u", "eval/run_eval.py",
        "--tag", exp["tag"],
        "--model", exp["model"],
        "--codex-config", exp["config"],
        "--tasks", exp["tasks"],
        "--k", str(exp.get("k", 1)),
        "--concurrency", str(exp.get("concurrency", 5)),
    ]
    if exp.get("prompt"):
        cmd += ["--prompt-template", exp["prompt"]]
    print(f"\n=== {exp['tag']}  model={exp['model']} tasks={exp['tasks']} est=${exp.get('est_usd', '?')}")
    print(" ".join(cmd))
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    if proc.stdout:
        tail = proc.stdout.strip().splitlines()[-12:]
        print("\n".join(tail))
    if proc.returncode != 0 and not proc.stdout.strip().endswith("}"):
        print(proc.stderr[-800:], file=sys.stderr)
    row = last_row() or {}
    row["_ok"] = proc.returncode == 0 and row.get("tag") == exp["tag"]
    row["_elapsed_min"] = round(elapsed / 60, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="config/queue.json")
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue = json.loads(Path(args.queue).read_text(encoding="utf8"))
    print(f"queue: {len(queue)} experiments, budget ${args.budget_usd}, spent ${spent():.2f}")

    table = []
    for exp in queue:
        if spent() + exp.get("est_usd", 10) > args.budget_usd:
            print(f"STOP: {exp['tag']} would exceed the budget (spent ${spent():.2f})")
            break
        if args.dry_run:
            print(f"  [dry] {exp['tag']}: {exp['model']} on {exp['tasks']} est ${exp.get('est_usd')}")
            continue
        row = run_one(exp)
        table.append(row)
        print(f"  -> pass1={row.get('pass1')} usd={row.get('usd')} ok={row['_ok']} "
              f"({row['_elapsed_min']} min)")

    if table:
        best = max(table, key=lambda r: float(r.get("pass1") or 0))
        print("\nleaderboard (by pass@1):")
        for r in sorted(table, key=lambda r: -float(r.get("pass1") or 0)):
            print(f"  {r['tag']:<24} pass1={r.get('pass1')} tasks={r.get('tasks')} "
                  f"usd={r.get('usd')} ok={r['_ok']}")
        print(f"\nKEEP: {best['tag']}")
        (Path("docs") / "autoresearch_runs.json").parent.mkdir(exist_ok=True)
        Path("docs/autoresearch_runs.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
