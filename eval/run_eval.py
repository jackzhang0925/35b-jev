"""Run a Harbor eval on a fixed subset and score it (pass@1 / pass@k), then log it.

Schema confirmed against Harbor 0.26.0 job output (2026-11-20):
  jobs/<job>/<task>__<id>/result.json
      .task_name                       "terminal-bench/fix-ocaml-gc"
      .verifier_result.rewards.reward  0.0 | 1.0
      .agent_result.cost_usd           float | null
      .exception_info                  present when the trial errored

Examples
    python eval/run_eval.py --tag FT1-base --k 4
    python eval/run_eval.py --parse-only jobs/2026-11-20__14-51-35 --tag FT0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REWARD_KEYS = ("reward", "score", "passed", "success", "resolved", "is_resolved")
COST_KEYS = ("cost_usd", "cost", "total_cost", "usd")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator (Chen et al. 2021). None when k > n."""
    if n <= 0 or k > n:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _num(record: dict, keys) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def collect(job_dir: Path) -> list[dict]:
    """One record per trial, from the per-trial result.json files."""
    out: list[dict] = []
    for result in sorted(job_dir.glob("*/result.json")):
        try:
            data = json.loads(result.read_text(encoding="utf8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        reward = _num(rewards, ("reward",)) 
        if reward is None:
            reward = _num(rewards, REWARD_KEYS)
        task = data.get("task_name") or data.get("trial_name")
        if task is None or reward is None:
            continue
        out.append(
            {
                "task": task.split("/")[-1],
                "reward": reward,
                "usd": _num(data.get("agent_result") or {}, COST_KEYS),
                "error": (data.get("exception_info") or {}).get("exception_type"),
            }
        )
    return out


def score(records: list[dict], k: int) -> tuple[dict, dict]:
    per_task: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        per_task[rec["task"]].append(rec["reward"])
    if not per_task:
        return {"tasks": 0, "attempts": 0, "pass1": None, "pass4": None, "usd": None,
                "errors": 0}, {}

    n_min = min(len(v) for v in per_task.values())
    kk = max(1, min(k, n_min))
    pass1 = sum(sum(v) / len(v) for v in per_task.values()) / len(per_task)
    passk = sum(pass_at_k(len(v), sum(1 for x in v if x >= 1.0), kk) for v in per_task.values()) / len(per_task)
    costs = [r["usd"] for r in records if r["usd"] is not None]
    return (
        {
            "tasks": len(per_task),
            "attempts": n_min,
            "pass1": round(pass1, 4),
            "pass4": round(passk, 4),
            "usd": round(sum(costs), 4) if costs else None,
            "errors": sum(1 for r in records if r["error"]),
        },
        {t: v for t, v in per_task.items()},
    )


def append_results(row: dict, path: Path) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow(row)


def build_cmd(args) -> list[str]:
    tasks = Path(args.tasks).read_text(encoding="utf8").split()
    cmd = ["harbor", "run", "-p", args.dataset_root]
    for task in tasks:
        # with -p (local dataset dir) Harbor matches on the directory name;
        # with -d (registry) it matches on the full "org/name" id.
        name = task.split("/")[-1] if args.dataset_root else task
        cmd += ["-i", name]
    cmd += ["-a", "codex", "-m", args.model, "--ak", f"config={args.codex_config}"]
    if args.prompt_template:
        cmd += ["--ak", f"prompt_template_path={args.prompt_template}"]
    for var in ("TINKER_API_KEY", "LITELLM_API_KEY"):
        value = os.environ.get(var)
        if value:  # the agent runs inside the sandbox, so credentials travel with it
            cmd += ["--ae", f"{var}={value}"]
    cmd += ["-k", str(args.k), "-n", str(args.concurrency), "-e", args.env]
    cmd += ["-o", args.jobs_dir]
    return cmd


def redact(cmd: list[str]) -> str:
    secrets = [os.environ[v] for v in ("TINKER_API_KEY", "LITELLM_API_KEY") if os.environ.get(v)]
    text = " ".join(cmd)
    for s in secrets:
        text = text.replace(s, "<redacted>")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="qwen9b")
    ap.add_argument("--tasks", default="eval/eval40.txt")
    ap.add_argument("--dataset-root", default="data/tb2/terminal-bench-2")
    ap.add_argument("--codex-config", default="config/codex_tinker_litellm.toml")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--env", default="modal")
    ap.add_argument("--jobs-dir", default="jobs")
    ap.add_argument("--parse-only", default=None)
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--prompt-template", default=None, help="Jinja2 file prepended to the task instruction")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.parse_only:
        job_dir = Path(args.parse_only)
    else:
        cmd = build_cmd(args)
        print(redact(cmd), flush=True)
        if args.dry_run:
            return 0
        before = {p.name for p in Path(args.jobs_dir).iterdir() if p.is_dir()}
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"harbor exited {proc.returncode}", file=sys.stderr)
        new = [p for p in Path(args.jobs_dir).iterdir() if p.is_dir() and p.name not in before]
        if not new:
            print("no new job directory produced", file=sys.stderr)
            return 3
        job_dir = max(new, key=lambda p: p.stat().st_mtime)

    records = collect(job_dir)
    if not records:
        print(f"no trial results found under {job_dir}", file=sys.stderr)
        return 3

    summary, _ = score(records, args.k)
    print(json.dumps(summary, indent=2))
    append_results(
        {
            "tag": args.tag,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "model": args.model,
            "tasks": summary["tasks"],
            "k": summary["attempts"],
            "pass1": "null" if summary["pass1"] is None else f"{summary['pass1']:.4f}",
            "pass4": "null" if summary["pass4"] is None else f"{summary['pass4']:.4f}",
            "usd": "null" if summary["usd"] is None else f"{summary['usd']:.4f}",
            "errors": summary["errors"],
            "job": str(job_dir),
        },
        Path(args.results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

