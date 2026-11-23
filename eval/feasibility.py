"""Feasibility judge: is this worth continuing, or should we pivot?

It reads a job's per-task attempts (from Harbor results, or a rewards JSON you
hand it) and returns one of three verdicts with the reason:

    GO       headroom exists and the reward signal is dense enough to learn from
    ADJUST   one specific knob is wrong (too easy / too hard / too sparse)
    PIVOT    this model x this harness x this task family has nothing to give

The thresholds are deliberately blunt and pre-registered here -- changing them
after seeing the numbers is how a project lies to itself.

Usage
    python eval/feasibility.py --job jobs/<id>          # real job
    python eval/feasibility.py --rewards rewards.json   # {"task": [0,1,0,1], ...}
    python eval/feasibility.py --demo                   # self-test on 3 scenarios
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# --- pre-registered thresholds ------------------------------------------------
MIN_PASS1 = 0.05          # below: the model cannot do this task family at all
MAX_PASS1 = 0.55          # above: subset too easy, ceiling hides the effect
MIN_HEADROOM = 0.08       # pass@k - pass@1 must be at least this
MAX_ZERO_VARIANCE = 0.70  # share of tasks where every attempt scored the same
MIN_RL_GAIN = 0.05        # holdout gain that justifies the main loop


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        return float("nan")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def metrics(rewards: dict[str, list[float]], k: int) -> dict:
    tasks = {t: r for t, r in rewards.items() if r}
    if not tasks:
        raise SystemExit("no rewards")
    pass1 = st.mean(st.mean(r) for r in tasks.values())
    kb = min(k, min(len(r) for r in tasks.values()))
    passk = st.mean(pass_at_k(len(r), sum(1 for x in r if x >= 1.0), kb) for r in tasks.values())
    zero_var = st.mean(1.0 if len(set(r)) == 1 else 0.0 for r in tasks.values())
    solved_all = st.mean(1.0 if all(x >= 1.0 for x in r) else 0.0 for r in tasks.values())
    return {
        "tasks": len(tasks),
        "attempts": kb,
        "pass1": round(pass1, 4),
        "pass_at_k": round(passk, 4),
        "headroom": round(passk - pass1, 4),
        "zero_variance": round(zero_var, 4),
        "solved_every_time": round(solved_all, 4),
    }


def verdict_prerun(m: dict) -> tuple[str, str, str]:
    if m["pass1"] < MIN_PASS1 and m["pass_at_k"] < 3 * MIN_PASS1:
        return (
            "PIVOT",
            "模型在这个任务族上基本不会：pass@1 和 pass@k 都在地板。RL 没有信号可放大。",
            "换基座（Qwen3.5-35B-A3B，MoE 3B 激活，更便宜更强）或换 agent 协议（mini-swe-agent）。",
        )
    if m["zero_variance"] > MAX_ZERO_VARIANCE:
        return (
            "ADJUST",
            f"{m['zero_variance']:.0%} 的 task 四次尝试全同分，reward 太稀疏。",
            "先做课程：只从 pass@4>0 的 task 训，或加部分分（测试通过比例）。再跑一次 FT-1。",
        )
    if m["headroom"] < MIN_HEADROOM:
        return (
            "PIVOT",
            f"pass@1={m['pass1']:.2f} 而 pass@4={m['pass_at_k']:.2f}，差 {m['headroom']:.2f} < "
            f"{MIN_HEADROOM}：多采样也救不回来，RL 同样没有可吃的方差。",
            "转 SFT：用强模型（Astra/Claude）在 train16 上造 200 条成功轨迹，蒸馏比 RL 更快见效果。",
        )
    if m["pass1"] > MAX_PASS1:
        return (
            "ADJUST",
            f"pass@1={m['pass1']:.2f} 太高，40 个 task 的天花板效应会淹没提升。",
            "只保留 hard 档（30 个）重抽子集，再跑 FT-1。",
        )
    return (
        "GO",
        f"pass@1={m['pass1']:.2f}，pass@4={m['pass_at_k']:.2f}，headroom={m['headroom']:.2f}，"
        f"零方差 {m['zero_variance']:.0%} —— 有方差可吃。",
        "跑 FT-2：16 个 train task 上 10 个 batch，6 个 holdout 上评。",
    )


def verdict_postrun(delta: float, pass4_delta: float) -> tuple[str, str, str]:
    beats_sampling = delta > pass4_delta
    if delta >= MIN_RL_GAIN and beats_sampling:
        return ("GO", f"holdout Δ={delta:+.3f}，且不弱于 best-of-4（{pass4_delta:+.3f}）。",
                "开放主预算：G2 $75，进 autoresearch 主循环。")
    if delta >= MIN_RL_GAIN:
        return ("PIVOT", f"holdout Δ={delta:+.3f}，但 best-of-4 已经能拿到 {pass4_delta:+.3f}。",
                "RL 没有打赢同等推理预算的采样，性价比不成立。转 best-of-4 + 蒸馏。")
    if delta > 0:
        return ("ADJUST", f"holdout Δ={delta:+.3f}，方向对但不够。", "只调一个超参再跑一次 10 步；仍不达标就 PIVOT。")
    return ("PIVOT", f"holdout Δ={delta:+.3f}，训练没带来提升。", "换奖励/课程再试一次；失败则写负面结果，回退到 SFT 或 harness 路线。")


def load_rewards(job: Path) -> dict[str, list[float]]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_eval  # sibling module

    records = run_eval.collect(job)
    if not records:
        raise SystemExit(f"no trial results under {job}; pass --rewards instead")
    _, per_task = run_eval.score(records, k=99)
    return {t: list(v) for t, v in per_task.items()}


def demo() -> int:
    scenarios = {
        "有希望": {f"t{i}": [1, 0, 0, 1] for i in range(6)} | {f"u{i}": [0, 0, 0, 0] for i in range(26)}
        | {f"v{i}": [1, 1, 0, 0] for i in range(8)},
        "太平（无方差）": {f"t{i}": [0, 0, 0, 0] for i in range(38)} | {f"u{i}": [1, 1, 1, 1] for i in range(2)},
        "太简单": {f"t{i}": [1, 1, 1, 1] for i in range(24)} | {f"u{i}": [1, 0, 1, 1] for i in range(16)},
    }
    for name, rewards in scenarios.items():
        m = metrics(rewards, 4)
        v = verdict_prerun(m)
        print(f"\n[{name}] pass1={m['pass1']} pass@4={m['pass_at_k']} headroom={m['headroom']} "
              f"zero_var={m['zero_variance']}")
        print(f"  -> {v[0]}: {v[1]}\n     next: {v[2]}")
    print("\n[post-RL] Δ=+0.09, best-of-4 Δ=+0.04 ->", verdict_postrun(0.09, 0.04)[0])
    print("[post-RL] Δ=+0.09, best-of-4 Δ=+0.12 ->", verdict_postrun(0.09, 0.12)[0])
    print("[post-RL] Δ=-0.02, best-of-4 Δ=+0.10 ->", verdict_postrun(-0.02, 0.10)[0])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--rewards")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--delta", type=float, help="post-RL holdout delta")
    ap.add_argument("--pass4-delta", type=float, default=0.0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        return demo()

    if args.delta is not None:
        v = verdict_postrun(args.delta, args.pass4_delta)
        print(json.dumps({"verdict": v[0], "why": v[1], "next": v[2]}, ensure_ascii=False, indent=2))
        return 0

    if args.rewards:
        rewards = json.loads(Path(args.rewards).read_text(encoding="utf8"))
    elif args.job:
        rewards = load_rewards(Path(args.job))
    else:
        raise SystemExit("need --job, --rewards, --delta or --demo")

    m = metrics(rewards, args.k)
    v = verdict_prerun(m)
    print(json.dumps({"metrics": m, "verdict": v[0], "why": v[1], "next": v[2]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
