# program.md — 自主 RL 研究协议（agent 读这个）

目标：在**固定预算**内，把 `Qwen3.5-9B` 用 RL 后训练，使它在 **Codex CLI + Terminal-Bench** 上的
`pass@1` 高于未训练版本，并且**打赢同等推理预算的 best-of-4**。只改一个变量，一次一个实验。

## 你唯一能改的东西

`experiments/next.py` 里的一个实验配置（数据配比 / 奖励 / 课程 / 超参 / 上下文，五选一）。
训练代码、评测任务集、判分脚本，一律不许动——动了就没有可比性。

## 一次实验的固定预算（不可协商）

- 训练：**10 个 batch × 32 条 rollout**（≈320 episode），硬上限 800 万 train tokens
- 评测：**40 个固定 task × 1 遍**（pass@1），外加 `-k 4` 的 best-of-4 对照
- 成本上限：**$12/实验**，开跑前必须写进 `budget.json` 的 reservation

## 记分与判定

```
主：pass@1 在 40 个固定 task 上的 paired 差值（vs 当前最好那个 checkpoint）
副：pass@4、$/solved、墙钟/solved、零方差 group 占比
keep ⟺ paired bootstrap 95% CI 下界 > 0 且 pass@4 不退步
```

**pass@4 是硬指标**：RL 只涨到 best-of-4 的水平等于没赚。任何实验都要同时报这两个数。

## 一轮循环

1. 读 `results.tsv`，定位当前最好 checkpoint 和最近一次失败的假设。
2. 提一个候选，**只改一个变量**，写清假设和它预测的观测（"如果 X 对，pass@1 应涨 ≥3pt 且 pass@4 不变"）。
3. 先查：这类改动能不能从已有 rollout 免费估个方向（重放/离线统计）。能估就别开跑。
4. 跑 `python eval/run_experiment.py --config experiments/next.py --tag <tag>`
5. 结算真实成本，append 到 `results.tsv`，按上面的 CI 规则 keep/discard。
6. 连续 2 次 discard → 停下来重读数据，不要盲目再试第 3 次。

## 闸门（写死，不要自己放宽）

- 总预算 $200；`budget.json` 每次实验后更新 `spent_usd`。
- 单实验实际 > 预估 2 倍 → 立刻停，查原因（多半是 rollout 太长或重试风暴）。
- 剩余 < $30 → 停止训练，用已有 checkpoint 出报告。
- **零方差 group 占比 > 70%** → 奖励太稀疏，先改课程，不要再加 batch。

## 什么时候找人类

1. 要充值（只有两次：$25 的闸门、$75 的续费）。
2. 需要他在真机上点一次审批/看一眼失败样本（人肉标注，最多 20 条）。

## 不要做

- 不要改评测 task 集、不要改判分、不要挑 task 汇报（reward hacking）
- 不要只看 pass@1 就宣布成功（必须给 pass@4）
- 不要在报告里省略置信区间
- 不要把 infra error 算成模型失败（重试并单独计数）
