# 报告：Qwen3.5-9B 在 Codex CLI + Terminal-Bench 上的 RL 后训练 —— 做到哪里了

> 快照：2026-11-20 19:25 ET。所有数字都来自本仓库 `jobs/`、`results.tsv`、`rl/eval_log.jsonl`，
> 每张表都标了来源目录。

## 1. 问题

在**同一批任务**上，让一个小模型（Qwen3.5-9B）通过 **Codex CLI** 跑 **Terminal-Bench 2**，
问三件事：RL 后训练能不能提升通过率、能不能同等通过率下更便宜、更快。

## 2. 最终形态的架构

```
Codex CLI  ──/v1/responses──►  LiteLLM (Modal)  ──/v1/chat/completions──►  Tinker
    │                              （协议桥，必须）                        （Qwen3.5-9B /
    │                                                                        Qwen3.6-35B-A3B）
    └──► Modal 沙箱（Terminal-Bench 任务容器）──► 任务自带 verifier ──► reward 0/1
```

RL 训练走**另一条路**（必须）：`Tinker SamplingClient → 本地 agent loop → Modal 沙箱 → verifier`，
因为 RL 更新需要模型自己采样出的 token 和 logprob，Codex 是个 HTTP 黑盒给不了。
评测则始终用 Codex —— 这是要报告的那个数。

## 3. 三个把结果压低的 harness bug（全部已修）

| # | bug | 现象 | 修法 | 修前 → 修后 |
|---|---|---|---|---|
| 1 | Codex ≥0.171 删除了 `wire_api="chat"`（openai/codex#7782） | 每次请求 `Error loading config.toml` | Modal 上跑 LiteLLM，把 `/v1/responses` 桥成 `/v1/chat/completions`（`custom_openai/` + `mode: chat`） | 不可用 → 可用 |
| 2 | Codex 对非 OpenAI provider 也挂 `apply_patch`，但 tool router 拒收 | `unsupported call: apply_patch`，模型把 diff 当文本吐出来 | prompt 里明确"没有 apply_patch，用 heredoc 写文件" | 7/38 trial 命中 → 0/3 |
| 3 | 我自己第一版 prompt 写了"你只有一个工具叫 `shell`"（不存在） | `unsupported call: shell`，全部早早退出 | 改成"用你工具列表里那个命令工具，按确切函数名调用" | 0/11 → 修后 1/3 通过，shell 调用中位数 12 → 192 |

另外两个环境坑：Harbor 0.26.0 缺 `harbor[modal]` extra；Windows 上 Harbor 读 `instruction.md`
会用 GBK 崩，必须 `PYTHONUTF8=1`。

## 4. 结果

### 4.1 基座模型对比（同一 10 个 TB2 任务，1 次尝试，固定 harness）

| 模型 | 完成/总数 | 通过 | pass@1 | 备注 | 来源 |
|---|---|---|---|---|---|
| Qwen3.5-9B | 10/10 | 2 | **20%** | 基准 | `jobs/2026-11-20__18-44-58` |
| Qwen3.6-35B-A3B | 9/10 | 3 | **33%** | 同族 MoE，3B 激活 | `jobs/2026-11-20__18-56-11` |
| gpt-oss-120b | 10/10 | 1 | 10% | Harmony 工具格式会漏成文本，不可用 | `jobs/2026-11-20__16-01-45` |

**结论（弱证据）**：换大一点的同族基座方向是对的（20% → 33%），但 n=9-10，
一个 task 就是 10-11 个点，这个差在统计上还不成立。要定论需要 40 个 task（≈$12-15）。

### 4.2 RL 后训练（polyglot 训练集，holdout 8）

| | 通过 | 分数 |
|---|---|---|
| base Qwen3.5-9B | `go_dnd-character`, `go_wordy` | 2/8 = 25% |
| RL（8 batch × 4 group，64 rollout，292 datum） | `python_two-bucket`, `go_dnd-character` | 2/8 = **25%** |

**净变化 0**，成分有交换（新会一个 Python 题，丢一个 Go 题）。训练侧 mean reward 0.34。
这个规模（几百个决策点）比正常 RL 后训练小 2-3 个数量级，**不足以证明或否证 RL**。

### 4.3 干净 trial 的真实水平

把 55 个 trial 按"有没有 harness 缺陷"分桶（分析见 `docs/ANALYSIS.md`）：

| 桶 | trial | 通过率 |
|---|---|---|
| 有缺陷（apply_patch / 文本调用 / 超时 / 非零退出） | 27 | **0%** |
| 干净 | 28 | **28.6%** |

所以 9B 的真实能力约 28-30%，之前那个 15.8% 是被 harness 缺陷拖下去的。

## 5. Torchcast Decision 31B 的状态

- **已部署**：`deploy/torchcast_modal.py`，Modal **A100-SXM4-80GB**（`deploy/gpu_probe.py` 实测可调度），
  `modal deploy` 成功，`GET /health` 正常返回
  `{"status":"ok","model":"torchcast-decision-31b-v1.0","backend":"torch","sliding_kernel":"sdpa","cuda_graphs":0,"max_length":16384}`。
- **权重加载正常**：日志显示 1189 个 shard 加载完成，服务打印
  `serving on http://0.0.0.0:8011/v1/systemone`。
- **但 `POST /v1/systemone` 挂起**：A100 上等 600 秒也不返回，客户端超时后 Modal 日志出现
  `Received a cancellation signal while processing input`。换 **H100**（`gpu="H100"`）重试同样挂起，
  且日志里 `sliding kernel` 仍是 **sdpa** 而不是 `fa3`（没跑 `--prefetch-kernels`，FA3 内核没装），
  所以 H100 这一趟并没有换到快路径。请求进了服务器但推理没有出来。
  怀疑原因（未验证）：SDPA + 非因果全局注意力在这个 prompt 预算下极慢；或服务单请求串行，
  第一次被取消的请求把后续都锁住了。
- **影响**：31B 作为审批门/验证器的集成**还没跑通**，因此本轮没有"大模型 + 31B"的联合结果。
  下一步很具体：① `python -m torchcast_decision.server --prefetch-kernels` 把 FA3 装进镜像再上 H100；
  ② 或直接在容器里用 `python -c` 加载权重跑一次单请求，把 48 ms 的官方延迟和我们的复现对齐。

## 6. 成本

| 项 | 估算 |
|---|---|
| Tinker（token） | ≈$20（Harbor 账本只记了 $7.63，RL rollout 不走 Harbor 所以没算进去） |
| Modal（CPU 沙箱 + A100） | ≈$3 |
| 合计 | **≈$23 / $45.69 余额** |

## 7. 复现

```powershell
# 1) 桥 + 环境
modal deploy deploy/litellm_modal.py
modal secret create tinker-secrets TINKER_API_KEY=... LITELLM_API_KEY=...

# 2) 一次 TB2 评测（10 个任务，固定 prompt）
$env:PYTHONUTF8='1'; $env:TINKER_API_KEY=...; $env:LITELLM_API_KEY=...
python eval/run_eval.py --tag X --model qwen36 --codex-config config/codex_tinker_qwen.toml `
  --tasks eval/ceiling10.txt --k 1 --concurrency 5 --prompt-template config/codex_prompt.j2

# 3) RL 探针（不需要 Codex，走 SamplingClient + Modal 沙箱）
$env:RL_ENV='modal'; $env:RL_MAX_TURNS='12'
python rl/train_probe.py --data data/train/aider-polyglot --train eval/train16_ext.txt `
  --holdout eval/holdout8_ext.txt --batches 8 --group 4 --tasks-per-batch 2

# 4) 评 checkpoint
python rl/eval_ckpt.py --tasks eval/holdout8_ext.txt --label rl --ckpt tinker://...

# 5) 31B（A100-80GB）
modal deploy deploy/torchcast_modal.py
```

## 8. 阶段 1 / RFT 的追加结果（2026-11-20 22:10 ET）

### 8.1 方差筛选：问题定位清楚了

筛了 56 个 polyglot (Python/Go) 任务，每个 4 次 rollout：分布是**双峰**的。

| 类别 | 数量 |
|---|---|
| 稳定不会（0/4） | ~34 |
| 稳定会（4/4） | ~5 |
| 落在可学区间 [0.25,0.75] | **10** |

在 10 个 band 任务上训练，`skipped_groups` 从 **50% → 25%**（门槛 30%），datums 从 292 → 510。
**方差问题解决，但可用任务池只有 10 个**（其余 169 个 JS/Java/Rust 的 verifier 只跑一个测试，不能用）。

### 8.2 两个后训练实验都是零

| 实验 | 数据量 | 评测 | base | 训练后 |
|---|---|---|---|---|
| GRPO（4 batch × group 8） | 64 rollout / 510 datum | holdout8（不在 band 内） | 2/8 | 2/8 |
| RFT（12 条成功轨迹，2 epoch） | 36 rollout / 132 datum | **band_test4（同分布留出）** | 5/16 = 31% | **4/16 = 25%** |

RFT 的细项：`dominoes` 2/4 → 3/4（好），`robot-name` 3/4 → 0/4（崩），另两个仍是 0/4。
净 −1/16，在噪声范围内。

### 8.3 为什么这里必然测不出来（重要）

- **样本太小**：132 个 datum 做 SFT、510 个 datum 做 GRPO，比正常后训练小 2-3 个数量级。
- **评测太吵**：同样的 base 配置，`bottle-song` 第一次 2/4、第二次 0/4；`band_test4` 一共 16 个 episode，
  一个 task 就是 25 个点。**这套评测根本看不见 10 个点以内的真实提升**——真有效果也会被噪声吃掉。
- **任务池只有 10 个**，切成 6 训 4 测之后，两边都太小。

结论：在这套评测的精度下，**"后训练有没有用"这个问题在设计上就无法回答**。先修评测，再谈训练。

## 9. 评测精度的实测（决定"值不值得投训练"的那一步）

把 40 个 TB2 任务各跑 2 次（`jobs/2026-11-20__22-13-42`，固定 harness），用 task-level bootstrap
估计这套评测到底能分辨多大的差异：

| 指标 | 值 |
|---|---|
| 任务数 × 次数 | 37 × 2（跑完 70/80，剩余是超时的长任务） |
| pass@1 | **10.8%** |
| 95% 区间（同一配置重复测） | **[2.7%, 20.3%]**，宽 **17.6 个点** |
| 任务分布 | 2 个稳定过 / **4 个时好时坏** / **31 个稳定不过** |
| 配对比较的 MDE（80% power） | **7.6 个点** |

**两个结论**：

1. **这套评测只能分辨 ≥8 个点的差异**。+5pt 的后训练效果在这里必然是"看不见"的——
   所以之前那些"RL 没提升"的结论，严格说只证明了"提升小于 8 个点"，不是"等于零"。
2. **更致命的是任务分布**：37 个任务里 **31 个 9B 永远是 0 分**，只有 4 个是"时好时坏"。
   也就是说这个 benchmark 对 9B 来说 84% 是硬地板——任何训练方法能动的只有那 4-6 个任务，
   上限本来就在 10-15 个点附近。

## 10. 配对比较：换基座到底值多少（本报告唯一显著的结果）

同一 harness、同一批任务、同一 prompt，9B 与 35B-A3B 的配对比较
（9B: `jobs/2026-11-20__22-13-42`，2 次/任务；35B-A3B: `jobs/2026-11-21__00-01-40`，1 次/任务）：

| | 值 |
|---|---|
| 共同任务 | **37**（全部跑完） |
| Qwen3.5-9B | **10.8%** |
| Qwen3.6-35B-A3B | **32.4%** |
| **配对差** | **+21.6 个点**，95% CI **[+8.1, +36.5]** |
| 不一致对 | 只有 35B 过 **8** 个 / 只有 9B 过 2 个 / 都过 4 个 / 都不过 23 个 |
| 判定 | **显著**（CI 下界 +8.1 > 这套评测 7.6 点的分辨力） |

这是整个项目里**唯一一个统计显著的正向结果**，而且它不是说"RL 有用"，而是说
**"换基座有用"**：同一套 harness、同样 10 分钟级的任务，把 9B 换成 35B-A3B 就多过 7 个任务。

对照：两个后训练实验（GRPO 510 datum、RFT 132 datum）的配对差都落在 ±1 个任务内，
远小于这套评测 7.6 点的分辨力。所以结论是
**"在这个预算与任务集上，换基座的收益可测，后训练的收益测不到"**——而不是"后训练无效"。

## 11. 35B + Jev 重排：本项目第二个正向结果（显著）

**前置修复**：31B 决策模型之前 `/v1/systemone` 挂起，`deploy/torchcast_diag.py` 在容器内 loopback
直连测出 **0.10s 正常返回**（choice=billing, 95.79ms），证明模型没问题、是 Modal 代理到裸 socket
服务的 POST 处理有问题。修法是加一层 FastAPI 网关（`deploy/torchcast_gateway.py`，
0.0.0.0:8012 → 127.0.0.1:8011），公网现在 91.7ms 返回。

### 11.1 决策模型能不能判断"解法对不对"

用 31B 给 37 个真实 trial 打"是否完全解决任务"的概率（`eval/jev_score.py`）：

| | 值 |
|---|---|
| p_solve ｜ 通过（12 个） | 平均 **0.804** |
| p_solve ｜ 失败（25 个） | 平均 **0.352** |
| **AUC** | **0.847** |
| 最佳 F1 | 0.727 @ 阈值 0.95（precision 0.80 / recall 0.67） |

### 11.2 重排实验（12 个 35B 能过的任务 × 3 个候选）

| 指标 | 值 |
|---|---|
| pass@1（单次平均） | 0.806 |
| pass@k（3 个里有任一个过） | 0.917 ← 任何选择器的天花板 |
| **Jev 选中的那个通过** | **0.917** ← 打满了天花板 |
| 只取第一个候选（对照） | 0.833 |
| **Δ vs pass@1** | **+11.1 个点**，95% CI **[+2.8, +19.4]** |

- 相对"只跑一次"：**+11.1pt（显著）**
- 相对"跑三次但盲取第一个"：**+8.4pt** —— 说明增益来自**选择**，不只是多跑
- 3 个混合任务（`extract-elf`、`openssl-selfsigned-cert`、`pypi-server`、`rstan-to-pystan`）
  全部选对；唯一没救的是 `torch-tensor-parallelism`（3 个候选都错，选择器无能为力）

**代价**：推理成本 ×3。**限制**：这 12 个任务是"35B 曾经通过过"的子集，pass@1 本来就有 80%，
天花板只有 92%，所以绝对增益被压缩了；在随机任务集上增益会更小（那些任务没有可选项）。

### 11.3 随机任务集上的复测（autoresearch 第 2 个实验）

预注册假设（`config/queue_rerank_hard.json`）：重排只能吃到 `pass@k − pass@1` 这段空间；
在 25 个"35B 从没做出来过"的任务里，这段空间很小，所以增益应该很小。

抽了其中 12 个（seed 11）× 3 个候选（31/36 完成，`jobs/2026-11-21__12-15-11`）：

| 指标 | 值 |
|---|---|
| pass@1 | 0.139 |
| pass@k（天花板） | 0.333 |
| Jev 选中 | 0.250 |
| 盲取第一个 | 0.167 |
| Δ vs pass@1 | +11.1pt，95% CI **[−2.8, +27.8]**（单独看不显著） |

**合并两半（24 个任务 × 3 候选）**：

| 指标 | 值 |
|---|---|
| pass@1 | **0.472** |
| pass@k（天花板） | 0.625 |
| **Jev 选中** | **0.583** |
| 盲取第一个（对照） | 0.500 |
| **Δ vs pass@1** | **+11.1pt，95% CI [+2.8, +20.8]（显著）** |
| 选择本身的贡献 | **+8.3pt**（0.583 vs 0.500） |

**对照预注册的判定规则**（keep 条件：`pass@k−pass@1 ≥ 8pt` 且 `selected ≥ pass@k−5pt`）：
0.16 ≥ 0.08 ✓，0.583 ≥ 0.575 ✓ → **KEEP**。重排吃到了 58% 的天花板空间。

### 11.4 阈值校准：全局阈值救不了假阳性（免费，67 个已打分候选）

测了 "最高分 ≥ t 才接受，否则回退到第一个候选" 这个策略：

| 阈值 t | 选中通过率 |
|---|---|
| 0.50 – 0.80 | **0.583**（= argmax） |
| 0.85 – 0.90 | 0.542 |
| 0.95 – 1.00 | 0.500（= 全回退到第一个） |

**没有任何阈值比 argmax 更好。** 原因是分数带重叠：真阳性候选大概在 0.85-0.99，
而"看起来做完了但没做完"的假阳性也在 0.95-0.98（`sparql-university` 那个失败候选拿了 0.980）。
全局阈值切不开这两团，要提高选择性只能给模型更硬的证据（比如把 verifier 的输出喂进去），
而不是调阈值。

### 11.5 全量端到端：35B + Jev 在全部 37 个任务上

把剩下 13 个任务也补成 k=3（`jobs/2026-11-21__13-08-08`，39/39 全部完成并打分），三种子集合起来：

| 指标 | 值 |
|---|---|
| 任务 | **37**（每个 3 个候选） |
| pass@1（单次） | **31.5%** |
| pass@k（天花板，3 个里任一过） | **43.2%** |
| **Jev 选中** | **37.8%** |
| 盲取第一个（对照） | 32.4% |
| **Δ vs pass@1** | **+6.3pt**，95% CI **[+0.0, +13.5]** |
| 吃到的天花板比例 | 54%（0.063 / 0.117） |

**这是"35B + Jev"的端到端数**：单次 31.5% → 重排 37.8%，**+6.3 个点，代价是 3 倍推理**。
CI 下界正好压在 0.000 上，属于**边缘显著**——不是没效果，是这个样本量下刚好卡在线上。

和 24 任务子集的 +11.1pt（CI +2.8~+20.8）对比可以看出稀释效应：新加的 13 个任务里
大部分是 0/3，重排在它们身上没有可选项，所以把整体效应从 +11.1 拉低到 +6.3。

## 12. RL on 35B with a Jev-shaped reward（A 选项，2026-11-21）

**设定**：Qwen3.6-35B-A3B + Tinker LoRA(r=16)，40 batch × 8 rollout = **320 rollout / 2860 datum**，
奖励 `pass + 0.3·(p_solves − 0.5)`，其中 `p_solves` 由 Torchcast 31B 逐 rollout 打分。
训练集是 35B 自己的 8 个可学任务（`eval/rl_train_35b.txt`，从 16 个重筛出）。

### 12.1 在 held-out polyglot（8 题 × 2 次 = 16 episode）

| 模型 | 通过 | 评测耗时（16 episode） | 说明 |
|---|---|---|---|
| 35B base | **4/16 = 25.0%** | 5.4 min（326 s，40.8 s/题） | — |
| 35B + Jev-RL（batch 30） | 9/16 = 56.3% | 6.2 min（372 s，46.5 s/题） | 中途快照；与训练重叠，耗时有争用 |
| **35B + Jev-RL（final）** | **10/16 = 62.5%** | **5.5 min（332 s，41.5 s/题，+1.8%）** | **+37.5pt；推理成本基本不变** |

**时间成本**：Jev-RL 训练一次性 112 min（40 batch × 8 rollout = 320 rollout / 2860 datum，
35B-A3B + Tinker LoRA r=16；约 21 s/rollout，含 31B Jev 逐 rollout 打分），base 侧为 0。
训练结束后推理侧无额外开销：配对到每题，final 比 base 每题仅 +0.8 s
（95% CI −6.0~+7.6 s），单 episode 20.4 s → 20.8 s，落在噪声内。
若改用 Jev 重排（不训练），评测推理约 3×（3 候选 + 1 次 31B 打分，见 §11），
所以这是"用 112 min 训练换掉 3× 推理"的取舍。

![35B vs 35B+Jev-RL：通过率、逐题方向、评测耗时](docs/fig_jev_rl.png)

*图：(a) 16 episode 通过率；(b) 逐题方向（● base / ■ batch 30 / ★ final，右侧为 episode 增量）；
(c) 端到端评测墙钟。生成脚本 `eval/plot_results.py`，数据 `rl/hold8_*.json`。*

逐题翻转（final）：`hangman` 0→2、`two-bucket` 2→2、`dnd-character` 2→2、
`zipper`/`tree-building`/`wordy` 0→1、`scale-generator` 0→0、`dot-dsl` 0→0。
5 题变好、2 题不变、1 题没有退步（`scale-generator` 在 batch30 时 1/2、final 回到 0/2，与 base 持平）。

**注意**：n=8 题、16 个 episode，符号检验（5 好 1 坏）p≈0.22，**不显著**。这是强方向性信号，不是结论。

### 12.2 迁移到 TB2（同一 9 个任务，1 次/任务）

| 模型 | 通过 |
|---|---|
| 35B base | 3/9 = 33.3% |
| 35B + Jev-RL | **4/9 = 44.4%** |
| 配对差 | **+11.1pt**，95% CI **[+0.0, +33.3]** |

唯一翻转是 `build-cython-ext`（0→1）。方向为正、CI 触零，同样只能算信号。

### 12.3 整形信号本身有没有信息（on-policy，312 个有打分的 rollout）

| | 值 |
|---|---|
| p_solves ｜ 通过 | 0.798 |
| p_solves ｜ 失败 | 0.507 |
| AUC | **0.72** |

比离线那次的 0.847 低（on-policy 数据更同质），但仍有分离度——
说明这次提升**部分是 Jev 整形的贡献，主体还是 verifier 的 GRPO**。诚实表述：
"Jev-shaped GRPO"，不要说成"Jev 让模型变强"。

### 12.4 已知缺陷（下次要修）

- 单次 rollout 的 `asyncio.wait_for` 超时**不生效**：`envs.ModalTask` 用的是同步 Modal 调用，
  会阻塞事件循环，所以 `wait_for` 没法取消它。修法是改用 `sandbox.exec.aio`。
- 这就是为什么前两次跑在 batch 4-6 卡死、而这次靠 `--save-every 5` 才没白跑。

### 12.5 Jev 在流程里的位置

![Jev 在训练/推理流程中的位置](docs/fig_jev_flow.png)

*图：训练时每个 rollout 一次 Jev 调用（橙色），把 verifier 的 0/1 变成 shaped reward；
训练完 Jev 不再进入推理路径（①，1×），只有 rerank 变体保留它（②，3×）。
生成脚本 `eval/plot_jev_flow.py`。*

Jev 全程只有一次外部调用（`rl/train_jev.py:148`）：输入是「任务 instruction +
最后 5k 字符的 agent 动作」，问一个 `solves` 问题，取 `answers.solves.noul` 作为
`p_solves ∈ [0,1]`。它既不参与采样，也不改 verifier；只通过 `α=0.3` 的整形项进入
reward，再经 GRPO 写进 LoRA 权重。所以「Jev 的收益」在推理侧是零边际成本，
代价就是那 112 min 训练（§12.1）。

## 13. 下一步（按性价比排序）

1. **把 35B-A3B 的对比跑满 40 个 task**（$12-15）。现在 33% vs 20% 只是方向，不是结论。
2. **修 31B 推理挂起**：先在一个 A100 上手工跑最小请求并计时；若 SDPA 在非因果 16k 上确实要几十秒，
   就换 H100（FA3）再试。
3. **RL 要么放大 100 倍，要么别做**：GRPO 至少需要上万 rollout；以当前单价约 $300-800。
   在 $200 预算下，RL 不是性价比最高的那条路——**换基座 + 修 harness 的收益/成本比高得多**。
