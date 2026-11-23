# Terminal-Bench 2 × Codex CLI 基线分析（qwen9b = Qwen3.5-9B）

**快照时间**：2026-11-20 16:05 ET（America/Toronto）。此后新写的 trial 不计入本文件。
**数据范围**：`jobs/` 下 5 个已结束的 job，共 55 个 `result.json` 齐全的 trial。
**排除**：`jobs/2026-11-20__16-01-45`（8 个 trial 目录）与 `jobs/2026-11-20__16-02-08`（6 个 trial 目录）
在快照时仍在运行（快照时分别只有 1 / 0 个 trial 完成），数据还会变，不进入任何表格。两个都是
`eval/ceiling10.txt` 的 10 task × 2 次、qwen9b + `config/codex_tinker_litellm.toml`——即把
`15-37-54` 的 10-task 探针改用 qwen 重跑一遍。
**只读**：本分析只读 `jobs/`、`eval/`、`scripts/`；未调用 `harbor`/`modal`/任何付费 API。

## 口径（先说清楚，否则表会读错）

- **cost**：`result.json` 的 `agent_result.cost_usd` 在全部 55 个 trial 里都是 `null`（job.log 里的原因是
  `No LiteLLM pricing entry for model 'qwen9b'; leaving codex cost_usd as None`）。因此所有金额都是**按
  `scripts/spend.py` 的方法从 token 用量估算**：读 `agent/codex.txt` 里 `type == "turn.completed"` 的
  `usage`，按 Tinker 公开价（prefill $0.30/M、cached $0.06/M、sample $0.807/M）折算。
- **median / mean $/trial**：只在「日志里有 usage（cost > 0）」的 trial 上取中位数/均值，与 `spend.py`
  的处理一致（它跳过 usage 全 0 的日志）；`total $` 则是该 job 全部 finished trial 的和（含 $0 的 trial）。
- **pass**：`verifier_result.rewards.reward >= 1.0`。`pass@1` = passes / finished。
- **apply_patch / leak**：沿用 `scripts/diagnose.py` 的判定——日志出现 `unsupported call: apply_patch`；
  泄漏用正则 `<tool_call>|<function=`。
- 自校验：`jobs/2026-11-20__15-07-24/result.json` 的 job 级 `stats.evals[...].metrics[0].mean = 0.15789`
  与本文的 6/38 = 15.8% 一致；job 级 `n_input_tokens = 68,893,799` 等于该 job 38 个 `result.json`
  的 `agent_result.n_input_tokens` 之和。解析口径对得上。

---

## T1 · 每个 job 汇总

数据来源：`jobs/2026-11-20__14-51-35`、`jobs/2026-11-20__14-59-17`、`jobs/2026-11-20__15-07-24`、
`jobs/2026-11-20__15-37-54`、`jobs/2026-11-20__15-44-02`（各自的 `config.json` + 每个 trial 的 `result.json`
+ `agent/codex.txt`）。

| job | model | dispatched | finished | passes | pass rate | timeouts | nonzero exits | apply_patch | text leak | median $/trial | mean $/trial | total $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2026-11-20__14-51-35` | Qwen/Qwen3.5-9B | 2 | 2 | 0 | 0.0% | 0 | 2 | 0 | 0 | null | null | 0.000 |
| `2026-11-20__14-59-17` | qwen9b | 2 | 2 | 1 | 50.0% | 0 | 0 | 0 | 1 | 0.174 | 0.174 | 0.348 |
| `2026-11-20__15-07-24` | qwen9b | 48 | 38 | 6 | 15.8% | 1 | 3 | 7 | 9 | 0.082 | 0.161 | 5.458 |
| `2026-11-20__15-37-54` | oss120b | 10 | 10 | 0 | 0.0% | 0 | 1 | 3 | 0 | 0.009 | 0.014 | 0.126 |
| `2026-11-20__15-44-02` | qwen9b | 3 | 3 | 1 | 33.3% | 1 | 0 | 0 | 0 | 0.093 | 0.093 | 0.186 |

读法：

- `14-51-35` 是 LiteLLM 桥接之前的失败尝试：两个 trial 的 `codex.txt` 第一行就是
  `Error loading config.toml: wire_api = "chat" is no longer supported`，请求从未发出，
  所以 token = 0、花费 $0.000 是真实值（不是缺数据）。
- `15-07-24` 是唯一的 baseline：48 个 trial 已派发、38 个跑完（job 在 38/80 时被中断，job 级
  `stats`：`n_completed_trials=38, n_running_trials=10, n_pending_trials=32`），10 个目录没有 `result.json`。
- `15-37-54` 是 gpt-oss-120b 的 10-task 上限探测：0/10，且日志中位数只有 ~15 KB、中位数 12 次
  shell 调用（基线是 ~64 次），远没把 task 跑完；3/10 撞上 `apply_patch` 拒绝。它衡量的是 harness
  兼容性，不是 qwen 的能力。
- `15-44-02` 是 harness 修复后的 3-task 复查：1/3（`constraints-scheduling` 过），另外一个
  `adaptive-rejection-sampler` 在 900 s 处超时。

## T2 · `jobs/2026-11-20__15-07-24` 每 task 明细

任务清单来自 `eval/ft1_fast.txt`（20 个 task），job 配置 `n_attempts = 4`；实际派发 48 个、跑完 38 个，
所以每 task 只有 2–3 次已派发、1–3 次跑完。**`pass@4` 无法计算**（见「null 清单」）。

| task | dispatched | finished | passes | pass@1 (finished) |
|---|---|---|---|---|
| adaptive-rejection-sampler | 3 | 2 | 0 | 0.0% |
| break-filter-js-from-html | 3 | 2 | 0 | 0.0% |
| build-cython-ext | 3 | 2 | 0 | 0.0% |
| build-pov-ray | 3 | 2 | 0 | 0.0% |
| code-from-image | 3 | 3 | 0 | 0.0% |
| constraints-scheduling | 3 | 2 | 1 | 50.0% |
| distribution-search | 3 | 2 | 0 | 0.0% |
| dna-assembly | 3 | 1 | 0 | 0.0% |
| dna-insert | 2 | 2 | 0 | 0.0% |
| extract-elf | 2 | 1 | 0 | 0.0% |
| feal-linear-cryptanalysis | 2 | 2 | 0 | 0.0% |
| filter-js-from-html | 2 | 2 | 0 | 0.0% |
| fix-code-vulnerability | 2 | 2 | 0 | 0.0% |
| fix-git | 2 | 2 | 0 | 0.0% |
| git-leak-recovery | 2 | 2 | 2 | 100.0% |
| gpt2-codegolf | 2 | 1 | 0 | 0.0% |
| modernize-scientific-stack | 2 | 2 | 2 | 100.0% |
| mteb-retrieve | 2 | 2 | 0 | 0.0% |
| openssl-selfsigned-cert | 2 | 2 | 1 | 50.0% |
| overfull-hbox | 2 | 2 | 0 | 0.0% |
| **total** | **48** | **38** | **6** | **15.8%** |

任务级：20 个 task 里只有 **4 个**拿到过至少一次 pass（`git-leak-recovery` 2/2、
`modernize-scientific-stack` 2/2、`constraints-scheduling` 1/2、`openssl-selfsigned-cert` 1/2），
即 task-level pass rate = 4/20 = 20.0%（这个 4/20 是从上表数出来的，不是从 job 级字段抄的）。

## T3 · 错误分类（同一批 55 个 trial）

数据来源：上表 5 个 job；按 trial 判定，一个 trial 可落进多个桶（实际只有 1 个重叠，见注）。

| bucket | trials (5 jobs) | passes | pass rate | trials (baseline) | passes | pass rate |
|---|---|---|---|---|---|---|
| apply_patch rejected | 10 | 0 | 0.0% | 7 | 0 | 0.0% |
| text tool-call leak | 10 | 0 | 0.0% | 9 | 0 | 0.0% |
| timeout (AgentTimeoutError) | 2 | 0 | 0.0% | 1 | 0 | 0.0% |
| non-zero exit (NonZeroAgentExitCodeError) | 6 | 0 | 0.0% | 3 | 0 | 0.0% |
| clean (none of the above) | 28 | 8 | 28.6% | 19 | 6 | 31.6% |

- 全部 55 个 trial：8 pass / 55 = 14.5%。
- 27 个 trial 至少命中上面 4 种缺陷之一（桶行加起来是 28，因为有 1 个 trial 同时命中
  `apply_patch` 和 timeout：`jobs/2026-11-20__15-07-24/adaptive-rejection-sampler__5PUQnNX`）。
- `apply_patch` 桶与 leak 桶**完全不重叠**（各 10 个，交集 0）：一个是「工具名不被路由接受」，
  一个是「把调用写成正文」，同一 trial 里通常只会走到其中一条退化路径。
- **所有 pass 都落在 clean 桶**：8/8。反过来，28 个 clean trial 里 8 个过（28.6%），
  27 个有缺陷的 trial 里 0 个过（0.0%）。这是相关性，不是因果——难 task 既更容易触发格式退化，
  也更容易失败；但**没有任何一个 trial 是在已经出现格式退化之后又通过 verifier 的**。

### T5 · 异常类型分布（按 job）

| exception_type | count | jobs |
|---|---|---|
| NonZeroAgentExitCodeError | 6 | 14-51-35 x2, 15-07-24 x3, 15-37-54 x1 |
| AgentTimeoutError | 2 | 15-07-24 x1, 15-44-02 x1 |

`NonZeroAgentExitCodeError` 的根因**不是模型答错**，逐条看 `exception.txt`：

- 14-51-35 的 2 个：`wire_api = "chat"` 配置被 Codex 0.171 拒绝，进程启动即退出（0 token）。
- 15-07-24 的 3 个（全部在 `code-from-image`）：`litellm.BadRequestError: Custom_openaiException - Error code: 400`。
  即 LiteLLM → Tinker 的上游 400，三次重试同一 task 全灭。
- 15-37-54 的 1 个（也是 `code-from-image`）：同一个 LiteLLM 400。

也就是说：6 个「非零退出」全是**基础设施/配置**问题，0 个是模型行为；`code-from-image` 这个 task
在 qwen 上从未真正开始跑过（3 次全 400），它的 0/3 不能当作模型在该 task 上的能力证据。

`AgentTimeoutError` 的 2 个都是 900 s 的 agent 超时（15-07-24 的 `adaptive-rejection-sampler__5PUQnNX`、
15-44-02 的 `adaptive-rejection-sampler__NZZrnLX`）；前者日志里已有 `unsupported call: apply_patch`，
看起来是「工具被拒 → 重试 → 烧完 15 分钟」的路径。

## T4 · 成本交叉校验（log 估算 vs result.json 记账）

`spend.py` 的方法依赖日志里的 `turn.completed`；两种断流会让它偏低：超时被杀时日志被截断、
上游 400 时压根没有事件。`result.json` 的 `agent_result` 是 Harbor 的记账（job 级 token 数就等于它的求和），
但它也会漏（见下表 3 处不一致）。两列都列出来，差异大的逐条解释。

| job | log method $ | result.json method $ | per-trial max $ | zero-usage trials |
|---|---|---|---|---|
| `2026-11-20__14-51-35` | 0.000 | 0.000 | 0.000 | 2/2 |
| `2026-11-20__14-59-17` | 0.348 | 0.348 | 0.348 | 0/2 |
| `2026-11-20__15-07-24` | 5.458 | 5.723 | 6.277 | 4/38 |
| `2026-11-20__15-37-54` | 0.126 | 0.128 | 0.128 | 1/10 |
| `2026-11-20__15-44-02` | 0.186 | 0.735 | 0.735 | 1/3 |

差异来源（逐 trial 核对过）：

- `15-07-24/adaptive-rejection-sampler__5PUQnNX`：log $0.000 vs result $0.561。900 s 超时被杀，
  `codex.txt` 里没有 `turn.completed`，但 result 记了 6.15 M input / 142 k output。**真实花费更接近 $0.56。**
- `15-07-24/build-pov-ray__fgfCi8A`：log $0.554 vs result $0.000。反过来——日志里有完整的
  7.65 M input / 39.8 k output，result 的 `agent_result` 却是 0。**这一笔应记 $0.554。**
- `15-07-24/dna-assembly__aP9Z6vw`：log $0.181 vs result $0.438（5.45 M input）。日志只记到 1.82 M。
- `15-44-02`：整 job 的差额几乎全部来自那个超时 trial（$0.549 只在 result 里）。
- 所以「per-trial max」列 = 逐 trial 取 `max(log, result)`，是对**实际 Tinker 账单**更保守的估计：
  baseline $6.277（不是 $5.458），全 5 job $7.488（不是 $6.118）。**结论不变**：量级 $0.1–0.2/trial。

baseline（`15-07-24`）的关键成本数：

- log 法：total $5.458 / 38 trial = **$0.1436/trial**（含 $0 的失败 trial）；usage>0 的 34 个 trial
  均值 $0.1605、中位数 **$0.0817**；**$/pass = $5.458 / 6 = $0.910**。
- max 法：total $6.277，均值 **$0.1652/trial**、中位数 $0.0768；**$/pass = $1.046**。
- 成本分布是长尾：中位数 ~$0.08，但单个长 trial（4–7 M token）能到 $0.5–0.8，超时 trial 是最贵的。

`scripts/spend.py` 原样输出（注意它 `glob` 所有 job，因此把两个在跑的 job 的日志也算进去了；
它的「latest job」也会指向正在跑的 `16-02-08`，显示 6 dispatched / 0 finished）：

```
latest job : 2026-11-20__16-02-08
  trials   : 6 dispatched, 0 finished, 0 passed
trials with usage : 48
  median $/trial  : 0.056
  max $/trial     : 0.768
estimated spend   : $6.12
gate G1           : $25.00  (remaining $18.88)
```

那个全局 median $0.056 不能当 baseline 用：它把 gpt-oss-120b 那 10 个日志中位数只有 ~15 KB 的
trial（median $0.009）混了进来，分母还是全仓 48 条有 usage 的日志（含正在跑的 job）。
**baseline 用 T1/T4 里的 $0.082 / $0.144。**

## 什么含义（what this implies）

1. **38 个 trial 里 19 个带缺陷，这 19 个全 0。** 拆开是：16 个 `apply_patch` 拒绝 / 文本化工具调用、
   3 个 LiteLLM 400、1 个 900 s 超时（其中超时那个同时带 `apply_patch` 拒绝，所以并集是 19）。
   6 个 pass 全部来自另外 19 个「干净」trial（31.6%）。也就是说，当前分数里有一半的 trial 根本
   没走到「解题」这一步，而是在测 Codex 0.171 + LiteLLM 桥 + 9B 之间的协议兼容性。任何
   「RL 提升 X pt」的结论，如果不先把这个 0% 的桶清掉，都会被噪声吃掉。
2. **失败原因分层很清楚**（下面用 5 个 job 的 55 个 trial 计数）：27 个有缺陷的 trial 里，
   20 个是格式退化（`apply_patch` 10 + 泄漏 10，互不重叠，其中 1 个同时超时），6 个是非零退出
   （全是 LiteLLM 400 / 配置错误，0 个是模型行为），2 个是 900 s 超时；三类并集 20 + 6 + 1 = 27。
   8 个 pass 全部落在 28 个 clean trial 里（28.6%），剩下 20 个 clean trial 才是真正的「不会做」。
   RL 的奖励会被与策略无关的 6/55（LiteLLM 400 直接杀死进程）压成 0。
3. **方差与 headroom 仍然未知。** 每 task 只跑完 1–3 次（多数 2 次），**pass@4 算不出来**，
   `pass@4 − pass@1` 这个 RUNBOOK 里预注册的 headroom 判据目前是 `null`。若按现有 4/20 的
   task-level 结果外推，`git-leak-recovery` / `modernize-scientific-stack` 是 2/2 的稳定通过，
   `constraints-scheduling` / `openssl-selfsigned-cert` 是 1/2 的边缘，其余 16 个 task 0 pass——
   **可学的 task 只有个位数**，课程学习的池子很浅，这一点必须先扩样本再谈 RL。
4. **钱不是瓶颈，样本量才是。** 有效成本 ~$0.14/trial（$0.08 中位数），跑满 20 task × 4 = 80 trial
   约 $11–13，在 G1 的 $25 之内；跑到 40 task × 4 也就 $23–26，刚好压线。真正的约束是
   Modal 上每个 trial 10–15 分钟的墙钟，以及 `code-from-image` 这种会 3/3 全灭在 400 上的 task。
5. **对 RL 的直接含义**：reward 若直接用 verifier 0/1，group 内会有大量「全 0」样本（格式退化 +
   基础设施），基线策略甚至还没进入「解题」这个动作空间。要么先把协议修好（SFT / 约束解码 /
   换文本协议 agent），要么在 reward 里把格式失败单独计分；否则 RL 最先学到的会是
   「别把 tool call 写成正文」——这恰好是 SFT 更擅长的部分。

## null 清单（算不出来就说算不出来）

| 量 | 值 | 为什么 |
|---|---|---|
| `agent_result.cost_usd` | `null`（全部 55 个 trial） | LiteLLM worker 没有 `qwen9b` 的价目表，Harbor 不填 |
| `pass@4` / `pass@k`（k≥2） | `null` | `15-07-24` 配置 `n_attempts=4`，但只派发 48/80、跑完 38 个；每 task 最多 3 次派发、3 次跑完（`code-from-image`，且全 400），多数只有 2 次 |
| task-level `pass@4>0` 的 task 数（课程池） | `null` | 同上，样本不够；只能报「至少 1 次 pass 的 task = 4/20」 |
| `15-37-54` 的「真实能力」 | `null` | 0/10 且 3/10 撞 `apply_patch`、1/10 撞 LiteLLM 400，harness 未修好，不能当能力测量 |
| `16-01-45` / `16-02-08` | 未纳入 | 快照时仍在跑（1/0 个 finished），数字会变 |
| `budget.json:spent_usd` | 0（与实测 $6.12 不符） | 台账没更新；以本文/spend.py 为准 |
| `results.tsv` 的 `CEILING-oss120b` 行 | 指向 `jobs/2026-11-20__15-07-24` | 该 job 的 `config.json` 里 `model_name=qwen9b`，tag 与实际不符；台账需修正 |

---

## 附录 A · 复现命令

所有命令都在仓库根目录 `C:\Users\15633\Downloads\blackbox` 下执行；设置 `PYTHONUTF8=1`
（Windows 上 Harbor 日志含 UTF-8 字符）。**不花一分钱**：只读本地文件。

### A.1 复现 T1–T5（一个脚本，逐字粘贴即可）

```powershell
cd C:\Users\15633\Downloads\blackbox
$env:PYTHONUTF8=1
@'
import glob, json, os, re, statistics, collections

JOBS = [
    "jobs/2026-11-20__14-51-35",
    "jobs/2026-11-20__14-59-17",
    "jobs/2026-11-20__15-07-24",
    "jobs/2026-11-20__15-37-54",
    "jobs/2026-11-20__15-44-02",
]
BASE = "jobs/2026-11-20__15-07-24"
PRICE = {"prefill": 0.30, "cached": 0.06, "sample": 0.807}
LEAK = re.compile(r"<tool_call>|<function=")


def usage(path):
    inp = cached = out = 0
    for line in open(path, encoding="utf8", errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed":
            continue
        u = event.get("usage", {})
        inp += u.get("input_tokens", 0)
        cached += u.get("cached_input_tokens", 0)
        out += u.get("output_tokens", 0)
    return inp, cached, out


def usd(inp, cached, out):
    return ((inp - cached) / 1e6 * PRICE["prefill"]
            + cached / 1e6 * PRICE["cached"]
            + out / 1e6 * PRICE["sample"])


def pct(p, n):
    return "n/a" if not n else "%.1f%%" % (100.0 * p / n)


records, jobs = [], []
for job in JOBS:
    cfg = json.load(open(os.path.join(job, "config.json"), encoding="utf8"))
    agent = (cfg.get("agents") or [{}])[0]
    dirs = sorted(d for d in glob.glob(job + "/*") if os.path.isdir(d))
    finished = [d for d in dirs if os.path.exists(os.path.join(d, "result.json"))]
    costs, result_costs, passes, exc = [], [], 0, collections.Counter()
    patch = leak = zero = 0
    for d in finished:
        rec = json.load(open(os.path.join(d, "result.json"), encoding="utf8"))
        reward = ((rec.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if reward is not None and reward >= 1.0:
            passes += 1
        etype = (rec.get("exception_info") or {}).get("exception_type")
        if etype:
            exc[etype] += 1
        log = os.path.join(d, "agent", "codex.txt")
        text = open(log, encoding="utf8", errors="ignore").read() if os.path.exists(log) else ""
        u = usage(log) if text else (0, 0, 0)
        cost = usd(*u)
        ar = rec.get("agent_result") or {}
        result_costs.append(usd(ar.get("n_input_tokens") or 0,
                                ar.get("n_cache_tokens") or 0,
                                ar.get("n_output_tokens") or 0))
        costs.append(cost)
        zero += 1 if sum(u) == 0 else 0
        patch += 1 if "unsupported call: apply_patch" in text else 0
        leak += 1 if LEAK.search(text) else 0
        records.append(dict(job=os.path.basename(job), dir=d,
                            task=(rec.get("task_name") or "").split("/")[-1],
                            reward=reward, exc=etype, cost=cost,
                            patch="unsupported call: apply_patch" in text,
                            leak=bool(LEAK.search(text))))
    jobs.append(dict(job=os.path.basename(job), model=agent.get("model_name"),
                     dirs=len(dirs), fin=len(finished), passes=passes, exc=exc,
                     costs=costs, result_costs=result_costs,
                     patch=patch, leak=leak, zero=zero))

print("### T1")
print("| job | model | dispatched | finished | passes | pass rate | timeouts | nonzero exits | apply_patch | text leak | median $/trial | mean $/trial | total $ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for j in jobs:
    positive = [c for c in j["costs"] if c > 0]
    median = "%.3f" % statistics.median(positive) if positive else "null"
    mean = "%.3f" % (sum(positive) / len(positive)) if positive else "null"
    print("| `%s` | %s | %d | %d | %d | %s | %d | %d | %d | %d | %s | %s | %.3f |" % (
        j["job"], j["model"], j["dirs"], j["fin"], j["passes"],
        pct(j["passes"], j["fin"]),
        j["exc"].get("AgentTimeoutError", 0),
        j["exc"].get("NonZeroAgentExitCodeError", 0),
        j["patch"], j["leak"], median, mean, sum(j["costs"])))

print()
print("### T2")
agg = collections.OrderedDict()
for d in sorted(glob.glob(BASE + "/*")):
    if os.path.isdir(d):
        agg.setdefault(os.path.basename(d).split("__")[0],
                       dict(disp=0, fin=0, passes=0))["disp"] += 1
for r in records:
    if r["job"] != os.path.basename(BASE):
        continue
    a = agg[r["task"]]
    a["fin"] += 1
    a["passes"] += 1 if (r["reward"] or 0) >= 1.0 else 0
print("| task | dispatched | finished | passes | pass@1 (finished) |")
print("|---|---|---|---|---|")
for t, a in agg.items():
    print("| %s | %d | %d | %d | %s |" % (t, a["disp"], a["fin"], a["passes"], pct(a["passes"], a["fin"])))
print("| **total** | **%d** | **%d** | **%d** | **%s** |" % (
    sum(a["disp"] for a in agg.values()),
    sum(a["fin"] for a in agg.values()),
    sum(a["passes"] for a in agg.values()),
    pct(sum(a["passes"] for a in agg.values()), sum(a["fin"] for a in agg.values()))))

print()
print("### T3")
base = [r for r in records if r["job"] == os.path.basename(BASE)]


def bucket(label, f):
    sel = [r for r in records if f(r)]
    bsel = [r for r in base if f(r)]
    p = sum(1 for r in sel if (r["reward"] or 0) >= 1.0)
    bp = sum(1 for r in bsel if (r["reward"] or 0) >= 1.0)
    print("| %s | %d | %d | %s | %d | %d | %s |" % (
        label, len(sel), p, pct(p, len(sel)), len(bsel), bp, pct(bp, len(bsel))))


print("| bucket | trials (5 jobs) | passes | pass rate | trials (baseline) | passes | pass rate |")
print("|---|---|---|---|---|---|---|")
bucket("apply_patch rejected", lambda r: r["patch"])
bucket("text tool-call leak", lambda r: r["leak"])
bucket("timeout (AgentTimeoutError)", lambda r: r["exc"] == "AgentTimeoutError")
bucket("non-zero exit (NonZeroAgentExitCodeError)", lambda r: r["exc"] == "NonZeroAgentExitCodeError")
bucket("clean (none of the above)", lambda r: not (r["patch"] or r["leak"] or r["exc"]))
passed = sum(1 for r in records if (r["reward"] or 0) >= 1.0)
print()
print("total: %d trials, %d passes, %s" % (len(records), passed, pct(passed, len(records))))
print("distinct defective: %d" % sum(1 for r in records if r["patch"] or r["leak"] or r["exc"]))

print()
print("### T4")
print("| job | log method $ | result.json method $ | per-trial max $ | zero-usage trials |")
print("|---|---|---|---|---|")
for j in jobs:
    best = [max(a, b) for a, b in zip(j["costs"], j["result_costs"])] or [0]
    print("| `%s` | %.3f | %.3f | %.3f | %d/%d |" % (
        j["job"], sum(j["costs"]), sum(j["result_costs"]), sum(best), j["zero"], j["fin"]))
b = [j for j in jobs if j["job"] == os.path.basename(BASE)][0]
positive = [c for c in b["costs"] if c > 0]
best = [max(a, bb) for a, bb in zip(b["costs"], b["result_costs"])]
print()
print("baseline logs : total=%.3f mean_all=%.4f mean_pos=%.4f median_pos=%.4f per_pass=%.3f" % (
    sum(b["costs"]), sum(b["costs"]) / b["fin"],
    sum(positive) / len(positive), statistics.median(positive),
    sum(b["costs"]) / b["passes"]))
print("baseline max  : total=%.3f mean=%.4f median=%.4f per_pass=%.3f" % (
    sum(best), sum(best) / len(best), statistics.median(best), sum(best) / b["passes"]))
allmax = [max(a, bb) for j in jobs for a, bb in zip(j["costs"], j["result_costs"])]
print("all 5 jobs    : log=%.3f result=%.3f max=%.3f n=%d" % (
    sum(c for j in jobs for c in j["costs"]),
    sum(c for j in jobs for c in j["result_costs"]),
    sum(allmax), len(records)))

print()
print("### T5")
exc = collections.Counter()
where = collections.defaultdict(list)
for r in records:
    if r["exc"]:
        exc[r["exc"]] += 1
        where[r["exc"]].append(r["job"].replace("2026-11-20__", ""))
for k, v in exc.items():
    parts = ", ".join("%s x%d" % (j, n) for j, n in collections.Counter(where[k]).items())
    print("| %s | %d | %s |" % (k, v, parts))
'@ | python -
```

### A.2 其它两条（官方脚本，原样跑）

```powershell
cd C:\Users\15633\Downloads\blackbox
$env:PYTHONUTF8=1
python scripts/spend.py
python scripts/diagnose.py jobs\2026-11-20__14-59-17 jobs\2026-11-20__15-07-24 jobs\2026-11-20__15-37-54 jobs\2026-11-20__15-44-02
```

`diagnose.py` 在 16:05 的输出（T1/T3 的 apply_patch / leak 列与它一致）：

```
2026-11-20__14-59-17  trials=2
  apply_patch rejected : 0/2
  text tool-call leak  : 1/2
2026-11-20__15-07-24  trials=38
  apply_patch rejected : 7/38
  text tool-call leak  : 9/38
2026-11-20__15-37-54  trials=10
  apply_patch rejected : 3/10
  text tool-call leak  : 0/10
2026-11-20__15-44-02  trials=3
  apply_patch rejected : 0/3
  text tool-call leak  : 0/3
```

### A.3 每个 trial 的原始字段（抽查单条结论用）

```powershell
python -c "import json,glob; [print(p, (json.load(open(p,encoding='utf8')).get('verifier_result') or {}).get('rewards'), (json.load(open(p,encoding='utf8')).get('exception_info') or {}).get('exception_type')) for p in sorted(glob.glob('jobs/2026-11-20__15-07-24/*/result.json'))]"
```

## 附录 B · 表格与命令的对应关系

| 表 | 来源行 | 复现 |
|---|---|---|
| T1 | `### T1` | A.1 |
| T2 | `### T2` | A.1 |
| T3 | `### T3` | A.1（并用 A.2 的 diagnose.py 交叉验证 apply_patch / leak 计数） |
| T4 | `### T4` | A.1（`baseline logs` / `baseline max` / `all 5 jobs` 三行是表下的汇总数字） |
| T5 | `### T5` | A.1 |

核对方式：写完本文件后，用 `[regex]` 从 A.1 的代码块里原样抽出 Python 代码再跑一次，检查输出中
每个以 `|` 开头的行是否逐字出现在本文件中——结果 `missing_lines=0`。A.1 脚本只列了 5 个 job 目录，
因此在 `16-01-45` / `16-02-08` 跑完之后重跑仍然给出同样的数字（不随新 trial 漂移）。
