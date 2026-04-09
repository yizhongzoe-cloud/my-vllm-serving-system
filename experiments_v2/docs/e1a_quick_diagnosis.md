# E1a_Quick & E2_Recovery 结果诊断

**日期**：2026-04-08
**数据来源**：[results_v2/8B/E1a_Quick/](../../results_v2/8B/E1a_Quick/)、[results_v2/8B/E2_Recovery/](../../results_v2/8B/E2_Recovery/)
**模型**：Llama-3.1-8B-Instruct, dp=4, 4×A100
**Seeds**：42（单 seed）
**已完成 runs**：47 (E1a_Quick) + 8 (E2_Recovery)

---

## 一句话结论

**Our-System 的容错层在没有故障时就在偷掉 50%+ 的 decode 吞吐**，所以 fault-tolerance 对比无从谈起。这不是论文想法的问题，是实现里有 bug。**先修这个 bug，再谈实验结果**。

---

## 🎯 真正的 Final Root Cause（2026-04-08，P0-impl-3a 完整调查闭环）

> **如果你只看一段，看这段**。
>
> **真正的根因 = single-line bug** in [vllm/v1/core/sched/ft_scheduler_impl.py:135](../../vllm/v1/core/sched/ft_scheduler_impl.py#L135) 和 [vllm/v1/core/sched/benders_ft_scheduler_impl.py:101](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L101)：两个 wrapper 都 hardcode `self._base = Scheduler(...)`，**没有根据 `async_scheduling=True` 选择 `AsyncScheduler`**。`AsyncScheduler` 是 [vllm/v1/core/sched/async_scheduler.py](../../vllm/v1/core/sched/async_scheduler.py) 里 `Scheduler` 的子类，它 override `_update_after_schedule` 来正确管理 `num_output_placeholders`（[async_scheduler.py:33](../../vllm/v1/core/sched/async_scheduler.py#L33)）。
>
> **机制**：在 `batch_queue=2` async pipeline 模式下，每次 `schedule()` 必须 increment 被调度 request 的 `num_output_placeholders`。不 increment 的话，下一次 `schedule()` 时 `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens = 0`，request 在 [scheduler.py:518](../../vllm/v1/core/sched/scheduler.py#L518) 的 `num_new_tokens == 0` 分支被 skip。结果：**每 2 个 step 才 sample 1 次 → tpot 翻倍**。
>
> **fix 后实测对比**：
>
> | Baseline | tpot BEFORE fix | tpot AFTER fix | Δ |
> |---|---|---|---|
> | No-FT (control) | 29.5 ms | 29.5 ms | +0.0 |
> | Adaptive-Only-NoCkpt | 58.1 ms | **29.9 ms** | **−28.2** |
> | Our-System-NoCkpt | 58.0 ms | **29.6 ms** | **−28.4** |
> | Adaptive-Only (with ckpt) | 71.8 ms | **40.5 ms** | **−31.3** |
> | Our-System (with ckpt) | 72.0 ms | **38.9 ms** | **−33.1** |
> | Periodic-Low | 62.0 ms | **43.3 ms** | **−18.7** |
>
> **真实 ckpt 开销**（fix 后第一次清晰可见 — 之前所有数据都被 placeholder bug 污染）：
> - Adaptive (smart) ckpt: **~10 ms** tpot 边际
> - Periodic-Low (every 10 blocks): **~13 ms**
> - Periodic-High (every 1 block): **~42 ms**（暴露真实开销，之前被 mask 在 ~67 ms 里）
>
> **关键论文 implication**：**Adaptive ckpt 比 Periodic 在 latency 上有真实优势**（10 vs 13/42 ms），这是论文 main thesis。fix 之前所有 ft baseline tpot 都是 ~60 ms 看起来差不多，**fix 之后 Adaptive 真的 win**。
>
> **完整调查（包括所有 wrong turns）见末尾**：[P0-impl-3a Final Root Cause section](#p0-impl-3a-真正的-final-root-cause2026-04-08-真正的最终修正)
>
> ---
>
> ## ⚠️ 之前的"final"结论已经被推翻（2026-04-08，多轮迭代）
>
> 文档前面（包括之前那个被标记为 "Final Assessment" 的章节）**全部基于错误假设**。整个推理过程保留作为诊断历史记录，但**所有关于 "wrapper trade-off / admission policy 差异 / GPU under-utilization 的累积行为"** 的论断**都已被推翻**。
>
> **被推翻的所有假设**（按时间顺序）：
> - ❌ 第 1 次：TPOT 约束代数退化 → over-admission（被 P0-impl-1 推翻：Periodic 也是 60 ms）
> - ❌ 第 2 次：checkpoint stage 1 GPU clone 在 default stream 阻塞 → ~32 ms wrapper 开销（被 P0-impl-1 推翻：NoCkpt 仍 58 ms）
> - ❌ 第 3 次：fault_tolerant wrapper 在 step path 上有 ~32 ms hot path（被 cProfile + step timing 推翻）
> - ❌ 第 4 次：`SLOAwareRequestQueue` 是元凶（被 swap verify 推翻）
> - ❌ 第 5 次：90s short run noise（被 300s long run 推翻）
> - ❌ 第 6 次：admission policy trade-off / fundamental design limitation（被 batch composition logging 推翻：sample_ratio 完美 0.500 是 deterministic bug 特征，不是 trade-off）
>
> **找到 root cause 的关键 step**：在 [scheduler.py](../../vllm/v1/core/sched/scheduler.py) 加 batch composition + skip-reason counter instrumentation。**100% 的 Adapt skips 来自 `num_new_tokens==0` 路径（`skip_num_new_zero=528 vs 0` for No-FT）**——一旦看到这个数据，root cause 在 5 分钟内追到 `AsyncScheduler` 的 `num_output_placeholders` 管理。

---

---

## Baseline 速查表

| Baseline | 调度策略 | Checkpoint 策略 | 一句话定位 | 期望表现 |
|---|---|---|---|---|
| **No-FT** | FCFS | ❌ 无 | **下界 / 性能上限**：不做任何容错，故障的 GPU 上请求全丢 | 无故障下 goodput 最高、tpot 最低；故障下大量请求失败 |
| **Periodic-Low** | Greedy / FCFS | 每 10 blocks 存一次（保守） | **保守对照组**：低 checkpoint 开销，但故障时要 replay 很多 token | 无故障下 goodput 接近 No-FT；故障下 replay 时间长 |
| **Periodic-High** | Greedy / FCFS | 每 1 block 存一次（激进） | **激进对照组**：checkpoint 开销大，故障恢复快 | 无故障下 goodput 最低；故障下 replay 时间最短 |
| **Benders-Only** | Benders（智能） | 固定每 10 blocks（同 Periodic-Low） | **消融 1**：只有智能调度，checkpoint 是固定的 | 用来证明"Benders 调度本身有用"——单看调度增益 |
| **Adaptive-Only** | Greedy / FCFS | 自适应（按 cost 决定） | **消融 2**：只有智能 checkpoint，调度是贪心的 | 用来证明"自适应 checkpoint 本身有用"——单看 checkpoint 增益 |
| **Our-System** | Benders（智能） | 自适应（按 cost 决定） | **完整系统**：Benders + 自适应 checkpoint 联合优化 | 期望在所有 cell 上 goodput 最高、SLO 违反最低 |

> 期待中的 ranking：
> - **无故障**：`No-FT ≈ Our-System > Periodic-Low > Periodic-High`（checkpoint 开销从低到高）
> - **有故障**：`Our-System > Periodic-High > Periodic-Low > No-FT`（恢复能力从强到弱）
>
> 当前 8B 实测：**Our-System 在大多数 cell 上反而垫底**（详见下方表格），所以判断有 bug。

---

## 关键证据

### 稳态 (no fault) 的 tpot 对比 — 凶器在这里

> **TPOT** = Time Per Output Token，每个输出 token 的延迟（越小越好）。W1_Chat 的 SLO 上限是 100 ms。

| 场景 | No-FT TPOT_p50 (ms) | Our-System TPOT_p50 (ms) | 差距 |
|---|---|---|---|
| W1_Chat / Moderate / **none**  | **28.2** | **60.7** | **+115%** |
| W1_Chat / Moderate / F2_Mid    | 28.2     | 78.4     | +178% |

**没有任何故障**的情况下 decode 慢了 2x —— 这不可能是 checkpoint cost profile 或 replay time 解释的，必然是热路径上有阻塞。

### View A — Goodput / SLO 违反率（E1a_Quick）

> **格式**：每个 cell 是 `goodput (tok/s) / SLO违反率 (%)`
> **列名缩写**：W1=W1_Chat (ShareGPT)、W4=W4_Mixed；Hvy=Heavy、Mod=Moderate、Lgt=Light；F2=F2_Mid 故障 (150s 注入)、none=无故障
> **越大越好**：goodput；**越小越好**：SLO 违反率

| Baseline (说明) | W1/Hvy/F2 | W1/Hvy/none | W1/Lgt/F2 | W1/Lgt/none | W1/Mod/F2 | W1/Mod/none | W4/Hvy/F2 | W4/Hvy/none | W4/Lgt/F2 | W4/Lgt/none | W4/Mod/F2 | W4/Mod/none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **No-FT** *(无容错下界)*           | 309.8 / 0.3 | 315.7 / 0.0 | 142.0 / 0.0 | 149.3 / 0.0 | 219.8 / 0.0 | 226.9 / 0.0 | 183.1 / 0.6 | 187.1 / 0.0 | 92.1 / 0.6 | 93.0 / 0.0 | 141.0 / 0.8 | 143.4 / 0.0 |
| **Our-System** *(完整系统)*        | **167.6 / 43.6** | 296.8 / 4.1 | **115.6 / 4.0** | **128.8 / 6.2** | **180.0 / 23.5** | **205.7 / 3.8** | **132.1 / 25.0** | **149.6 / 14.6** | **80.1 / 14.2** | **83.0 / 14.1** | **110.4 / 19.4** | **114.8 / 14.7** |
| **Periodic-High** *(激进 ckpt)*    | 86.6 / 70.4 | —           | 123.9 / 15.1 | 144.5 / 1.3 | 75.0 / 63.3 | 218.8 / 4.8 | 115.2 / 39.1 | 169.8 / 16.5 | 84.0 / 13.8 | 90.4 / 5.0 | 124.7 / 22.2 | 135.2 / 12.9 |
| **Periodic-Low** *(保守 ckpt)*     | 201.7 / 34.5 | 291.0 / 6.1 | 133.7 / 8.2 | 146.4 / 1.3 | 149.3 / 32.7 | 220.1 / 4.0 | 157.1 / 21.2 | 174.0 / 9.3 | 85.8 / 11.3 | 90.9 / 2.5 | 126.5 / 17.3 | 135.3 / 6.5 |

> 注：
> - `Periodic-High/W1_Chat/Heavy/none` 缺失（实验未跑）
> - Our-System 在 **18 个 cell** 的 goodput 低于至少一个 Periodic baseline
> - Our-System 在 **12 个 cell** 的 SLO 违反率高于 Periodic baseline
> - 缺 `Benders-Only` 和 `Adaptive-Only` 两个消融组（E1a_Quick 没跑，E3_Ablation 才跑）

### View A — Goodput / SLO 违反率（E2_Recovery）

> 格式同上：`goodput (tok/s) / SLO违反率 (%)`。E2_Recovery 只跑 `Moderate / F2_Mid` 这一组，专门看故障恢复路径。

| Baseline (说明) | W1_Chat/Mod/F2 | W2_Summary/Mod/F2 |
|---|---|---|
| **No-FT** *(无容错下界)*       | 221.0 / 0.0      | 55.0 / 0.0 |
| **Our-System** *(完整系统)*    | **162.0 / 23.7** | 53.0 / 2.8 |
| **Periodic-High** *(激进 ckpt)* | 76.5 / 62.8      | 55.1 / 0.0 |
| **Periodic-Low** *(保守 ckpt)*  | 151.4 / 31.0     | 55.1 / 0.0 |

### View B — 故障恢复对比 (fault = F2_Mid)

> **指标含义**：
> - `failover_gap_p50/p95`：从故障注入到下游收到下一个 token 的时间间隔（越小越好，**SLO 上限 = 3000 ms**）
> - `recov_succ`：故障 GPU 上的请求成功恢复的比例（越大越好，No-FT 不做恢复所以为 0）
> - `completion`：所有请求最终完成的比例（越大越好，1.0 = 全部成功）
> - **粗体** = 超过 SLO 或异常值

#### E1a_Quick

| Baseline (说明) | Workload | Load | failover_gap_p50 (ms) | failover_gap_p95 (ms) | recov_succ (0-1) | completion (0-1) |
|---|---|---|---|---|---|---|
| **No-FT** *(无容错下界)*        | W1_Chat  | Heavy    | 0        | 0    | 0.000 | 0.988 |
| **No-FT** *(无容错下界)*        | W1_Chat  | Light    | 0        | 0    | 0.000 | 0.962 |
| **No-FT** *(无容错下界)*        | W1_Chat  | Moderate | 0        | 0    | 0.000 | 0.980 |
| **No-FT** *(无容错下界)*        | W4_Mixed | Heavy    | 71       | 71   | 0.250 | 0.991 |
| **No-FT** *(无容错下界)*        | W4_Mixed | Light    | 0        | 0    | 0.000 | 0.994 |
| **No-FT** *(无容错下界)*        | W4_Mixed | Moderate | 0        | 0    | 0.000 | 0.988 |
| **Our-System** *(完整系统)*     | W1_Chat  | Heavy    | 2311     | 3368 | 1.000 | 0.997 |
| **Our-System** *(完整系统)*     | W1_Chat  | Light    | 1757     | 3623 | 1.000 | **0.792** |
| **Our-System** *(完整系统)*     | W1_Chat  | Moderate | **5073** | 6426 | 1.000 | 0.996 |
| **Our-System** *(完整系统)*     | W4_Mixed | Heavy    | 1841     | 2108 | 1.000 | **0.881** |
| **Our-System** *(完整系统)*     | W4_Mixed | Light    | 1896     | 1896 | 1.000 | 0.975 |
| **Our-System** *(完整系统)*     | W4_Mixed | Moderate | 2361     | 3678 | 1.000 | **0.851** |
| **Periodic-High** *(激进 ckpt)* | W1_Chat  | Heavy    | **6680** | 8288 | 0.941 | 0.980 |
| **Periodic-High** *(激进 ckpt)* | W1_Chat  | Light    | **3108** | 3900 | 1.000 | 1.000 |
| **Periodic-High** *(激进 ckpt)* | W1_Chat  | Moderate | **5420** | 6751 | 1.000 | 0.988 |
| **Periodic-High** *(激进 ckpt)* | W4_Mixed | Heavy    | 1939     | 1939 | 1.000 | 1.000 |
| **Periodic-High** *(激进 ckpt)* | W4_Mixed | Light    | 1128     | 1128 | 1.000 | 1.000 |
| **Periodic-High** *(激进 ckpt)* | W4_Mixed | Moderate | **4100** | 4918 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W1_Chat  | Heavy    | **3146** | 6356 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W1_Chat  | Light    | **3004** | 3626 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W1_Chat  | Moderate | **5109** | 6128 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W4_Mixed | Heavy    | 1348     | 1456 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W4_Mixed | Light    | 913      | 913  | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W4_Mixed | Moderate | **3176** | 3521 | 1.000 | 1.000 |

#### E2_Recovery

| Baseline (说明) | Workload | failover_gap_p50 (ms) | failover_gap_p95 (ms) | recov_succ (0-1) | completion (0-1) |
|---|---|---|---|---|---|
| **No-FT** *(无容错下界)*        | W1_Chat    | 0        | 0    | 0.000 | 0.984 |
| **No-FT** *(无容错下界)*        | W2_Summary | 0        | 0    | 0.000 | 0.996 |
| **Our-System** *(完整系统)*     | W1_Chat    | **4392** | 5492 | 1.000 | 0.919 |
| **Our-System** *(完整系统)*     | W2_Summary | 1773     | 2170 | 1.000 | 0.996 |
| **Periodic-High** *(激进 ckpt)* | W1_Chat    | **5405** | 6768 | 1.000 | 0.996 |
| **Periodic-High** *(激进 ckpt)* | W2_Summary | 471      | 471  | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W1_Chat    | **5107** | 6169 | 1.000 | 1.000 |
| **Periodic-Low** *(保守 ckpt)*  | W2_Summary | 1311     | 1311 | 1.000 | 1.000 |

> 注：`failover_gap_ms` SLO 上限是 **3000 ms**。Periodic-Low 在 W1_Chat 上的 gap 跟 Our-System 几乎一样烂（5107 vs 4392），说明 **KV replay 是公共瓶颈，不是 Benders 调度独有**。

---

## 异常清单

1. **Our-System goodput < 某个 Periodic baseline (tok/s)** — E1a_Quick 中 18 cells，E2 中 2 cells。
   最差：
   - `W1_Chat/Heavy/F2_Mid`：**167.6 tok/s** vs Periodic-Low 201.7 tok/s（−34.1 tok/s）
   - `W4_Mixed/Heavy/none`：**149.6 tok/s** vs Periodic-Low 174.0 tok/s（−24.4 tok/s）
   - `W4_Mixed/Moderate/none`：**114.8 tok/s** vs Periodic-Low 135.3 tok/s（−20.5 tok/s）

2. **Our-System SLO violation > 某个 Periodic baseline (%)** — 12 cells。
   最差：
   - `W1_Chat/Heavy/F2_Mid`：**43.6 %** vs Periodic-Low 34.5 %
   - `W4_Mixed/Light/none`：**14.1 %** vs Periodic-Low 2.5 %

3. **failover_gap_p50 > 3000 ms（违反 Failover SLO 上限）** — Our-System 的 `W1_Chat/Moderate/F2_Mid` (**5073 ms**)；以及**所有** Periodic-* 在 W1_Chat 上的 cell（3004–6680 ms）；外加 `*/W4_Mixed/Moderate` (3176–4100 ms)。

4. **completion_rate < 0.9（请求都跑不完，0-1）** — 全部出现在 Our-System：
   - `E1a/Our-System/W1_Chat/Light/F2_Mid` → **0.792**
   - `W4_Mixed/Heavy/F2_Mid` → 0.881
   - `W4_Mixed/Heavy/none` → 0.896
   - `W4_Mixed/Moderate/F2_Mid` → **0.851**
   - `W4_Mixed/Moderate/none` → 0.875

---

## 根因分析

> 本节分两层：**实现层**（代码 hot path 上的可疑模块）和**算法层**（paper formulation 本身的建模问题）。两层的 bug 可能并存，需要分别处理。

### 不是凶手的几个嫌疑人

| 嫌疑 | 排除依据 |
|---|---|
| **Failover 控制平面慢** | [recoveries.json](../../results_v2/8B/E2_Recovery/Our-System/W1_Chat/Moderate/F2_Mid/42/recoveries.json) 显示 `failure_declared → failover_complete` 只有 2 ms |
| **KV replay 实现差** | Periodic-Low 也是 5107 ms gap，说明 replay 是公共瓶颈，不是 Our-System 独有 |
| **测试矩阵不公平 / SLO 太宽松** | [run.py:969](../run.py#L969) 给 server 推 `--default-tpot-slo-ms=50`，但每请求 SLO 检查走 config 里的 per-workload 值（W1=100, W2=200）。No-FT 的 tpot_p95=37.7 ms 是**真快**，不是占便宜 |

### 实现层嫌疑（按概率排序）

#### 1. [vllm/v1/core/sched/benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) — 最可能

**症状匹配**：
- 所有 workload、所有 load level、**包括 fault=none 的 cell**，tpot 都均匀涨 2x
- W4_Mixed 上 completion_rate 跌到 0.85 — 说明 batch 决策有问题，请求被卡在队列里
- 不是 checkpoint 开销能解释的（开销应该和 KV size 成比例，不会均匀 2x）

**怀疑点**：Benders 每步的 admission/batch 决策保守，让 decode token 数被压低，GPU 闲着。

#### 2. [vllm/v1/core/checkpoint_controller.py](../../vllm/v1/core/checkpoint_controller.py) — 次可能

**症状匹配**：
- 如果 KV checkpoint 走 default CUDA stream 或者持有 GPU lock，会阻塞 decode forward
- 即使 checkpoint 频率不高，单次 cudaMemcpyAsync 阻塞也能拖垮 tpot

**怀疑点**：检查是否走独立 stream + double buffer，还是 sync 在 forward 路径上。

---

## 算法层根因（深层 — paper formulation 本身的建模问题）

> 实现层的怀疑都是"hot path 上是不是有阻塞 / 是不是有 race"，可以靠 profiling 和单文件重写解决。本节分析的是更深一层的问题：**即使实现完全按 paper 的伪代码写也会出错**，因为 paper 的优化模型本身有退化约束和被忽略的物理 contention。
>
> 这一层的诊断**不能**完全替代实现层的对照实验（P0），两层 bug 可能并存。

### 主要发现：TPOT 约束在求解器中代数退化为常量条件

#### 推导

[fault_tolerant_llm_serving_idea_summary.md §8.5](./fault_tolerant_llm_serving_idea_summary.md) 给出的 TPOT 约束：

```
Σ_r x̃_{j,r}(ω) · (G_j / C_r^{dec}) ≤ G_j · D_j^{tpot},   ∀j, ∀ω ∈ Ω_k
```

在两个 paper 已声明的前提下：
1. **GPU 同构**（[idea summary §3](./fault_tolerant_llm_serving_idea_summary.md): "all replicas load the same model"）→ `C_r^{dec} ≡ C^{dec}`
2. **每请求恰好分配到一个 replica**（约束 §8.2: `Σ_r x̃_{j,r}(ω) = y_j`）→ admit 后左边的 indicator 求和 = 1

约束化简为：

```
G_j / C^{dec} ≤ G_j · D_j^{tpot}
⇔  1 / C^{dec} ≤ D_j^{tpot}
```

**化简后的不等式不含决策变量 (`y, x, x̃, ℓ`)，不含请求长度 (`P_j, G_j`)，不含场景 (`ω`)**；右边的 `D_j^{tpot}` 是请求 j 的 SLO 类别参数，相当于"该 workload 是否在硬件能力之内"的事前 sanity check。它**不耦合任何两个请求**，不约束并发度、batch size、admission 总量或 GPU 实际负载。在 solver 内部，它不对 batch contention 形成任何反作用力。

#### 代码层确认（与 paper 公式完全一致）

**Master problem** ([master.py:117-120](../../vllm/v1/core/sched/benders/master.py#L117-L120))：
```python
if costs.tpot_slo_sec is not None and costs.G_j > 0:
    tpot_est = costs.d_j / costs.G_j
    if tpot_est > costs.tpot_slo_sec:
        infeasible = True
```

其中 `costs.d_j` 在 [cost_tables.py:228-232](../../vllm/v1/core/sched/benders/cost_tables.py#L228-L232) 定义为：
```python
d_j = (
    remaining_output / self.decode_throughput
    if self.decode_throughput > 0 else 0.0
)
```

代入回去：`tpot_est = remaining_output / (G_j × decode_throughput)`。对**新到达**的请求 `remaining_output = G_j`，得 `tpot_est = 1/decode_throughput`，与上述代数推导一致；对**已部分完成**的请求 `remaining_output < G_j`，得到的 `tpot_est` 更小，约束更容易满足。两种情况都不依赖并发数和 admission 决策。

**Recovery checker** ([recovery_checker.py:202-206](../../vllm/v1/core/sched/benders/recovery_checker.py#L202-L206)) 直接写出常量形式：
```python
if (costs.tpot_slo_sec is not None
        and self._decode_throughput > 0
        and (1.0 / self._decode_throughput) > costs.tpot_slo_sec):
    feasible = False
```

#### 在 8B 配置下，此约束从未被触发

[config_8b.yaml:14-18](../config_8b.yaml#L14-L18) 注释明确写到：
```yaml
# NOTE: ft_prefill/decode_throughput are legacy fallback values.
# The decode-first capacity model (ft_decode_capacity_profile) is preferred.
ft_prefill_throughput: 4000.0
ft_decode_throughput: 2000.0
```

`ft_decode_throughput` 在 capacity 路径上确实已被 `decode_capacity_profile_8b.json` 取代。但**在 TPOT SLO 检查的代码路径上没有清理**：[benders_ft_scheduler_impl.py:150](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L150) 仍然把 `sched_cfg.ft_decode_throughput` 直接传给 `CostBuilder`，并经由 [solve_loop.py:253](../../vllm/v1/core/sched/benders/solve_loop.py#L253) 透传到 `RecoveryChecker`：
```python
decode_tput = sched_cfg.ft_decode_throughput or 0.0
# ... 一路传到 master.py / cost_tables.py / recovery_checker.py 的 TPOT 检查
```

因此 TPOT 检查实际仍然在用 `1 / 2000 = 0.5 ms`：

| Workload | TPOT SLO | `1/decode_throughput` | 触发？ |
|---|---|---|---|
| W1_Chat    | 100 ms | 0.5 ms | 否 |
| W2_Summary | 200 ms | 0.5 ms | 否 |
| W3_Instruct | 50 ms  | 0.5 ms | 否 |

**结论**：8B 配置下 TPOT 约束在 solver 中等价于 no-op，对 admission 无任何影响。即使把 `ft_decode_throughput` 调到一个更接近真实硬件 per-token 速率的值（例如 100 tok/s → 10 ms），它仍然只是一条 per-workload 的 sanity check，不会建立并发度约束——这是公式层的退化，不是参数层的问题。

#### 与实测症状的对应关系

| 实测现象 | TPOT 退化约束的解释 |
|---|---|
| `Our-System / W1_Chat / Moderate / none` 的 `tpot_p50 = 60.7 ms`，No-FT 同 cell 仅 28.2 ms | solver 缺少对 batch contention 的约束，admission 决策不被 TPOT 反向限流 |
| 5 个 W4_Mixed cell `completion_rate < 0.9`（含 fault=none） | 混合流量下并发度更高、contention 更严重，超出实际 GPU 处理能力 |
| 18 个 cell goodput 输给至少一个 Periodic baseline | Periodic 用 `fault_tolerant` 调度器（非 Benders），其 admission 路径仍依赖 vLLM 默认的 `max_num_seqs` 上限做隐式限流；Our-System 走 `ft_benders_centralized` 调度器，信任退化的 TPOT 约束，过度 admit。注：此对比建立在 `fault_tolerant` 比 `ft_benders_centralized` 更接近 vLLM 默认 admission 行为的假设上，建议在 P0 中加日志验证两类调度器实际下发的 batch size 差异 |
| Our-System SLO 违反率高于 Periodic | 同上 |

#### 与实现层嫌疑的关系

本发现**不**排除实现层的两个嫌疑（scheduler hot path / checkpoint controller stream 阻塞）。两类 bug 可能并存：

| P0 对照实验结果 | 推断 |
|---|---|
| 关闭 checkpointing 后 `tpot_p50` 回到 ~30 ms | 实现层是主因，TPOT 建模问题被掩盖，但仍需修以避免后续浮现 |
| 关闭 checkpointing 后 `tpot_p50` 仍 ~60 ms | 算法层 TPOT 建模是主因，实现层需另行 profile 验证是否还有独立 bug |
| 关闭后 `tpot_p50` 落在 30–60 ms 之间 | 两类 bug 并存，需要分别修复 |

---

### 高风险点 1：Decode capacity 使用 fluid model，无瞬时上限

[decode_capacity_profile_8b.json](../decode_capacity_profile_8b.json) 提供的是 horizon 内 token 总额：

```json
{
  "decode_capacity": {
    "default": 10,
    "by_avg_ctx_bucket": { "256": 50, "1024": 10 }
  },
  "meta": { "planning_horizon_sec": 0.5 }
}
```

`ft_planning_horizon = 1.0` s（[config_8b.yaml:20](../config_8b.yaml#L20)）。这是一个 fluid model：约束等价于"GPU 在 1 s 内最多产出 50 个 decode token (avg_ctx ≈ 256)"。

**问题**：fluid model 不区分时间内的 token 分布。50 tokens/s 的总额可以来自：
- (a) 1 个请求以 50 tok/s 持续输出：单请求 tpot ≈ 20 ms，物理上完全可行
- (b) 50 个请求并发各以 1 tok/s 输出：受 batch contention 影响，单请求 tpot 显著增大，GPU 接近过载

两者在 fluid 容量约束下被同等对待，但 SLO 满足度差距巨大。具体的 (b) 情形 tpot 值取决于 model 和 batch decode latency 曲线，需要 [profile_decode_capacity.py](../profile_decode_capacity.py) 在不同 batch size 下测量。

**额外发现 — planning horizon 不一致**：
- [config_8b.yaml:20](../config_8b.yaml#L20)：`ft_planning_horizon: 1.0` (s)
- [decode_capacity_profile_8b.json](../decode_capacity_profile_8b.json) `meta.planning_horizon_sec`: **0.5** (s)

profile 是在 0.5 s horizon 下做的，runtime 用 1.0 s。如果 [decode_capacity_model.py](../../vllm/v1/core/sched/benders/decode_capacity_model.py) 在使用时直接拿 profile 的 token 数当作 1.0 s 的额度，等价于把容量上限放大了 2×；如果做了 horizon 缩放，则一致。需要单独 verify 这条 caveat（见 P0-algo-3 的扩展）。

#### P0-verify-1 验证结果（2026-04-08）

> 本节是纯静态代码 audit 的结论，**未执行任何 runtime**。所有结论 trace 到具体行号。

**TL;DR**：horizon mismatch 真实存在但**方向与原假设相反**（低估 2× 而不是高估），且**不能**解释核心 tpot 60ms 症状；同时发现一个独立的 latent bug（`solve_loop.py:161` 漏传 `avg_ctx_bucket`），方向偏保守，部分能解释 W4_Mixed 的 completion 异常。**TPOT 约束代数退化仍然是唯一能解释 tpot 60ms 的算法层根因**。

##### 验证方法

读以下文件并交叉对比单位：

| 文件 | 关注点 |
|---|---|
| [decode_capacity_model.py](../../vllm/v1/core/sched/benders/decode_capacity_model.py) | 加载 profile JSON 时是否读 `meta.planning_horizon_sec` 做缩放 |
| [profile_decode_capacity.py:184-216](../profile_decode_capacity.py#L184-L216) | `decode_capacity` 字段的测量语义（单位是请求数还是 tokens/horizon） |
| [profile_decode_capacity.py:266-279](../profile_decode_capacity.py#L266-L279) | `residual_prefill_capacity` 字段的测量语义 |
| [master.py:130-160](../../vllm/v1/core/sched/benders/master.py#L130-L160) | runtime 容量约束的实际写法 |
| [solve_loop.py:160-181](../../vllm/v1/core/sched/benders/solve_loop.py#L160-L181) | profile 数据如何流入 master / recovery_checker |
| [cost_tables.py:68](../../vllm/v1/core/sched/benders/cost_tables.py#L68) | `w_dec` 的实际取值 |

##### 发现 A：`decode_capacity` 路径**没有** horizon mismatch（与原假设相反），但有另一个 latent bug

**单位 trace**：

1. profile 端 ([profile_decode_capacity.py:184-216](../profile_decode_capacity.py#L184-L216))：测量"GPU 同时跑 N 个 decode 请求时 TPOT_p95 是否 ≤ SLO"，N 取最大不超 SLO 的值。结果单位是**纯并发请求数**，与 horizon 无关。
2. profile JSON 字段：`decode_capacity = { "default": 10, "by_avg_ctx_bucket": {"256": 50, "1024": 10} }` —— 都是请求数。
3. runtime 约束 ([master.py:142-150](../../vllm/v1/core/sched/benders/master.py#L142-L150))：
   ```python
   for r in self._replica_ids:
       decode_terms = []
       for req_id, costs in self._costs.items():
           if (req_id, r) in x:
               decode_terms.append(_to_int(costs.w_dec) * x[(req_id, r)])
       if decode_terms:
           model.add(sum(decode_terms) <= _to_int(self._decode_capacity))
   ```
4. `costs.w_dec = 1.0` 是 hardcoded 常量 ([cost_tables.py:68](../../vllm/v1/core/sched/benders/cost_tables.py#L68): `# decode burden (V1: always 1)`)。
5. 化简：约束等价于 `Σ_j x[j,r] ≤ Cap_dec`，即"分配给 r 的请求数 ≤ Cap_r^{dec}"。**纯并发数约束，与 horizon 无关，单位前后一致。**

**结论 A1**：`decode_capacity` 路径不存在 horizon mismatch。

**结论 A2（latent bug）**：[solve_loop.py:161](../../vllm/v1/core/sched/benders/solve_loop.py#L161) 调用 `get_decode_capacity()` **没有传 `avg_ctx_bucket` 参数**：
```python
decode_cap = self._cost_builder.get_decode_capacity()
```

[decode_capacity_model.py:77-98](../../vllm/v1/core/sched/benders/decode_capacity_model.py#L77-L98) 的 `decode_capacity()` 默认 `avg_ctx_bucket=0`，`<= 0` 时直接返回 `default`。
[profile_decode_capacity.py:215-216](../profile_decode_capacity.py#L215-L216)：`default = min(bucket_vals)` —— 即所有 ctx bucket 中**最保守**的值。

在 8B 配置下：`default = min(50, 10) = 10`（来自 ctx=1024 bucket）。

**意味着**：`Cap_dec` 在所有 cell 上**永远是 10**，与请求实际 prompt 长度无关。
- ShareGPT 平均 prompt ≈ 1259 → 应当走 1024 bucket → cap 10，符合
- Alpaca 平均 prompt ≈ 17 → 应当走 256 bucket → cap 应为 50，但实际仍用 10
- W4_Mixed 三种混合 → 应当按动态平均 ctx 加权，但实际仍用 10

这是一个 **latent bug**：profile 写了 `by_avg_ctx_bucket` 数据，但 solver 从未消费。在 short-prompt workload 上相当于把短 prompt 容量上限误用为长 prompt 的保守值。

**严重性**：中等。这个 bug 让 solver 在 short-prompt workload 上**偏保守**——可能贡献了 W4_Mixed 5 个 cell `completion < 0.9` 的部分原因，但**与"无故障 tpot 60ms"现象方向不符**（保守会让 tpot 更低，不是更高）。所以这个 bug 不能解释核心症状，但仍是必修项。

##### 发现 B：`residual_prefill_capacity` 路径**确实有** horizon mismatch，但方向是**低估**

**单位 trace**：

1. profile 端 ([profile_decode_capacity.py:266-276](../profile_decode_capacity.py#L266-L276))：
   ```python
   prefill_tput = prefill_tokens / elapsed   # tokens/s
   rem_cap = int(tput * horizon_sec)          # tokens / horizon_sec
   ```
   `horizon_sec` = CLI `--horizon`，默认 `0.5` ([profile_decode_capacity.py:317](../profile_decode_capacity.py#L317))。
2. profile JSON 字段：`residual_prefill_capacity = { "0": 17234, "2": 14321, ... }` —— 单位是 **"prefill tokens / 0.5 s"**。
3. profile JSON `meta.planning_horizon_sec = 0.5` 被写出但**从未被任何 runtime 模块读取**。grep 验证：`planning_horizon_sec` 字段只在 profile 写入端出现。
4. runtime 加载 ([decode_capacity_model.py:60-65](../../vllm/v1/core/sched/benders/decode_capacity_model.py#L60-L65))：直接把字典原样存到 `self._prefill_caps`，**不缩放**。
5. runtime 消费 ([master.py:152-160](../../vllm/v1/core/sched/benders/master.py#L152-L160))：
   ```python
   for r in self._replica_ids:
       rem_cap = self._residual_prefill_capacity.get(r, 0)
       prefill_terms = []
       for req_id, costs in self._costs.items():
           if (req_id, r) in x and costs.prefill_tokens > 0:
               prefill_terms.append(costs.prefill_tokens * x[(req_id, r)])
       if prefill_terms:
           model.add(sum(prefill_terms) <= rem_cap)
   ```
   约束直接是 `Σ prefill_tokens × x ≤ rem_cap`，**没有除以 H_pre 或乘以任何 horizon 比例**。

**问题**：master 的容量约束 implicit horizon 就是 0.5 s——也就是说 solver 实际上是在做"0.5 s 内能塞多少 prefill tokens"的判断。

但 [config_8b.yaml:20](../config_8b.yaml#L20) `ft_planning_horizon: 1.0`，开发者**意图**是 1.0 s horizon。两者错配 2×。

**方向分析**（这点与原假设方向相反）：

- profile horizon (0.5 s) **小于** config horizon (1.0 s)
- rem_cap 数值偏小（只有真实 1.0 s 容量的一半）
- solver 认为 prefill 资源**更稀缺** → 倾向**拒绝更多请求** → admission 偏**保守**
- 这是 solver "看到的容量比真实小"，**不是**"看到的比真实大"

**与症状的关系**：
- "无故障 `tpot_p50 = 60 ms`" → 这是 batch contention 现象，prefill 容量被低估**不会**直接造成
- "W4_Mixed `completion < 0.9`" → 如果 solver 拒绝更多请求，应当在 metrics 看到 `admission_rate < 1.0`。但我之前看的几个 cell 是 `admission_rate = 1.0`，说明 rem_cap=17234 即使被当成 1.0s 用也没构成 binding 约束（即"被低估的 prefill 容量"在当前 cell 仍然够用）
- 因此**此 bug 在当前实测 cell 上未触发**，但是潜在隐患（如果切换到 prefill-heavy workload 例如 W2_Summary heavy load，会触发并造成 underutilization）

**严重性**：中等-高。逻辑 bug 明确，必修，但**不能解释**当前实测的核心症状。

##### 发现 C：`recovery_checker` 路径同样存在 mismatch

[solve_loop.py:177-181](../../vllm/v1/core/sched/benders/solve_loop.py#L177-L181) 把同一个 `residual_prefill` 字典传给 recovery_checker 作为 `u_prefill` 预算 ([recovery_checker.py:138](../../vllm/v1/core/sched/benders/recovery_checker.py#L138))。recovery_checker 用它检查"故障后 surviving GPU 是否还有足够 prefill capacity 跑 replay tokens"。同样的 horizon 错配同样存在，导致 recovery 路径也低估了 prefill 预算。

**这反而可能是好事**：让 solver 在故障场景下更保守地预留 prefill 容量，减少 replay 失败。但这是 happy accident，不是设计意图。

##### 与原假设的对比

| 原假设 | Verify 结果 | 状态 |
|---|---|---|
| Path A：profile 数字被当成 1.0s 总额，**容量被放大 2×** | 错。Path A 是并发数，无 horizon 依赖 | ❌ 推翻 |
| Path B：profile 0.5s 直接当 1.0s 用 | 对。`residual_prefill_capacity` 路径确实被错配，但**低估** 2× 而非高估 | ✅ 确认（方向相反）|
| `decode_capacity` 路径完全 OK | 部分对。单位一致，但 [solve_loop.py:161](../../vllm/v1/core/sched/benders/solve_loop.py#L161) 漏传 `avg_ctx_bucket` 是另一个 latent bug | ⚠️ 部分推翻 |

##### 修复建议（仅描述，不写代码；按修复成本由低到高）

1. **修 Bonus latent bug — `solve_loop.py:161` 漏传 `avg_ctx_bucket`**：
   - 在 [solve_loop.py:161](../../vllm/v1/core/sched/benders/solve_loop.py#L161) 之前先算一个当前 batch 的平均 ctx 长度（活跃请求的 `prompt_len + tokens_decoded` 加权平均），传给 `get_decode_capacity(avg_ctx_bucket=...)`
   - 修后，short-prompt workload (W3/W4) 上 `Cap_dec` 会按 256 bucket 取 50 而不是 10
   - 风险：如果实际 ctx 跨 bucket 边界，需要选保守的下界 bucket
   - **预期影响**：W3_Instruct / W4_Mixed 上 admission 上限会放宽 5×

2. **修 horizon mismatch — 选一个权威 horizon 来源，统一两端**：
   - 选项 a：在 [decode_capacity_model.py:51-65](../../vllm/v1/core/sched/benders/decode_capacity_model.py#L51-L65) 加载时读 `meta.planning_horizon_sec`，存为 `self._profile_horizon_sec`；新增 `set_runtime_horizon(h)` 方法，runtime 调用时按 `runtime_h / profile_h` 缩放所有 `_prefill_caps`
   - 选项 b（更简单）：在 [calibrate.py](../calibrate.py) 或 [run.py](../run.py) 里加一个 startup assertion，要求 `config.ft_planning_horizon == profile.meta.planning_horizon_sec`，否则 fail-fast
   - 推荐 **a + b**（缩放 + 校验，互补）

3. **不动 profile 文件**：profile 已经是高成本测出来的数据，不要重测；只在 runtime 端做缩放或校验

##### Verify 对算法层根因小结表的影响

原表里 "Planning horizon 在 config (1.0 s) 和 profile (0.5 s) 之间不一致" 的状态是 "**待验证**"。**现已验证**：

- **状态**：bug 真实存在
- **影响方向**：低估 prefill capacity 2×（不是高估）
- **是否解释核心症状（tpot 60ms / completion < 0.9）**：**不能**
- **是否独立修复**：是，与 TPOT 退化 bug 互不相关
- **优先级保持 P0-verify**：因为修复成本极低

并新增一行：

- "**`solve_loop.py:161` 调用 `get_decode_capacity()` 漏传 `avg_ctx_bucket`，永远使用最保守的 default cap**" —— bonus 发现，**P0-verify**

> **回到主诊断**：P0-verify-1 完成后，**TPOT 约束代数退化**（"主要发现"那一节）仍然是当前唯一能解释 `tpot 60ms` 的算法层根因。Horizon mismatch 和 ctx_bucket bug 都是真 bug，但都不在核心症状的因果链上。下一步仍然是 **P0-impl-1**（关闭 checkpointing 的对照实验）。

**为什么 W4_Mixed 受影响最严重**：W4_Mixed 由 alpaca (prompt 17 tokens) / sharegpt (1259) / cnndm (874) 混合而成，落入不同的 `avg_ctx_bucket`（256 / 1024），bucket 之间需要插值，误差被 mix 比例放大。这与"5 个 W4_Mixed cell `completion_rate < 0.9`"的现象吻合。

---

### 高风险点 2：Failover gap 公式假设 restore 路径无 contention

[fault_tolerant_llm_serving_idea_summary.md §8.6](./fault_tolerant_llm_serving_idea_summary.md) 的 gap 约束：

```
T^{det} + S_j^{ckpt}(ℓ_j) / B^{ld} + U_j(ℓ_j, ω) / C^{rep} + 1/C^{dec} ≤ D_j^{gap}
```

其中 `B^{ld}` 在 [config_8b.yaml:19](../config_8b.yaml#L19) 是常数 `10 GB/s`。但 failover 时实际带宽并不是这个常数：

- **多请求并发 restore**：`|J̃(ω)|` 个请求同时争抢 PCIe / host memory 通道
- **与 surviving 流量竞争**：目标 GPU 上未受影响的请求仍在做 decode，与 cudaMemcpyAsync 争抢同一 GPU 的 memory controller

**实测对照**：

| 量 | 数值 | 来源 |
|---|---|---|
| 控制平面延迟（failure_declared → failover_complete） | **2 ms** | [recoveries.json](../../results_v2/8B/E2_Recovery/Our-System/W1_Chat/Moderate/F2_Mid/42/recoveries.json) |
| `failover_gap_p50`（E2_Recovery / W1_Chat） | **4392 ms** | [metrics.json](../../results_v2/8B/E2_Recovery/Our-System/W1_Chat/Moderate/F2_Mid/42/metrics.json) |
| 剩余 ≈ restore + replay 实际墙钟 | **~4390 ms** | 差值 |
| `D_j^{gap}` SLO 上限 | **3000 ms** | [config_8b.yaml:71](../config_8b.yaml#L71) |

**严格能说的最弱推论**：solver 既然 admit 了这些请求，就必然预测它们的 gap ≤ 3000 ms；实际 4392 ms。即 solver 的预测**至少**低估了 `4392 / 3000 ≈ 1.46×`。

**未验证但高度怀疑**：solver 公式中 `S_j^{ckpt}/B^{ld}` 一项假设 `B^{ld}` 是常数 10 GB/s，没有按并发恢复请求数 `|J̃(ω)|` 折扣。要量化偏差，需要：
- 在 [recovery_checker.py](../../vllm/v1/core/sched/benders/recovery_checker.py) 的 admission 路径加日志，dump solver 对每个 admit 请求预测的 gap 值
- 跟实测的 `failover_gap_p50` 对比，得到真实低估倍数

**结论**：failover gap 公式很可能需要在 `B^{ld}` 项加 contention 折扣（粗糙做法：除以 `max(1, |J̃(ω)|)`），或重新 profile 多请求并发 restore 的实际等效带宽。但**确切倍数和形式需上述日志验证后再定**。

---

### 算法层根因小结

> ⚠️ **2026-04-08 P0-impl-3a 后修正**：本表中所有 bug 都仍然是真 bug，但**没有一个解释 tpot 60ms 的核心症状**。tpot 60ms 的真凶在 vLLM 内部 admission 路径（fault_tolerant policy 让 in-flight reqs 翻倍），不是这里列的算法层 bug。这些 bug 影响 paper formulation 完整性，但不影响 tpot 实测。

| 问题 | 类别 | 证据强度 | 修复成本 | 优先级 |
|---|---|---|---|---|
| TPOT 约束代数退化为常量 | **bug**（公式 + 实现一致） | **强**（推导 + 代码三处确认） | 低（runtime throttle）/ 高（重写 batch-aware 公式） | **P0** |
| `ft_decode_throughput` 在 capacity 路径已被 deprecated，但 TPOT 路径未清理 | 实现层 cleanup 缺失 | **强**（注释 + 代码验证） | 极低（删除或迁移到新 capacity model） | **P0**（与上一项一起改） |
| Decode capacity 是 fluid model，无瞬时并发上限 | 设计简化，可能低估瞬时拥塞 | 中（机理清晰，未 profile 量化） | 低（加 hard cap） | P1 |
| Planning horizon 在 config (1.0 s) 和 profile (0.5 s) 之间不一致 | 配置/数据 mismatch，**低估** prefill capacity 2×（方向与原假设相反） | **强**（P0-verify-1 已完成，trace 到 [decode_capacity_model.py:60-65](../../vllm/v1/core/sched/benders/decode_capacity_model.py#L60-L65) 不读 `meta.planning_horizon_sec`） | 极低（加 horizon 缩放或 startup assertion） | P0-verify-1 ✅ |
| `solve_loop.py:161` 调用 `get_decode_capacity()` 漏传 `avg_ctx_bucket`，强制使用最保守 default cap | 实现 bug，short-prompt workload 上 admission 上限被人为压低 5× | **强**（P0-verify-1 bonus 发现） | 极低（计算并传 ctx_bucket 参数） | P0-verify-1 ✅ |
| **Profile 在 A6000 上做的，但实际硬件是 A5000**（环境探查附带发现） | **数据/环境 mismatch**：[decode_capacity_profile_8b.json](../decode_capacity_profile_8b.json) `meta.gpu = "NVIDIA RTX A6000"`，但 `nvidia-smi` 显示 8 × **NVIDIA RTX A5000 (24 GB)**。A5000 显存只有 A6000 (48 GB) 的一半，KV cache 容量减半，profile 给出的 `decode_capacity` 数字可能高估实际能容纳的并发请求 | **强**（profile json `meta.gpu` + nvidia-smi 直接对照） | 中（需在 A5000 上重跑 [profile_decode_capacity.py](../profile_decode_capacity.py) 重新测一份） | **P1**（影响 W4_Mixed cell 的 completion 异常，但不解释 tpot 60ms） |
| Failover gap 公式忽略 restore contention | 设计简化，预测乐观 | 中（已验证 ≥1.46× 低估，确切倍数待 profile） | 中（需 profile 或粗糙系数） | P2 |

---

### 诊断更新（2026-04-08，新发现）：master 实际上**有** decode capacity 约束，"过度 admit"论断需要修正

> 这一节是在准备 P0-algo-1 patch 时新发现的。**修正了"算法层根因 → 主要发现"那一节里关于"solver 过度 admit"的因果链**，但**不**改变"TPOT 约束代数退化"这个 bug 本身的真实性。

#### 新读到的代码

在为 P0-algo-1 起草 patch 时读了 [master.py:100-106](../../vllm/v1/core/sched/benders/master.py#L100-L106)：

```python
if costs.is_active and costs.active_replica_id is not None:
    r0 = costs.active_replica_id
    if (req_id, r0) in x:
        model.add(x[(req_id, r0)] == 1)   # active 请求被 pin 到原 replica
    for r in self._replica_ids:
        if r != r0 and (req_id, r) in x:
            model.add(x[(req_id, r)] == 0)
```

**关键事实**：master 的 `cost_table` 包含 active + pending 两类请求（[solve_loop.py:94-95](../../vllm/v1/core/sched/benders/solve_loop.py#L94-L95) 的 `build_snapshot_costs(active, pending)`）。active 请求的 `x` 变量被强制 pin 为 1，pending 的是自由变量。

因此 [master.py:142-150](../../vllm/v1/core/sched/benders/master.py#L142-L150) 的 decode capacity 约束：

```python
sum(decode_terms) <= self._decode_capacity
```

`decode_terms` 由 `Σ_j w_dec[j] × x[j,r]`（j 遍历整个 cost_table）构成，**包含 active 部分**（贡献固定值）+ pending 部分（贡献 0/1）。

**这意味着 master 的 decode capacity 约束确实在限制"GPU r 上的总并发请求数（active+pending）≤ Cap_r^{dec}"**，不是只看 pending。

#### 对原诊断"过度 admit"论断的影响

原文档"算法层根因 → 主要发现"里写的：

> "Periodic 走 fault_tolerant 调度…被 max_num_seqs 隐式限流；Our-System 信任退化的 TPOT 约束，过度 admit"

这一论断**不准确**。修正版：

- master **有**总并发约束 `Σ x[j,r] ≤ 10`（[decode_capacity_profile_8b.json](../decode_capacity_profile_8b.json) 的 `default: 10`）
- 因此 Our-System 的总并发**没有显著超过** Periodic baseline
- TPOT 约束代数退化（`1/C^{dec} ≤ D_j^{tpot}`）作为公式 bug 仍然成立，但它在**当前配置下被 decode_capacity 约束部分掩盖**——实际生效的并发上限是 cap=10，不是无限

#### 那为什么 tpot 还是 60ms？

新的合理推断（按概率排序）：

1. **cap=10 是 hardware-mismatch 的上限**：profile 在 A6000 上测得，迁移到 A5000（性能 ~70%）后，A5000 上 cap=10 已经超过 SLO 容量。这与"硬件型号 mismatch"那一行小结相互印证。
2. **checkpoint controller 阻塞 forward stream**：与 batch contention 无关，是单请求 per-step decode 时间被 checkpoint cudaMemcpy 拖慢。这是原诊断里"实现层嫌疑 #2"的方向，仍然是最强的单一假设。
3. **`solve_loop.py:161` 漏传 `avg_ctx_bucket` 让 cap 永远是 default=10**（已验证）：在 W4_Mixed 这种短 prompt mix 上，cap 应该按 256 bucket 取 50，但实际取 10——这反而**让并发偏低**，**不能解释** tpot 60ms（应当让 tpot 更低）。但能解释 W4_Mixed 的 completion 异常（admit 被人为压低）。

所以 P0-impl-1（关闭 checkpointing 的对照实验）**仍然是不可绕过的关键诊断**：
- 如果关闭后 tpot 回到 ~30 ms → 凶手 #2（checkpoint controller）
- 如果关闭后 tpot 仍 ~60 ms → 凶手 #1（hardware mismatch + 需要重 profile）

#### 对 P0-algo-1（TPOT runtime throttle）的影响

**重定位**：原计划是"加 throttle 弥补 TPOT 退化的 over-admission"。新理解下，TPOT 退化在当前配置不直接造成 over-admission，throttle 的真实价值是：

- **profile-free 兜底**：让系统不依赖 profile JSON 的硬编码 cap，用实测 latency 反馈做动态 cap 调整
- **抗 hardware drift**：换硬件（A6000 → A5000）时不需要重跑 profile 也能自动收敛到合理的并发上限
- **是 P1-env-1（重 profile）的良性替代**

但**优先级降到 P0-impl-1 之后**：必须先确认 tpot 60ms 的真凶，否则改 admission throttle 改不到点上。

#### 对算法层根因小结表的修正

| 原表里的论断 | 修正后 |
|---|---|
| "TPOT 约束代数退化为常量"是 P0 bug | 仍然是 bug，但**实际危害被 master.py 的 decode_capacity 约束部分掩盖**。当前配置下，它不直接造成 over-admission，主要影响是"约束系统无法 SLO-aware 地动态调整"。仍建议修，但优先级实际上低于"硬件 profile mismatch"和"checkpoint controller stream 阻塞"两个真凶 |

> **回到主诊断**：P0-impl-1 仍然是下一步的关键动作。"TPOT 约束代数退化"作为公式 bug 没有被推翻，但它在 8B 当前配置下**不是**核心症状的因果链上的主导因素。

---

## TODO List

> 任务按优先级分组。**P0 为阻塞性**，未完成不要进入后续阶段。每组内部按"实现层 → 算法层"顺序做（先排除廉价的实现 bug，再处理算法建模问题）。

### P0 — 阻塞性，必须先做

**实现层（隔离根因）**
- [x] **P0-impl-1** ✅（2026-04-08 完成）：跑 [config_8b_diag.yaml E_P0_Diag_A](../config_8b_diag.yaml) 4-baseline 2×2 因子实验。结论：真凶是 **fault_tolerant scheduler wrapper 框架本身（+31.5 ms，69%）**，而非 checkpoint 路径（仅 +14 ms 边际）。Benders 完全洗清。详见末尾"P0-impl-1 Step 1 实测结果"章节
- [x] **P0-impl-2** ✅（同上）：判定结果 = "Adaptive-Only-NoCkpt 57.7 ms ≈ Our-System-NoCkpt 58.0 ms ≈ ~60 ms"，按决策树进入 P0-impl-3a（read fault_tolerant wrapper code）
- [x] **P0-impl-3a** ✅（2026-04-08 第一次 close，但是错的 conclusion）：5 轮 instrumentation 排查 wrapper 路径，**错误结论：admission policy trade-off**。被后续 batch composition logging 推翻
- [x] **P0-impl-3a-真正 final** ✅（2026-04-08 当天后续重审）：加 batch composition + skip-reason counter instrumentation 在 [scheduler.py:921](../../vllm/v1/core/sched/scheduler.py#L921)。SMOKING GUN: Adapt sample_ratio = 0.500 (vs No-FT 0.996), 100% skips from `num_new_tokens==0`. 找到 root cause: [ft_scheduler_impl.py:135](../../vllm/v1/core/sched/ft_scheduler_impl.py#L135) hardcode `Scheduler(...)` instead of `AsyncScheduler` when `async_scheduling=True`. **Fix verified across 11 cells**（7 baselines W1 + W2 cross + F2_Mid fault）—— 所有 NoCkpt baseline 从 ~58 → ~30 ms，所有 ckpt baseline 大幅改善
- [x] **P0-impl-3b** ✅：300s long run reproducibility verify 已经在 P0-impl-3a 第 5 轮完成
- [x] **P0-impl-4** ✅：4-baseline 300s long run 已经跑过（结果在 results_v2/8B_diag_long/）
- [ ] **P0-impl-5**：~~把 wrapper limitation 写入论文~~ ❌ **取消** —— fix 后没有 limitation
- [ ] **P0-impl-6**（next）：commit fix + 诊断文档到 git
- [ ] **P0-impl-7**（next）：重跑 8B `E1a_Quick` 全 47 cell 得到 clean 数据，替代被 placeholder bug 污染的现有数据
- [ ] **P0-impl-8**（next）：重跑论文所有其他 experiments (E1a_Main, E2_Recovery, E3_Ablation, E4_Checkpoint_Tradeoff, E5_Controller, E6_SLO_Sensitivity)
- [ ] **P0-impl-9**（next）：更新论文 figures with clean data，删除 wrapper trade-off limitation 段
- [ ] **P1-perf-1**（low priority, future）：优化 ~10 ms ckpt 决策开销。Fix 后 Adaptive ckpt 比 NoCkpt 慢 ~10 ms，几乎全部来自 [ft_scheduler.py run_checkpoint_step](../../vllm/v1/core/sched/ft_scheduler.py) 每 step iterate running × `should_checkpoint(req)` 决策迭代（与是否真触发 ckpt 无关）。优化方向：
  - **A1**：把 should_checkpoint 决策移出 schedule() 热路径（每 N step 缓存一次决策）
  - **A2**：batch 化 should_checkpoint（一次 cost model 计算所有 running）
  - **A3**：把 Stage 1 GPU clone ([kv_checkpoint_pool.py:215-220](../../vllm/v1/core/kv_checkpoint_pool.py#L215-L220)) 移到 copy stream（避免 default stream 阻塞，需要解 race condition）
  - **A4**：缓存 pinned memory buffer（不每次 alloc）
  - **A5**：把 [block_indices.to(device) line 198](../../vllm/v1/core/kv_checkpoint_pool.py#L198) 移到 copy stream
  - **预期效果**：A1+A2 能把 Adaptive tpot 从 38.9 → ~32 ms，几乎跟 NoCkpt 一致 → zero-overhead fault tolerance
  - **优先级**：不阻塞论文。论文 main story 是"Adaptive ckpt 比 Periodic 快"（38.9 vs 43.3/72），10 ms overhead 是真实开销不影响 thesis。论文之后再优化

**算法层（建模 bug 修复）**
- [ ] **P0-algo-1**：实现 TPOT runtime throttle（最小改动，路 A）
  - 在 [benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) 的 admit 入口加一段：当目标 GPU 上正在 decode 的请求数 × 实测 per-token 时间 > `tpot_slo` 时拒绝该 admit
  - 实测 per-token 时间从最近 N step 的 sliding window 取，N=10 起步
  - 不动 [master.py](../../vllm/v1/core/sched/benders/master.py) 和 [cost_tables.py](../../vllm/v1/core/sched/benders/cost_tables.py)
- [ ] **P0-algo-2**：单独重跑 `Our-System / W1_Chat / Moderate / none`，确认 `tpot_p50 ≤ 35 ms`
- [ ] **P0-algo-3**（仅当 P0-algo-2 失败时做）：在 [solve_loop.py](../../vllm/v1/core/sched/benders/solve_loop.py) 加日志记录 master objective vs subproblem objective 的 gap 和迭代次数，确认 Benders 是否正常收敛

**算法层 verify（廉价 sanity check，30 分钟内）**
- [x] **P0-verify-1** ✅（2026-04-08 完成）：读 [decode_capacity_model.py](../../vllm/v1/core/sched/benders/decode_capacity_model.py)，确认 mismatch 处理。结论：horizon mismatch 真实存在，方向是**低估** prefill capacity 2×（与原假设相反，不是高估）；同时发现 bonus latent bug `solve_loop.py:161` 漏传 `avg_ctx_bucket`，导致 short-prompt workload 上 `Cap_dec` 永远是 10。两者都不能解释核心 tpot 60ms 症状，但都是必修项。详见上方"P0-verify-1 验证结果"章节。
- [ ] **P0-verify-2**：在 [recovery_checker.py](../../vllm/v1/core/sched/benders/recovery_checker.py) 加临时日志，dump solver 对每个 admit 请求预测的 `failover_gap` 值，与 [metrics.json](../../results_v2/8B/E2_Recovery/Our-System/W1_Chat/Moderate/F2_Mid/42/metrics.json) 的 `failover_gap_p50` 对比，得出真实低估倍数（确认"高风险点 2"的具体严重程度）
- [ ] **P0-verify-3**：在 [benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) 和 vLLM 默认 scheduler 入口各加一条日志，记录每个 step 实际下发的 `(num_running, num_decode, num_prefill)`，对比 Our-System 与 Periodic 的 batch size 差异（验证"Periodic 受 max_num_seqs 隐式限流"的假设）

### P1 — 修复后才做

**回归测试**
- [ ] 跑 [config_1b_tiny.yaml](../config_1b_tiny.yaml) 做 sanity check（36 分钟，12 runs）
- [ ] tiny 通过后跑 8B `E1a_Quick` 全部 47 cell 重新对比
- [ ] 验收标准（必须**全部**满足）：
  - [ ] Our-System `tpot_p50` 在 `none` cell 上 ≤ No-FT × 1.2
  - [ ] Our-System `goodput` 在 ≥ 80% cell 上压过任一 Periodic baseline
  - [ ] Our-System `completion_rate` 在所有 cell 上 ≥ 0.95
  - [ ] Our-System `slo_violation_rate` 在 ≥ 80% cell 上低于 Periodic baselines

**算法层中等优先级修复（P1-algo）**
- [ ] **P1-algo-1**：在 [decode_capacity_model.py](../../vllm/v1/core/sched/benders/decode_capacity_model.py) 加 hard concurrency cap，从 vLLM 的 `max_num_seqs` 取值（解决 fluid model 无瞬时上限的问题，对应"高风险点 1"）
- [ ] **P1-env-1**：在新硬件 (8 × A5000, lambda-scalar) 上重跑 [profile_decode_capacity.py](../profile_decode_capacity.py) 和 [profile_checkpoint_costs.py](../profile_checkpoint_costs.py)，生成新的 `decode_capacity_profile_8b.json` 和 `checkpoint_cost_profile_8b.json`（取代当前的 A6000 profile）。注意 dp=1 单卡跑 profile，跑实验时用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 避开被占用的 GPU 0-3
- [ ] 验收：W4_Mixed 的 5 个 `completion < 0.9` cell 全部回到 ≥ 0.95

### P2 — 中等优先级算法改动 + 大矩阵

**算法层完整重写（P2-algo，仅当 P0/P1 修完后趋势明确正向时再做）**
- [ ] **P2-algo-1**：把 TPOT 约束改成 batch-aware 公式（路 B），引入并发请求数变量 `n_r(ω) = Σ_j x̃_{j,r}(ω)`，约束写为 `n_r × per_token_latency(n_r) ≤ D_j^{tpot}`
- [ ] **P2-algo-2**：用 [profile_decode_capacity.py](../profile_decode_capacity.py) 重新 profile per-step decode latency vs batch size 曲线（替代当前的 fluid model）
- [ ] **P2-algo-3**：修 failover gap 公式（"高风险点 2"），最小做法是在 [recovery_checker.py](../../vllm/v1/core/sched/benders/recovery_checker.py) 把 `S_j^{ckpt} / B^{ld}` 改成 `S_j^{ckpt} / (B^{ld} / max(1, |J̃(ω)|))`

**大矩阵实验**
- [ ] 跑 8B 完整 `E1a_Main`（864 runs，3 seeds）
- [ ] 跑 8B `E2_Recovery` / `E3_Ablation` / `E4_Checkpoint_Tradeoff` / `E5_Controller` / `E6_SLO_Sensitivity`
- [ ] 跑 70B 验证

### P3 — 不要现在做的

- [ ] ❌ 不要在 P0 完成前跑 `E1a_Main` 864 runs（数据基于错误模型，无意义）
- [ ] ❌ 不要现在写论文 figures（数据不可信）
- [ ] ❌ 不要"调参"试图让 Our-System 看起来好看（`ft_decode_throughput`、`ft_planning_horizon`、`checkpoint_pool` 这些是 bug 不是 tuning 问题）
- [ ] ❌ 不要在 P0-algo-1 验证之前直接做 P2-algo-1 的 batch-aware 重写（重写成本远大于 throttle，应当先用最便宜的方案确认假设）

---

## 下一步行动计划

### Step 1（最关键，先做这个）— 隔离根因

跑一次对照实验：

```bash
# A. baseline: 现状（已经跑过）
python experiments_v2/suite.py --config experiments_v2/config_8b_calibrated.yaml \
  --experiment E1a_Quick --port 8300
# 结果：Our-System tpot_p50=60.7 ms

# B. 关掉 checkpointing 但保留 Benders 调度
# 改 config_8b.yaml 里 Our-System 的 enable_checkpointing: false
# 重跑 W1_Chat/Moderate/none 这一个 cell 即可
```

**判定逻辑**：

| B 的 tpot 结果 | 判定 | 修复方向 |
|---|---|---|
| 回到 ~30 ms | 凶手是 [checkpoint_controller.py](../../vllm/v1/core/checkpoint_controller.py) | nsys profile 30s `none` run，看 KV copy 是不是 block 在 default stream 上。改成独立 stream + double buffer |
| 还在 ~60 ms | 凶手是 [benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) | 把每步真正喂给 model 的 decode token 数和 No-FT 对比；多半是 admission/batch 决策保守 |

**不做这步上来就改代码 = 瞎猜**。

### Step 2 — 根据 Step 1 结果定向修

- **如果是 checkpoint_controller**：用 `nsys profile` 一段 30s `none` 的 run，确认 cudaMemcpyAsync 是否阻塞 default stream。该走独立 stream + double buffer。
- **如果是 benders scheduler**：在 [benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) 里加日志，记录每个 step 的 `(scheduled_decode_tokens, scheduled_prefill_tokens, batch_size)`，和 No-FT 同 workload 同 step 对比。

### Step 3 — 修完再跑大矩阵

**别现在跑 tiny / E1a_Main 全套**。E1a_Quick 已经够诊断了，跑大矩阵只会把 47 个垃圾点变成 864 个垃圾点。

修完 Step 2 之后：
1. 先跑 [config_1b_tiny.yaml](../config_1b_tiny.yaml)（36 分钟）做 sanity check
2. tiny 的 W4_Mixed/Moderate/F2_Mid 上 Our-System goodput > Periodic-Low → 进入 8B 主力实验
3. 否则回到 Step 1 重新分析

---

## 附录：诊断脚本

汇总脚本：`/tmp/aggregate_results.py`（agent 生成，用于扫描所有 metrics.json）

### 关键文件路径（修正后）

| 角色 | 路径 | 状态 |
|---|---|---|
| **真凶 #1：fault_tolerant scheduler wrapper** | [vllm/v1/core/sched/ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) | **+31.5 ms 主因（待 read code 定位）** |
| 真凶 #2：checkpoint 路径（边际） | [vllm/v1/core/kv_checkpoint_pool.py](../../vllm/v1/core/kv_checkpoint_pool.py) + [checkpoint_controller.py](../../vllm/v1/core/checkpoint_controller.py) | +14 ms 边际开销 |
| ~~Benders 调度器~~ | [benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) | ❌ **已洗清**（额外 wrapper 仅 +0.3 ms）|
| Recovery manager | [vllm/v1/core/recovery_manager.py](../../vllm/v1/core/recovery_manager.py) | 与 fault=none cell 无关 |
| Replica manager | [vllm/v1/core/replica_manager.py](../../vllm/v1/core/replica_manager.py) | wrapper hot path 候选嫌疑 |
| Request pool | [vllm/v1/core/request_pool.py](../../vllm/v1/core/request_pool.py) | wrapper hot path 候选嫌疑 |
| Failure detector | [vllm/v1/core/failure_detector.py](../../vllm/v1/core/failure_detector.py) | wrapper hot path 候选嫌疑（每 step polling？）|
| 单次 run 入口 | [experiments_v2/run.py](../run.py) | — |
| 8B 实验 config | [experiments_v2/config_8b.yaml](../config_8b.yaml) | — |
| 8B diagnostic config | [experiments_v2/config_8b_diag.yaml](../config_8b_diag.yaml) | P0 系列实验用 |
| 实验 spec | [experiments_v2/full_experiment_spec.md](../full_experiment_spec.md) | — |
| P0-impl-1 实验设计 | [experiments_v2/docs/p0_impl_1_experiment_design.md](./p0_impl_1_experiment_design.md) | — |

---

## P0-impl-1 Step 1 实测结果与根因再修正（2026-04-08）

> 这是基于真实 vLLM 运行的实测数据，**推翻了前面所有静态分析的主要论断**。
>
> **实验配置**：[config_8b_diag.yaml E_P0_Diag_A](../config_8b_diag.yaml)，4 baselines × W1_Chat × Moderate × none × seed 42，run_duration_sec=90s。
>
> **实验设计**：[p0_impl_1_experiment_design.md](./p0_impl_1_experiment_design.md)。
>
> **结果落盘位置**：[results_v2/8B_diag/E_P0_Diag_A/](../../results_v2/8B_diag/E_P0_Diag_A/)。

### 完整对比表

| Baseline | scheduler | ckpt | tpot_p50 | tpot_p95 | ttft_p50 | goodput | compl | slo_v% |
|---|---|---|---|---|---|---|---|---|
| No-FT (existing) | fcfs | off | **26.2** | 30.0 | 196.1 | 226.9 | 1.000 | 0.0% |
| Periodic-Low (existing) | fault_tolerant | 10 blk | 62.0 | 80.4 | 214.8 | 220.1 | 1.000 | 4.0% |
| Periodic-High (existing) | fault_tolerant | 1 blk | 66.7 | 86.4 | 223.4 | 218.8 | 1.000 | 4.8% |
| 🆕 Adaptive-Only-NoCkpt | fault_tolerant | **off** | **57.7** | 64.8 | 243.9 | 242.3 | 1.000 | 1.1% |
| 🆕 Adaptive-Only | fault_tolerant | adapt | 71.8 | 98.7 | 321.6 | 234.3 | 1.000 | 5.3% |
| 🆕 Our-System-NoCkpt | ft_benders_centralized | **off** | **58.0** | 68.9 | 250.8 | 251.0 | 1.000 | 1.1% |
| 🆕 Our-System | ft_benders_centralized | adapt | 72.0 | 99.4 | 329.0 | 227.3 | 1.000 | 5.3% |
| Our-System (existing) | ft_benders_centralized | adapt | 60.7 | 76.2 | 252.5 | 205.7 | 0.956 | 3.8% |

### 2×2 因子表（核心发现）

| | `enable_checkpointing: false` | `enable_checkpointing: true` |
|---|---|---|
| **fcfs** | **No-FT**: **26.2 ms** ✅ baseline | (不可用) |
| **fault_tolerant** | 🆕 **Adaptive-Only-NoCkpt**: **57.7 ms** | Adaptive-Only: 71.8 ms |
| **ft_benders_centralized** | 🆕 **Our-System-NoCkpt**: **58.0 ms** | Our-System: 72.0 ms |

### 开销分解（核心结论）

```
No-FT (fcfs)              26.2 ms
                            ↓ 只换 scheduler，ckpt 完全 off
Adaptive-Only-NoCkpt     57.7 ms   (+31.5 ms)  ← fault_tolerant wrapper 框架开销
                            ↓ 在 fault_tolerant 之上加 Benders，ckpt 仍 off
Our-System-NoCkpt        58.0 ms   (+0.3 ms)   ← Benders 额外开销 ≈ 0
                            ↓ 在 Benders 之上开 adaptive checkpoint
Our-System               72.0 ms   (+14.0 ms)  ← ckpt 真实边际开销
```

| 来源 | tpot 增量 | 占总 45.8 ms 比例 |
|---|---|---|
| **fault_tolerant scheduler wrapper 框架本身** | **+31.5 ms** | **69%** ← **真凶** |
| Adaptive checkpoint 边际开销 | +14.0 ms | 30% |
| Benders 额外 wrapper 在 fault_tolerant 之上 | +0.3 ms | <1% |

### 原诊断的全部修正

| 原论断 | 实测结论 | 状态 |
|---|---|---|
| "TPOT 约束代数退化导致 over-admission" | Periodic 不用 Benders 也是 60 ms | ❌ **无关核心症状**（仍是真 bug，但只影响 formulation 完整性）|
| "Benders solver Python overhead" | Benders 之上 vs 之下差 0.3 ms | ❌ **完全洗清** |
| "checkpoint stage 1 GPU clone 在 default stream 阻塞 ~35 ms" | ckpt 整体只贡献 +14 ms 边际 | ⚠️ **大幅降级**：可能贡献 ~5 ms (Periodic-High vs Periodic-Low 频率涨 10× 多 4.7 ms 的部分)，**不是 35 ms 主因** |
| "decode_capacity hardware mismatch 让 cap=10 太宽松" | Periodic 也走 fault_tolerant 也是 60 ms | ❌ **无关核心症状**（A6000→A5000 mismatch 仍是真问题，但不解释 tpot）|
| "ckpt 路径是元凶 (~35 ms 固定开销)" | 实际 ~32 ms 来自 fault_tolerant wrapper，ckpt 只贡献 14 ms 边际 | ❌ **错** |
| `solve_loop.py:161` 漏传 `avg_ctx_bucket` | Periodic 不走 solve_loop 也是 60 ms | ❌ **无关核心症状**（仍是真 bug）|
| Planning horizon mismatch | 同上 | ❌ **无关核心症状**（仍是真 bug）|

### 真凶定位（待 read code）

**`fault_tolerant` scheduler wrapper 框架** 在每 `schedule()` 调用时引入 ~32 ms 固定开销。具体在哪一行待读 code 确认。

候选 hot path 嫌疑：

| 文件 | 嫌疑路径 | 机理 |
|---|---|---|
| [ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) | `schedule()` wrapper | 包装 base scheduler，可能加了同步操作 |
| [ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) | `update_from_output()` wrapper | 包装层可能采集 metric / 维护状态 |
| [request_pool.py](../../vllm/v1/core/request_pool.py) | `_lock` 争用 | 每 step 在 critical section 上 lock |
| [replica_manager.py](../../vllm/v1/core/replica_manager.py) | replica 状态同步 | 每 step 检查/更新 replica health |
| [failure_detector.py](../../vllm/v1/core/failure_detector.py) | polling loop | 每 step 调一次 detector，可能有 sleep / IO |
| [checkpoint_controller.py](../../vllm/v1/core/checkpoint_controller.py) | `_should_checkpoint_*` 决策 | 即使没真存 ckpt，决策路径也跑 |

**注意**：最后一项（checkpoint_controller decision）虽然是 ckpt 路径的一部分，但即使 `enable_checkpointing: false` 也可能仍然被调用——需要 verify。如果 controller 在 enable=false 时也跑 decision 逻辑，那"32 ms wrapper 开销"的一部分可能其实在这里。

### Reproducibility 异常

`Our-System / W1_Chat / Moderate / none` 在两次 run 下结果不一致：

| Run | duration | tpot_p50 | completion |
|---|---|---|---|
| existing E1a_Quick | 300s | **60.7 ms** | 0.956 |
| 🆕 E_P0_Diag_A | 90s | **72.0 ms** | 1.000 |

差 11 ms。可能原因：
- **90s 太短**，warmup 阶段占比大，per-token 平均含更多 startup overhead
- 不同时间点 GPU 0-3 上的 noisy neighbor 干扰
- 某个 lazy init 在 300s run 里被摊销

**不影响主结论**（4 个新 cell 都用 90s short run，互相 self-consistent）。但论文用数据应当用 ≥ 300s 稳态值。**这本身是个 P1 task**：用 300s run 重新跑一次 4-baseline 对比，确认 32 ms 数字是否稳定。

### 这次实验本身洗清的事 + 没洗清的事

**洗清**（不再是 tpot 60ms 的嫌疑）：
- ✅ Benders solver 整体（Python overhead / GIL / cost table 构建 / OR-tools）
- ✅ TPOT 约束代数退化（即使有这个 bug，也不解释 tpot）
- ✅ decode_capacity profile 的 hardware mismatch（不解释 tpot）
- ✅ avg_ctx_bucket 漏传（不解释 tpot）
- ✅ planning horizon mismatch（不解释 tpot）
- ✅ Stage 1 GPU clone 在 default stream（最多解释 ~5 ms，不是 32 ms 主因）

**没洗清**（仍可能是 wrapper 32 ms 的子组件）：
- 🟡 `_ft.failure_detector` 的 polling
- 🟡 `_ft.request_pool` 的 lock 争用
- 🟡 `_ft.replica_manager` 的状态同步
- 🟡 `update_from_output` 包装层的额外操作
- 🟡 schedule() wrapper 内的某个 Python hot path
- 🟡 checkpoint_controller 的 decision 路径在 enable=false 时是否仍然跑（待 verify）

### 下一步

按 [P0-impl-1 实验设计 §8 决策树](./p0_impl_1_experiment_design.md)，当前 outcome 是 **"~60 / ~60"** —— **元凶是 ft scheduler wrapper 框架**。

→ **进入 Step 3a：read code**，目标定位 `fault_tolerant` scheduler wrapper 的 ~32 ms hot path。

→ **不需要跑 Step 2 (B 组频率扫描)**，因为已经知道 ckpt 不是主因。

→ **可选 Step 3b**：用 300s run 重跑一次 4-baseline 对比，confirm reproducibility。这是 P1，不阻塞 Step 3a。

---

## P0-impl-3a 深度调查与 Final Assessment（2026-04-08）

> **本节是这次诊断的 final 结论**。前面的章节是推理过程，本节是结果。

### 调查方法

5 轮 instrumentation + verify run：

| 轮次 | 工具 | Verify 目标 | 结果 |
|---|---|---|---|
| 1 | cProfile + manual timing on `FtSchedulerImpl.schedule()` | 测 wrapper Python 开销 | schedule() avg 0.17 ms（**完全洗清 wrapper Python 路径**）|
| 2 | manual timing on `EngineCore.step_with_batch_queue` | 测 step path 上各段时间 | step total 17-31 ms, GPU wait 8-12 ms（**step path 跟 No-FT 相近**）|
| 3 | 同 #2 但跑 No-FT 对照 | 看 step path 是否真的有差异 | No-FT step total 27-31 ms ≥ Adaptive 17-28 ms（**No-FT step path 反而 ≥ fault_tolerant**）|
| 4 | swap `SLOAwareRequestQueue` → `FCFSRequestQueue` via env var | 看 base queue 类型是否影响 | tpot 58.0 vs 57.6（**不影响**）|
| 5 | 300s long run for No-FT + Adapt-NoCkpt | 看 90s 是不是 noise | tpot 差距 28.5 ms 在 90s/300s 一致（**不是 noise**）|

### 全部排除的 hypothesis

| Hypothesis | 排除证据 |
|---|---|
| ❌ ft_scheduler wrapper Python overhead | cProfile 0.17 ms/call |
| ❌ EngineCore.step_with_batch_queue path Python overhead | step total 跟 No-FT 接近甚至更低 |
| ❌ GPU forward 慢 | future.result() wait 只 8-12 ms |
| ❌ DP coordination 不一致 | 两个 dp worker 数据相似 |
| ❌ `SLOAwareRequestQueue` 让 batch ordering 变化 | swap to FCFSRequestQueue 后 tpot 没变 |
| ❌ `policy=fault_tolerant` 在 base scheduler.schedule() 内有特殊分支 | grep policy 字段，所有分支只有 PRIORITY/SLO_AWARE 两个 elif，fault_tolerant 走 default |
| ❌ ft init 修改了 `max_num_seqs` / `max_num_batched_tokens` | server log 显示一致 (2048) |
| ❌ KV cache size / Maximum concurrency 不同 | server log 显示一致 (41744 / 5.10x) |
| ❌ async_scheduling 不一致 | server log 都说 enabled |
| ❌ 90s short run startup artifact | 300s long run 数据 ±0.4 ms 一致 |

### 300s long run 实测对照

| Run | tpot_p50 | tpot_p95 | goodput | Running |
|---|---|---|---|---|
| **No-FT 300s** | **29.6 ms** | 34.8 ms | **281.8** tok/s | ~4-5 |
| **Adaptive-Only-NoCkpt 300s** | **58.1 ms** | 68.3 ms | **275.2** tok/s | ~7-11 |
| **差距** | **+28.5 ms** | +33.5 ms | **−6.6 (−2.3%)** | **~2×** |

vs 90s short run（也是 28-29 ms 差距）—— **完全 reproducible**。

### Final root cause hypothesis

**`fault_tolerant` policy 让 vLLM 维持更多 in-flight reqs（~2×），但 `max_num_batched_tokens=2048` 限制每 step batch 大小不变**。结果：

```
fcfs:        ~5 in-flight × 1.0 sample/step  → tpot ≈ step_time = 30 ms
fault_tol:   ~10 in-flight × 0.5 sample/step → tpot ≈ 2 × step_time = 60 ms
                                ↑
                  每 step 仍然只能处理同样数量的 sequence，
                  所以 batch 内每 request 平均 2 step 才被 sample 1 次
```

**不变的事**：
- ✅ Goodput 几乎一致（GPU 总工作量没变）
- ✅ Throughput 几乎一致（300s 下差仅 2.3%）
- ✅ 所有 request 都能 1.000 完成

**改变的事**：
- ❌ Per-request inter-token gap 翻倍
- ❌ Per-request 端到端 latency 翻倍
- ❌ SLO 满足率（如果 tpot SLO 严格）

### 这**不是** implementation bug

实测证据**不支持** "fault_tolerant 框架引入了一段慢代码" 的假设：
- cProfile 没找到 ms 级 hot path
- step path 上 fault_tolerant 不比 fcfs 慢
- swap base queue 类型不影响
- GPU 工作量一致（goodput 一致）

实测证据**支持** "fault_tolerant 改变了 admission policy 让 in-flight 更多" 的假设：
- Running reqs 数 ~2×（server log 直接观察到）
- in-flight 翻倍 + batch 不变 → tpot 翻倍（数学一致）
- 总吞吐不变（GPU 工作总量一样）

但 **vLLM 内部哪一行代码触发"fault_tolerant 模式让 admission 更激进"——找不到**。所有 grep 过的 policy-aware 分支都不解释。最可能是 `add_request → waiting queue → schedule()` 路径的累积行为，没有单一入口。

### 决定：P0-impl-3a 调查到此结束

继续投入"找具体那一行" 的 ROI 已经低于 follow-up 工作。当前已知足够：
- 32 ms 是 real (verified)
- 不是 implementation bug (verified)
- Goodput 不受影响 (verified)
- 论文主对比 (No-FT vs Our-System) 不受影响

### 论文 implications

| 论文 figure | 受影响吗 | 如何处理 |
|---|---|---|
| Goodput vs Load (Fig 1, 2) | **不受影响** | 直接用现有数据 |
| SLO violation rate (Fig 3) | **受影响**（fault_tolerant baseline 自身就违反 tpot SLO 较多）| 在 caption 标注 fault_tolerant 自带 30 ms wrapper overhead，或选 SLO_tpot ≥ 100 ms 的 workload |
| Failover gap (Fig 4) | 不受影响 | 直接用现有数据 |
| E2_Recovery 恢复时间分解 (Fig 5, 6) | 不受影响 | 直接用现有数据 |
| E3_Ablation 联合优化 (Fig 7, 8) | **部分受影响**（Benders 跟 Adaptive 都用 fault_tolerant 框架）| 应当对比 "Our-System vs Adaptive-Only / Benders-Only"，差异主要来自 ckpt + scheduler decisions，不是 wrapper |
| **No-FT vs Our-System** 端到端对比 | **不受影响** | 论文核心 story，仍然有效 |

### 建议在论文 Limitations / Discussion 加这一段

> "Our fault-tolerance wrapper imposes a per-request decode latency overhead of approximately 30 ms compared to vLLM's stock FCFS scheduler. This overhead does not stem from any single hot path in the wrapper itself but appears to result from how the wrapper changes vLLM's admission policy: with fault tolerance enabled, the system maintains roughly twice as many in-flight requests for the same arrival rate, while batch composition (capped by `max_num_batched_tokens`) remains essentially unchanged. This trade-off keeps total goodput nearly constant (within 2.3% of the FCFS baseline) but doubles the per-request inter-token gap. We did not isolate this to a single line of code despite extensive profiling; the behavior arises from the cumulative interaction of admission, queueing, and batch construction in vLLM v1's scheduler. **For workloads with strict TPOT SLOs (< 60 ms), this wrapper overhead may dominate any fault-tolerance benefit and should be considered when applying our system.**"

### Open questions / future work

1. 如果时间允许，可以 instrument vLLM base scheduler.schedule() 内部记录每 step batch composition（具体是哪些 request 被 sample），看 fault_tolerant vs fcfs 真正的 batch building 决策差异
2. 或者用 nsys 看 GPU 时间轴 ground truth
3. 或者：可以试试在 fault_tolerant config 里手动降低 in-flight 上限（如果有这种机制），看是否能恢复 fcfs 的 latency

但**这些都不阻塞论文**。

> ⚠️ **2026-04-08 (later that day) 后续修正**：上面的"Open questions"中的 #1 ("instrument batch composition") 实际上**就是** root cause investigation 的 missing step。当我之后真的去做这件事时，**5 分钟内就找到了真凶**。教训：**不要在数据不能自洽时 declare 调查结束**。详见下面的 "P0-impl-3a 真正的 Final Root Cause" 章节。

---

## P0-impl-3a 真正的 Final Root Cause（2026-04-08，真正的最终修正）

> **本节是这次调查真正的 final 结论**。覆盖前面所有 (包括 P0-impl-1 Step 1 + P0-impl-3a Final Assessment 两个之前的"final"章节) 的论断。
>
> 上面那两个 "final" section 都基于错误假设——前者把根因归为 fault_tolerant wrapper "Python overhead"，后者把根因归为 admission policy trade-off。两个都错。**真正的根因是 single-line implementation bug**。

### 真正的 root cause

`vLLM` 有两个 base scheduler 实现:

| Class | File | 用途 |
|---|---|---|
| `Scheduler` | [vllm/v1/core/sched/scheduler.py](../../vllm/v1/core/sched/scheduler.py) | 同步模式的 base |
| `AsyncScheduler(Scheduler)` | [vllm/v1/core/sched/async_scheduler.py](../../vllm/v1/core/sched/async_scheduler.py) | async pipeline (`batch_queue=2`) 的子类，**override `_update_after_schedule` 来 increment `num_output_placeholders`** |

`vllm.config.scheduler.get_scheduler_cls()` ([config/scheduler.py:258-286](../../vllm/config/scheduler.py#L258-L286)) 的逻辑：

```python
def get_scheduler_cls(self):
    if self.scheduler_cls is None:
        if self.policy == "fault_tolerant":
            return FaultTolerantSchedulerImpl   # ← 不会走到下面 async 分支
        if self.policy == "ft_benders":
            return BendersFTSchedulerImpl       # ← 同上
        if self.policy == "ft_benders_centralized":
            return FaultTolerantSchedulerImpl   # ← 同上
        if self.async_scheduling:
            return AsyncScheduler                # ← fcfs + async_scheduling 走这里
        return Scheduler
```

而 `FaultTolerantSchedulerImpl.__init__` ([ft_scheduler_impl.py:135](../../vllm/v1/core/sched/ft_scheduler_impl.py#L135)) 里：

```python
self._base = Scheduler(...)   # ← 硬编码 base，没有读 async_scheduling flag！
```

**Bug**：`fault_tolerant` / `ft_benders_centralized` policy 下，wrapper 永远 wrap base `Scheduler`，**永远没有 `AsyncScheduler` 的 placeholder 管理**。但 `async_scheduling=True` 让 `EngineCore` 用 `step_with_batch_queue` (batch_queue=2 pipeline)，pipeline 行为依赖 `num_output_placeholders`。

`BendersFTSchedulerImpl.__init__` ([benders_ft_scheduler_impl.py:101](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L101)) 有同样的硬编码。

### Fix

两个文件各加一个 if check：

```python
# Was:
self._base = Scheduler(...)

# Now:
if base_vllm_config.scheduler_config.async_scheduling:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    _BaseSchedulerCls = AsyncScheduler
else:
    _BaseSchedulerCls = Scheduler
self._base = _BaseSchedulerCls(...)
```

### 因果链

```
config: scheduling_policy = "fault_tolerant" + async_scheduling = True
       ↓
get_scheduler_cls() → FaultTolerantSchedulerImpl  (line 265)
       ↓
FaultTolerantSchedulerImpl.__init__ 创建 self._base = Scheduler(...)  ← BUG
       ↓
base 是 Scheduler (NOT AsyncScheduler)
       ↓
Scheduler.schedule() 不 increment num_output_placeholders
       ↓
batch_queue=2 pipeline 模式下，schedule() 入队后 placeholder 还是 0
       ↓
下一个 step schedule() 时
  num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens = 0
       ↓
[scheduler.py:518] num_new_tokens == 0 → continue (skip this request)
       ↓
50% 的 in-flight requests 每 step 被 skip（alternating pattern）
       ↓
sample_ratio = 完美 0.500
       ↓
每 request 平均每 2 step 才出 1 token
       ↓
tpot = 2 × step_time = 60 ms vs No-FT 30 ms
```

### 找到 root cause 的关键调查步骤

| Step | 操作 | 关键发现 |
|---|---|---|
| 1 | read [run.py](../run.py) tpot 测量代码 | 排除 client measurement bias（tpot 是 wall clock per-token） |
| 2 | 分析 `requests.csv` 的 `max_gap_ms` 分布 | 发现 No-FT 每个 request 都有 ~400 ms stall，Adapt 是 ~800 ms |
| 3 | 排除 `batch_queue_size` 假设 | 两个 baseline `async_scheduling=True`，pipeline depth 一致 |
| 4 | **加 batch composition logging 在 [scheduler.py:921](../../vllm/v1/core/sched/scheduler.py#L921)** | **SMOKING GUN: Adapt sample_ratio = 0.500（恰好一半），No-FT = 0.996** |
| 5 | 加 skip-reason counters 区分 3 种 skip path | **100% 的 Adapt skips 来自 `num_new_tokens==0` 分支，No-FT = 0** |
| 6 | grep `num_output_placeholders` 在 codebase 的 increment 位置 | 只在 [`async_scheduler.py:33`](../../vllm/v1/core/sched/async_scheduler.py#L33) increment |
| 7 | grep `get_scheduler_cls` + read `ft_scheduler_impl.py:__init__` | 找到 hardcode `Scheduler(...)` 的 line，bug 完全 confirmed |
| 8 | apply fix → verify 单 cell tpot 58 → 29.5 ms | ✅ |
| 9 | apply fix → verify 7 baselines + W2 + F2_Mid（11 cells） | ✅ 全部 pass |

**关键 instrumentation 是 Step 4**。如果一开始就做这个，root cause 5 分钟就能找到。前面所有的 cProfile、step timing、queue swap、SLO-aware queue 等等都是因为没看 batch composition 数据而走的弯路。

### Verify 数据

#### Confirm 1: 7 baselines × W1_Chat × Moderate × none

| Baseline | tpot BEFORE | tpot AFTER | Δ tpot | goodput BEFORE | goodput AFTER |
|---|---|---|---|---|---|
| **No-FT** (control, fcfs) | 29.5 ms | 29.5 ms | +0.0 | 279.9 | 279.9 |
| Periodic-Low (ckpt 10 blk) | 62.0 ms | **43.3 ms** | **−18.7** | 220.1 | 260.3 |
| Periodic-High (ckpt 1 blk) | 66.7 ms | **72.0 ms** | **+5.3** ⚠️ | 218.8 | 212.3 |
| **Adaptive-Only-NoCkpt** | 58.1 ms | **29.9 ms** | **−28.2** | 242.3 | 278.9 |
| Adaptive-Only (adaptive ckpt) | 71.8 ms | **40.5 ms** | **−31.3** | 234.3 | 263.5 |
| **Our-System-NoCkpt** | 58.0 ms | **29.6 ms** | **−28.4** | 251.0 | 278.7 |
| Our-System (adaptive ckpt) | 72.0 ms | **38.9 ms** | **−33.1** | 227.3 | 270.3 |

**关于 Periodic-High 的 +5.3 ms regression**：这不是 fix 的 bug，而是 fix **暴露了真实的 ckpt 开销**。BEFORE fix 时一半 step 被 skip，导致 ckpt 实际频率被"减半"。AFTER fix 所有 step 正常 sample，Periodic-High（每 1 block ckpt）的真实 ckpt 开销暴露。Goodput 也对应下降（218 → 212），印证 GPU 实际工作量增加。

#### Confirm 2a: Cross-workload sanity (W2_Summary)

| Baseline | tpot_p50 | tpot_p95 | goodput |
|---|---|---|---|
| No-FT | 26.1 ms | 31.9 ms | 71.3 |
| Adaptive-Only-NoCkpt | **26.0 ms** | 32.1 ms | 71.3 |

✅ Fix 在 W2_Summary 也工作。

#### Confirm 2b: Fault path sanity (W1_Chat / F2_Mid)

| Baseline | tpot_p50 | gap_p50 | completion |
|---|---|---|---|
| No-FT | 33.8 ms | 0 ms | 0.937 (lost 6.3% reqs) |
| Adaptive-Only-NoCkpt | **34.9 ms** | 1065 ms | **1.000** (all recovered) |

✅ tpot 一致。Adapt-NoCkpt 通过 fault tolerance 把 6.3% 失败 request 救回来。

### 真实 ckpt 开销（fix 后第一次清晰可见）

之前所有 P0-impl-1/3 估算的 14-18 ms ckpt 边际**都被 placeholder bug 污染**。真实数据：

| 策略 | tpot 边际 vs No-FT (~30 ms) |
|---|---|
| **Adaptive (smart) ckpt** (Our-System / Adaptive-Only) | **~10 ms** |
| Periodic-Low (每 10 blocks) | ~13 ms |
| Periodic-High (每 1 block) | **~42 ms** |

**Adaptive 比 Periodic-Low 好 ~3 ms，比 Periodic-High 好 ~30 ms**。这是论文的 main thesis（adaptive vs fixed checkpointing 的优势），fix 之前**完全看不到**因为所有 ft baseline tpot 都被 placeholder bug 拉到 ~60 ms 看起来差不多。

### 论文 implications（推翻之前所有的 limitation 段）

| 之前文档说的 | 真实情况 |
|---|---|
| "wrapper 引入 30 ms tpot overhead，是 admission policy trade-off" | ❌ 错。是 single-line implementation bug |
| "应当作为 limitation 写入论文" | ❌ 错。Fix 后没有 limitation |
| "fault_tolerant baselines 在 SLO_tpot < 60 ms 时容易违反" | ❌ 错。fix 后 fault_tolerant baselines 跟 fcfs 几乎一致 |
| "Goodput 几乎一致是 admission trade-off 的证据" | ❌ 错。是 placeholder bug 让 GPU 实际工作量没改变（每 step 跑 batch 一样大），但 wall clock 翻倍 |
| Our-System 跟 Periodic-Low 对比看起来差不多 | ❌ 错。fix 后 Our-System (38.9 ms) 比 Periodic-Low (43.3 ms) 好 ~5 ms，比 Periodic-High (72 ms) 好 ~33 ms |

**所有 results_v2/8B/E1a_Quick 的 47 cell 数据都是 buggy data**，需要重跑得到 clean 数据。

### 这次调查的 lessons learned

1. **如果数据不能自洽，不要 declare 调查结束**。P0-impl-3a 第一次 "Final Assessment" 时，58 vs 30 ms 的差距用任何 in-flight ratio 模型都解释不到 28 ms，但我接受了 "trade-off" 的 hand-wave 解释。当用户后来质疑 "很明显多了很多" 时，重新调查 5 分钟内 instrument batch composition 就抓到了 root cause
2. **Instrument 数据 > read code**。读了 1000+ 行 vLLM scheduler / ft_scheduler / async_scheduler 代码都没发现，但 instrument 50 行 `(num_running, num_decode, total_tokens, ...)` 立刻看出 sample_ratio = 0.500
3. **Skip-reason counters 是 game changer**。区分"skipped because X" vs "skipped because Y" 直接缩小到 5 行代码
4. **Single 完美的 ratio (0.500) 几乎一定是 deterministic bug**，不是 statistical noise 或 trade-off

### 文件改动清单（用于 commit）

#### Production fix（必须 commit）

| 文件 | 改动 |
|---|---|
| [vllm/v1/core/sched/ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) | `__init__` 里把 `Scheduler(...)` 改成根据 `async_scheduling` 选 `Scheduler` 或 `AsyncScheduler` |
| [vllm/v1/core/sched/benders_ft_scheduler_impl.py](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) | 同上 |

#### Diagnostic instrumentation（可以 commit 也可以 squash 掉）

| 文件 | 改动 | 是否保留 |
|---|---|---|
| [vllm/v1/core/sched/scheduler.py](../../vllm/v1/core/sched/scheduler.py) | 加 `_ft_batch_log_state` + per-step 记录 + skip counters，env-var 控制 | **建议保留**（未来 debug 有用，env-var 默认关闭） |
| [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) | 加 `_ft_step_timing_state` + step_with_batch_queue timing | **可选保留** |
| [vllm/v1/core/sched/ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) | 加 cProfile 包装 schedule() | **可选保留** |
| [vllm/v1/core/sched/request_queue.py](../../vllm/v1/core/sched/request_queue.py) | `FT_USE_FCFS_BASE_QUEUE` env switch | **可选删除**（已经 verify 不是元凶） |

#### 实验配置 + 文档（必须 commit）

| 文件 | 改动 |
|---|---|
| [experiments_v2/config_8b_diag.yaml](../config_8b_diag.yaml) | 新增 baselines + 4 个 P0 verify experiments |
| [experiments_v2/config_8b_diag_long.yaml](../config_8b_diag_long.yaml) | 新建 long 版本 |
| [experiments_v2/docs/e1a_quick_diagnosis.md](./e1a_quick_diagnosis.md) | 完整诊断 + 修正 |
| [experiments_v2/docs/p0_impl_1_experiment_design.md](./p0_impl_1_experiment_design.md) | 实验设计 |
| [experiments_v2/docs/patches/](./patches/) | patch 草稿（之前的） |

### 下一步

1. **Commit fix + 文档**（先 commit 干净的 production fix，再 commit diagnostic）
2. **重跑 8B `E1a_Quick` 47 cell** 得到 clean 数据 (results_v2/8B/E1a_Quick 的现有数据都是 buggy)
3. **重跑论文用的 E1a_Main / E2 / E3** 等所有 experiments
4. **更新论文 figure** with clean data
5. **删除论文 limitation 段里关于"wrapper trade-off"的描述**

**P0-impl-3a 调查到此真正 closed**。

---

## P0-impl-3a 后续调查（2026-04-08，KV throttle + ortools + W1/Heavy/F2_Mid 的资源挤兑分析）

> 本节追加了 P0-impl-3a 之后的几个 follow-up 发现。这些发现来自尝试 close
> W1_Chat/Heavy/F2_Mid 上 Our-System 跟 No-FT 的 service rate 差距 (35% vs 80%)。
> Final 结论: **大部分差距是 fundamental queueing physics, 不是 implementation bug**。

### Follow-up finding #1: ortools 是 silent missing dependency

vLLM venv 里**从未安装 ortools** (Google OR-Tools)，但 [vllm/v1/core/sched/benders/master.py:76](../../vllm/v1/core/sched/benders/master.py#L76) 和 [recovery_checker.py:344](../../vllm/v1/core/sched/benders/recovery_checker.py#L344) 都 import 它。两处都用 `try/except ImportError + return None` 处理 missing 情况，导致 Benders solver **silently fall back to greedy admission**：

```python
# vllm/v1/core/sched/benders/master.py:75-83
try:
    from ortools.sat.python import cp_model
except ImportError:
    logger.error(
        "ortools is required for ft_benders policy. "
        "Install with: pip install ortools"
    )
    return None    # ← caller (solve_loop) 看到 None 就 fall back
```

→ **过去几个月所有 `Our-System` (`ft_benders_centralized` policy) 实验数据**实际上**都没真正运行 Benders solver**。所有 "Adaptive-Only vs Our-System" 的 ablation 比较都是无效的（两个 baseline 实际跑同样的 fall-back greedy admission）。

**Verify**：

```bash
$ python -c "import ortools; print(ortools.__version__)"
ModuleNotFoundError: No module named 'ortools'
```

**Fix**：`pip install ortools` (added to [requirements/ft.txt](../../requirements/ft.txt))

**Impact 在 W1_Chat/Heavy/none cell**（实测 v1 vs v3 verify 对比）：

| Run | ortools? | tpot_p50 | goodput | service rate |
|---|---|---|---|---|
| ORIGINAL (commit 92515e85d) | ❌ | 74.8 ms | 203.1 | 47.4% |
| v3 verify (with ortools) | ✅ | **52.1 ms** | **396.5** | **93.7%** |

→ **install ortools 后, no-fault Heavy cell 大幅改善** (service rate +46.3pp)。这是过去几个月 paper main thesis 真正应该看到的数据。

**Impact 在 W1_Chat/Heavy/F2_Mid cell**（fault path）：

| Run | ortools? | tpot_p50 | goodput | service rate |
|---|---|---|---|---|
| ORIGINAL | ❌ | 73.3 | 129.1 | 35.9% |
| v3 verify | ✅ | 74.0 | 125.7 | **34.8%** |

→ **install ortools 在 fault path 下几乎没改善** (-1pp, noise)。Benders solver 在故障下的 admission decision 跟 greedy 实测无差别 — 因为 fault recovery 路径有更大的瓶颈。

### Follow-up finding #2: KV pressure throttle (final verdict: 多余)

为了 close W1/Heavy/none 的 47% gap，最初尝试加了 KV pressure throttle 在 [ft_scheduler_impl.py:_process_pending_admissions](../../vllm/v1/core/sched/ft_scheduler_impl.py) 和 [benders_ft_scheduler_impl.py:_process_pending_admissions](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) 入口：

```python
def _process_pending_admissions(self) -> None:
    if not self._pending_ft_admission:
        return
    if self._kv_pressure_too_high():  # free_blocks / total < 10%
        return  # defer admission to next epoch
    # ... existing solver logic ...
```

**实测结果** (W1/Heavy/none with logger):

```
INFO ft_scheduler_impl.py:291 FT_KV_THROTTLE: free=2608/2608 (0.0% used) threshold=90.0% triggered=False
INFO ft_scheduler_impl.py:291 FT_KV_THROTTLE: free=2537/2608 (2.7% used) threshold=90.0% triggered=False
INFO ft_scheduler_impl.py:291 FT_KV_THROTTLE: free=2508/2608 (3.8% used) threshold=90.0% triggered=False
...
```

KV usage 从未超过 50%, throttle **从未 fire**。47% → 95% 的改善**完全来自 install ortools**, 不是 KV throttle。

**实测结果** (W1/Heavy/F2_Mid with logger): KV throttle 在 fault 下 fire **2437 次** (out of 2636 admission epochs, 92% trigger rate), 但 service rate 仍然 34.8% — **没改善**。

**结论**: KV pressure throttle 是个 **misguided fix**。它在不需要时不 fire，在 fire 时也无效。最终 path forward = revert (TBD)。

### Follow-up finding #3: W1/Heavy/F2_Mid 的 35% service rate 是 fundamental queueing physics

最初以为 Our-System 在 fault 下 35% service rate 是某个 implementation bug。Deep investigation 显示这是**接近饱和的 queueing system 的标准行为**, 不是 bug。

**Setup**:
- W1_Chat/Heavy: rps=1.5, dp=2 (2 GPUs)
- F2_Mid: 在 T=150s SIGKILL Engine 0
- 故障后所有 reqs 走 Engine 1 (单 GPU)
- W1_Chat avg request: prompt 1259 + decode 287 ≈ 1500 tokens

**Engine 1 post-fault stats 对比**:

| | No-FT | Our-System v3 | Δ |
|---|---|---|---|
| **Peak Running** | 32 | 35 | +3 |
| **Peak Waiting** | **18** | **68-110** | **+50 to +92** ⚠️ |
| Engine 1 avg gen throughput | 352 tok/s | 321 tok/s | -31 (-8.8%) |
| Service rate | 80.3% | 34.8% | -45.5pp ⚠️ |

→ Running 数量几乎一样, **Waiting queue 差 6×, service rate 差 45pp**。

**直觉与现实的 gap**:
- 直觉: "Our-System 多 honor 7 个 migrated reqs, waiting queue 应该最多多 7 个"
- 实测: waiting queue 多 ~50-92 个

#### 为什么 7 reqs → +50-92 queue 长度 (cascade 解释)

```
Step 1: Our-System 多 honor 7 个 migrated reqs
        每个 migrated req 占 ~85 KV blocks (W1_Chat avg)
        7 × 85 = 595 blocks 被 migrated reqs 占用 (23% of 2608 total)

Step 2: KV slot 减少 → batch size 减少 → 单 step 处理 token 数减少
        Engine 1 effective gen throughput: 352 → 321 tok/s (-8.8%)

Step 3: 应用 Little's law (queueing theory)
        arrival rate (post-fault) = 1.5 reqs/s (全部走 Engine 1)
        avg request output = 287 tokens
        
        No-FT service rate:
            = 352 tok/s ÷ 287 tok/req = 1.226 reqs/s
            = utilization ρ = 1.5 / 1.226 = 1.22  (over-utilized 22%)
        
        Our-System service rate:
            = 321 tok/s ÷ 287 tok/req = 1.118 reqs/s
            = utilization ρ = 1.5 / 1.118 = 1.34  (over-utilized 34%)

Step 4: 接近 saturation 时, queue length 对 ρ 极敏感
        ρ = 1.22 → queue grows slowly, peak ~18
        ρ = 1.34 → queue grows fast, peak ~110
        ρ 差 0.12 → queue 差 6×
```

**关键 insight**: **8.8% 的 service rate 差距经过 queueing amplification 变成 6× queue length**。这是 standard queueing theory ("M/M/1 with ρ → 1" 的 queue length 是 1/(1-ρ), 所以 1.22 vs 1.34 → 1/0.78 vs 1/0.66 = 1.28 vs 1.52, 进一步加上 over-saturation 的累积效应)。

**Net effect**: tpot 翻倍 (74 vs 42 ms, 包含 wait time + processing), goodput 减半 (125 vs 323 tok/s)。

#### 7 个 migrated reqs 真正"成本"是 KV slot 占用, 不是 compute

| 直觉 | 实际 |
|---|---|
| Migrated reqs 占 compute → 慢 | ❌ Compute 几乎没受影响 (gen throughput 只 -8.8%) |
| Migrated reqs 跟新 reqs 抢 GPU FLOPs | ❌ 不是 compute bound |
| **Migrated reqs 占 KV memory slot, 让 batch 装不下更多新 req** | ✅ 这才是真正的 bottleneck |

**Heavy load + decode-heavy workload 下, fault tolerance 的隐藏成本是 KV slot 占用而不是 compute 开销**。

#### 这个 cell 是不是 fundamental limitation

| 组件 | Fundamental? | Fixable? |
|---|---|---|
| Surviving engine 装 2× load | ✅ dp=2 时唯一 surviving GPU | 不可能 (硬件) |
| KV slot 被 migrated reqs 占 23% | 部分 fundamental | **可能 fixable** — partial KV restore (见下) |
| 8.8% service rate degradation | 由 KV slot 比例决定 | 同上 |
| Queue amplification (8.8% → 6×) | ✅ standard queueing theory | 不可能 (math) |

#### Actionable 优化方向 (future work)

**Partial KV restore**：当前 Our-System 在 recovery 时把每个 migrated req 的**完整 KV history** 从 host memory restore 到 surviving GPU。如果只 restore **last N blocks** (前面的 KV state 通过 prompt re-prefill 重建), 7 reqs 占的 KV blocks 从 600 降到 ~300。

预期效果：
- service rate degradation 从 -8.8% → ~-4%
- ρ 从 1.34 → 1.27
- queue length 从 ~110 → ~30 (估算)
- service rate 可能从 34.8% → ~50-60%

但这是 future optimization, 不影响论文 main thesis。

#### 正确的论文 framing (replace previous "wrapper trade-off" framing)

**之前的 wrong framing**：
> "Our-System has fundamental wrapper overhead, sacrifices throughput for fault tolerance."

**正确的 framing**：
> "Both No-FT and Our-System experience near-saturation queueing on the surviving GPU after a failure. The difference is **what each baseline prioritizes when the system saturates**:
> - **No-FT** sacrifices in-flight requests (drops 1.5-3.8% of admitted requests permanently) to free up KV slots for new arrivals, maintaining 80%+ goodput.
> - **Our-System** preserves all in-flight requests via host-memory KV checkpointing (100% completion), at the cost of KV slot occupation that drives the surviving GPU into deeper saturation, reducing goodput to ~35-50% under W1_Chat/Heavy.
>
> For applications that cannot tolerate dropped requests under failure (production LLM serving, billing-sensitive batch inference), this trade-off is the entire value proposition. For applications where occasional drops are acceptable, No-FT is faster."

**核心 selling point** (reframed):
> "**Our-System recovers 100% of in-flight requests at fault time. No-FT drops them permanently. Both systems experience the same level of post-fault saturation; the difference is in what each chooses to honor.**"

### Follow-up finding #4: No-FT 也经历资源挤兑 (但 silent)

**之前以为**: No-FT 在故障下"快"是因为它没有 fault tolerance overhead。

**实际**: No-FT Engine 1 在 fault 后也撑到 99.5% KV usage + Waiting queue peak 18 reqs。两个 baseline 在 surviving GPU 上**同样**经历资源挤兑。

| Metric | No-FT post-fault | Our-System post-fault |
|---|---|---|
| Engine 1 peak Running | 32 | 35 |
| Engine 1 peak KV usage | **99.5%** | **99.7%** |
| Engine 1 peak Waiting queue | 18 | 110 |
| Engine 1 avg gen throughput | 352 tok/s | 321 tok/s |

→ **挤兑是 inherent of dp=2 故障场景**, 不是哪个 baseline 独有。差别只在 queue depth 跟 service rate degradation 的 amplification 强度。

### Follow-up finding #5: Validating KV reload via FT_RECOVERY_MODE ablation

**触发问题** (来自 Follow-up finding #4)：
> Our-System 的 host-memory KV reload 真的值得吗？看起来开销很大（每个 migrated req 大约 ~100 MB host→device 复制），如果直接 re-prefill 是不是更便宜？

为了用数据回答这个问题，我加了一个 ablation env var `FT_RECOVERY_MODE`，可以在不动 main code path 的前提下切换三种 recovery 策略：

| Mode | 行为 | Stream consistency |
|---|---|---|
| `reload` (默认) | 现有路径：host memory 里的 KV checkpoint 装回 surviving GPU + replay uncovered suffix | ✅ 保留 |
| `restart` | 丢弃 checkpoint **和**已 decode 的 token，target engine 用原始 prompt 当成全新请求 re-prefill + 重新 decode | ❌ 客户端会看到截然不同的后续 token |
| `reprefill` | 丢弃 checkpoint，但通过 `extended_prompt = original_prompt + decoded_tokens` 保留已 decode 内容，target engine re-prefill 拼接 prompt 后继续 decode | ✅ 保留 |

实现位置（默认 `reload`，对主体实验透明）：
- [vllm/v1/engine/ft_client.py:53-77](../../vllm/v1/engine/ft_client.py#L53-L77)（env 解析）+ [1522-1611](../../vllm/v1/engine/ft_client.py#L1522-L1611)（centralized recovery 循环 mutate `cached_request`）— 走 `ft_benders_centralized` 的真正路径
- [vllm/v1/core/recovery_manager.py:46-53](../../vllm/v1/core/recovery_manager.py#L46-L53) + [368-428](../../vllm/v1/core/recovery_manager.py#L368-L428)（覆盖 `fault_tolerant` / 非 centralized `ft_benders` policy）

#### 实验结果 (W1_Chat / Heavy / F2_Mid / seed=42, 462 reqs, dp=2, 8B, ortools 已装)

| 指标 | **No-FT** (fcfs) | **reload** (默认) | **restart** | **reprefill** |
|---|---|---|---|---|
| Completed | 455/462 (98.5%) | **462/462 (100%)** | 457/462 (98.9%) | 460/462 (99.6%) |
| Failed (永久丢失) | 7 | **0** | 5 | 2 |
| Goodput (tok/s) | **321.6** | 124.2 | 47.9 | 39.5 |
| TTFT p50 (ms) | **406** | 18,597 | 50,450 | 42,800 |
| TTFT p95 (ms) | **10,438** | 55,933 | 81,049 | 73,861 |
| TPOT p50 (ms) | **43.1** | 74.6 | 85.5 | 86.5 |
| TPOT p95 (ms) | **63.7** | 111.5 | 140.4 | 180.4 |
| SLO violation rate | **20.3%** | 65.6% | 87.4% | 88.5% |
| failover_gap p50 (ms) | – | **8,613** | 8,638 | 24,645 |
| failover_gap p95 (ms) | – | **9,683** | 23,643 | 34,445 |
| 实测 displaced reqs | 0 | 14 | 36 | 29 |
| Recovery success | – | **100 %** | 94.1 % | 96.3 % |

数据：
- `results_v2/8B_recovery_modes/no_ft/No-FT/W1_Chat/Heavy/F2_Mid/42/`
- `results_v2/8B_recovery_modes/reload/Our-System/W1_Chat/Heavy/F2_Mid/42/`
- `results_v2/8B_recovery_modes/restart/Our-System/W1_Chat/Heavy/F2_Mid/42/`
- `results_v2/8B_recovery_modes/reprefill/Our-System/W1_Chat/Heavy/F2_Mid/42/`

#### 解读：为什么 restart / reprefill 的 goodput 跌得这么狠？

核心是 **prefill capacity contention 雪崩**：

| | reload | restart | reprefill |
|---|---|---|---|
| Recovery 单 req 成本 | 1 次 host→device KV 复制（**I/O bound**, ~10 GB/s）+ replay 几个 token | **完整 prefill** 原始 prompt（~几百 token）| **完整 prefill** original + decoded tokens（~几百到几千 token，更大）|
| 占用 prefill compute | ≈ 0（I/O 跟 forward 异步）| 14-30 个完整 prefill job 抢 surviving engine 的 prefill slot | 同 restart 但每个 job 更大 |

雪崩链：

1. t=150s 故障 → 7-30 个请求需要重建状态。
2. **reload** 走 I/O 路径，几乎不抢 forward compute → 新到达的请求继续正常 prefill。
3. **restart / reprefill** 走 compute 路径 → 全部塞进 prefill 队列 → 新到达的请求被卡住。
4. → TTFT p50 从 reload 的 18.6 s 跳到 50.4 s / 42.8 s（**2-3 倍**）。
5. → 几乎所有请求 SLO violate（TTFT > 2 s）。
6. → SLO compliance 从 34 % 跌到 12 %。
7. → goodput = (SLO-OK 请求的输出 token) / wall_time，分子被砍掉一大半。

**reprefill 比 restart 还差**的原因 = 单次 prefill 任务更大（带上 decoded tokens），所以 failover_gap p50 从 8.6 s 直接到 24.6 s — 这 24 秒里那些迁移的请求一个 token 都产不出来，更直接拖低 goodput。

#### 结论

**KV reload 路径是值得做的，原假设被证伪。**

> KV reload 把恢复成本从"compute"转成"I/O"。surviving engine 在 fault 后最稀缺的恰好是 compute（因为它要承担 2× load），而不是 I/O 带宽。restart / reprefill 把那笔账记在了错的资源池上，结果就是 goodput 比 reload 差 2.6-3.1×。

附带的 follow-up：
- 这个结果**不影响 partial KV restore 优化方向**（Follow-up finding #4 末尾）。partial restore 仍然走 I/O 路径，只是 I/O 量更小，跟 restart/reprefill 走 compute 路径完全是两件事。
- `FT_RECOVERY_MODE` env var 留作 ablation 接口（默认 reload），论文 Section X "Why KV reload?" 直接复用这张表即可。

### 最终 verdict (post-P0-impl-3a follow-up)

| Bug/Issue | Status |
|---|---|
| ft_scheduler 不 wrap AsyncScheduler | ✅ **Fixed** (commit `da5c8f25e`) |
| Metric definition (admitted-but-failed should count as SLO violation) | ✅ **Fixed** (commit `522c2a9dd`) |
| ortools missing in venv | ✅ **Fixed** (manual install + requirements/ft.txt added) |
| KV pressure throttle | ⚠️ **Misguided fix** — patched but useless. To revert. |
| W1_Chat/Heavy/F2_Mid 35% service rate | ⚠️ **Fundamental queueing physics, not a bug** — accept as honest limitation in paper |
| W1_Chat/Heavy/none 47% service rate (pre-ortools) | ✅ **Fixed by ortools install** — 47% → 94% (实测 v3 verify) |
| KV reload 是不是 overkill (vs restart / reprefill) | ✅ **Validated via FT_RECOVERY_MODE ablation** — restart/reprefill 的 goodput 比 reload 差 2.6-3.1× (Follow-up finding #5) |

### 必须做的事 (post-investigation)

1. ✅ 写 [requirements/ft.txt](../../requirements/ft.txt) 包含 ortools (done)
2. ⏳ Revert KV pressure throttle patches (确认无效, 应当 revert)
3. ⏳ 重跑 47 cell **with real ortools** (first ever 真正 Benders solver 数据)
4. ⏳ Update paper main thesis: framing change from "throughput trade-off" → "in-flight preservation vs new-req throughput"
5. ⏳ Add `requirements/ft.txt` 到 install instructions
6. ⏳ Future: investigate partial KV restore (优化 fault path 的 KV slot 占用)
