# 实验设计方案 — MLSys 论文

> 容错多 GPU LLM Serving：自适应 KV Checkpointing + Benders 准入控制

---

## 1. 实验环境

### 1.1 硬件

| 组件 | 配置 |
|---|---|
| GPU | NVIDIA A100 80GB SXM4，NVLink 互联 |
| CPU | AMD EPYC 7763（或同级），64+ 核 |
| 主机内存 | 256+ GB DDR4（用于 KV checkpoint pool） |
| 软件 | CUDA 12.x, PyTorch 2.x, vLLM (our fork) |

### 1.2 模型

| 模型 | TP | dp_size | 总 GPU 数 | 挂 1 个 replica 容量损失 | 角色 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 1 | 4 | 4×A100 | 25%（4 个 replica 挂 1 个） | 主力模型：全扫 |
| Llama-3.1-70B-Instruct | 4 | 2 | 8×A100 | 50%（2 个 replica 挂 1 个） | 验证模型：可扩展性 |

选择理由：
- 同一模型家族，控制架构变量。
- 8B（dp=4）：挂 1 个 = 25% 容量损失，恢复场景现实，适合全扫。
- 70B（dp=2）：挂 1 个 = 50% 容量损失，KV cache 约为 8B 的 8 倍，checkpoint/恢复开销显著。
- 4/5 篇相关论文都测了 7B-8B 和 70B 级别模型。

其他参数：
- `max_model_len`: 4096
- `dtype`: float16
- `gpu_memory_utilization`: 0.90

### 1.3 数据集与工作负载

| ID | 数据集 | 应用场景 | 输入 tokens (mean±std) | 输出 tokens (mean±std) | TPOT SLO | 对 FT 系统的压力 |
|---|---|---|---|---|---|---|
| W1_Chat | **ShareGPT** | 聊天 | 227±264 | 256±249 | 100ms（中等） | decode-heavy：大量 KV 需要 checkpoint，恢复时间长 |
| W2_Summary | **CNN/DailyMail** | 摘要 | 1067±537 | 94±47 | 200ms（宽松） | prefill-heavy：prefill 阶段故障 = 全部重做 |
| W3_Instruct | **Alpaca** | 指令跟随 | ~100-300 | ~100-300 | 50ms（严格） | 严格 SLO + 中等 decode，考验恢复速度 |
| W4_Mixed | 50% W1 + 20% W3 + 30% W2 | 生产混合 | 混合 | 混合 | per-request | 测 Benders 的 per-request SLO 差异化调度 |

各数据集选择理由：
- **ShareGPT**（3/5 篇论文使用）：聊天场景的事实标准。重尾长度分布天然给 checkpoint 决策施加压力——有的请求积累了大量 KV cache，有的很快就结束。
- **CNN/DailyMail**（3/5 篇论文使用）：长输入主导 prefill 开销。prefill 阶段故障时需要在存活 GPU 上完全重做 prefill——checkpoint 对 prefill 阶段无帮助。
- **Alpaca**（AdaServe 使用）：中等 I/O 长度 + 严格 SLO。迫使系统快速恢复否则违反 SLO，直接测试恢复效率。
- **W4_Mixed**：不同请求有不同 SLO 紧急程度。Benders solver 可以优先恢复严格 SLO 的请求、放弃宽松 SLO 的——这是我们系统的核心卖点。

数据处理方式：
- ShareGPT：采样 2000 条对话，用用户发言作为 prompt，`max_tokens` 设为真实回复长度（截断到 max_model_len）。
- CNN/DailyMail：采样 2000 篇文章，追加 "Summarize the above article in one paragraph." 作为指令。
- Alpaca：使用全部 52k 条 instruction-response 对，有放回采样。
- Mixed：每次到达时，按比例采样工作负载类型，再从对应数据集中采样。

### 1.4 请求到达模式

| 模式 | 说明 | 用途 |
|---|---|---|
| **Poisson** | 指数间隔到达，目标 RPS | 主实验（所有实验） |
| **Azure LLM Trace** | Azure 真实生产 trace（coding + chatting） | 补充验证（E1 附录） |

Poisson 作为主到达过程（SOLA、TurboSpec 均使用）。Azure trace（SLOs-Serve 使用）作为真实场景的补充验证。

### 1.5 负载级别

负载定义为**单 GPU 利用率**（相对于单 GPU 饱和容量的百分比），不是绝对 RPS。

**校准流程：**
1. 对每个（模型，工作负载）组合：跑 No-FT baseline，逐步增大 RPS，直到 SLO violation > 10%，记为 `RPS_sat`。
2. 单 GPU 容量 = `RPS_sat / dp_size`。
3. 负载级别是单 GPU 容量的百分比：

| 级别 | 单 GPU 利用率 | 故障后负载 (dp=4, 挂1) | 故障后负载 (dp=2, 挂1) | 场景 |
|---|---|---|---|---|
| Light | 25% | 33% | 50% | 轻松恢复 |
| Moderate | 40% | 53% | 80% | 有压力但可恢复 |
| Heavy | 55% | 73% | 110% | 接近/超容量，需要准入控制 |

### 1.6 SLO 配置

**校准流程：**
1. **TTFT_base**：跑单请求（无竞争）→ 测 TTFT。
2. **TPOT_base**：同上 → 测 TPOT。
3. **Gap_base**：跑 Periodic-High baseline + F2 故障，Moderate 负载 → 测恢复请求的 failover gap P50。

| SLO | 值 | 校准方式 | 参考 |
|---|---|---|---|
| TTFT | 5× TTFT_base | 跟模型走，倍数校准 | SOLA (5-10×), SLOs-Serve (3-5×) |
| TPOT | per-workload: 50 / 100 / 200 ms | 绝对值，按应用场景 | AdaServe (40-150ms), SLOs-Serve (50-100ms) |
| Failover Gap | 3× Gap_base | 跟模型走，自动校准 | 我们独有的 SLO |

E6（SLO 敏感度实验）额外测试：
- Tight：TTFT = 3× base，Gap = 1.5× base
- Moderate：TTFT = 5× base，Gap = 3× base（默认）
- Loose：TTFT = 10× base，Gap = 5× base

### 1.7 故障注入

| ID | 时间 | 系统状态 | 说明 |
|---|---|---|---|
| none | — | 无故障 | 测正常运行时的 checkpoint 开销 |
| F1_Early | 60s | 刚进稳态 | 少量 in-flight 请求，KV cache 小 |
| F2_Mid | 150s | 稳态中期 | 典型故障场景，大部分实验的默认设置 |
| F3_Late | 240s | 稳态后期 | 大量累积的 KV cache，checkpoint pool 压力最大 |

机制：对第一个 EngineCore worker 进程发送 `SIGKILL`（模拟突发 GPU 故障 / CUDA 错误）。

### 1.8 Baselines

| ID | 简称 | 调度策略 | Checkpoint 策略 | 代表什么 |
|---|---|---|---|---|
| B1 | **No-FT** | FCFS | 不存 | 下界：无容错，故障时所有 in-flight 请求丢失 |
| B2 | **Periodic-Low** | Greedy | 固定每 10 blocks | 保守周期 checkpoint（低开销，慢恢复） |
| B3 | **Periodic-High** | Greedy | 固定每 1 block | 激进周期 checkpoint（高开销，快恢复） |
| B4 | **Benders-Only** | Benders | 固定每 10 blocks | 消融：智能调度 + 固定 checkpoint |
| B5 | **Adaptive-Only** | Greedy | 自适应 | 消融：贪心调度 + 智能 checkpoint |
| B6 | **FTServe (Ours)** | Benders | 自适应 | 完整系统：联合优化 |

### 1.9 评估指标

| 指标 | 定义 | 使用场景 |
|---|---|---|
| **Goodput** | 满足 SLO 的请求的 output tokens / 运行时间 (tok/s) | E1, E3, E4 的主要指标 |
| **SLO Violation Rate** | 完成请求中违反任何 SLO 的比例 | E1, E6 的主要指标 |
| **Failover Gap (P95)** | 故障影响请求的最大 token 间隔的 P95 (ms) | E2 的主要指标 |
| **Recovery Success Rate** | 受故障影响且成功恢复完成的请求比例 | E1, E2 中报告 |
| **Time-to-Stable** | 从故障到吞吐恢复至 pre-fault 90% 并持续 10s 的时间 (s) | E2 中报告 |
| **Completion Rate** | 所有请求中成功完成的比例 | E1 中报告 |
| **TTFT (P50, P95, P99)** | 首 token 延迟的百分位 (ms) | E1 表格中报告 |
| **TPOT (P50, P95, P99)** | 每 token 生成延迟的百分位 (ms) | E1 表格中报告 |
| **Solver Overhead** | Benders 求解时间 / 总决策周期时间 (%) | E5 的主要指标 |
| **Checkpoint Overhead** | 无故障时相对 No-FT 的 goodput 下降 (%) | E4 的主要指标 |

### 1.10 运行参数

| 参数 | 值 |
|---|---|
| 单次运行时长 | 300s (5 min) |
| Warmup | 30s（排除在指标计算之外） |
| Seeds | 3 个 (42, 123, 456) |
| 最大并发连接 | 200 |
| 请求超时 | 120s |
| Checkpoint pool 大小 | 32 GB (8B 模型), 128 GB (70B 模型) |

---

## 2. 实验与图表设计

### E1: 端到端性能

**问题**：FTServe 能否在所有工作负载、负载水平和故障场景下保持最高 goodput 和最低 SLO violation？

**规模**：
- E1a (8B)：6 baselines × 4 workloads × 3 loads × 4 faults × 3 seeds = **864 runs**
- E1b (70B)：3 baselines (No-FT, Periodic-High, Ours) × 2 workloads (W1, W4) × 2 loads (Moderate, Heavy) × 2 faults (none, F2) × 3 seeds = **72 runs**

---

#### Figure 1: Goodput vs 负载（无故障）

- **图类型**：折线图 + 误差棒 (mean ± std across seeds)
- **布局**：1 行 × 4 列 (W1_Chat, W2_Summary, W3_Instruct, W4_Mixed)
- **X 轴**：负载级别 (Light / Moderate / Heavy)
- **Y 轴**：Goodput (tokens/s)
- **数据线**：每个 baseline 一条线（6 条），按颜色 + 标记区分
- **数据来源**：E1a, fault=none
- **想传达的信息**：FTServe 的 checkpoint 开销极小（接近 No-FT）；Periodic-High 即使无故障也有显著开销。

```
预期趋势：
- No-FT 是天花板（零开销）
- Periodic-High 比 No-FT 低 10-20%（checkpoint 开销）
- Periodic-Low 比 No-FT 低 2-5%（不频繁的 checkpoint）
- FTServe 与 No-FT 差距在 1-5% 以内（自适应：只在有收益时才 checkpoint）
```

---

#### Figure 2: Goodput vs 负载（F2_Mid 故障）

- **图类型**：折线图 + 误差棒
- **布局**：1 行 × 4 列 (W1-W4)
- **X 轴**：负载级别
- **Y 轴**：Goodput (tokens/s)
- **数据线**：每个 baseline 一条线（6 条）
- **数据来源**：E1a, fault=F2_Mid
- **想传达的信息**：故障下 FTServe 保持最高 goodput。No-FT 大幅下跌（丢失请求产生零 goodput）。Periodic-High 恢复快但预先付出了开销。FTServe 两全其美。

```
预期趋势：
- No-FT goodput 下降 25-50%（故障 GPU 上的请求全丢）
- Periodic-Low 恢复部分请求但 replay 慢
- Periodic-High 恢复快但起点就低（开销）
- FTServe 恢复好 + 起点高 → 整体 goodput 最优
```

---

#### Figure 3: SLO Violation Rate

- **图类型**：分组柱状图
- **布局**：2 行 × 4 列。行 1：无故障。行 2：F2 故障。列：W1-W4。
- **X 轴**：Baseline（6 组）
- **Y 轴**：SLO violation rate (%)
- **分组依据**：负载级别 (Light / Moderate / Heavy 作为柱子簇)
- **数据来源**：E1a
- **想传达的信息**：FTServe 在所有条件下 SLO violation 最低。优势在 fault + heavy load 下最大。

---

#### Figure 4: Failover Gap (P95) vs 故障时机

- **图类型**：分组柱状图
- **布局**：1 行 × 2 列 (W1_Chat, W4_Mixed — 两个 decode-heavy 工作负载)
- **X 轴**：故障时机 (F1_Early, F2_Mid, F3_Late)
- **Y 轴**：Failover gap P95 (ms)，Gap SLO 阈值用虚线标出
- **柱子**：每个 baseline 一个（No-FT 除外，因为请求直接丢了）
- **标注**：每个柱子上方标注 "恢复 X/Y"（成功/总受影响数）
- **数据来源**：E1a, load=Moderate
- **想传达的信息**：FTServe 的 failover gap 最短，因为自适应 checkpoint 同时减少了 restore 和 replay 时间。

---

#### Table 1: 端到端汇总（8B, Moderate 负载）

| Baseline | 故障 | Goodput | TTFT P95 | TPOT P95 | SLO Viol. | 完成率 | 恢复率 |
|---|---|---|---|---|---|---|---|
| No-FT | none | — | — | — | — | — | — |
| No-FT | F2 | — | — | — | — | — | — |
| Periodic-Low | none | — | — | — | — | — | — |
| Periodic-Low | F2 | — | — | — | — | — | — |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **Ours** | none | — | — | — | — | — | — |
| **Ours** | F2 | — | — | — | — | — | — |

- 每个 workload 一个子表，或一个合并表取 workload 平均。
- 报告 3 seeds 的 mean ± std。

---

#### Table 2: 70B 模型验证 (E1b)

| Baseline | 工作负载 | 负载 | 故障 | Goodput | SLO Viol. | Gap P95 | 恢复率 |
|---|---|---|---|---|---|---|---|
| No-FT | W1 | Mod | F2 | — | — | — | — |
| Periodic-High | W1 | Mod | F2 | — | — | — | — |
| Ours | W1 | Mod | F2 | — | — | — | — |
| ... | ... | ... | ... | ... | ... | ... | ... |

- 精简表格，确认趋势在 70B 规模下依然成立。
- 重点：KV cache 大 8 倍 → checkpoint/恢复更耗时 → 自适应策略优势更明显。

---

### E2: 恢复时间分解

**问题**：恢复时间由什么主导？各 baseline 的恢复时间构成有何不同？

**规模**：4 baselines × 2 models × 2 workloads (W1, W2) × 1 load (Moderate) × 1 fault (F2) × 3 seeds = **48 runs**

---

#### Figure 5: 恢复时间分解

- **图类型**：堆叠水平柱状图
- **布局**：2 行 × 2 列。行：8B, 70B。列：W1_Chat, W2_Summary。
- **X 轴**：时间 (ms)
- **Y 轴**：Baseline (Periodic-Low, Periodic-High, Ours)
- **堆叠段**（4 种颜色）：
  - 检测时间 (T_det)：从故障到失败声明
  - KV 恢复时间 (T_restore)：host → GPU checkpoint 拷贝
  - Replay 时间 (T_replay)：重新执行未覆盖的 tokens
  - 首 token 开销：调度器重新准入 + 首次 decode 步骤
- **数据来源**：E2，在所有恢复请求上取平均
- **想传达的信息**：
  - Periodic-Low：T_restore 小但 T_replay 大（checkpoint 的少，要 replay 的多）
  - Periodic-High：T_restore 大但 T_replay 小（checkpoint 的多，replay 少）
  - FTServe：平衡——自适应 checkpoint 选了最合适的量

---

#### Figure 6: 单请求恢复 Gap 分布

- **图类型**：CDF 图
- **布局**：1 行 × 2 列 (8B, 70B)
- **X 轴**：Failover gap (ms)，对数刻度
- **Y 轴**：CDF（恢复请求中的累积比例）
- **数据线**：每个 baseline 一条线
- **垂直虚线**：Gap SLO 阈值
- **数据来源**：E2, W1_Chat
- **想传达的信息**：FTServe 的 CDF 曲线最靠左（恢复最快），且在 SLO 线以下的比例最高。

---

#### Table 3: 恢复统计

| 模型 | Baseline | 平均恢复 tokens | 平均 replay tokens | 中位 Gap (ms) | P95 Gap (ms) | 恢复成功率 | 恢复稳定时间 (s) |
|---|---|---|---|---|---|---|---|
| 8B | Periodic-Low | — | — | — | — | — | — |
| 8B | Periodic-High | — | — | — | — | — | — |
| 8B | Ours | — | — | — | — | — | — |
| 70B | ... | ... | ... | ... | ... | ... | ... |

---

### E3: 消融实验

**问题**：Benders 调度和自适应 checkpoint 各贡献多少？联合优化是否优于任何单独组件？

**规模**：5 baselines（去掉 No-FT）× 1 model (8B) × 2 workloads (W1, W4) × 2 loads (Moderate, Heavy) × 2 faults (none, F2) × 3 seeds = **120 runs**

---

#### Figure 7: 消融 — Goodput 对比

- **图类型**：分组柱状图
- **布局**：2 行 × 2 列。行：无故障 / F2 故障。列：W1_Chat, W4_Mixed。
- **X 轴**：Baseline (Periodic-Low, Periodic-High, Benders-Only, Adaptive-Only, Ours)
- **Y 轴**：Goodput (tokens/s)
- **分组依据**：负载 (Moderate, Heavy)
- **数据来源**：E3
- **想传达的信息**：
  - Benders-Only > Periodic（智能路由有帮助）
  - Adaptive-Only > Periodic（智能 checkpoint 有帮助）
  - **Ours > Benders-Only 且 Ours > Adaptive-Only**（联合优化是超加性的）

---

#### Figure 8: 消融 — SLO Violation 热力图

- **图类型**：热力图
- **布局**：行 = baselines (5)，列 = (workload × load × fault) 组合 (8)
- **单元格值**：SLO violation rate (%)，颜色从绿（低）到红（高）
- **数据来源**：E3
- **想传达的信息**：FTServe（底行）在所有条件下一致为绿色。

---

#### Table 4: 消融提升分解

| 对比 | Goodput Δ (无故障) | Goodput Δ (有故障) | SLO Viol. Δ |
|---|---|---|---|
| Ours vs Periodic-High | +X%（减少 ckpt 开销） | +Y%（更好的恢复） | -Z pp |
| Ours vs Benders-Only | +A%（自适应 ckpt 省开销） | +B%（恰到好处的 ckpt 量） | -C pp |
| Ours vs Adaptive-Only | +D%（Benders 准入控制） | +E%（恢复感知路由） | -F pp |

- 量化每个组件的边际贡献。

---

### E4: Checkpoint 开销 vs 恢复收益

**问题**：checkpoint 频率如何在正常运行开销和故障恢复质量之间权衡？我们的自适应策略是否找到了 sweet spot？

**规模**：4 baselines × 2 models × 1 workload (W1) × 2 loads × 2 faults (none, F2) × 3 seeds = **96 runs**

---

#### Figure 9: Checkpoint Tradeoff 曲线

- **图类型**：散点图 + 连线
- **布局**：1 行 × 2 列 (8B, 70B)
- **X 轴**：正常运行 goodput 相对 No-FT 的下降 (%) — checkpoint 开销
- **Y 轴**：故障下 goodput 相对 No-FT 的提升 (%) — 恢复收益
- **数据点**：每个 baseline 是一个带标签的点。理想位置 = 右下角（低开销，高恢复收益）。
  - No-FT：(0%, 0%) — 锚点
  - Periodic-Low：（低开销，低恢复）
  - Periodic-High：（高开销，高恢复）
  - Ours：（低开销，高恢复）— **sweet spot**
- **数据来源**：E4, load=Moderate
- **想传达的信息**：FTServe 的恢复质量接近 Periodic-High，但开销接近 Periodic-Low。

---

#### Figure 10: Goodput 时间线

- **图类型**：时间序列折线图
- **布局**：1 行 × 2 列 (8B, 70B)
- **X 轴**：时间 (s)，故障注入点 t=150s 用红色垂直虚线标记
- **Y 轴**：滑动窗口 goodput (tokens/s, 5s 窗口)
- **数据线**：No-FT, Periodic-High, Ours（3 条）
- **数据来源**：E4, W1, Moderate, F2, 选一个代表性 seed
- **想传达的信息**：展示实际恢复动态。No-FT 跌下去后丢失的 token 再也回不来。Periodic-High 短暂下跌后恢复。FTServe 恢复同样快但 baseline 更高（故障前开销更小）。

---

### E5: Solver 开销

**问题**：Benders solver 是否给调度决策增加了不可接受的延迟？

**规模**：1 baseline (Ours) × 2 models × 4 workloads × 3 loads × 1 fault (none) × 3 seeds = **72 runs**

---

#### Figure 11: Solver 开销 vs 负载

- **图类型**：柱状图 + 误差棒
- **布局**：1 行 × 2 列 (8B, 70B)
- **X 轴**：工作负载 (W1-W4)
- **Y 轴（左）**：平均每 epoch 求解时间 (ms)
- **Y 轴（右）**：solver 开销占 epoch 时间的百分比
- **分组依据**：负载级别 (Light, Moderate, Heavy)
- **水平虚线**：10% 开销阈值
- **数据来源**：E5
- **想传达的信息**：solver 开销始终 < 5%，即使在 heavy load 下。

---

#### Figure 12: Solver 收敛行为

- **图类型**：箱线图
- **布局**：1 行 × 2 列 (8B, 70B)
- **X 轴**：负载级别
- **Y 轴**：Benders 收敛迭代次数
- **数据来源**：E5，跨 workload 聚合
- **想传达的信息**：solver 通常 2-5 次迭代收敛，heavy load 下偶有 8-10 次的 outlier。

---

#### Table 5: Solver 统计

| 模型 | 负载 | 平均求解时间 (ms) | P99 求解时间 (ms) | 平均迭代次数 | 开销占比 (%) | Fallback 率 (%) |
|---|---|---|---|---|---|---|
| 8B | Light | — | — | — | — | — |
| 8B | Moderate | — | — | — | — | — |
| 8B | Heavy | — | — | — | — | — |
| 70B | ... | ... | ... | ... | ... | ... |

---

### E6: SLO 敏感度分析

**问题**：系统性能如何随 SLO 松紧变化？FTServe 相对 No-FT 的优势在什么范围最大？

**规模**：2 baselines (No-FT, Ours) × 1 model (8B) × 2 workloads (W1, W4) × 1 load (Moderate) × 1 fault (F2) × 3 seeds × 3 SLO levels = **36 runs**

---

#### Figure 13: SLO 敏感度

- **图类型**：分组柱状图
- **布局**：1 行 × 2 列 (W1_Chat, W4_Mixed)
- **X 轴**：SLO 松紧级别 (Tight, Moderate, Loose)
- **Y 轴（左柱）**：Goodput (tokens/s)
- **Y 轴（右柱）**：SLO violation rate (%)
- **分组依据**：Baseline (No-FT vs Ours)
- **数据来源**：E6
- **想传达的信息**：FTServe 在 moderate SLO 下优势最大。SLO 太松时 No-FT 存活请求也大多满足；SLO 太紧时恢复的请求也可能违反。

---

#### Figure 14: Gap SLO 敏感度

- **图类型**：折线图
- **布局**：单图
- **X 轴**：Gap SLO 倍数 (1×, 2×, 3×, 4×, 5× Gap_base)
- **Y 轴**：恢复请求中满足 Gap SLO 的比例
- **数据线**：Periodic-Low, Periodic-High, Ours
- **数据来源**：E6 + E2 数据，在不同 gap 阈值下重新计算
- **想传达的信息**：FTServe 即使在紧 gap 阈值下也能保持高达标率，而 Periodic-Low 急剧下降。

---

## 3. 图表完整索引

### 正文

| ID | 类型 | 内容 | 实验 | 核心信息 |
|---|---|---|---|---|
| **Fig 1** | 折线图 (1×4) | Goodput vs 负载，无故障 | E1a | FTServe checkpoint 开销极小 |
| **Fig 2** | 折线图 (1×4) | Goodput vs 负载，F2 故障 | E1a | FTServe 故障下保持最高 goodput |
| **Fig 3** | 柱状图 (2×4) | SLO violation rate | E1a | FTServe 在所有条件下 violation 最低 |
| **Fig 4** | 柱状图 (1×2) | Failover gap P95 vs 故障时机 | E1a | FTServe 恢复 gap 最短 |
| **Fig 5** | 堆叠柱状图 (2×2) | 恢复时间分解 | E2 | 自适应 checkpoint 平衡 restore 和 replay |
| **Fig 6** | CDF 图 (1×2) | 单请求恢复 gap 分布 | E2 | FTServe CDF 最靠左 |
| **Fig 7** | 柱状图 (2×2) | 消融：goodput 对比 | E3 | 联合优化 > 任何单独组件 |
| **Fig 8** | 热力图 | 消融：SLO violation 全景 | E3 | FTServe 始终最低 |
| **Fig 9** | 散点图 (1×2) | Checkpoint 开销 vs 恢复收益 | E4 | FTServe 在 Pareto 最优 sweet spot |
| **Fig 10** | 时间线 (1×2) | 故障前后 goodput 变化 | E4 | 可视化恢复动态 |
| **Fig 11** | 柱状图 (1×2) | Solver 开销 vs 负载 | E5 | 开销 < 5% |
| **Table 1** | 表格 | 端到端汇总 (8B, Moderate) | E1a | 完整数值 |
| **Table 2** | 表格 | 70B 验证 | E1b | 趋势在大模型上成立 |
| **Table 3** | 表格 | 恢复统计 | E2 | 恢复细节 |
| **Table 4** | 表格 | 消融提升分解 | E3 | 各组件边际贡献 |
| **Table 5** | 表格 | Solver 统计 | E5 | 开销数据 |

### 附录

| ID | 类型 | 内容 | 实验 |
|---|---|---|---|
| Fig 12 | 箱线图 | Solver 收敛迭代次数 | E5 |
| Fig 13 | 柱状图 | SLO 敏感度 | E6 |
| Fig 14 | 折线图 | Gap SLO 敏感度 | E6 |
| Table S1 | 表格 | E1a 完整结果 (所有 workload × load × fault) | E1a |
| Table S2 | 表格 | Azure trace 验证结果 | E1 补充 |
| Fig S1 | 折线图 | Azure trace 下 Goodput vs 负载 | E1 补充 |

---

## 4. 实验矩阵总览

| 实验 | 问题 | Baselines | 模型 | 工作负载 | 负载 | 故障 | Seeds | Runs |
|---|---|---|---|---|---|---|---|---|
| **E1a** | 端到端 (8B) | 6 | 1 | 4 | 3 | 4 | 3 | **864** |
| **E1b** | 端到端 (70B) | 3 | 1 | 2 | 2 | 2 | 3 | **72** |
| **E2** | 恢复时间分解 | 4 | 2 | 2 | 1 | 1 | 3 | **48** |
| **E3** | 消融实验 | 5 | 1 | 2 | 2 | 2 | 3 | **120** |
| **E4** | Checkpoint 权衡 | 4 | 2 | 1 | 2 | 2 | 3 | **96** |
| **E5** | Solver 开销 | 1 | 2 | 4 | 3 | 1 | 3 | **72** |
| **E6** | SLO 敏感度 | 2 | 1 | 2 | 1 | 1 | 3×3 | **36** |
| | | | | | | | **合计** | **~1308** |

预估时间：
- 8B runs：约 10 min/run（含启动）。~1100 runs × 10 min ≈ 183 小时 ≈ 8 天。
- 70B runs：约 15 min/run。~200 runs × 15 min ≈ 50 小时 ≈ 2 天。
- **总计：单 8-GPU 节点约 10 天**（串行执行），多节点可并行加速。

---

## 5. 执行顺序

按以下顺序执行，最大化早期反馈：

```
Phase 0: Profiling (1 天)
├─ 跑 profile_checkpoint_costs.py，分别测 8B 和 70B
├─ 跑 prescan (suite.py --prescan) 找每个 (model, workload) 的 RPS_sat
├─ 跑 recovery prescan：Periodic-High + F2 + Moderate → 拿到 Gap_base
└─ 输出：checkpoint_cost_profile_{8B,70B}.json、config.yaml 中的 load_levels 和 SLO 值

Phase 1: Smoke Test (半天)
├─ E1a 跑 1 个 seed，2 个 baseline (No-FT, Ours)，仅 W1，Moderate，仅 F2
├─ 验证流水线：server 启动、请求发送、故障注入、恢复日志、指标计算
└─ 修 bug，确保全流程正确

Phase 2: 核心结果 (4 天)
├─ E1a：8B 全扫 (864 runs) — 主结果
├─ E3：消融实验 (120 runs) — 验证各组件贡献
└─ 每组实验跑完后跑 analyze.py 检查趋势

Phase 3: 深度分析 (3 天)
├─ E2：恢复时间分解 (48 runs)
├─ E4：Checkpoint 权衡 (96 runs)
├─ E5：Solver 开销 (72 runs)
└─ E6：SLO 敏感度 (36 runs)

Phase 4: 大模型验证 (2 天)
├─ E1b：70B 验证 (72 runs)
└─ 补充：Azure trace 验证

Phase 5: 收尾 (1 天)
├─ 重跑失败/异常的 runs
├─ 生成最终图表
└─ 计算所有汇总统计
```

---

## 6. 本方案不覆盖的内容（局限性 / Future Work）

| 方向 | 为什么不做 | 论文中写在哪 |
|---|---|---|
| 多节点容错 | 当前系统是单机多 GPU | Limitations |
| 多 GPU 同时故障 | 系统目前支持 max_gpu_failures=1 | Future work |
| 投机解码集成 | 正交技术，可以结合 | Future work |
| 模型级 checkpoint 压缩 | 可进一步降低 checkpoint 开销 | Future work |
| Prefill/Decode 分离部署 | 正交架构（参考 DistServe） | Related work |
| 真实 GPU 硬件故障 | 我们用 SIGKILL 模拟；真实故障检测特征可能不同 | Threats to validity |
