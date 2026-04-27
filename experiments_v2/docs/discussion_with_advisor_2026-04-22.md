# 项目现状与方向调整讨论（给老板看）

> 日期：2026-04-22
> 目的：汇报当前 idea 实验结果、诊断遇到的问题、提出调整方案

---

## 一、当前 idea 和做法

我们的系统做的是 **fault-tolerant LLM serving**：
- **Benders solver** 做 admission control（决策哪些请求接、分给哪个 GPU），每 20ms 重新规划一次
- 规划时**考虑所有可能的 GPU 故障场景**，保证"万一某张卡挂了，幸存的卡能在 SLO 内恢复"
- **Adaptive checkpoint** 把 KV cache 定期存到 CPU 内存
- 故障发生时从 checkpoint 恢复 in-flight 请求

跟 baseline 的对比对象：
- **No-FT (NR)**：vLLM 原版，不做任何容错
- **Periodic-Low / High**：简单的定期 checkpoint（每 10 block / 每 1 block）

---

## 二、实验结果：现状打不过 baseline

### 2.1 主要 benchmark 数据（Llama-3.1-8B，A5000 dp=2，W7 Heavy workload，9 seeds）

| Recovery | Checkpoint | Solver | 配置 | W7 Heavy 胜出次数 |
|---|---|---|---|---|
| reload | ✓ | ✓ | V2 完整系统 | **0/9 (输 -11%)** |
| reprefill | ✓ | ✓ | V2-reprefill | **0/8 (输 -14%)** |
| reprefill | ✗ | ✓ | V2-NoCkpt（去掉 checkpoint） | 4/9（-9.3%，不显著） |
| reprefill | ✗ | ✗ | NR（baseline） | 参照线 |

**核心事实**：
- **带 checkpoint 的所有版本都输 NR**
- **去掉 checkpoint 后性能最接近 NR**（但还是略输）
- 9 seed 下 paired t-test p≈0.11，统计上不显著

**相关图（可给老板看）**：
- `experiments_v2/figures/8B/E1a_Quick/goodput_by_load_F2_Mid.pdf` — 故障下各方法 goodput 对比
- `experiments_v2/figures/8B/E1a_Quick/slo_violation_F2_Mid.pdf` — SLO 违反率
- `experiments_v2/figures/8B/E1a_Quick/failover_gap_p95.pdf` — 故障恢复时间

### 2.2 唯一正向结果：W5 长文档过载场景

| seed | V2-NoCkpt completion | NR completion | 差值 |
|---|---|---|---|
| 42 | 78% | 90% | -12 |
| 123 | 89% | 90% | -1 |
| 456 | **85%** | **24%** | **+61** |
| 789 | 84% | 84% | 0 |
| 1234 | 96% | 96% | 0 |
| 22222 | 100% | 88% | +12 |
| **平均** | **88.5%** | **78.6%** | **+9.9%** |

在 ArXiv 长文档过载场景下，V2 的 admission control 能保护 completion rate（最极端 seed 救回 61%）。**这是目前唯一真实可靠的正向结果**。

---

## 三、为什么打不过：深入诊断

### 3.1 根因 1：实验场景下 FT 的 motivation 本身就弱

- 8B 模型在 A5000 上 re-prefill 便宜（~300ms）
- Checkpoint load 也要 ~200ms
- **两者几乎打平，checkpoint 基本没价值**
- GPU 硬件故障一年才几次，不值得为这罕见事件付常态 overhead

换句话说：**在当前实验 setting 下，"做 fault tolerance" 本身就是个弱需求**。

### 3.2 根因 2：Benders solver 在 critical path 上拖慢主线程

- Solver 每 20ms 触发一次
- 每次执行需要：snapshot building + cost table + MIP 求解 + dispatch ≈ 10-25ms Python 工作
- **全在 API server 主线程上**（Python GIL 限制，后台线程也救不了）
- 导致 decode step rate 下降 → goodput 跌

**Profile 数据（合作者 cProfile 跑出来的）**：
- API server 主线程 CPU 利用率只 13%（不是算不过来，是被串行操作卡住）
- Section 2（schedule + exec launch）占 step time 62%

### 3.3 根因 3：决策时考虑 failure，但代价不值

Benders 的核心机制是 **subproblem 验证"任意 GPU 挂了都能恢复"**。
- 这个验证每 epoch 都跑
- 但 GPU 故障实际罕见
- **为罕见事件每 20ms 付一次代价**

### 3.4 根因 4：其他同类工作都不这么做

调研了 LLM serving 领域的优化方向：

| 系统 | 决策触发 | 求解方式 | 是否 runtime critical path |
|---|---|---|---|
| AlpaServe (OSDI'23) | 离线 | ILP | 否 |
| AdaServe (EuroSys'26) | 请求到达 | 闭式公式 | 是（O(1)）|
| FastServe | 请求到达 + quota | 启发式 MLFQ | 是（O(log n)）|
| Sarathi-Serve (OSDI'24) | 每 step | 固定参数 | 是（O(1)）|
| Llumnix (OSDI'24) | 后台定期（秒级）| 贪心 | 否（后台）|
| **我们的系统** | **每 20ms** | **Benders MIP** | **是（10-25ms）** |

**我们是唯一在 runtime critical path 上跑 MIP 的**。这个架构选择跟 serving 系统的"decode 必须快"矛盾。

---

## 四、调整方向

### 4.1 明确要砍掉的

**Benders solver**：在当前场景下没体现价值
- Subproblem 的 feasibility check 靠的 capacity model 不准
- dp=2 下只有 2 个 failure scenario，用 Benders 分解得不偿失
- 跑 solver 本身的 Python overhead 拖慢主线程

**决策时对 failure 的考虑**：
- 改成 **reactive recovery**——正常运行不考虑 failure，故障发生后才处理
- 这是 vLLM、FastServe、Llumnix 等主流 serving 系统的做法

### 4.2 保留和延伸的部分

保留：
- **Adaptive checkpoint 机制**（KV pool、经济模型、recovery path）
- 实验框架

延伸成 paper 核心 contribution：
- **Per-request adaptive KV preservation**：不是每个请求都同等对待，而是根据**请求特征**动态决定 checkpoint 策略
- 关键维度：
  - **SLO 紧迫度**（紧 SLO 值得多存）
  - **Decode 进度**（已经跑了很久的丢了贵）
  - **Prompt 长度**（长 prompt 重算贵）
  - **系统负载**（系统忙时少存）

### 4.3 场景收窄

把 paper 故事从"通用 FT serving"收窄到 **long-running LLM inference 下的 runtime failure recovery**：

- 短请求：丢了 client retry 就行，不需要我们系统
- 长请求（长 reasoning、agent workflow、长文档处理）：丢了代价大，才需要保留状态

把 failure 的定义从"GPU 硬件故障"扩展到**各种 runtime event**：
- OOM crash（真实频发）
- 进程 crash
- vLLM 内部 preemption（内存压力触发）
- Spot instance 回收（云平台场景）

这样 motivation 从"为罕见事件买保险"变成"为日常事件做 recovery"。

### 4.4 实验 setting 调整

当前问题：
- W7 Heavy 短 prompt，re-prefill 不贵，checkpoint 没优势
- KV cache 使用率峰值只 2.1%，preemption 根本不触发

新 setting：
- 换 W5 LongDoc 长文档 workload（目前**唯一有正向结果的场景**）
- 或者构造长 prompt + 高 RPS 场景让 preemption 真正触发
- 可能需要提高 max_model_len（32K+）

---

## 五、老板提过的方向：Sparse KV / KV reuse

### 5.1 KV reuse（prefix caching）

跟我们 idea 的联系比较弱：
- Prefix cache 是跨请求共享 KV，重点在"前面 prompt 一样的请求不用重复算"
- 我们的 checkpoint 是单请求 state 的保留
- 两者基本正交

可以作为 **orthogonal 的 future extension** 提一下，但不是 paper 主线。

### 5.2 Sparse KV（选择性保留重要 KV）

跟我们 idea **天然契合**：

- Sparse KV 的 insight：不是所有 KV 都同等重要（H2O、StreamingLLM）
- 我们的 insight：不是所有 KV 都值得 checkpoint
- **两者核心思想一致**

可能的融合方式：

**方式 1：用 sparse 压缩 checkpoint 内容**
- Checkpoint 时不存全部 KV，只存 attention 重要的那部分
- 存储开销降 50-80%
- 但 recovery 后可能有 accuracy loss

**方式 2：二维选择（时间 × 重要性）**
- 时间维度：存哪些 token 位置（我们原来 partial save 的思路）
- 重要性维度：每个 token 里存哪些 head
- 两者联合决策

### 5.3 但查了相关工作：二维已经有人做

| 论文 | 二维？ | 会议/年份 |
|---|---|---|
| **SAGE-KV** (2025) | ✓ token + head 联合 top-k | arXiv 2025 |
| **Ada-KV** | ✓ per-head token budget | NeurIPS 2025 |
| **RazorAttention** | ✓ head-conditional eviction | ICLR 2025 |
| **DuoAttention** | ✓ head-type (full/streaming) | ICLR 2025 |
| **HeadKV / CriticalKV** | ✓ per-head importance | 2024-2025 |

**SAGE-KV 已经做了我们想做的"token × head 联合 top-k"**。单纯讲 2D 不是 novel 方向。

### 5.4 要用 sparse KV 的话，需要加第三个角度

纯 sparse + checkpoint 已经不够。可能的差异化：
- **SLO-aware 的 sparse budget 分配**（紧 SLO 的请求保留得更完整）
- **Runtime failure 场景下的 sparse checkpoint**（recovery latency 是 first-class metric）
- **动态 workload 下的 sparsity 调整**

这些还没人专门做，但每个都是单独的新方向，工作量不小。

---

## 六、三个可能的路径

### 路径 A：快速调整当前 idea，冲 SoCC（7 月中）

**做法**：
- 砍掉 Benders，决策改成 per-request
- 换长 prompt workload（W5 LongDoc）
- Framing 改成 "adaptive KV preservation for long-running LLM inference"
- 覆盖 runtime event（OOM / preempt / crash），不光是 GPU 故障

**工作量**：9-12 周
**预期录取率**：中等（~25-35%）
**风险**：跟现有 preempt-aware 工作（FastServe、QLM、CacheOpt、QLLM）的差异化要仔细划

### 路径 B：加 sparse KV，冲 EuroSys 27 fall（10 月）

**做法**：
- 路径 A 的基础上加 sparse KV（跟老板方向一致）
- 但必须加第三个角度（比如 SLO-aware sparse），不然跟 SAGE-KV 等已有工作重合
- 完整做 accuracy 评估

**工作量**：4-5 个月
**预期录取率**：类似（~25-35%），novelty 稍高但竞争也强
**风险**：跟 SAGE-KV / Ada-KV / RazorAttention 等 sparse KV 工作的差异化更难讲

### 路径 C：接受窄 scope，写 focused paper 投 SoCC

**做法**：
- 聚焦在 W5 LongDoc 过载场景的 +9.9% completion rate 保护
- 定位为 "admission control for overload protection in long-context serving"
- 不卖 fault tolerance，只卖 overload 下的 completion 保证

**工作量**：6-8 周（大部分现有数据可用）
**预期录取率**：中等偏低（~20-30%），scope 小 reviewer 可能挑
**风险**：paper 卖点弱

---

## 七、我的倾向和想听老板意见的问题

**我的倾向**：路径 A。理由：
- 时间上 SoCC 还赶得上
- 现有代码复用度最高（60-70%）
- 故事清晰（per-request adaptive checkpoint for runtime recovery）
- 不用处理 sparse KV 的 accuracy loss 评估

**但有几个问题想听老板意见**：

1. **Benders 是否真的要砍？** 我自己已经 convince 自己砍掉，但数据还没完整验证。老板觉得有没有 salvage 的可能？

2. **Sparse KV 方向**：老板坚持的话，我走路径 B。但时间会拉长到 4-5 个月，SoCC 赶不上，要等 EuroSys fall。老板觉得值得吗？

3. **实验 setting 怎么选**：
   - 当前 8B + A5000 + 短 prompt → 结果打不过 NR
   - 换 8B + 长 prompt（W5）→ 有正向结果但 scope 窄
   - 扩到 30B/70B → 硬件门槛高（需要 L40S 多卡）
   - 老板觉得硬件预算能支持到什么程度？

4. **Paper venue 预期**：老板希望投 SoCC 7 月还是 EuroSys fall 或更后面？不同选择对应不同 scope。

5. **负面结果 paper 可以接受吗**：如果最终还是打不过 NR，把 "我们实验了一圈发现 FT overhead 在 8B 规模不值" 作为一个诚实的 finding 发出去，老板觉得可行吗？

---

## 附录：关键图表位置

- **主实验 goodput 对比**：`experiments_v2/figures/8B/E1a_Quick/goodput_by_load_F2_Mid.pdf`
- **SLO 违反率**：`experiments_v2/figures/8B/E1a_Quick/slo_violation_F2_Mid.pdf`
- **故障恢复 gap**：`experiments_v2/figures/8B/E1a_Quick/failover_gap_p95.pdf`
- **组件贡献 ablation**：`experiments_v2/figures/8B/E1a_Quick/ablation_F2_Mid.pdf`
- **Controller（Benders）overhead**：`experiments_v2/figures/8B/E1a_Quick/controller_overhead.pdf`
- **Recovery 时间分解**：`experiments_v2/figures/8B/E2_Recovery/recovery_breakdown.pdf`

详细分析和完整数据在：
- `experiments_v2/docs/overnight_2026-04-21_summary.md`
- `experiments_v2/docs/paper_claim_v3_2026-04-21.md`
- `experiments_v2/docs/investigation_summary_2026-04-08-09.md`
