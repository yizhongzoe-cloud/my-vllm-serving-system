# Motivation: Runtime Failure State Preservation for Long-Running LLM Inference

## 2.1 The Rise of Long-Running Inference and Runtime Volatility

**背景升级**：LLM serving 正在从短请求的 Q&A 模式演变为**长程推理**（Long-horizon Inference）。Long reasoning 模型（如 DeepSeek-R1、o1-style）单次思考可持续 30 秒至数分钟；agent workflow 中的单次 LLM 调用伴随长 prompt 和多轮工具交互；长文档问答与代码仓库分析动辄涉及 32K 至 128K 的上下文。**单次请求的生命周期从秒级延长到分钟级**。

**状态的重要性**：在长程请求中，KV Cache 不再是临时加速器，而是**请求执行进度的唯一载体**（In-flight Working State）。一次请求已生成的数万 token，其代表的全部计算进度均凝结在 GPU 显存里的 KV Cache 中。一旦进程崩溃或实例消失，这部分进度**无法以任何外部状态重建**，只能从原始 prompt 重新开始。

## 2.2 The Problem: The High Cost of Runtime Failures

**失效的代价**：现代 LLM serving 运行时的"失效"远不止稀有的硬件故障。在实际生产部署中，**以下 runtime event 每天都会发生**：

- **Out-of-Memory Crash**：长 prompt 到达时 KV 预算预测失败，vLLM 来不及 preempt，进程被 CUDA OOM killer 杀掉
- **Process Crash**：driver bug、CUDA memory corruption、vLLM 自身缺陷导致 engine 进程退出
- **Involuntary Preemption**：vLLM 感知到 KV 压力主动踢出运行中的请求
- **Spot Instance Reclamation**：云厂商 30 秒至 2 分钟预告回收 GPU 实例（AWS Spot 省成本 50-90%）
- **Auto-Scaling Events**：平台 autoscaler 根据负载下线 replica

每一次 event 都让 replica 上的全部 in-flight KV Cache 瞬间蒸发。

**性能挑战**：对短请求（几秒完成），丢失后 client retry 代价低，用户几乎无感。但对**长请求**，丢失意味着几十秒到几分钟的 stall——用户可感知的体验断层。现有的 serving 系统（vLLM、Sarathi-Serve）假设稳态运行，**没有 principled 的 in-flight state preservation 机制**。

## 2.3 The Gap: Limitations of Prior Art

**现有方案的局限性**：

- **Client-Side Retry（业界默认做法）**：请求失败由客户端重发。简单但**用户等待重跑整个长请求**，长 reasoning 场景下代价巨大。
- **vLLM RECOMPUTE**：preempt 时丢弃 KV，resume 时全量 re-prefill。**长 prompt 下 re-prefill 成本随 context length 二次增长**（attention 的 O(N²)），32K context 下 re-prefill 可达数秒至数十秒。
- **vLLM SWAP**：preempt 时整个请求的 KV 搬到 host memory，resume 时搬回。代价是**巨大的 PCIe 带宽占用和 CPU 内存开销**，且仅覆盖 vLLM 内部 preempt 事件——无法应对进程 crash 或实例消失。
- **Full Replication（如 DejaVu 风格）**：持续复制 KV 到多个 replica。**3.0× 的显存开销**在 long-context serving 中迅速耗尽 GPU 预算，严重限制并发量。
- **ServerlessLLM (OSDI'24)**：关注模型权重的 cold start，**不处理 in-flight 请求状态**，与本文正交。
- **Llumnix (OSDI'24)**：做 cross-instance live migration，但 migration 是**稳态调度手段**（负载均衡、优先级迁移），不是 runtime failure 下的状态恢复。

**结论**：现有系统无法在 **正常态开销** 和 **Runtime Failure 下的 Recovery Latency** 之间取得平衡。

## 2.4 The Insight: Not All KV Is Worth Preserving

**核心观察**：一次长 LLM 调用中，不同位置的 KV Cache 在 **"保留价值"** 上存在显著差异。

- **Decode 进度深的 KV**：已生成数千 token，重算需要完整 prefill + 重新 decode，**保留价值最高**
- **Prefill 刚完成的 KV**：仅包含 prompt 的状态，重算成本等于一次 prefill，中等价值
- **Stale / 已完成的 KV**：请求已临近结束，剩余生成时间短，保留边际价值低
- **紧 SLO 的请求**：用户可感知延迟大，保留价值高
- **宽 SLO 的请求（后台批处理）**：可容忍重跑，保留价值低

**机会点**：通过对"关键状态"（Critical KV）进行选择性保留，可以用**远低于 Full Replication 的内存/带宽成本**，换取接近其恢复速度的用户体验。

## 2.5 Key Insights & Motivational Experiments

以下三组实验（在 Llama-3.1-8B / L40S 集群上测量）量化上述观察：

### Figure A: Runtime Failure Rates in Production-Like Configurations

- **设计思路**：在三种典型配置（生产调优 / 默认配置 / 高密度部署）下跑 24 小时 workload，记录 OOM、进程 crash、preempt 事件频率
- **预期结果**：长 context + 高密度部署下 **runtime event 达每小时数次至数十次**；即使生产调优，**长 prompt 压力下仍有显著的 tail event**
- **支撑结论**：runtime failure 不是罕见假设，是真实的高频扰动

### Figure B: Re-prefill Cost vs Context Length

- **设计思路**：测量 8K、16K、32K、64K、128K prompt 的 prefill 时间，对比不同模型（8B、70B）
- **预期结果**：re-prefill 时间**随 context length 超线性增长**；32K 上 8B 需数秒，70B 需十几秒至分钟级
- **支撑结论**：长 context 下 retry 或 RECOMPUTE 的用户可感知代价达到不可接受水平

### Figure C: Memory-Latency Trade-off Landscape

- **设计思路**：在同一 workload 下横向对比 Client Retry / vLLM RECOMPUTE / vLLM SWAP / Full Replication，绘制 **(Memory Overhead, P99 Resume Latency) 散点图**
- **预期结果**：现有方案形成两个极端——低内存高恢复延迟 vs. 低延迟高内存
- **支撑结论**：存在明显的 **Pareto gap**，为 selective preservation 留下空间

## 2.6 Opportunities: Selective State Preservation

**关键洞察**：runtime failure 频繁发生 × 长请求重算代价高 × KV 保留价值异构 → **adaptive, selective state preservation** 是填补 Pareto gap 的合理设计点。

**设计目标**：

1. **Selective Preservation（Step 1）**：基于 per-request cost model（decode 进度、SLO 紧迫度、剩余预估时间）决定哪些 KV 值得保留、保留到什么粒度
2. **Runtime-Failure-Transparent Recovery（Step 2）**：涵盖 OOM、进程 crash、preempt、spot reclamation 等多种事件类型，统一的 state preservation 抽象
3. **Cost-Bounded Overhead（Step 3）**：总体内存与带宽开销控制在 1.2–1.5× 基线水平，远低于 Full Replication 的 3.0×

---

## 写作说明

### 跟相关工作的区分（related work 要写清楚）

- **QLLM (EuroMLSys'25)**：layer-level preemption for MoE 模型。我们做 dense 模型 + token/block 粒度 + 覆盖多种 runtime event（QLLM 只管 preempt）。
- **ServerlessLLM (OSDI'24)**：模型权重 cold start。我们管 in-flight 请求状态。互补不冲突。
- **Llumnix (OSDI'24)**：live migration 作为稳态调度手段。我们做 runtime failure 下的恢复，场景不同。
- **FastServe / QLM / CacheOpt**：SLO-aware preempt policy。我们不只是 preempt policy，是统一的 failure recovery 机制。

### 数据风险点

Paper 成立的关键前提是 **Figure A 的数据**——如果实测发现 runtime event 并不频繁（比如生产调优下一天才几次），motivation 要弱化：

- **强版本（数据支持）**：strong motivation "event 频繁发生 + 每次代价大"
- **弱版本（数据不支持）**：只讲"一旦发生代价大"，承认频率低但后果严重

Figure A 的数据是 **Phase 0 benchmark 的核心目标**。先验证数据，再决定走哪个版本。

### 遗留问题

1. 系统名（System Name）还没定。原 paper 用 PRLI/PARS。可以后面想。
2. Section 2.4 的 "selective preservation" 跟原 paper 的 "criticality" 有点像，可能要在措辞上更区分，避免被误认为是 follow paper。
3. 长期看可能需要 expand 到 cross-replica state migration 的讨论（如果走 serverless spot 方向的扩展版）。
