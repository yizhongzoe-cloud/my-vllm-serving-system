# SLO-Aware LLM Serving 相关论文实验设置分析

---

## 1. AdaServe (EuroSys'26)

### 模型配置

| Target Model | Draft Model | GPU 配置 |
|---|---|---|
| Llama3.1-70B-Instruct (TP=4) | LLaMA-3.2-1B-Instruct | 4×A100 80GB |
| Qwen2.5-32B-Instruct (TP=2) | Qwen2.5-0.5B-Instruct | 2×A100 80GB |

Draft model 和 target model 同族，共置在同一 GPU 上。

### 数据集

三类请求，模拟不同应用场景：

| 类别 | 场景 | 数据集 | 说明 |
|---|---|---|---|
| Cat.1 | Coding Copilot | **HumanEval** (164条) | 延迟敏感 |
| Cat.2 | Chatbot | **Alpaca** (52k条) | 中等延迟要求 |
| Cat.3 | Summarization | **CNN/DailyMail** | 延迟要求最宽松 |

默认混合比例：**60% Coding / 20% Chatbot / 20% Summarization**。

### 负载生成

- 采用**真实 trace** 的请求时间戳，截断缩放得到不同 RPS
- 总持续约 **20 分钟**，带波动
- RPS 范围约 **2.4 ~ 4.8 req/s**
- 还构造了合成 trace，不同类别在不同时段出现峰值

### SLO 定义

核心指标：**TPOT (Time Per Output Token)**

| 类别 | TPOT SLO | 依据 |
|---|---|---|
| Cat.1 (Coding) | 1.2× baseline latency | 与 MLPerf v5.0 的 40ms/token 对齐 |
| Cat.2 (Chatbot) | 50ms/token | 人类阅读速度 |
| Cat.3 (Summarization) | 150ms/token | 宽松场景 |

SLO scale 实验：scale 从 0.6 到 1.6，测试不同严格程度。

### Baselines

1. **vLLM v0.8.4** — continuous batching + PagedAttention
2. **Sarathi-Serve** — chunked prefill
3. **vLLM-Spec(4/6/8)** — 固定投机长度 4/6/8

### 硬件

- 单节点，4×NVIDIA A100 80GB，NVLink 互联
- CPU: AMD EPYC 7763 64核
- 256 GB DRAM
- 基于 **FlexFlow Serve** + FlashInfer

### 核心指标

- **SLO Attainment**：满足 TPOT SLO 的请求比例
- **Goodput**：只计算满足 SLO 的请求产生的 tokens/s
- Mean accepted tokens、延迟分解、CPU overhead

---

## 2. TurboSpec (Arxiv 2025)

### 模型配置

| Target Model | Draft Model | GPU |
|---|---|---|
| Vicuna-7B (TP=1) | Llama-160M (fine-tuned on ShareGPT) | 1×L40S |
| Vicuna-13B (TP=1 或 TP=2) | Llama-160M | 1×L40S 或 2×H100 |
| Vicuna-33B (TP=2) | Llama-160M | 2×H100 |
| LLaMA3.1-70B (TP=4) | Llama3.2-1B | 4×H100 |

还测了 **Prompt Lookup Decoding (PLD)**，不需要额外 draft model。

### 数据集

| 数据集 | 场景 | 输入长度 (avg±std) | 输出长度 (avg±std) |
|---|---|---|---|
| **ShareGPT** | 在线聊天 | 227±264 | 227~274±208~255 |
| **Sonnet** | 写作补全 | 517±10 | 135~145±15~22 |
| **CNN/Daily Mail** | 文本摘要 | 1067±537 | 82~108±41~48 |

Token acceptance rate：ShareGPT 0.56~0.78，Sonnet 0.53~0.92，CNN/DM 0.54~0.61。

### 负载生成

- **Poisson 分布**到达
- QPS 取值：0.5, 1, 2, 4, 8, 16
- 动态 QPS 实验：QPS 按 1→16→48 切换，每段 40s，总 2 分钟
- 数据集分布变化实验：前 60s Sonnet，后 60s ShareGPT，固定 6 QPS
- **Greedy sampling**

### SLO 定义

**没有定义传统 TTFT/TPOT SLO**。核心优化目标是：

$$\text{Goodput} = \frac{\text{Generated Tokens}}{\text{Execution Time}}$$

不区分请求是否满足延迟约束，纯粹优化有效 token 生成速率。

### Baselines

1. **No Speculative Decoding** — 标准自回归
2. **Fixed-k SD** — k=1, 3, 5, 7 的穷举
3. 三种投机方法：**PLD**、**Eagle**、**Draft model-based**

### 硬件

- NVIDIA H100 80GB 和 L40S 48GB
- 配置从单卡到 4 卡不等

### 核心指标

- **Goodput** (tokens/s，核心指标)
- Average request latency
- Speedup（相对非投机基线）
- Throughput（离线 batch 场景）
- Request latency CDF
- Token acceptance rate
- Memory usage

---

## 3. AdaSpec (SoCC'25)

### 模型配置

| Target Model (TP) | Draft Model (TP) | GPU |
|---|---|---|
| Vicuna-7B-v1.5 (1) | Vicuna-68M (1) | 1×L40 或 1×A100 40GB |
| Vicuna-33B-v1.3 (4) | Vicuna-160M (1) | 4×A100 |
| Llama-3.3-70B (8) | Llama-3.2-1B (1) | 8×A100 |

### 数据集

使用 **SpecBench**，涵盖 6 类任务，每类随机抽 **80 条**，共 **480 条**请求：

| 任务 | 数据集 |
|---|---|
| 翻译 | WMT14 DE-EN |
| 摘要 | CNN/Daily Mail |
| 问答 | Natural Questions |
| 数学推理 | GSM8K |
| RAG | DPR |
| 通用对话 | MTbench |

输入长度大部分 0-500 tokens，部分（RAG）可达 1000+；输出长度 0-400 tokens，不同任务差异明显。

### 负载生成

- 从 **BurstGPT** 和 **Mooncake** 真实生产 trace 中提取 3 条代表性 trace
- Trace 1: 剧烈波动；Trace 2: 高频周期性震荡；Trace 3: 围绕中心值快速波动
- 截取 **10 分钟** 中等请求速率窗口

### SLO 定义

- 指标：**TPOT**
- 阈值：autoregressive decoding 的 **P90 TPOT** 作为基准
- Scale factor 从 **0.8（最严格）到 1.2（最宽松）**

### Baselines

1. **Autoregressive decoding** — 标准自回归
2. **Vanilla SD (k=1,3,5)** — 固定投机长度
3. **Threshold-based dynamic SD** — HuggingFace 内置方法
4. **SpecInfer*** — tree width=1 的变体

### 硬件

- L40 48GB、A100 40GB
- 最多 8×A100
- 基于 **vLLM** 扩展，约 2000 行 Python

### 核心指标

- **端到端 Speedup**（相对 autoregressive，主要指标）
- SLO attainment
- 平均 speculative length / accepted tokens / acceptance rate
- 不同 batch size 下的表现

---

## 4. SOLA (MLSys'25)

### 模型配置

**纯调度系统，不用 speculative decoding，没有 draft model。**

| 模型 | TP |
|---|---|
| Llama3-8B | 1 |
| Llama3-70B | 4 |
| Qwen1.5-14B | 2 |
| Vicuna1.5-7B-16k | 1 |
| Qwen1.5-72B | 4 |

### 数据集

| 场景 | 数据集 | 请求数 | 特征 |
|---|---|---|---|
| Chatbot | **ShareGPT** | 2000 条 | 常规对话 |
| 长文本理解 | **LongBench** | 832 条 | 平均输入 3.3k tokens |

### 负载生成

- **Poisson 分布**
- 请求率因模型而异：
  - Llama3-8B: 8~10.5 req/s
  - Llama3-70B: 5~7 req/s
  - Qwen1.5-14B: 5~6.5 req/s
  - Vicuna1.5-7B (LongBench): 0.5~1.1 req/s
  - Qwen1.5-72B (LongBench): 0.25~0.55 req/s

### SLO 定义

同时约束 **TTFT** 和 **TPOT**。基于单请求延迟的倍数：

| 模型 | Tight SLO (TTFT/TPOT) | Loose SLO (TTFT/TPOT) |
|---|---|---|
| Llama3-8B | 0.4s / 0.15s | 0.6s / 0.225s |
| Llama3-70B | 0.5s / 0.2s | 1.0s / 0.4s |
| Qwen1.5-14B | 0.8s / 0.2s | 1.2s / 0.3s |
| Vicuna1.5-7B-16k | 2.55s / 0.16s | 3.825s / 0.24s |
| Qwen1.5-72B | 2.425s / 0.2s | 4.85s / 0.4s |

7B-14B: tight=10×单请求延迟，loose=15×；70B-72B: tight=5×，loose=10×。
还测了 very tight (5×) 和 very loose (20×)。

### Baselines

1. **vLLM-D** — 默认 prefill-prioritized
2. **vLLM-S** — SplitFuse (chunked prefill + decode 混合)，chunk size 1024/4096
3. **SJF** — Shortest-Job-First

全部基于 **vLLM v0.4.2**。

### 硬件

- 单节点，8×NVIDIA A100 80GB SXM4
- CUDA 12.1, NCCL 2.18, PyTorch 2.3.0

### 核心指标

- **SLO attainment**（核心指标）
- **Goodput**：90%/99% SLO attainment 下的最大请求率
- TTFT-TPOT 二维分布
- 调度开销（0.40%~0.45%）
- Cost model 精度（误差 4.98%~5.92%）

---

## 5. SLOs-Serve (Arxiv 2025)

### 模型配置

| Target Model | Draft Model | GPU 配置 |
|---|---|---|
| OPT-7B | OPT-125m | 单卡 A100 40GB |
| OPT-13B (TP=2) | OPT-125m | 2×A100 40GB |
| OPT-30B (TP=4) | OPT-125m | 4×A100 40GB |
| ToolLlama-7B | — | 单卡 |
| DeepSeek-R1-Qwen-1.5B | — | 单卡 |

Draft model (OPT-125m) 复制到每张 GPU。

### 数据集

5 个数据集，6 个场景：

| 场景 | 数据集 | Prompt (avg/P99/std) | Output (avg/P99/std) |
|---|---|---|---|
| ChatBot | **ShareGPT** | 763/1591/424 | 266/619/160 |
| Coder | **HumanEval** | 847/2010/617 | 26/232/47 |
| Summarizer | **Arxiv Summary** | 1333/1946/444 | 202/1508/234 |
| Mixed | ChatBot+Coder+Summarizer 混合 | — | — |
| ToolLLM | **ToolBench** | 690/2131/356 | 116/363/66 |
| Reasoning | **s1K** | 127/421/83 | thinking 4693, response 803 |

### 负载生成

- 使用 **Azure LLM inference traces**：
  - Coding trace：突发性强，峰值约 30 req/s
  - Chatting trace：相对稳定，峰值约 15 req/s
- 通过缩放请求率测试不同负载水平

### SLO 定义

**多阶段 SLO**，同时约束 TTFT 和 TPOT：

| SLO 级别 | Max TTFT Slowdown (vs 零负载) | Max TPOT |
|---|---|---|
| Tight | 3× | 50ms |
| Loose | 5× | 100ms |

不同应用对不同阶段有不同松紧：
- **Summarizer**: prefill tight, decode loose
- **Coder**: prefill loose, decode tight
- **ChatBot**: 都 loose
- **ToolLLM**: prefill tight, tool交互 tight, response loose
- **Reasoning**: prefill tight, thinking decode tight, response decode loose

**请求必须每个阶段的 SLO 都满足才算达标。**

### Baselines

1. **vLLM** — prefill-oriented
2. **vLLM (Spec)** — vLLM + 投机解码
3. **Sarathi-Serve** — decode-oriented, chunked prefill
4. **DistServe** — prefill/decode 分离部署，测了 1:1/2:1/1:2 设备比

### 硬件

- Google a2-highgpu-4g: 4×A100 40GB (主实验)
- Google a3-highgpu-8g: 8×H100 80GB (扩展性实验)

### 核心指标

- **Serving Capacity**（核心指标）：SLO violation < 10% 下的最大 per-GPU req/s
- SLO violation rate
- p99 TTFT / p99 TPOT
- Normalized capacity（跨场景比较）
- Scheduling overhead（大部分 < 10ms）
- 还测了 2% violation 上限的严格场景

---

## 汇总对比

### 一、模型选择

| 论文 | 用了 Speculative Decoding? | Target Models | Draft Models |
|---|---|---|---|
| AdaServe | 是 | Llama3.1-70B, Qwen2.5-32B | LLaMA-3.2-1B, Qwen2.5-0.5B |
| TurboSpec | 是 | Vicuna-7B/13B/33B, LLaMA3-70B | Llama-160M, Llama3.2-1B, PLD |
| AdaSpec | 是 | Vicuna-7B/33B, Llama-3.3-70B | Vicuna-68M/160M, Llama-3.2-1B |
| SOLA | **否** | Llama3-8B/70B, Qwen1.5-14B/72B, Vicuna-7B | — |
| SLOs-Serve | 是（部分场景） | OPT-7B/13B/30B, ToolLlama-7B, DeepSeek-R1 | OPT-125m |

**共同点**：都覆盖了 7B 和 70B 级别的模型。
**差异**：SOLA 纯调度不涉及 speculative decoding；SLOs-Serve 用了较老的 OPT 系列；AdaServe 和 AdaSpec 用了最新的 Llama3 系列。

### 二、数据集

| 论文 | ShareGPT | HumanEval | CNN/DM | Alpaca | SpecBench | LongBench | Arxiv Summary | ToolBench | Azure Trace |
|---|---|---|---|---|---|---|---|---|---|
| AdaServe | | ✓ | ✓ | ✓ | | | | | |
| TurboSpec | ✓ | | ✓ | | | | | | |
| AdaSpec | | | ✓(间接) | | ✓ | | | | |
| SOLA | ✓ | | | | | ✓ | | | |
| SLOs-Serve | ✓ | ✓ | | | | | ✓ | ✓ | ✓ |

**共同点**：ShareGPT 和 CNN/DailyMail 是最常用的数据集；ChatBot/Summarization 是标配场景。
**差异**：SLOs-Serve 场景最丰富（6个），包括 ToolLLM 和 Reasoning 这种新兴场景；AdaSpec 用了 SpecBench 这个专门的投机解码 benchmark。

### 三、负载生成

| 论文 | 到达模式 | Trace 来源 | 持续时间 |
|---|---|---|---|
| AdaServe | 真实 trace 缩放 | 引用 [39] 的 trace | ~20 min |
| TurboSpec | Poisson 分布 | 合成 | 2 min |
| AdaSpec | 真实 trace | BurstGPT + Mooncake | 10 min |
| SOLA | Poisson 分布 | 合成 | 未明确 |
| SLOs-Serve | 真实 trace | **Azure LLM inference traces** | 按 trace 长度 |

**共同点**：都通过调节请求率来测试不同负载水平。
**差异**：AdaServe、AdaSpec、SLOs-Serve 用真实 trace（更现实）；TurboSpec 和 SOLA 用 Poisson（更可控）。

### 四、SLO 定义

| 论文 | SLO 指标 | SLO 设定方式 | 多级 SLO? |
|---|---|---|---|
| AdaServe | TPOT only | 按场景固定阈值 (1.2×baseline / 50ms / 150ms) | 是，3个类别不同 SLO |
| TurboSpec | **无 SLO** | — | — |
| AdaSpec | TPOT only | P90 TPOT × scale (0.8~1.2) | 否 |
| SOLA | TTFT + TPOT | 单请求延迟 × 倍数 (5×~20×) | 是，tight/loose 两级 |
| SLOs-Serve | TTFT + TPOT | TTFT: 零负载的 3×/5×; TPOT: 50ms/100ms | 是，per-phase SLO |

**共同点**：TPOT 是最普遍的 SLO 指标（4/5 篇都关注）。
**差异**：
- TurboSpec 完全不定义 SLO，只优化 goodput
- SOLA 和 SLOs-Serve 同时约束 TTFT 和 TPOT
- SLOs-Serve 最细粒度，支持 per-phase（prefill/decode/tool）的不同 SLO
- SLO 阈值设定没有统一标准——有的用 baseline 延迟的倍数，有的用绝对值，有的用 percentile

### 五、Baselines

| 论文 | vLLM | Sarathi-Serve | DistServe | 固定投机 SD | 其他 |
|---|---|---|---|---|---|
| AdaServe | ✓ | ✓ | | ✓ (k=4,6,8) | |
| TurboSpec | | | | ✓ (k=1,3,5,7) | No SD, PLD, Eagle |
| AdaSpec | | | | ✓ (k=1,3,5) | Autoregressive, Threshold SD, SpecInfer* |
| SOLA | ✓ (D+S) | | | | SJF |
| SLOs-Serve | ✓ | ✓ | ✓ | ✓ (vLLM Spec) | |

**共同点**：vLLM 是最常见的 baseline（4/5 篇）；固定投机长度的 SD 也是标配对比。
**差异**：SLOs-Serve baseline 最全面（4个系统）；SOLA 因为不涉及 SD 所以只比较调度策略。

### 六、硬件

| 论文 | 主要 GPU | 数量 |
|---|---|---|
| AdaServe | A100 80GB | 4 |
| TurboSpec | H100 80GB + L40S 48GB | 1~4 |
| AdaSpec | A100 40GB + L40 48GB | 1~8 |
| SOLA | A100 80GB | 8 |
| SLOs-Serve | A100 40GB + H100 80GB | 4~8 |

**共同点**：A100 是最常用的 GPU（5/5 篇都用了）。
**差异**：TurboSpec 额外测了 L40S；SLOs-Serve 做了 H100 的扩展性实验。

### 七、核心指标

| 论文 | SLO Attainment | Goodput | Latency | Throughput | Capacity |
|---|---|---|---|---|---|
| AdaServe | ✓ (主) | ✓ (主) | | | |
| TurboSpec | | ✓ (主) | ✓ avg + CDF | ✓ | |
| AdaSpec | ✓ | | ✓ speedup (主) | | |
| SOLA | ✓ (主) | ✓ | | | |
| SLOs-Serve | ✓ | | ✓ p99 | | ✓ (主) |

**共同点**：SLO attainment 是最核心的评估维度（4/5 篇）。
**差异**：TurboSpec 以 goodput 为核心（不关心 SLO）；SLOs-Serve 引入了 serving capacity 的概念（SLO 约束下的最大吞吐）。

### 八、关键差异总结

1. **方法定位不同**：AdaServe/TurboSpec/AdaSpec 做的是 speculative decoding 的自适应优化；SOLA 做的是纯调度优化；SLOs-Serve 做的是系统级资源编排（调度+SD+分离部署）
2. **SLO 粒度不同**：从"不关心SLO"(TurboSpec) → "单一TPOT"(AdaServe/AdaSpec) → "TTFT+TPOT"(SOLA) → "per-phase多阶段SLO"(SLOs-Serve)，粒度逐渐精细
3. **实验规模差异**：SLOs-Serve 场景最多（6个）、数据集最多（5个）；AdaSpec 请求数最少（480条）；SOLA 请求数最多（2000条）
4. **实现基础**：4/5 篇基于 vLLM，只有 AdaServe 基于 FlexFlow Serve
5. **模型选择代差**：SLOs-Serve 用了较老的 OPT 系列，其他都用了 Llama/Vicuna/Qwen 等较新模型
