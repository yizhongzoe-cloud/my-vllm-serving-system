# Step 5 Results — End-to-End on LongBench (8B / A6000)

**Date**: 2026-04-26
**Hardware**: 2× NVIDIA RTX A6000 (48GB), dp=2
**Model**: meta-llama/Llama-3.1-8B-Instruct, max_model_len=32768
**Workloads**: LongBench narrativeqa (W6) + qmsum/musique mix (W8)
**Cells**: 48 (4 baselines × 2 workloads × 1 load × 2 faults × 3 seeds)
**Wall time**: 298.6 minutes (4.97 h), 0 failed runs

---

## TL;DR

**Step 5 的设计目的是验证「长上下文下 V2（Benders + 自适应 checkpoint）能赢 NR」。结论是：没赢，反而完败。**

- **V2-full** 在 12 个对比场景里**全输 NR**（per-seed 9/9 全输），goodput 落后 19%–82%；W8/F2 故障下完成率崩到 32%-39%（3 个 seed 中 2 个）。
- **V2-NoCkpt**（去掉 checkpoint，只剩 Benders solver）和 NR 持平（差距 ≤5%，落噪声内）。
- **Periodic-Low** 全场崩盘，无可救药。
- 智能 checkpoint 机制在 8B/A6000/长上下文下**仍然不成立**——和导师讨论里 Mode 5 的判断一致，长上下文没有改变本质。

**根本原因不是 re-prefill 不够贵**（Step 2/4 测过 21–83× 比例），而是 **V2 的 checkpoint 复制在 decode 期间和正常生成抢资源**（V2-full 的 TPOT p95 从 NR 的 50ms 飙到 1700–3500ms）。长上下文反而让 KV checkpoint 更大（每请求 ~4GB），干扰更严重。

---

## 1. 配置

### 1.1 工作负载

| Workload | 数据集 | 池子大小 | Prompt P50 | Prompt P95 |
|---|---|---|---|---|
| W6_NarrativeQA | LongBench narrativeqa（小说/电影脚本 QA） | 85 | 12K | 23K |
| W8_LongMix | LongBench qmsum + musique 混合 | 391 | 15.8K | 18.4K |

### 1.2 基线

| Baseline | Routing | Checkpoint | 角色 |
|---|---|---|---|
| **NoFT-Reprefill (NR)** | fault_tolerant | 关 | 故障时 re-prefill 重启 — 对照 |
| **Periodic-Low** | fault_tolerant | 每 10 blocks 周期 | 周期 checkpoint 基线 |
| **V2-NoCkpt** | Benders centralized | **关** | 仅 Benders solver |
| **V2-full** | Benders centralized | **开（自适应）** | 完整系统 |

### 1.3 负载与故障

- Load: Moderate (rps=0.4)
- Fault: none / F2_Mid（150s 注入故障）
- Run duration: 300s (warmup 30s)
- Seeds: [42, 123, 456]
- SLO: TTFT 12s, TPOT 200ms, failover gap 12s（长上下文设定，比短 prompt 宽松）

---

## 2. Goodput 总览

![Goodput](../figures/8B/Step5_Main/goodput_2x2.png)

| Workload | Fault | NR | PerLow | V2-NoCkpt | V2-full | V2 vs NR |
|---|---|---|---|---|---|---|
| W6 | none | **6.8** | 3.7 | 6.8 | 2.6 | **−61%** |
| W6 | F2_Mid | **6.8** | 2.2 | 6.8 | 1.9 | **−72%** |
| W8 | none | **17.1** | 3.3 | 16.3 | 10.8 | **−37%** |
| W8 | F2_Mid | **6.9** | 0.5 | 6.6 | 4.1 | **−40%** |

**观察**：
- NR 在所有场景都最高
- V2-NoCkpt 和 NR 几乎贴平（差距 ≤2%）
- V2-full 落后 37%–72%
- Periodic-Low 全部场景都垫底，特别是 W8/F2 几乎 0 goodput

---

## 3. Per-seed 配对：每个 seed V2 都输

![Paired seeds](../figures/8B/Step5_Main/paired_seeds.png)

每条线连接同一 seed 下 NR 和 V2-full 的 goodput。**所有 12 个 (workload × fault × seed) 组合里 V2 全部低于 NR**——不是抖动，是稳定输。

| Workload | Fault | seed=42 | seed=123 | seed=456 |
|---|---|---|---|---|
| W6 | none | 6.9 → 2.5 (−63%) | 6.8 → 2.5 (−63%) | 6.6 → 2.6 (−60%) |
| W6 | F2_Mid | 6.9 → 1.2 (−82%) | 6.8 → 1.6 (−77%) | 6.7 → 2.7 (−59%) |
| W8 | none | 18.0 → 10.8 (−40%) | 17.3 → 11.4 (−34%) | 15.8 → 10.1 (−36%) |
| W8 | F2_Mid | 6.5 → 2.2 (−67%) | 7.6 → 6.1 (−20%) | 6.6 → 3.9 (−42%) |

---

## 4. 完成率：V2-full 在故障下大量丢请求

![Completion rate](../figures/8B/Step5_Main/completion_rate.png)

W8_LongMix / F2_Mid 下：

| Baseline | seed=42 | seed=123 | seed=456 | mean |
|---|---|---|---|---|
| NR | 100% | 100% | 100% | 100% |
| V2-NoCkpt | 99.2% | 100% | 100% | 99.7% |
| **V2-full** | **32.3%** | **39.2%** | 99.2% | **56.9%** |
| Periodic-Low | — | — | — | 60.2% |

V2-full 在 3 个 seed 里 2 个崩到 32%-39%——意味着故障注入后系统恢复不过来，大量 in-flight 请求超时丢弃。第 3 个 seed 居然恢复到 99.2%，说明**行为高度不稳定**。

---

## 5. TPOT p95：V2-full 的"罪证"

![TPOT p95](../figures/8B/Step5_Main/tpot_p95.png)

| Workload | Fault | NR | V2-NoCkpt | V2-full |
|---|---|---|---|---|
| W6 | none | 48 ms | 29 ms | **1706 ms** |
| W6 | F2_Mid | 50 ms | 34 ms | **3489 ms** |
| W8 | none | 403 ms | 415 ms | **575 ms** |
| W8 | F2_Mid | 461 ms | 462 ms | **595 ms** |

W6 上 V2-full 的 TPOT 比 NR 慢 **35–70 倍**（绿色虚线是 SLO=200ms）——这是 **V2-full 失败的直接证据**：

> checkpoint 机制在 decode 期间复制 KV，和正常 token 生成抢 GPU 计算+PCIe 带宽，让每 token 间隔从 50ms 拉到 1700ms。

V2-NoCkpt 关掉 checkpoint 后立刻和 NR 持平——证明问题在 checkpoint 持续运行的开销，不在 Benders solver。

---

## 6. SLO 违规率

![SLO violation](../figures/8B/Step5_Main/slo_violation.png)

V2-full 在 W6 上 SLO 违规率 60%–73%（即便 SLO 已经放宽到 12s/200ms），原因就是上面 TPOT 飙到秒级——大量请求 TPOT 超过 200ms 阈值。

---

## 7. 诊断：为什么长上下文也救不了

Step 5 的初衷源于 Step 4 的测量：长上下文下 re-prefill 比 reload 贵 21–83 倍，理论上 checkpoint 应该赢。**这个比例没错，但前提是 checkpoint 能干干净净地恢复**。问题是：

1. **V2 的 checkpoint 是连续运行的**，每出现一个新稳定 KV block 就触发判断+复制，**整个 run 期间都在抢资源**。即使没故障也付出 60% 以上的 goodput 代价。
2. **长上下文让 KV 单次复制变大**：32K context × 8B 模型 ≈ 4GB KV/请求，复制一次更慢，对 decode 的干扰窗口更长。
3. **PCIe 带宽是双向共享的**：vLLM 的 prefetch、output 回传、checkpoint 复制都过 PCIe。长上下文 prefill 阶段本就对 PCIe 重压，再叠加 checkpoint 复制就崩了。
4. **故障下 V2-full 反而更糟**：本来应该靠 reload 节省时间，但 reload 期间累积的延迟把 in-flight 请求全推过 SLO，结果丢得比 NR 直接 re-prefill 还多。

之前导师讨论里识别的 **Mode 4（系统级争用）+ Mode 5（价值不成立）依然成立，长上下文不仅没改变本质，反而放大了问题**。

---

## 8. 选项

| 方向 | 思路 | 代价 | 概率 |
|---|---|---|---|
| **A. 等 L40S 结果** | L40S KV 带宽快 16%（20.4 vs 17.6 GB/s） | 几小时 | 不会改变结论 |
| **B. 异步 checkpoint** | 让 checkpoint 复制脱离 critical path（独立线程、独立 PCIe channel） | 大改架构（数周） | 中等 |
| **C. 接受 V2-NoCkpt 路线** | 丢掉 checkpoint，纯 Benders + admission control | 故事好讲度低（V2-NoCkpt 也只是和 NR 持平） | 论文难发 |
| **D. 换设定** | 70B / 多节点 TP / PCIe 慢速链路，让 re-prefill 真的秒级到分钟级 | 需要新硬件 | 较高 |

**建议**：今天把这个数据带去和导师讨论方向。Step 5 的主要价值是**严格证伪**了「长上下文能救 V2」这个假设——这本身是有用的负面结果，但不是论文卖点。

---

## 附：数据原始路径

- 实验结果：`results_v2/8B/Step5_Main/<baseline>/<workload>/Moderate/<fault>/<seed>/`
  - `metrics.json`：聚合指标
  - `requests.csv`：每请求记录
  - `server.log`：vLLM 服务端日志
- 图：`experiments_v2/figures/8B/Step5_Main/`
- Config：`experiments_v2/config_8b_step5.yaml`
