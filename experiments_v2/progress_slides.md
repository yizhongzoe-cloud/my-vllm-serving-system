# Fault-Tolerant Multi-GPU LLM Serving with Adaptive KV-Cache Checkpointing

## Progress Update — Yi Zhong

---

## Slide 1: 上次 → 这次

### 上次（v1 实验框架）
- 1B 模型 + 合成数据（模板文本拼接，均匀长度分布）
- 3 个简单 workload（W1 Short / W2 Long / W3 Bursty）
- 负载固定 1/3/5 rps，SLO 手动设定
- 70s 运行时长，1 个 seed

### 这次（v2 实验框架）
- **真实数据集**：ShareGPT (5000条) / CNN/DailyMail (3000条) / Alpaca (50k条)
- **4 个 workload**：W1_Chat / W2_Summary / W3_Instruct / **W4_Mixed**（per-request SLO）
- **负载按饱和百分比**：Light 25% / Moderate 40% / Heavy 55%（calibrate.py 自动校准）
- 300s 运行时长，3 seeds，6 baselines
- **新增 baseline**：Benders-Only / Adaptive-Only（消融用）
- **发现并修复 solver 参数不匹配问题**（拒绝率从 33% → 正常）

---

## Slide 2: 实验框架 v2

### 数据集（真实分布，重尾）
| 数据集 | 条数 | Prompt (mean±std) | Output (mean±std) |
|---|---|---|---|
| ShareGPT | 5000 | 1259±953 | 287±174 |
| CNN/DailyMail | 3000 | 874±448 | 67±27 |
| Alpaca | 50154 | 17±13 | 57±55 |

### Mixed Workload（W4_Mixed）
| 子 workload | 数据集 | 占比 | TPOT SLO |
|---|---|---|---|
| W1_Chat | ShareGPT | 50% | 100ms |
| W3_Instruct | Alpaca | 20% | 50ms (严格) |
| W2_Summary | CNN/DailyMail | 30% | 200ms (宽松) |

每个请求的 TPOT SLO 不同 → Benders solver 可以差异化调度

---

## Slide 3: 6 个 Baselines

| Baseline | 调度 | Checkpoint | 角色 |
|---|---|---|---|
| No-FT | FCFS | 不存 | 下界 |
| Periodic-Low | Greedy | 每 10 blocks | 保守 ckpt |
| Periodic-High | Greedy | 每 1 block | 激进 ckpt |
| **Benders-Only** | Benders | 固定 10 blocks | **消融：只有智能调度** |
| **Adaptive-Only** | Greedy | 自适应 | **消融：只有智能 ckpt** |
| **Our-System** | **Benders** | **自适应** | **完整系统** |

新增 Benders-Only 和 Adaptive-Only 用于消融实验，分离两个组件各自的贡献。

---

## Slide 4: Tiny 实验结果（E_Tiny, 12 runs）

设置：1B 模型, dp=2, W4_Mixed, Moderate (2 rps), 60s, seed=42

| Baseline | 无故障 Goodput | 有故障 Goodput | 完成率(无故障) | 恢复率 |
|---|---|---|---|---|
| No-FT | 258.2 | 253.6 | 95.7% | 0% |
| Periodic-Low | 246.8 | 246.0 | 95.7% | 100% |
| Periodic-High | 249.8 | 247.4 | 95.7% | 100% |
| Adaptive-Only | 247.7 | 251.2 | 95.7% | 100% |
| Benders-Only | 213.7 | 209.0 | **66.1%** | 100% |
| Our-System | 217.8 | 198.5 | **64.3%** | 100% |

**问题**：Benders 系列完成率只有 64-66%，远低于其他 95%。

---

## Slide 5: 根因分析 — Solver 参数不匹配

Solver 计算恢复时间时使用的吞吐参数和实际值差了 10 倍：

| 参数 | Config 原值 | 实际值 | 偏差 |
|---|---|---|---|
| ft_prefill_throughput | 4,000 tok/s | ~45,000 tok/s | **低 11 倍** |
| ft_decode_throughput | 2,000 tok/s | ~143 tok/s | 高 14 倍|

Solver 估算 replay 143 tokens 要 35.8ms（实际 3.2ms）→ 认为恢复不可行 → 拒绝 33% 请求。

### 修复
```
ft_prefill_throughput: 4000 → 40000
ft_decode_throughput:  2000 → 150
ft_planning_horizon:   1.0  → 0.5
```

**教训**：solver 参数必须从实测数据校准，不能用默认值。8B/70B 切换时需重新校准。

---

## Slide 6: Tiny 其他确认

### No-FT + 故障不会崩溃
- 109/115 完成（94.8%），只丢故障 GPU 上的请求
- Server 存活，存活 engine 继续正常服务

### force_output_len 生效
- 85/85 成功请求的 output_tokens == expected_output_len
- 保证 KV cache 积累量与数据集一致

### SLO violation 几乎为 0
- 实际 TPOT 7-25ms，SLO 最严 50ms
- 需要在 Heavy 负载 + 300s 下观察是否出现 violation

---

## Slide 7: 今晚跑的实验（E1a_Tonight）

### 设置
- 6 baselines × W4_Mixed × 3 loads (Light/Moderate/Heavy) × 3 faults (none/F1/F2) × 1 seed
- **54 runs，预计 ~7 小时**
- 已用调参后的 solver 参数

### 产出
- Goodput vs Load 折线图（无故障 / 有故障）
- SLO Violation 柱状图
- Failover Gap 对比
- 调参后 Benders 系列是否正常工作

### 预期
- 调参后 Benders-Only / Our-System 完成率应接近其他 baseline
- Our-System 在有故障时 goodput 最高
- Heavy 负载下开始出现 baseline 分化

---

## Slide 8: 完整实验计划

| 实验 | 目的 | Runs/seed | 产出 |
|---|---|---|---|
| E1a_Main | 端到端全扫 | 288 | Fig 1-4: Goodput, SLO, Gap |
| E2_Recovery | 恢复时间分解 | 8 | Fig 5-6: 堆叠柱状图, CDF |
| E3_Ablation | 消融 | 40 | Fig 7-8: 各组件贡献 |
| E4_Checkpoint | 开销 vs 收益 | 16 | Fig 9-10: Tradeoff 散点图 |
| E5_Controller | Solver 开销 | 12 | Fig 11: Solver latency |
| E6_SLO | SLO 敏感度 | 12 | Fig 13-14: 敏感度曲线 |

### 三个模型
| 模型 | dp | GPU | 角色 |
|---|---|---|---|
| 1B | 2 | 2×A100 | 开发调试（当前） |
| **8B** | **4** | **4×A100** | **论文主力** |
| 70B | 2 | 8×A100 | 规模验证 |

---

## Slide 9: Next Steps

### 短期（本周）
1. 今晚 E1a_Tonight (54 runs) 跑完 → 分析调参效果
2. 如果 Benders 正常 → 跑完整 E1a (288 runs, 1 seed)
3. 跑 E3 消融实验 → 验证"联合优化 > 单独组件"

### 中期
4. 切到 **8B 模型**（论文主力）
   - 重新 profile checkpoint costs
   - 重新校准 solver 参数（ft_prefill/decode_throughput）
   - dp=4 → 故障丢 25% 容量，更现实的恢复场景
5. 补充 3 seeds → 报告 mean ± std

### 要解决的问题
- solver 参数如何自动校准（加入 calibrate.py）
- SLO 是否需要收紧（当前 violation 全 0）
- 1B dp=2 Heavy 负载故障后过载问题（8B dp=4 不存在）
