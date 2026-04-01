# 完整实验说明

Config: `experiments_v2/config_1b_full.yaml`（1B 开发验证）
正式论文用 `config_8b.yaml`（主力）+ `config_70b.yaml`（验证）

---

## 硬件 & 模型

| 项目 | 1B (开发) | 8B (论文主力) | 70B (论文验证) |
|---|---|---|---|
| 模型 | Llama-3.2-1B | Llama-3.1-8B | Llama-3.1-70B |
| dp_size | 2 | 4 | 2 |
| GPU | 2×A100 | 4×A100 | 8×A100 |
| max_model_len | 2048 | 4096 | 4096 |
| checkpoint pool | 8GB | 32GB | 128GB |

---

## 数据集

| 数据集 | 条数 | Prompt (mean±std) | Output (mean±std) |
|---|---|---|---|
| ShareGPT | 5000 | 1259±953 | 287±174 |
| CNN/DailyMail | 3000 | 874±448 | 67±27 |
| Alpaca | 50154 | 17±13 | 57±55 |

---

## 4 个 Workload

| ID | 数据集 | TPOT SLO | 特点 |
|---|---|---|---|
| W1_Chat | ShareGPT | 100ms | decode-heavy，大量 KV 要 checkpoint |
| W2_Summary | CNN/DailyMail | 200ms (宽松) | prefill-heavy，故障时要重做 prefill |
| W3_Instruct | Alpaca | 50ms (严格) | 短请求严格 SLO，考验恢复速度 |
| W4_Mixed | 50% W1 + 20% W3 + 30% W2 | per-request | 测 Benders 差异化调度 |

---

## 6 个 Baseline

| Baseline | 调度 | Checkpoint | 代表什么 |
|---|---|---|---|
| No-FT | FCFS | 不存 | 下界：故障全丢 |
| Periodic-Low | Greedy | 每 10 blocks | 保守 checkpoint（低开销，慢恢复） |
| Periodic-High | Greedy | 每 1 block | 激进 checkpoint（高开销，快恢复） |
| Benders-Only | Benders | 固定每 10 blocks | 消融：只有智能调度 |
| Adaptive-Only | Greedy | 自适应 | 消融：只有智能 checkpoint |
| **Our-System** | **Benders** | **自适应** | **完整系统** |

---

## 3 个负载级别

| 级别 | 单 GPU 利用率 | 1B RPS (手动) |
|---|---|---|
| Light | 25% | 1.5 |
| Moderate | 40% | 2.5 |
| Heavy | 55% | 3.5 |

8B/70B 需要先跑 `calibrate.py` 自动校准。

---

## 4 个故障时机

| ID | 时间 | 说明 |
|---|---|---|
| none | — | 无故障，测正常运行开销 |
| F1_Early | 60s | 刚进稳态 |
| F2_Mid | 150s | 稳态中期（主实验） |
| F3_Late | 240s | 稳态后期，checkpoint pool 压力最大 |

---

## SLO

| SLO | 值 | 来源 |
|---|---|---|
| TTFT | 2000ms | 5× TTFT_base |
| TPOT | per-workload: 50/100/200ms | 按应用场景 |
| Failover Gap | 3000ms | 3× Gap_base |

E6 额外测 3 档松紧：Tight (3×/1.5×)、Moderate (5×/3×)、Loose (10×/5×)

---

## 7 个实验

### E0_Smoke — 跑通测试
- 不写论文，纯验证
- 2 baselines (No-FT, Our-System) × 1 workload (W1_Chat) × 1 load (Moderate) × 2 faults (none, F2_Mid) = **4 runs/seed**

---

### E1a_Main — 端到端主结果

**论文核心实验**，产出 Figure 1-4。

**问题**：Our-System 在所有条件下 goodput 最高、SLO violation 最低吗？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | No-FT, Periodic-Low, Periodic-High, Benders-Only, Adaptive-Only, Our-System | 6 |
| Workloads | W1_Chat (ShareGPT), W2_Summary (CNN/DM), W3_Instruct (Alpaca), W4_Mixed | 4 |
| Loads | Light (1.5 rps), Moderate (2.5 rps), Heavy (3.5 rps) | 3 |
| Faults | none, F1_Early (60s), F2_Mid (150s), F3_Late (240s) | 4 |
| Seeds | 42, 123, 456 | 3 |
| **Total** | 6×4×3×4×3 | **864** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 1 | 折线图 (1×4) | Goodput vs Load，无故障，每个 workload 一列 |
| Fig 2 | 折线图 (1×4) | Goodput vs Load，F2_Mid 故障 |
| Fig 3 | 柱状图 (2×4) | SLO violation rate，无故障/有故障 × 4 workloads |
| Fig 4 | 柱状图 (1×2) | Failover gap P95，按故障时机分组 |

**预期趋势**：
- 无故障：No-FT ≈ Our-System > Periodic-Low > Periodic-High（checkpoint 开销从低到高）
- 有故障：Our-System > Periodic-High > Periodic-Low > No-FT（恢复能力从强到弱）

---

### E2_Recovery — 恢复时间分解

**产出 Figure 5-6**（堆叠柱状图、CDF）

**问题**：GPU 挂了后恢复时间花在哪？检测 / 加载 checkpoint / replay 各占多少？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | No-FT, Periodic-Low, Periodic-High, Our-System | 4 |
| Workloads | W1_Chat, W2_Summary | 2 |
| Loads | Moderate | 1 |
| Faults | F2_Mid | 1 |
| **Total/seed** | | **8** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 5 | 堆叠柱状图 | 恢复时间分解：T_detect + T_restore + T_replay |
| Fig 6 | CDF 图 | 单请求恢复 gap 的累积分布 |

**预期趋势**：
- Periodic-Low：T_restore 小但 T_replay 大（存的少，要重跑的多）
- Periodic-High：T_restore 大但 T_replay 小（存的多，重跑少）
- Our-System：两者之间，自适应平衡，总恢复时间最短

---

### E3_Ablation — 消融实验

**产出 Figure 7-8**（柱状图、热力图）

**问题**：Benders 调度和自适应 checkpoint 各贡献多少？联合优化是否 > 单独任一？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | Periodic-Low, Periodic-High, Benders-Only, Adaptive-Only, Our-System | 5 |
| Workloads | W1_Chat, W4_Mixed | 2 |
| Loads | Moderate, Heavy | 2 |
| Faults | none, F2_Mid | 2 |
| **Total/seed** | | **40** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 7 | 分组柱状图 (2×2) | Goodput 对比，行=fault，列=workload |
| Fig 8 | 热力图 | SLO violation，行=baseline，列=条件组合 |

**预期趋势**：
- Benders-Only > Periodic（智能调度有帮助）
- Adaptive-Only > Periodic（智能 checkpoint 有帮助）
- **Our-System > Benders-Only 且 > Adaptive-Only**（联合优化超加性）

---

### E4_Checkpoint_Tradeoff — 开销 vs 收益

**产出 Figure 9-10**（散点图、时间线）

**问题**：存 checkpoint 的正常运行开销 vs 故障时省的恢复时间，值不值？自适应策略是不是 sweet spot？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | No-FT, Periodic-Low, Periodic-High, Our-System | 4 |
| Workloads | W1_Chat | 1 |
| Loads | Moderate, Heavy | 2 |
| Faults | none, F2_Mid | 2 |
| **Total/seed** | | **16** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 9 | 散点图 | X=正常运行开销%, Y=故障恢复收益%，每个 baseline 一个点 |
| Fig 10 | 时间线 | 故障前后 goodput 变化（5s 滑动窗口） |

**预期趋势**：
- 散点图上 Periodic-High 在左上（高开销高收益），Periodic-Low 在右下（低开销低收益）
- Our-System 在**右上角 sweet spot**（低开销 + 高收益）

---

### E5_Controller — Solver 开销

**产出 Figure 11**（柱状图）

**问题**：Benders 求解器每轮算多久？会不会拖慢系统？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | Our-System | 1 |
| Workloads | W1_Chat, W2_Summary, W3_Instruct, W4_Mixed | 4 |
| Loads | Light, Moderate, Heavy | 3 |
| Faults | none | 1 |
| **Total/seed** | | **12** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 11 | 柱状图 | 每 epoch 平均 solve time + overhead 占比，按 workload × load 分组 |

**预期**：solver 开销 < 5% of decision epoch time

---

### E6_SLO_Sensitivity — SLO 敏感度

**产出 Figure 13-14**（柱状图、折线图）

**问题**：SLO 设松/设紧，系统表现怎么变？优势在什么区间最大？

**矩阵**：
| 维度 | 值 | 数量 |
|---|---|---|
| Baselines | No-FT, Our-System | 2 |
| Workloads | W1_Chat, W4_Mixed | 2 |
| Loads | Moderate | 1 |
| Faults | F2_Mid | 1 |
| SLO scales | Tight (3×/1.5×), Moderate (5×/3×), Loose (10×/5×) | 3 |
| **Total/seed** | | **12** |

**产出图表**：
| Figure | 类型 | 内容 |
|---|---|---|
| Fig 13 | 分组柱状图 | Goodput + SLO violation，按 SLO 松紧分组 |
| Fig 14 | 折线图 | Gap SLO 倍数 vs 恢复请求达标比例 |

**预期**：
- Tight SLO：两者都差，但 Our-System 仍优于 No-FT
- **Moderate SLO：Our-System 优势最大**
- Loose SLO：No-FT 也凑合过，差距缩小

---

## 规模汇总

| 实验 | Runs/seed | ×3 seeds | 预估时间(1B) |
|---|---|---|---|
| E0_Smoke | 4 | 12 | 1.5h |
| E1a_Main | 288 | 864 | 5 天 |
| E2_Recovery | 8 | 24 | 3h |
| E3_Ablation | 40 | 120 | 16h |
| E4_Checkpoint | 16 | 48 | 6h |
| E5_Controller | 12 | 36 | 5h |
| E6_SLO | 12 | 36 | 5h |
| **合计** | **380** | **1140** | **~8 天** |

建议：先跑 1 seed 看趋势（380 runs ≈ 2.5 天），确认合理后再补 2 个 seed。

---

## 执行命令

```bash
# 逐个跑（推荐）
python experiments_v2/suite.py --config experiments_v2/config_1b_full.yaml --experiment E0_Smoke --port 8300
python experiments_v2/suite.py --config experiments_v2/config_1b_full.yaml --experiment E1a_Main --resume --port 8300
python experiments_v2/suite.py --config experiments_v2/config_1b_full.yaml --experiment E2_Recovery --resume --port 8300
# ...

# 分析
python experiments_v2/analyze.py results_v2/1B/E1a_Main --output experiments_v2/figures/E1a_Main
```

---

## 输出文件

每个 run 产出：
```
results_v2/1B/{实验名}/{baseline}/{workload}/{load}/{fault}/{seed}/
├── metrics.json       # 聚合指标（goodput, TTFT, TPOT, SLO violation...）
├── requests.csv       # 每条请求详情（30+ 字段）
├── epochs.csv         # Benders solver 每轮数据
├── recoveries.json    # 恢复事件时间线
├── run_meta.json      # 实验元数据
└── server.log         # vLLM server 日志
```

图表输出到 `experiments_v2/figures/{实验名}/`

---

## Tiny 实验诊断 (E_Tiny, 12 runs)

### 结果

| Baseline | 无故障 Goodput | 有故障 Goodput | 完成率(无故障) | 完成率(有故障) | 恢复率 |
|---|---|---|---|---|---|
| No-FT | 258.2 | 253.6 | 95.7% | 94.8% | 0% |
| Periodic-Low | 246.8 | 246.0 | 95.7% | 95.7% | 100% |
| Periodic-High | 249.8 | 247.4 | 95.7% | 95.7% | 100% |
| Adaptive-Only | 247.7 | 251.2 | 95.7% | 95.7% | 100% |
| Benders-Only | 213.7 | 209.0 | **66.1%** | **79.1%** | 100% |
| Our-System | 217.8 | 198.5 | **64.3%** | **75.7%** | 100% |

### 发现的问题：Benders solver 拒绝 33% 请求

**现象**：Benders-Only 和 Our-System 的完成率只有 64-79%，远低于其他 baseline 的 95%。36/110 请求被标记为 `empty_stream`（admitted 到 server 但 solver reject 了，0 output tokens）。

**根因**：solver 的吞吐参数和实际性能严重不匹配。

| 参数 | Config 原值 | 实际值 | 偏差 | 后果 |
|---|---|---|---|---|
| `ft_prefill_throughput` | 4,000 tok/s | ~45,000 tok/s | **低 11 倍** | Solver 高估 replay 开销 → 认为恢复不可行 → 拒绝 |
| `ft_decode_throughput` | 2,000 tok/s | ~143 tok/s | 高 14 倍 | Solver 低估 decode 时间（反而让 admission 更宽松） |

Solver 计算 recovery replay 时间：`T_replay = 143 tokens / 4000 tok/s = 35.8ms`。实际只要 `143 / 45000 = 3.2ms`。Solver 觉得恢复太贵 → 拒绝请求。

### 调参修复

```yaml
# 修改前（导致 33% 拒绝）
ft_prefill_throughput: 4000.0
ft_decode_throughput: 2000.0
ft_planning_horizon: 1.0

# 修改后
ft_prefill_throughput: 40000.0   # 接近实测 ~45000
ft_decode_throughput: 150.0      # 接近实测 ~143
ft_planning_horizon: 0.5        # 缩短前瞻窗口，减少保守性
```

已同步更新所有 1B config 文件。

### 其他确认

- **No-FT + 故障不会崩**：109/115 完成（94.8%），只丢故障 GPU 的请求，server 存活。~~问题 #2 已解决~~。
- **HTTP 400 (5 个)**：所有 baseline 都有，是 prompt 超 max_model_len 被 server 拒绝，不是 FT 问题。
- **SLO violation 几乎为 0**：TPOT 实际 7-25ms，SLO 最严 50ms，太松。Heavy 负载 + 300s 下可能开始有 violation。

### 对今晚实验的影响

调参后 Benders 系列应该不再过度拒绝。但 8B/70B 模型的吞吐参数完全不同，**切换模型时必须重新测量并调整**。建议：
- 8B: 先跑 calibrate.py 或手动 benchmark 一次 prefill/decode throughput
- 把 ft_prefill_throughput / ft_decode_throughput 加入 calibrate.py 的自动校准流程

---

## 已知问题 & 注意事项

| # | 严重度 | 问题 | 状态 |
|---|---|---|---|
| 1 | 中 | F3_Late (240s) 只剩 60s 数据 | 今晚实验已去掉 F3_Late |
| 2 | ~~高~~ | ~~No-FT + 故障时 server 崩溃~~ | **已验证不会崩**（94.8% 完成率） |
| 3 | 低 | dp=2 + Heavy → 故障后过载 | 1B 开发接受，8B (dp=4) 没这问题 |
| 4 | 低 | 跨实验重复 runs | 不影响正确性 |
| 5 | **已修** | Solver 参数不匹配导致 33% 拒绝 | 调参：prefill 4000→40000, decode 2000→150, horizon 1.0→0.5 |
| 6 | 低 | SLO 可能太松（violation 全 0） | 先跑看 Heavy 负载下是否有 violation |
| 5 | 低 | E5 没测故障时 solver 开销 | 只测 fault=none。故障时 recovery_checker 可能让 solver 更慢，可选加 F2_Mid |
