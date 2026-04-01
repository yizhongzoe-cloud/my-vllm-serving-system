# Tiny 实验设置

快速验证全流程 + 看 6 个 baseline 在有/无故障下的趋势差异。

---

## 实验矩阵

| 维度 | 值 | 数量 |
|---|---|---|
| 模型 | Llama-3.2-1B-Instruct (dp=2, 2×GPU) | 1 |
| Baselines | No-FT, Periodic-Low, Periodic-High, Benders-Only, Adaptive-Only, Our-System | 6 |
| Workloads | W4_Mixed (50% Chat + 20% Instruct + 30% Summary) | 1 |
| Load | Moderate (2 rps) | 1 |
| Faults | none, F2_Mid (30s) | 2 |
| Seeds | 42 | 1 |
| **Total runs** | 6 × 1 × 1 × 2 × 1 = **12** | |

---

## 运行参数

| 参数 | 值 |
|---|---|
| run_duration | 60s |
| warmup | 10s |
| fault 时机 (F2_Mid) | 30s |
| 每 run 请求数 | ~120 (2 rps × 60s) |
| request_timeout | 120s |
| force_output_len | true |
| checkpoint_pool | 8 GB |
| max_model_len | 2048 |

---

## SLO 设置

| SLO | 值 | 说明 |
|---|---|---|
| TTFT | 2000ms | 手动设定（未经 calibrate） |
| TPOT | per-request: 50 / 100 / 200ms | W3=50, W1=100, W2=200 |
| Failover Gap | 3000ms | 手动设定 |

---

## 6 个 Baseline

| Baseline | 调度 | Checkpoint | 测什么 |
|---|---|---|---|
| No-FT | FCFS | 不存 | 下界：故障全丢 |
| Periodic-Low | Greedy | 每 10 blocks | 保守 checkpoint |
| Periodic-High | Greedy | 每 1 block | 激进 checkpoint |
| Benders-Only | Benders | 每 10 blocks（固定） | 消融：智能调度 + 固定 checkpoint |
| Adaptive-Only | Greedy | 自适应 | 消融：贪心调度 + 智能 checkpoint |
| Our-System | Benders | 自适应 | 完整系统 |

---

## W4_Mixed 构成

| 子 workload | 数据集 | 占比 | TPOT SLO |
|---|---|---|---|
| W1_Chat | ShareGPT (5000条) | 50% | 100ms |
| W3_Instruct | Alpaca (50154条) | 20% | 50ms (严格) |
| W2_Summary | CNN/DailyMail (3000条) | 30% | 200ms (宽松) |

---

## 预期观察

### 无故障 (fault=none)
- No-FT goodput 最高（零开销）
- Periodic-High goodput 最低（checkpoint 开销大）
- Our-System 接近 No-FT（自适应：开销小）

### 有故障 (fault=F2_Mid)
- No-FT goodput 大幅下降（故障 GPU 上的请求全丢）
- Periodic-High 恢复快但起点低
- Our-System 恢复好 + 起点高 → 整体 goodput 最优

---

## 文件位置

- Config: `experiments_v2/config_1b_tiny.yaml`
- 结果: `results_v2/1B/E_Tiny/{baseline}/W4_Mixed/Moderate/{fault}/42/`
- 图表: `experiments_v2/figures/E_Tiny/`

---

## 预估时间

12 runs × ~3 min/run ≈ **36 分钟**
