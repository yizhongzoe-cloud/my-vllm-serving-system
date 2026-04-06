# 8B 实验结果

Bug 修复后（删掉 max_gpu_failures 钳制）的结果。

---

## E0_Smoke (4 runs)

| Baseline | Fault | Goodput | 完成率 | SLO Viol | TPOT P95 | 恢复率 |
|---|---|---|---|---|---|---|
| No-FT | none | 223.1 | 98.8% | 0.0% | 29.9ms | 100% |
| No-FT | F2_Mid | 216.0 | 96.8% | 0.0% | 37.8ms | 0% |
| Our-System | none | 174.8 | 98.4% | 18.9% | 80.3ms | 100% |
| Our-System | F2_Mid | 120.5 | 98.4% | 45.5% | 125.2ms | 100% |

### 分析

- Checkpoint 现在正常 publish 了（`max_gpu_failures=1`，不再被钳制）
- Our-System 完成率 98.4%（正常），恢复率 100%
- **但 SLO violation 仍然很高**（18-45%），TPOT P95 80-125ms 超过了 50ms SLO
- Goodput 差距：Our-System 174.8 vs No-FT 223.1（正常态差 22%）

---

## E2_Recovery (8 runs)

| Baseline | Workload | Fault | Goodput | 完成率 | SLO Viol | TPOT P95 | 恢复率 |
|---|---|---|---|---|---|---|---|
| No-FT | W1_Chat | F2_Mid | 216.0 | 96.8% | 0.0% | 37.8ms | 0% |
| No-FT | W2_Summary | F2_Mid | 55.0 | 99.6% | 0.0% | 31.3ms | 0% |
| Our-System | W1_Chat | F2_Mid | 118.9 | 98.8% | 44.9% | 126.2ms | 100% |
| Our-System | W2_Summary | F2_Mid | 47.4 | 98.8% | 11.0% | 98.9ms | 100% |
| **Periodic-High** | **W1_Chat** | **F2_Mid** | **0.0** | **0.0%** | — | — | — |
| **Periodic-High** | **W2_Summary** | **F2_Mid** | **0.0** | **0.0%** | — | — | — |
| **Periodic-Low** | **W1_Chat** | **F2_Mid** | **0.0** | **0.0%** | — | — | — |
| **Periodic-Low** | **W2_Summary** | **F2_Mid** | **0.0** | **0.0%** | — | — | — |

### 分析

**Periodic-Low/High 完成率 0%——全部请求被拒绝。**

根因：`fault_tolerant` 模式下 EngineCore 的 `dp_size=1`（只看到自己），但 `max_gpu_failures=1`。
`check_capacity_under_failures` 算存活 replica = dp_size - max_gpu_failures = 1 - 1 = 0。容量为 0，所有请求被拒。

这个问题**只影响 `fault_tolerant` 模式**（Periodic-Low/High/Adaptive-Only）。
`ft_benders_centralized` 模式（Our-System/Benders-Only）不受影响——它跳过了本地 admission。

---

## 待修 Bug

| Bug | 影响 | 根因 | 修法 |
|---|---|---|---|
| `fault_tolerant` 模式 EngineCore dp_size=1 | Periodic-Low/High/Adaptive-Only 所有请求被拒 | `check_capacity_under_failures` 用本地 dp_size（=1）而不是全局值（=2）| EngineCore 需要知道全局 dp_size，或者 `fault_tolerant` 模式也跳过 capacity-under-failures 检查 |
| Our-System SLO violation 高（18-45%） | TPOT 80-125ms 超过 50ms SLO | FT 调度框架的请求处理链路延迟 + checkpoint 拷贝开销 | 需要进一步诊断：异步 checkpoint / 异步 solver / 降低负载 |
