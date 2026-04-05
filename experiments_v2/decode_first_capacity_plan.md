# Decode-First Capacity Admission 改造方案

## 问题

现在 solver 的容量约束把 decode 时间串行相加，导致过度拒绝请求：

```
现在：Σ (d_j + c_j) · x[j,r] ≤ H_dec     ← 串行加时间，和 batch 现实差 3-10 倍
      Σ p_j · x[j,r] ≤ H_pre              ← prefill 独立拿一份时间预算
      H_pre = H_dec = planning_horizon     ← 两个预算互不影响
```

结果：1B 模型 Moderate 负载下 Benders 拒绝了 33% 请求。

---

## 新方案

镜像 vLLM 的 decode-first 调度策略：

```
新：Σ w_j · x[j,r] ≤ Cap_r^dec             ← decode slot 数量约束
    Σ p_j · x[j,r] ≤ RemPreCap_r(L_r^dec)  ← prefill 用 decode 剩余容量
```

### 核心思路

不问"每个请求要多久"，而问：

1. 这个 replica 还能扛多少个 decode 请求？
2. 在当前 decode 负载下，还能做多少 prefill？

---

## 符号定义

| 符号 | 含义 |
|---|---|
| `x_{j,r} ∈ {0,1}` | 请求 j 是否分配到 replica r |
| `w_j^dec` | 请求 j 的 decode 负担（最简版 = 1） |
| `p_j` | 请求 j 的 prefill 需求（= prompt token 数） |
| `Cap_r^dec` | replica r 在 planning horizon 内的 decode 容量 |
| `RemPreCap_r(L)` | decode 负载为 L 时，replica r 剩余的 prefill 容量 |
| `L_r^dec` | replica r 当前的 decode 负载 = `Σ w_j · x[j,r]` |

---

## 约束

### 约束 1: Decode 容量

```
Σ_j w_j^dec · x_{j,r} ≤ Cap_r^dec    ∀r
```

这个 replica 上的 decode 请求总负担不超过容量。

最简版 `w_j = 1` 时，就是数人头：最多同时跑多少个 decode 请求。

### 约束 2: Residual Prefill 容量

```
Σ_j p_j · x_{j,r} ≤ RemPreCap_r(L_r^dec)    ∀r
```

decode 越多 → 留给 prefill 的越少。

实际使用时，epoch 开头冻结 `L_r^dec`（当前 active decode 数已知），查表得到 `RemPreCap_r` 作为常数，约束保持线性。

---

## Profile 数据

需要两张表，离线测一次。

### Table A: Decode Capacity

> 在 SLO 约束下，这个 replica 最多同时跑多少个 decode 请求？

Profile 方法：
1. 固定模型和 GPU
2. 逐步增加并发 decode 请求数
3. 记录 TPOT 什么时候超过 SLO
4. 超 SLO 前的最大并发数 = `Cap_r^dec`

可选：按 context length 分 bucket（短 context vs 长 context 的负担不同）。

### Table B: Residual Prefill Capacity

> 当 decode 负载为 L 时，还能做多少 prefill？

Profile 方法：
1. 固定 L 个 decode 请求在跑
2. 逐步增加 pending prefill tokens
3. 记录 prefill 能在 planning horizon 内完成的最大 token 数
4. 不同 L 值各测一次 → 得到 `RemPreCap_r(L)` 表

```
L=0   → RemPreCap = 很大（GPU 全给 prefill）
L=10  → RemPreCap = 中等
L=30  → RemPreCap = 很小（decode 几乎占满）
```

---

## 在线 Admission 流程

每个 epoch（0.5s）：

```
Step 1: 统计当前 active decode 数 → L_r_current（已知常数）
Step 2: 查 Table A → Cap_r^dec
Step 3: 查 Table B → RemPreCap_r(L_r_current + 预估新 decode)
Step 4: 对每个 pending 请求，检查：
        - decode: L_r_current + Σ w_j · x[j,r] ≤ Cap_r^dec ?
        - prefill: Σ p_j · x[j,r] ≤ RemPreCap_r ?
Step 5: 通过的接，不通过的拒
Step 6: 跑 recovery checker 验证故障恢复可行性
```

---

## 求解方法

约束是线性的（冻结 L_r 后右边是常数），0/1 变量。

| 方法 | 速度 | 最优性 |
|---|---|---|
| **贪心**（按 SLO 紧急度排序，逐个塞） | < 0.1ms | 近似，实践够好 |
| **LP 松弛 + rounding** | < 1ms | 接近最优，有理论保证 |
| MIP (CP-SAT) | ~15ms | 精确最优 |

推荐 LP 松弛。二部图 assignment 约束接近全幺模，LP 解大概率天然是整数，不需要 rounding。

---

## 代码改动

| 文件 | 改什么 | 行数 |
|---|---|---|
| **新增** decode_capacity_profiler.py | 测 Cap_r^dec 和 RemPreCap_r(L) 表 | ~150 行 |
| cost_tables.py | `d_j` → `w_j`，`H_pre/H_dec` → 查 profile 表 | ~30 行 |
| master.py | 两个约束改写 | ~20 行 |
| recovery_checker.py | `w_total`/`u_r` 用新模型 | ~30 行 |
| config / arg_utils | 新增 profile 表路径参数 | ~10 行 |
| **合计** | | **~240 行** |

---

## 和现在代码的对比

| | 现在 | 改后 |
|---|---|---|
| decode 建模 | `d_j = output_tokens / throughput`（串行时间） | `w_j = 1`（并发 slot） |
| prefill 建模 | 独立预算 `H_pre`（和 decode 无关） | 依赖 decode 的剩余 `RemPreCap_r(L)` |
| 容量来源 | 两个固定常数（`ft_prefill/decode_throughput`） | 两张 profile 表（离线实测） |
| prefill/decode 关系 | 互不影响 | decode 先占，prefill 用剩余（和 vLLM 对齐） |
| 准确性 | 误差 3-10 倍 | 误差 ~20-30% |
| 过度拒绝 | 严重（33% 拒绝率） | 应该消除 |

---

## Recovery Checker 同步改动

recovery_checker.py 的 `w_total` 和 `u_r` 也需要用新模型：

```python
# 现在
w_total[j] = restore_time + replay_time + d_j + ckpt_overhead  # 串行时间
u_r = H_dec - surv_load[r]                                      # 时间预算余量

# 改后
w_total[j] = w_j^dec  # decode slot 占用（恢复到存活 replica 上要占一个 slot）
             + restore_work + replay_work  # 恢复工作量（用 token 数或 profile 查表）
u_r = Cap_r^dec - current_decode_on_r     # 存活 replica 剩余 decode slot
```

---

## 实施步骤

### Phase 1: Profile（1 天）
1. 写 decode_capacity_profiler.py
2. 对 1B 模型测 Table A 和 Table B
3. 保存为 JSON

### Phase 2: 改 Solver（1-2 天）
1. cost_tables.py: 加载 profile 表，改 `w_j` 和容量计算
2. master.py: 改两个约束
3. recovery_checker.py: 同步改
4. 跑现有测试确认不 break

### Phase 3: 验证（1 天）
1. 用 1B tiny 实验验证：Benders 完成率应从 64% → 90%+
2. 对比 No-FT / Periodic baselines，趋势应合理
3. 如果 `w_j = 1` 不够准，升级到 `w_j = α + β·ctx_j`

---

## 论文描述

> We model each replica's capacity using a decode-first policy that mirrors vLLM's default scheduling behavior. Each replica has a profiled decode capacity — the maximum number of concurrent decode requests it can sustain within SLO. Prefill capacity is modeled as the residual: given the current decode load, how many prefill tokens can still be processed within the planning horizon. A request assignment is feasible only if both the decode capacity and residual prefill capacity constraints are satisfied. This design captures the interference between prefill and decode workloads without requiring per-step latency simulation.

---

## 方案评估

### 1. 准确性

| | 现在（串行时间模型） | 新方案（decode-first capacity） |
|---|---|---|
| 误差来源 | 把并行 decode 当串行相加 | profile 表粒度、`w_j=1` 忽略 context 差异、epoch 内负载变化 |
| 误差量级 | **3-10 倍** | **20-30%** |
| 后果 | 拒绝 33% 请求 | 偶尔多接或少接几个请求 |

20-30% 误差对 admission 决策够用——你不需要精确算出"还剩 147.3ms"，只需要判断"还能不能塞下这个请求"。

如果 20-30% 不够，可升级：
- `w_j = 1` → `w_j = α + β·ctx_j`（context-sensitive），误差降到 10-15%
- profile 表加密 bucket（L 从 5 个点加到 15 个点）

### 2. 求解难度

**比现在更简单。**

冻结 `L_r^dec` 后，约束变成：

```
Σ w_j · x[j,r] ≤ 常数（Cap_r^dec）
Σ p_j · x[j,r] ≤ 常数（RemPreCap_r）
x[j,r] ∈ {0,1}
```

这是**带容量的二部图 assignment**。0/1 变量的处理：

- 约束矩阵接近**全幺模**（totally unimodular）
- LP 松弛的解大概率**天然是整数**，不需要 rounding
- 即使需要 rounding，0/1 变量是 LP 松弛效果最好的情况（参考 "On-Demand or On-Premises" DDoS paper 的 LP 松弛 + 随机化 rounding 方法）

| 方法 | 速度 | 最优性 | 推荐？ |
|---|---|---|---|
| 贪心 | < 0.1ms | 近似 | 最简单，先用这个验证 |
| **LP 松弛** | **< 1ms** | **接近最优，有理论保证** | **推荐用于论文** |
| MIP | ~15ms | 精确最优 | 没必要，LP 已经够好 |

### 3. 代码改动量

**~240 行，集中在 4 个文件 + 1 个新文件。**

| 文件 | 改什么 | 行数 |
|---|---|---|
| **新增** decode_capacity_profiler.py | 测 Cap_r^dec 和 RemPreCap_r(L) 两张表 | ~150 行 |
| cost_tables.py | `d_j` → `w_j`，`H_pre/H_dec` → 查 profile 表 | ~30 行 |
| master.py | 两个约束改写 | ~20 行 |
| recovery_checker.py | `w_total`/`u_r` 用新模型 | ~30 行 |
| config / arg_utils | 新增 profile 表路径参数 | ~10 行 |
| **合计** | | **~240 行** |

不需要改 Benders 分解结构、solve_loop.py、cuts.py、recovery checker 的 ILP/flow 求解逻辑。改动量小，风险可控。

### 4. 论文优雅性

**非常好。** 这是这个方案最大的优点。

叙事链自然流畅：

```
vLLM 的默认策略是 decode-first
  → 我们的 admission 也 decode-first
    → decode 容量从 profile 来（实测，不是拍常数）
      → prefill 容量是 decode 剩余（查表，不是独立估计）
        → prefill/decode 互相影响被自然建模了
```

对比现在论文要解释的：
- 为什么 `d_j = output_tokens / decode_throughput`？（解释不通，串行假设不成立）
- 为什么 `H_pre = H_dec = planning_horizon`？（为什么 prefill 和 decode 各拿一份？）
- 为什么有两个独立的 throughput 常数？（这两个数怎么来的？）

新方案一句话就能说清楚：

> "Each replica has a profiled decode capacity. Prefill capacity is the residual under the current decode load. This mirrors vLLM's decode-first scheduling."

Reviewer 不需要理解 MIP 的细节就能理解约束为什么这么写。

---

## Profile 的演进路径

先查表，后拟合。接口不变，内部实现随时可换。

| 阶段 | Cap_r^dec | RemPreCap_r(L) | 精度 |
|---|---|---|---|
| **V1（查表）** | 测 1 个值 | 测 5-10 个 L 值，插值 | 够用 |
| V2（线性拟合） | 不变 | `RemPreCap ≈ a - b·L` | 更平滑 |
| V3（context-sensitive） | 按 ctx bucket 分多个值 | `RemPreCap(L, avg_ctx)` 二维表 | 更准 |

从 V1 升级到 V2/V3 只需要改一个函数内部实现：

```python
# V1: 查表
def get_residual_prefill_cap(self, L):
    return interpolate(self.table, L)

# V2: 线性拟合（只改这个函数）
def get_residual_prefill_cap(self, L):
    return self.a - self.b * L

# V3: 二维拟合
def get_residual_prefill_cap(self, L, avg_ctx):
    return self.a - self.b * L - self.c * avg_ctx
```

调用方（master.py、recovery_checker.py）不需要改。

---

## 参考

| 来源 | 借鉴了什么 |
|---|---|
| **vLLM** | decode-first + residual prefill 的调度策略 |
| **SLOs-Serve** | batch/iteration 级别思考，不用 per-request 串行时间 |
| **Sarathi-Serve** | chunked prefill 和 decode 共存的建模 |
| **MuxWise** | context-sensitive decode burden、profile-based 估计 |

Decode capacity + residual prefill capacity 的组合是我们的综合设计，不是直接从某篇论文搬来的。
