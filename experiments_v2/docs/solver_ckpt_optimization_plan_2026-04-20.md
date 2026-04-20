# Solver + Checkpoint Optimization Plan (2026-04-20)

## 核心原则 — 不跳过算法

Paper 的核心 idea 是 **Benders ILP admission + proactive checkpoint recovery**。这两个组件必须始终运行、是 contribution 本体。**优化目标：保持算法完整的同时提速，不是绕开它们。**

## 之前的错误方向（已回滚）

| 错误方向 | 问题 |
|---|---|
| `FT_SOLVER_TRIVIAL_SKIP=1` | 跳过 MIP，审稿人会说"你没在跑 ILP" |
| `FT_SKIP_SOLVER=1` | 同上，更糟 |
| `FT_FAST_FAILOVER=1` (故障期) | 故障期不 reroute，失去 paper 主打点 |
| `FT_GATED_SOLVER=1` | 低负载不跑 = 多数时间跳过 |
| `Our-System-NoCkpt` | 直接关 ckpt 基础设施 |

**这些之后禁用**。保留的代码改动：`time_cap`（合法时限参数）和 `warm-start`（合法热启动），它们是**优化**，不是跳过。

## 真正的优化方向

### A. Solver 提速（保持 ILP 始终运行）

#### A1. 增量 cost_table（避免每 epoch 重建）
现状：`CostTableBuilder.build_snapshot_costs(active, pending)` 每次全量重建。
优化：缓存 cost_table，只对新增/离开的 request 增量更新。
预计：Python loop O(N) → O(ΔN)，N=活跃请求数时节省显著。

#### A2. 模型结构复用（Cache CpModel）
现状：每次 `MasterProblem(...).solve()` 新建 `cp_model.CpModel()`，重新 `new_bool_var` 所有变量、重新 `model.add()` 所有约束。
优化：只要 replica_ids 和 active set 不变，复用同一个 model，只更新变量的 objective 系数（CP-SAT 支持）。
预计：模型构造 O(R × replica) 省掉。

#### A3. 减少 MIP 问题规模 — 合法预处理
- 在 ILP 前 drop 显然 infeasible 的 request（TTFT 已经过期）
- Replica 过滤：排除容量为 0 的 replica
- 均非跳过 — 是 "presolve" 步骤，标准做法

#### A4. Recovery checker 并行
现状：recovery checker 对每个 failure scenario 串行解一个子 MIP。
dp=2 场景下只有一个 scenario，但改成 multi-threaded 求解可降低未来扩展到 dp=4+ 的延迟。

#### A5. 快速收敛：multi-cut Benders
现状：每 iter 只生成 1 cut。
优化：同时对所有 infeasible scenarios 生成 cuts，加速 Benders 收敛。

### B. Checkpoint 基础设施提速（保持 ckpt 始终启用）

之前诊断发现 **17% 开销来自 ckpt 基础设施** — 并非写 ckpt 本身。

#### B1. 缩小 checkpoint_pool_bytes
现状：`checkpoint_pool_bytes: 34359738368` = **32GB**。A5000 24GB，这个池需要额外大内存。
**对 8B 模型**：KV per token ≈ 128KB。32GB 池能装 256k tokens，远超 `max_model_len × max_reqs` 所需。
优化：4-8GB 池对 max_model_len=8192 的 8B 模型足够用。
预计：直接节省 24GB 内存压力，ckpt 相关分配 / 映射大幅加速。

#### B2. Lazy pool 分配
现状：启动即预分配全部 32GB 池。
优化：启动时预分配 4GB，按需增长。
预计：启动快，稳态无空闲浪费。

#### B3. Snapshot 增量构建
现状：engine 每步调用 `_build_active_snapshots()` 全量重建 `outputs.active_request_snapshots`。
优化：只发送变化（new/finished），engine 和 client 各维护状态。
预计：per-step O(active_reqs) → O(ΔN)。

#### B4. Snapshot 异步构建
现状：snapshot 构建在 engine 主循环里（阻塞）。
优化：移到独立 thread，主循环只 enqueue。
代码变动中等。

#### B5. Checkpoint KV 压缩
现状：KV blocks 以 fp16 直接 memcpy 到 /dev/shm。
优化：fp8 量化 → 体积减半，写入带宽翻倍。
paper 角度：这是**新 mechanism contribution**。

### C. 调度循环优化（保持 epoch 模式）

#### C1. 增大 epoch_interval_sec
现状：`benders_epoch_interval_sec: 0.02` = 20ms。稳态下 20ms poll 太频繁。
优化：稳态 50-100ms，故障期触发即时缩短到 5ms。
不是"跳过"，是"自适应频率"。

#### C2. Event-driven pending 触发
现状：epoch timer 驱动。
优化：新 request 到达 + active request 完成 → 立即触发 solver。timer 仅作 safety net。
Solver 仍**每次被正确触发**，只是触发更及时（非 periodic lag）。

### D. Python 热路径 JIT

#### D1. CostTableBuilder cython-ize
cost_table 构造是纯 Python loop。改 cython 或 numba JIT，预计 3-5× 加速。

## 实施优先级（按"最短收益时间"排序）

| # | 优化 | 代码改动 | 预计收益 | 风险 |
|---|---|---|---|---|
| 1 | B1: 缩小 ckpt pool (32GB→4GB) | 1 行 config | 恢复 17% ckpt 开销大头 | 低 |
| 2 | A2: Cache CpModel | ~40 行 master.py | 每次 solve -20-40ms | 低 |
| 3 | C1: 增大 epoch_interval | 1 行 | 平时 CPU 省 | 低 |
| 4 | A1: 增量 cost_table | ~60 行 cost_tables.py | per-epoch -10ms | 中 |
| 5 | B3: Snapshot 增量 | ~80 行 | 持续 CPU 省 | 中 |
| 6 | C2: Event-driven | ~30 行 ft_client.py | TTFT 响应快 | 中 |
| 7 | B4: 异步 snapshot | ~100 行 | 主循环不阻塞 | 高 |
| 8 | D1: cython cost_table | 大改 | python 热路径 3× | 高 |
| 9 | B5: fp8 ckpt 量化 | ~200 行 | ckpt IO 减半 | 高（精度） |

## 今晚先做

**今晚计划**：B1 + C1（都是 config 调整，几分钟能测）+ A2（Cache CpModel，核心代码优化）

三者合起来覆盖最大 17% ckpt + solver 重复构造开销，**不跳过任何算法逻辑**，**保持 Benders MIP 每次都跑**。

### 实验
- baseline: V2（当前带 warm-start、time_cap，**去除 trivial_skip**）
- `smaller_pool`: baseline + `checkpoint_pool_bytes: 4294967296` (4GB)
- `longer_epoch`: baseline + `benders_epoch_interval_sec: 0.1`
- `cached_cpmodel`: baseline + A2 代码改动
- `all_three`: smaller_pool + longer_epoch + cached_cpmodel

全部在 W5_LongDoc × F2_Mid × 3 seeds。判决：gp ≥ NR 30.4 AND fg_p95 ≤ NR 6115。

## 清理之前的 env flag

- `FT_SOLVER_TRIVIAL_SKIP` — 默认改为 0（关）
- `FT_SKIP_SOLVER` — 保留但默认 0
- `FT_FAST_FAILOVER` — 保留但默认 0
- `FT_GATED_SOLVER` — 保留但默认 0
- `FT_SOLVER_TIME_CAP_MS` — 保留（合法优化）
- Warm-start — 保留（合法优化）
- `FT_SOLVER_TRIVIAL_MAX` — 保留但默认 0（相当于关）
