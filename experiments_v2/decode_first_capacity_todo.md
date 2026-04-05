# Decode-First Capacity Admission — 实现 TODO (v2)

---

## Phase 1: Profile 脚本 (1 天)

### T1.1 Profile 数据格式

```json
{
  "meta": {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "gpu": "A100-80GB",
    "dp_size": 2,
    "max_model_len": 2048,
    "tpot_slo_ms": 50.0,
    "planning_horizon_sec": 0.5
  },
  "decode_capacity": {
    "default": 30,
    "by_avg_ctx_bucket": {
      "256": 35,
      "512": 30,
      "1024": 22,
      "2048": 15
    }
  },
  "residual_prefill_capacity": {
    "0": 2048,
    "5": 1700,
    "10": 1400,
    "15": 1100,
    "20": 800,
    "25": 500,
    "30": 200
  }
}
```

- [ ] **T1.1.1** 确定 JSON schema（字段名、类型、含义）

### T1.2 新建 `experiments_v2/profile_decode_capacity.py`

- [ ] **T1.2.1** `measure_decode_capacity(model, port, tpot_slo_ms) -> dict`
  - **必须用 dp=1 跑**（测单 replica 容量，多 replica 系统用 Cap = 测出值）
  - 对每个 ctx_bucket（至少测 256 和 1024 两个）：
    - 同时发 N 个请求（prompt_len=ctx_bucket, output_len=500），每组跑到稳态（20s+）
    - 测稳态下的 TPOT P95
    - 逐步增大 N = [5, 10, 15, 20, 25, 30, 40, 50]
    - 记录 TPOT P95 < tpot_slo_ms 的最大 N
  - 返回 `{ctx_bucket: max_N}` 和 `default = min(各 bucket 的 max_N)`

- [ ] **T1.2.2** `measure_residual_prefill_capacity(model, port, decode_count, horizon) -> int`
  - 测量方法（修正：不需要精确控制 decode/prefill 时序）：
    1. 同时发 `decode_count` 个长请求（output_len=1000，确保一直在 decode）
    2. 等待 10s 让它们全部进入 decode 稳态
    3. 开始持续发短 prefill 请求（max_tokens=1，只测 prefill 吞吐）
    4. 测 30s 内的 prefill token 吞吐量
    5. `RemPreCap = prefill_throughput * planning_horizon`
  - 对 decode_count = [0, 5, 10, 15, 20, 25, 30] 各测一次

- [ ] **T1.2.3** CLI 接口
  ```bash
  python experiments_v2/profile_decode_capacity.py \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --port 8300 \
      --tpot-slo-ms 50 \
      --horizon 0.5 \
      --output experiments_v2/decode_capacity_profile_1b.json
  ```

- [ ] **T1.2.4** 对 1B 模型跑一次（**dp=1**），验证：
  - decode_capacity 在合理范围（1B 大约 20-50）
  - 不同 ctx_bucket 的 capacity 有差异（长 context < 短 context）
  - RemPreCap(L) 随 L 单调递减
  - RemPreCap(0) ≈ max_num_batched_tokens * horizon / single_prefill_step_time

---

## Phase 2: 加载 Profile (0.5 天)

### T2.1 新建 `vllm/v1/core/sched/benders/decode_capacity_model.py`

- [ ] **T2.1.1** `DecodeCapacityModel` 类
  ```python
  class DecodeCapacityModel:
      def __init__(self, profile_path: str | None):
          """加载 profile JSON。profile_path=None 时启用 fallback。"""
          if profile_path is None or not os.path.exists(profile_path):
              logger.warning(
                  "Decode capacity profile not found: %s. "
                  "Using legacy fallback (planning_horizon model).",
                  profile_path,
              )
              self._fallback = True
              return
          self._fallback = False
          # 加载 JSON ...

      @property
      def is_fallback(self) -> bool:
          return self._fallback

      def decode_capacity(self, avg_ctx_bucket: int = 0) -> int:
          """返回 Cap_r^dec"""

      def residual_prefill_capacity(self, decode_load: int) -> int:
          """返回 RemPreCap_r(L)，线性插值"""
  ```

- [ ] **T2.1.2** 插值逻辑
  ```python
  # 例：profile 有 L=10 → 1400, L=15 → 1100
  # 查 L=12 → 1400 + (1100-1400) * (12-10)/(15-10) = 1280
  # L 超出最大 profile 点时返回 0（无剩余容量）
  # L 小于最小 profile 点时返回最大值
  ```

- [ ] **T2.1.3** Fallback 模式
  - profile 不存在时，`decode_capacity()` 返回一个保守大值（如 100）
  - `residual_prefill_capacity()` 返回 `max_model_len`（不限制）
  - 效果：退回到无容量约束的行为，不会崩，但也不会做 admission 控制

- [ ] **T2.1.4** 预留拟合接口（V2 升级用）
  ```python
  def residual_prefill_capacity(self, decode_load: int) -> int:
      if self._use_fitted:
          return max(0, int(self._a - self._b * decode_load))
      return self._interpolate(decode_load)
  ```

---

## Phase 3: 改 cost_tables.py (0.5 天)

### T3.1 加载 decode capacity model

- [ ] **T3.1.1** `CostTableBuilder.__init__()` 新增参数 `decode_capacity_profile_path: str | None`
- [ ] **T3.1.2** 加载 `DecodeCapacityModel` 实例存为 `self._decode_cap_model`
- [ ] **T3.1.3** 新增方法：
  ```python
  def decode_capacity(self) -> int:
      """Cap_r^dec（所有同构 replica 共享同一个值）"""
      return self._decode_cap_model.decode_capacity()

  def residual_prefill_capacity(self, decode_load: int) -> int:
      """RemPreCap_r(L)"""
      return self._decode_cap_model.residual_prefill_capacity(decode_load)

  @property
  def use_decode_first_model(self) -> bool:
      """是否使用新模型（vs fallback 到旧的 H_pre/H_dec）"""
      return not self._decode_cap_model.is_fallback
  ```

### T3.2 改 RequestCosts

- [ ] **T3.2.1** `RequestCosts` dataclass 新增字段：
  ```python
  w_dec: float = 1.0           # decode burden（最简版 = 1）
  prefill_tokens: int = 0      # prompt token 数（= p_j）
  replay_tokens: int = 0       # 恢复时需要 replay 的 token 数
  ```

- [ ] **T3.2.2** `build_request_costs()` 里填充新字段
  ```python
  costs.w_dec = 1.0  # V1: 每个 decode 请求权重 1
  costs.prefill_tokens = prompt_len if not is_active else 0
  # active 请求已经 prefill 完了，不再需要 prefill 容量

  # replay_tokens: 恢复时未被 checkpoint 覆盖的 token 数
  recovery_tokens = num_computed_tokens if is_active else prompt_len
  published_tokens = min(num_checkpointed_tokens, recovery_tokens) if is_active else 0
  costs.replay_tokens = max(0, recovery_tokens - published_tokens)
  ```

### T3.3 保留旧字段（兼容）

- [ ] **T3.3.1** `d_j` 字段保留（recovery checker 的 gap_time 计算还在用）
- [ ] **T3.3.2** `H_pre` 和 `H_dec` 属性保留（fallback 模式使用）

---

## Phase 4: 改 master.py (0.5 天)

### T4.1 修改构造函数

- [ ] **T4.1.1** `MasterProblem.__init__()` 参数变更：
  ```python
  # 保留（fallback 用）
  H_pre: float,
  H_dec: float,

  # 新增
  decode_capacity: int = 0,                       # Cap_r^dec（同构，所有 replica 共享）
  residual_prefill_capacity: dict[int, int] | None = None,
      # {replica_id: RemPreCap_r}  per-replica，各自 decode 负载不同
  use_decode_first: bool = False,                  # 是否启用新模型
  ```

### T4.2 改约束

- [ ] **T4.2.1** 如果 `use_decode_first=True`，用新约束：

  **Decode capacity 约束**（同构 replica 共享同一个 Cap）：
  ```python
  for r in self._replica_ids:
      decode_terms = []
      for req_id, costs in self._costs.items():
          if (req_id, r) in x:
              decode_terms.append(_to_int(costs.w_dec) * x[(req_id, r)])
      if decode_terms:
          model.add(sum(decode_terms) <= _to_int(self._decode_capacity))
  ```

  **Residual prefill 约束**（per-replica，各自不同）：
  ```python
  for r in self._replica_ids:
      rem_cap = self._residual_prefill_capacity.get(r, 0)
      prefill_terms = []
      for req_id, costs in self._costs.items():
          if (req_id, r) in x and costs.prefill_tokens > 0:
              prefill_terms.append(costs.prefill_tokens * x[(req_id, r)])
      if prefill_terms:
          model.add(sum(prefill_terms) <= rem_cap)
  ```

- [ ] **T4.2.2** 如果 `use_decode_first=False`（fallback），保留旧约束不动

---

## Phase 5: 改 solve_loop.py (0.5 天)

### T5.1 统计 per-replica decode 负载

- [ ] **T5.1.1** 每个 epoch 开始时，从 request snapshot 统计每个 replica 的 active decode 数：
  ```python
  L_current: dict[int, int] = {}  # {replica_id: active_decode_count}
  for req_id, costs in request_costs.items():
      if costs.is_active and costs.active_replica_id is not None:
          r = costs.active_replica_id
          L_current[r] = L_current.get(r, 0) + int(costs.w_dec)
  ```

- [ ] **T5.1.2** 查表得到 per-replica 的 residual prefill capacity：
  ```python
  # 保守估计：当前 active decode + pending 请求数都算进去
  # 因为 pending 请求一旦被接，很快会进入 decode 阶段
  pending_count = sum(1 for c in request_costs.values() if not c.is_active)
  residual_prefill = {
      r: self._cost_builder.residual_prefill_capacity(
          L_current.get(r, 0) + pending_count // len(replica_ids)
      )
      for r in replica_ids
  }
  ```
  注意：这里用 `pending / replica_count` 做均摊估计，偏保守但安全。

- [ ] **T5.1.3** 传给 MasterProblem：
  ```python
  master = MasterProblem(
      request_costs=...,
      replica_ids=...,
      decode_capacity=self._cost_builder.decode_capacity(),
      residual_prefill_capacity=residual_prefill,   # per-replica dict
      use_decode_first=self._cost_builder.use_decode_first_model,
      # fallback 参数保留
      H_pre=..., H_dec=...,
  )
  ```

- [ ] **T5.1.4** 同样传给 RecoveryChecker

---

## Phase 6: 改 recovery_checker.py (0.5 天)

### T6.1 构造函数新增参数

- [ ] **T6.1.1** 新增参数：
  ```python
  decode_capacity: int = 0,
  decode_cap_model: DecodeCapacityModel | None = None,
  use_decode_first: bool = False,
  ```

### T6.2 改 residual budget 计算

- [ ] **T6.2.1** `surv_load` 从时间累加改为 decode slot 计数：
  ```python
  # 现在
  surv_load[r] += costs.d_j + costs.checkpoint_overhead_sec

  # 改为（use_decode_first=True 时）
  surv_decode_count[r] += costs.w_dec  # 就是 +1
  ```

- [ ] **T6.2.2** `u_r` 双维度 budget：

  **Decode slot 余量**：
  ```python
  u_decode[r] = max(0, self._decode_capacity - surv_decode_count[r])
  ```

  **Prefill 余量**（恢复请求要做 replay，需要 prefill 容量）：
  ```python
  # 最坏情况估计：假设所有 affected 请求都恢复到这个 replica
  # decode 数增加 len(affected) → prefill 容量相应缩小
  u_prefill[r] = self._decode_cap_model.residual_prefill_capacity(
      surv_decode_count[r] + len(affected)
  )
  # 这是保守的（实际不会全恢复到一个 replica），但保证 ILP 解出的方案一定可行
  ```

### T6.3 Pool overload 检查改为双维度

- [ ] **T6.3.1** decode slot 维度：
  ```python
  total_decode_slots_needed = len(affected)  # 每个恢复请求占 1 个 slot
  total_decode_slots_available = sum(u_decode.values())
  if total_decode_slots_needed > total_decode_slots_available:
      return PoolOverload ...
  ```

- [ ] **T6.3.2** prefill 维度：
  ```python
  total_replay_tokens = sum(costs.replay_tokens for j in affected)
  total_prefill_available = sum(u_prefill.values())
  if total_replay_tokens > total_prefill_available:
      return PoolOverload ...
  ```

### T6.4 改 feasible_edge（双维度 + TTFT/TPOT）

- [ ] **T6.4.1** decode slot 检查：
  ```python
  if costs.w_dec > u_decode.get(replica_id, 0):
      feasible = False
  ```

- [ ] **T6.4.2** prefill/replay 检查：
  ```python
  if costs.replay_tokens > u_prefill.get(replica_id, 0):
      feasible = False
  ```

- [ ] **T6.4.3** TTFT 检查（补算法对齐）：
  ```python
  if (costs.ttft_slo_sec is not None
          and costs.p_j > costs.ttft_slo_sec):
      feasible = False
  ```

- [ ] **T6.4.4** TPOT 检查（补算法对齐）：
  ```python
  if (costs.tpot_slo_sec is not None
          and costs.resume_time_sec > costs.tpot_slo_sec):
      feasible = False
  ```

### T6.5 改 ILP 约束

- [ ] **T6.5.1** work 约束改为 decode slot 约束：
  ```python
  # 现在
  model.add(sum(work_terms) <= residual_budget[r])

  # 改为
  model.add(sum(slot_terms) <= u_decode[r])
  ```

- [ ] **T6.5.2** 新增 prefill 约束：
  ```python
  for r in surviving:
      replay_terms = []
      for req_id in affected:
          if (req_id, r) in assign:
              replay_terms.append(
                  costs[req_id].replay_tokens * assign[(req_id, r)])
      if replay_terms:
          model.add(sum(replay_terms) <= u_prefill[r])
  ```

### T6.6 Fallback

- [ ] **T6.6.1** `use_decode_first=False` 时保留旧逻辑不动

---

## Phase 7: 改 config 和参数传递 (0.5 天)

### T7.1 config 新增参数

- [ ] **T7.1.1** `vllm/config/scheduler.py` 新增：
  ```python
  ft_decode_capacity_profile: Optional[str] = None
  ```

- [ ] **T7.1.2** `vllm/engine/arg_utils.py` 新增 CLI 参数：
  ```
  --ft-decode-capacity-profile  path/to/decode_capacity_profile.json
  ```

### T7.2 参数传递链

- [ ] **T7.2.1** `benders_ft_scheduler_impl.py`：从 config 读取 profile path，传给 CostTableBuilder
- [ ] **T7.2.2** `ft_scheduler_impl.py`：同上
- [ ] **T7.2.3** experiments_v2 的 config yaml 里加 `ft_decode_capacity_profile` 字段

---

## Phase 8: 验证 (1 天)

### T8.1 单元测试

- [ ] **T8.1.1** `test_decode_capacity_model.py`：
  - 加载 profile JSON
  - 插值正确性（边界值、中间值、超范围值）
  - fallback 模式不崩

- [ ] **T8.1.2** `test_master_new_constraints.py`：
  - 新约束下 master 能正常求解
  - decode capacity 约束生效（超了会拒绝）
  - residual prefill 约束生效（per-replica 不同值）
  - fallback 模式退回旧约束

- [ ] **T8.1.3** `test_recovery_checker_new_model.py`：
  - 双维度 budget（decode slot + prefill）正确
  - TTFT/TPOT 边检查生效
  - fallback 模式不崩

### T8.2 集成测试

- [ ] **T8.2.1** 用 1B 模型跑 E_Tiny（12 runs），对比改前改后：
  - 改前：Benders 完成率 64%，goodput 218
  - **改后目标**：Benders 完成率 > 90%，goodput 接近 Periodic baselines（~247）

- [ ] **T8.2.2** 检查 recovery checker 在故障场景下仍能正确验证恢复方案

### T8.3 回归测试

- [ ] **T8.3.1** 跑 `tests/ft/test_benders_solver.py` 确认不 break
- [ ] **T8.3.2** 跑 `tests/ft/test_benders_integration.py` 确认不 break
- [ ] **T8.3.3** 跑 `tests/ft/test_centralized_benders.py` 确认不 break

---

## 依赖关系

```
T1 (profile 脚本) → T1.2.4 (跑 profile) → T2 (DecodeCapacityModel)
                                                    ↓
                                              T3 (cost_tables)
                                                    ↓
                                    ┌───────────────┼───────────────┐
                                    ↓               ↓               ↓
                              T4 (master)     T5 (solve_loop)  T6 (recovery_checker)
                                    └───────────────┼───────────────┘
                                                    ↓
                                              T7 (config 传参)
                                                    ↓
                                              T8 (验证)
```

---

## 注意事项

| # | 说明 |
|---|---|
| 1 | `Cap_r^dec` 假设同构 GPU（所有 replica 共享同一个值）。异构 GPU 需要 per-replica profile，目前不支持 |
| 2 | `residual_prefill_capacity` 是 **per-replica** 的（各 replica decode 负载不同）|
| 3 | recovery_checker 需要**双维度**检查：decode slot + prefill replay。不能只查一个 |
| 4 | 所有改动都有 fallback 到旧模型的路径（profile 不存在时）。不会因为缺 profile 而崩 |
| 5 | `d_j` 字段保留，因为 `gap_time_sec` 的计算还在用它（failover gap SLO 检查）|
| 6 | Profile 必须用 **dp=1** 跑（测单 replica 容量），至少测 **2 个 ctx bucket**（256, 1024）|
| 7 | `residual_prefill_capacity` 查表时用**保守估计**：L = 当前 active + pending 均摊，避免偏乐观 |
| 8 | recovery_checker 的 `u_prefill` 用**最坏估计**：假设所有 affected 都恢复到该 replica |
| 9 | `RequestCosts` 新增 `replay_tokens` 字段（从 checkpoint 状态推算），recovery_checker 的 prefill 检查需要它 |

---

## 文件清单

| 文件 | 操作 | 行数 |
|---|---|---|
| `experiments_v2/profile_decode_capacity.py` | 新建 | ~150 |
| `vllm/v1/core/sched/benders/decode_capacity_model.py` | 新建 | ~80 |
| `vllm/v1/core/sched/benders/cost_tables.py` | 改 | ~30 |
| `vllm/v1/core/sched/benders/master.py` | 改 | ~30 |
| `vllm/v1/core/sched/benders/solve_loop.py` | 改 | ~20 |
| `vllm/v1/core/sched/benders/recovery_checker.py` | 改 | ~50 |
| `vllm/config/scheduler.py` | 改 | ~3 |
| `vllm/engine/arg_utils.py` | 改 | ~5 |
| `vllm/v1/core/sched/benders_ft_scheduler_impl.py` | 改 | ~5 |
| `vllm/v1/core/sched/ft_scheduler_impl.py` | 改 | ~5 |
| `experiments_v2/config_1b*.yaml` | 改 | ~5 |
| `tests/ft/test_decode_capacity_model.py` | 新建 | ~80 |
| **合计** | | **~463 行** |

---

## 时间预估

| Phase | 内容 | 时间 |
|---|---|---|
| Phase 1 | Profile 脚本 + 跑一次 | 1 天 |
| Phase 2 | DecodeCapacityModel 类 + fallback | 0.5 天 |
| Phase 3 | cost_tables.py 改动 | 0.5 天 |
| Phase 4 | master.py 改约束 | 0.5 天 |
| Phase 5 | solve_loop.py per-replica 统计 + 传参 | 0.5 天 |
| Phase 6 | recovery_checker.py 双维度 + TTFT/TPOT | 0.5 天 |
| Phase 7 | config 和参数传递 | 0.5 天 |
| Phase 8 | 测试和验证 | 1 天 |
| **合计** | | **~5 天** |

---

## 执行命令

代码已全部实现。按以下顺序执行：

### Step 1: Profile（约 30 分钟）

测 1B 模型的 decode capacity 和 residual prefill capacity，产出 profile JSON。

```bash
python experiments_v2/profile_decode_capacity.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --port 8300 \
    --tpot-slo-ms 50 \
    --output experiments_v2/decode_capacity_profile_1b.json
```

### Step 2: 配置 profile 路径

把 Step 1 产出的 profile 路径填到实验 config 里：

```bash
# 修改 config_1b_tiny.yaml 的 ft_decode_capacity_profile 字段
# 从 "" 改成 "experiments_v2/decode_capacity_profile_1b.json"
```

### Step 3: 跑 E_Tiny 验证（约 20 分钟）

```bash
python experiments_v2/suite.py \
    --config experiments_v2/config_1b_tiny.yaml \
    --experiment E_Tiny \
    --port 8300
```

### 预期结果

| 指标 | 改前（旧模型） | 改后目标（decode-first） |
|---|---|---|
| Benders 完成率 | 64% | > 90% |
| Our-System goodput | 218 tok/s | 接近 Periodic baselines (~247) |
| Solver 开销 | ~15ms | 不变或更低 |
