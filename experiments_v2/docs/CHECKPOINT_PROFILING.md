# Checkpoint Profiling & Scheduling Design

## Phase 0: Checkpoint Cost Profiling

### Overview

Phase 0 implements an offline profiling pipeline to measure checkpoint system costs and build a cost model for adaptive checkpoint scheduling. The goal is to quantify four key cost functions:

1. **T_prefill(n)**: Latency to prefill n tokens (via HTTP + first-token generation)
2. **T_load(S)**: Latency to restore S bytes of KV cache from checkpoint
3. **T_ckpt(S)**: Latency to save S bytes of KV cache to checkpoint
4. **c0**: Fixed publication overhead (independent of checkpoint size)

These measurements enable the **profile-driven checkpoint decision rule**:
```
Publish checkpoint if: T_replay(L, u) > T_load(ΔS) + λ * T_ckpt(ΔS) + c0
```

Where:
- **L**: Number of previously published tokens (stable)
- **u**: Number of unpublished tokens (at risk during failure)
- **ΔS**: Incremental KV cache size since last checkpoint
- **λ**: Tuning parameter (default 1.0)

---

### Implementation Details

#### 1. Prefill Microbench (`PrefillBenchmark`)

**Purpose**: Measure T_prefill(n) by benchmarking end-to-end HTTP latency with max_tokens=1.

**Method**:
```python
PrefillBenchmark(model="meta-llama/Llama-3.2-1B-Instruct", port=8300).run(
    token_lengths=[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
    num_warmup=2,
    num_trials=5
)
```

**Key Design Decisions**:

1. **Tokenizer-based token generation**: Uses `transformers.AutoTokenizer` to generate input text with exact target token count via iterative refinement (max 5 iterations). This ensures T_prefill measurements are tied to true token counts, not character-based approximations.

2. **Fallback to character estimation**: If tokenizer loading fails (e.g., model not available locally), falls back to character-based estimation (4 chars ≈ 1 token) with a warning. This is tracked in `tokenizer_failed` flag to mark measurements as placeholder data.

3. **HTTP latency measurement**: Includes network roundtrip, JSON serialization, and first-token decode. This is intentional—the cost model should account for end-to-end amortization of communication costs in checkpoint decisions.

4. **Retry logic**: Up to 3 retries with clear error messages if server is unavailable or unresponsive.

**Output**: Dictionary `{token_count: median_latency_ms}`

**Validation**:
- Profile marked `is_real_measurement=False` if tokenizer fails (falls back to character estimation)
- This prevents using inaccurate T_prefill data in checkpoint decisions

---

#### 2. KV Checkpoint/Restore Microbench (`KVCheckpointBenchmark`)

**Purpose**: Measure T_load(S) and T_ckpt(S) using actual `KVCheckpointPool` API.

**Method**:
```python
kv_bench = KVCheckpointBenchmark()  # Uses Llama-3.2-1B geometry by default
load_data, checkpoint_data = kv_bench.benchmark_save_restore(
    block_counts=[1, 2, 4, 8, 16, 32, 64]
)
```

**Key Design Decisions**:

1. **Model-specific geometry**: Uses correct Llama-3.2-1B-Instruct parameters:
   - `num_kv_heads=8` (from model config, extracted from config.num_key_value_heads)
   - `head_size=64` (from config.hidden_size // num_attention_heads)
   - `num_layers=16` (hardcoded as constant for this model)

   This ensures measurements reflect realistic KV cache sizes for the target model. Since the model is fixed, hardcoding these parameters is sufficient. **Future generalization**: If supporting multiple models, extract these from model config dynamically.

2. **Per-layer tensor structure**: Creates tensors matching actual GPU KV cache layout:
   ```
   Shape per layer: (2, num_blocks, block_size, num_kv_heads, head_size)
   where 2 = K and V dimensions
   ```
   This ensures `KVCheckpointPool` API calls exercise realistic code paths.

3. **Synchronous measurement**: Uses `async_copy=False` and `torch.cuda.synchronize()` for accurate wall-clock latency. Avoids overlapping with other GPU work.

4. **Median over trials**: 5 trials per block count to reduce noise.

**Output**: Two dictionaries:
- `save_results: {total_bytes: median_latency_ms}` for checkpoints
- `restore_results: {total_bytes: median_latency_ms}` for restores

**Fallback**: If benchmark fails (e.g., CUDA not available), uses placeholder data and marks `kv_measurement_failed=True`.

---

#### 3. Publication Overhead Estimation (`estimate_c0_from_checkpoint_data`)

**Purpose**: Decompose checkpoint latency into fixed overhead c0 and size-dependent slope a.

**Assumes**: `T_ckpt(S) = c0 + a*S` (linear model)

**Method**: Linear regression on smallest 3-5 checkpoint measurements (most reliable, less measurement noise on small sizes).

**Output**: `(c0, a)` where c0 is publication overhead in milliseconds.

---

#### 4. Profile Validation (`inspect_checkpoint_profile.py`)

**Purpose**: Validate checkpoint cost profile JSON and allow interactive inspection of checkpoint decisions.

**Validation Checks**:

1. **Measurement reality**: Rejects profiles with `is_real_measurement=False` (placeholder data)
   ```
   Profile marked as PLACEHOLDER if:
   - skip_prefill=True (no actual measurement)
   - tokenizer_failed=True (T_prefill uses char estimation)
   - skip_kv=True (no actual measurement)
   - kv_measurement_failed=True (benchmark failed)
   ```

2. **Monotonicity**: Checks that all three cost functions are monotonically increasing (realistic property).

3. **Convexity** (prefill only): Checks that T_prefill has non-negative second derivative (realistic property for typical GPUs—prefill per-token cost usually increases with sequence length due to cache effects).

**Interactive Mode**: Test checkpoint decisions under various (L, u, S, ΔS) scenarios:
```
> 128 256 262144 131072
  T_replay(L,u)  = +1234.56
  T_load(ΔS)     =    50.25
  T_ckpt(ΔS)     =    75.80
  c0             =    10.00
  total_cost     =   136.05
  Decision: ✅ PUBLISH (margin=+1098.51ms)
```

---

### Output Files

#### checkpoint_cost_profile.json

```json
{
  "meta": {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "dtype": "float16",
    "device": "cuda",
    "block_size_tokens": 16,
    "layout": "standard_kv_cache",
    "generated_at": "2026-03-30T...",
    "is_real_measurement": true,
    "notes": "REAL measurements"
  },
  "prefill_ms_by_tokens": {
    "16": 1.5,
    "32": 2.0,
    ...
  },
  "load_ms_by_bytes": {
    "131072": 0.5,
    ...
  },
  "checkpoint_ms_by_bytes": {
    "131072": 0.1,
    ...
  },
  "publication_overhead_ms": 10.0
}
```

**Key Fields**:
- `is_real_measurement`: Boolean flag—reject profiles with `false` as unreliable for checkpoint decisions
- `checkpoint_ms_by_bytes`: Already has c0 subtracted (size-dependent latency only)
- `publication_overhead_ms`: Fixed overhead needed by decision rule

---

### Usage

#### Generate Profile

```bash
# Full measurement (requires running vLLM server)
python experiments/profile_checkpoint_costs.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --port 8300 \
  --output experiments/checkpoint_cost_profile.json

# Skip prefill (only measure KV)
python experiments/profile_checkpoint_costs.py --skip-prefill

# Use placeholder data (no measurements)
python experiments/profile_checkpoint_costs.py --skip-prefill --skip-kv
```

#### Validate and Inspect Profile

```bash
# Validate profile structure and check decision rule
python experiments/inspect_checkpoint_profile.py experiments/checkpoint_cost_profile.json

# With custom λ (checkpoint cost weight)
python experiments/inspect_checkpoint_profile.py experiments/checkpoint_cost_profile.json --lambda 2.0
```

---

### Known Limitations & Future Work

#### Phase 0 Limitations

1. **HTTP latency includes communication overhead**: T_prefill is end-to-end latency, not pure GPU computation. This is acceptable for cost amortization but may overestimate pure prefill time.

2. **Fixed model geometry**: Currently hardcoded for Llama-3.2-1B. Generalizing to arbitrary models requires extracting config dynamically (low effort, deferred to future phases).

3. **Placeholder data on failure**: If tokenizer or KV benchmark fails, profile contains placeholder data. Validation rejects these for real experiments.

4. **Single token generation**: max_tokens=1 may not reflect multi-token generation patterns. Acceptable for checkpoint decisions (amortization) but not for throughput modeling.

#### Planned Extensions

- [ ] **Dynamic model geometry**: Extract num_kv_heads, head_size, num_layers from model config
- [ ] **Batch prefill measurement**: Measure T_prefill with larger batch sizes if needed
- [ ] **Checkpoint traffic measurement**: Model upload/download costs for remote checkpoints
- [ ] **Host memory footprint**: Track peak memory usage during checkpoint/restore
- [ ] **Integrated validation**: Run validation automatically after profiling

---

### Testing

```bash
# Test profile inspection (validates mock profiles)
python -m pytest tests/ft/test_experiment_log_parser.py -v -k "checkpoint"
```

---

## Phase 1: Profile-Driven Checkpoint Decision in Runtime

### Overview

Phase 1 integrates the checkpoint cost profile into the runtime's adaptive checkpoint controller. The goal is to replace hardcoded linear throughput/bandwidth assumptions with real measured cost functions.

**Design Principle**: Only change the adaptive (`economic_policy`) path in CheckpointController. Fixed baselines (Fixed-High/Low, Robust-Routing-Only) are unaffected.

### Implementation Details

#### 1. CheckpointCostModel Runtime Module

**File**: `vllm/v1/core/checkpoint_cost_model.py`

- **Loads profile JSON** and validates `is_real_measurement` flag (rejects placeholder data immediately with ValueError)
- **Provides three cost functions** via piecewise linear interpolation:
  - `t_prefill(n)` → T_prefill(n)
  - `t_load(S)` → T_load(S)
  - `t_ckpt(S)` → T_ckpt(S)
- **Implements decision rule**:
  ```python
  def should_publish(L, u, S, delta_S, lambda_=1.0) -> bool:
      replay = t_prefill(L+u) - t_prefill(L)
      load   = t_load(S+delta_S) - t_load(S)
      ckpt   = t_ckpt(delta_S)
      return replay > load + lambda_ * ckpt + c0
  ```

#### 2. CheckpointController Integration

**File**: `vllm/v1/core/checkpoint_controller.py`

- **New parameter**: `cost_profile_path: str = ""` in `__init__`
- **Gate logic update**: `_economic_policy_available` now returns True if EITHER:
  - Profile is loaded (cost_model is not None), OR
  - Linear model inputs are provided (replay_throughput > 0 AND load_bandwidth > 0 AND checkpoint_bandwidth > 0)
- **Decision logic**: In `_should_checkpoint_by_economic_policy()`:
  - If `_cost_model is not None`: use profile-driven costs
  - Else: fall back to linear model

**Key insight**: Profile-driven takes precedence when both are available, but system works with either or neither.

#### 3. Config Chain (10 touchpoints)

All components from CLI to CheckpointController properly thread the profile path:

| Component | Field Name | Type |
|-----------|-----------|------|
| CLI / EngineArgs | `ft_checkpoint_cost_profile` | str |
| SchedulerConfig | `ft_checkpoint_cost_profile` | str |
| FTSchedulerConfig | `checkpoint_cost_profile` | str |
| FaultTolerantScheduler | `cost_profile_path` param | str |
| CheckpointController | `cost_profile_path` param | str |

Both FaultTolerantSchedulerImpl and BendersFTSchedulerImpl pass the config through.

#### 4. Experiment Integration

**File**: `experiments/run.py`

Server launch command now includes:
```bash
--ft-checkpoint-cost-profile <path>  # If provided in experiment config
```

This allows experiments to specify profile paths in YAML for Checkpoint-Only and Our-System baselines.

### Validation & Safeguards

1. **Fail-fast on placeholder data**: If `is_real_measurement=False`, CheckpointCostModel raises ValueError immediately during construction. This prevents silent fallback to broken data.

2. **Graceful fallback**: If profile path is empty or invalid, system falls back to linear model (if available) or legacy level-based checkpointing.

3. **No gate coupling**: Profile and linear inputs are independent. Profile path doesn't require throughput values, and vice versa.

### Test Coverage

**File**: `tests/ft/test_checkpoint_cost_model.py` (10 tests)

- ✅ Real profile loads successfully
- ✅ Placeholder profile rejected with clear error
- ✅ Interpolation correctness (T_prefill, T_load, T_ckpt)
- ✅ Decision rule (should_publish) correctness
- ✅ CheckpointController accepts profile
- ✅ CheckpointController rejects placeholder profile
- ✅ Fallback to linear model when no profile
- ✅ Economic policy unavailable without profile or linear inputs
- ✅ Profile takes precedence over linear

**Regression**: All 32 existing tests in test_ft_system.py pass unchanged.

### Baseline Impact

| Baseline | Affected | Reason |
|----------|----------|--------|
| Fixed-High (1 block) | No | Uses fixed blocks dispatch, never reaches economic policy |
| Fixed-Low (10 blocks) | No | Uses fixed blocks dispatch |
| Robust-Routing-Only | No | Uses fixed blocks dispatch |
| Checkpoint-Only | Yes | Walks economic policy path; profile preferred over linear |
| Our-System | Yes | Walks economic policy path; profile preferred over linear |

---

## Phase 2: Solver 对齐 Profile-Driven 成本模型

### Overview

Phase 1 把 CheckpointCostModel 接入了 runtime 的 CheckpointController，但 Benders solver（`CostTableBuilder`）还在用线性吞吐/带宽模型算成本。这导致 runtime 和 solver 对同一个状态的"要不要 checkpoint"判断不一致。

Phase 2 让 solver 的成本表也读同一个 profile JSON，替换掉线性近似。不改 checkpoint 触发逻辑（Phase 1 已完成），只改 solver 侧的成本估计。

### Implementation Details

#### 1. CostTableBuilder 加 cost_model 参数

**File**: `vllm/v1/core/sched/benders/cost_tables.py`

构造函数新增 `cost_model: CheckpointCostModel | None = None`，存为 `self._cost_model`。有 profile 时用插值，没有时走原有线性 fallback。

#### 2. 替换三处成本计算

在 `_compute_costs_raw()` 中，对 replay / load / checkpoint overhead 三处成本加 profile 分支：

**Replay cost**:
```python
if self._cost_model is not None and replay_tokens > 0:
    replay_time_sec = (
        self._cost_model.t_prefill(recovery_tokens)
        - self._cost_model.t_prefill(published_tokens)
    ) / 1000.0  # ms → sec
```

**Load cost**:
```python
if self._cost_model is not None and restore_bytes > 0:
    restore_time_sec = self._cost_model.t_load(restore_bytes) / 1000.0
```

**Checkpoint overhead**:
```python
if self._cost_model is not None and is_active:
    # Total cost = c0 (fixed) + t_ckpt(ΔS) (size-dependent)
    # checkpoint_ms_by_bytes in profile has c0 subtracted, must add back
    checkpoint_overhead_sec = (self._cost_model._c0 + self._cost_model.t_ckpt(delta_S)) / 1000.0
```

**关键细节**:
- CheckpointCostModel 返回毫秒，CostTableBuilder 内部用秒，需要 `/1000.0` 转换
- checkpoint overhead 必须加回 c0（profile 的 `checkpoint_ms_by_bytes` 存的是已减去 c0 的 size-dependent 部分）
- 三张插值表都有 `(0, 0.0)` 锚点，避免 `t_prefill(0)` clamp 到第一个测量点导致新请求 replay cost 被低估

#### 3. 按 Baseline 条件注入

**File**: `vllm/v1/core/sched/benders_ft_scheduler_impl.py`

只对 adaptive baselines（`fixed_checkpoint_blocks == 0`）启用 profile-driven 成本：

```python
solver_cost_model = (
    self._ft.checkpoint_controller._cost_model
    if sched_cfg.fixed_checkpoint_blocks == 0
    else None
)
```

这保证 `Robust-Routing-Only`（`fixed_checkpoint_blocks=10`）的 solver 继续用线性成本，不会被 profile 改变行为。

#### 4. 实验配置接入

**File**: `experiments/config.yaml`

在 adaptive baselines 中加入 profile path：
```yaml
Checkpoint-Only:
    ft_checkpoint_cost_profile: "experiments/checkpoint_cost_profile.json"

Our-System:
    ft_checkpoint_cost_profile: "experiments/checkpoint_cost_profile.json"
```

Fixed baselines（Fixed-High/Low, Robust-Routing-Only）和 No-FT 不加。

### Baseline Impact

| Baseline | Solver 成本模型 | 原因 |
|----------|----------------|------|
| Fixed-High/Low | 不受影响 | 不走 Benders solver |
| No-FT | 不受影响 | 不走 Benders solver |
| Robust-Routing-Only | 线性 fallback | `fixed_checkpoint_blocks=10` → cost_model=None |
| Checkpoint-Only | Profile-driven | `fixed_checkpoint_blocks=0` → 用 CheckpointCostModel |
| Our-System | Profile-driven | `fixed_checkpoint_blocks=0` → 用 CheckpointCostModel |

### Test Coverage

**File**: `tests/ft/test_checkpoint_cost_model.py` — 新增 4 个 solver 测试：

- ✅ Profile-driven replay cost 用插值而非线性
- ✅ Profile-driven load cost 用插值
- ✅ 无 profile 时 fallback 行为不变
- ✅ Pending request 用 T_prefill(prompt_len) 作为 replay cost

**Regression**: 全部 151 个 FT 测试通过。

---

## References

- Checkpoint decision rule: Adaptive checkpoint scheduling based on replay cost vs restore+overhead cost
- KVCheckpointPool: vLLM's in-memory checkpoint storage with save/restore operations
- Linear interpolation: Evaluate cost functions for arbitrary token counts / cache sizes
