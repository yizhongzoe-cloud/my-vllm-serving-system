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

## Phase 1: [To be added...]

---

## References

- Checkpoint decision rule: Adaptive checkpoint scheduling based on replay cost vs restore+overhead cost
- KVCheckpointPool: vLLM's in-memory checkpoint storage with save/restore operations
- Linear interpolation: Evaluate cost functions for arbitrary token counts / cache sizes
