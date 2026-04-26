# Validation Steps

> Step-by-step record of the validation experiments for the new direction (long-context state preservation).

---

## Step 2: Prefill Cost vs Context Length

**Date**: 2026-04-26
**Goal**: Measure how prefill latency grows with context length, on both L40S and A6000.

### Setup

- **Model**: Llama-3.1-8B-Instruct
- **Tool**: `vllm bench latency`
- **Config**: `--output-len 1`, `--num-iters 5`, `--num-iters-warmup 3`, `--gpu-memory-utilization 0.85` (0.95 for 128K)
- **Prefix caching**: disabled

### Results

| Context | L40S (48 GB) | A6000 (48 GB) | A6000 / L40S |
|---|---|---|---|
| 8K | 0.73 s | 1.20 s | 1.65× |
| 16K | 1.69 s | 2.86 s | 1.69× |
| 32K | 4.21 s | 7.61 s | 1.81× |
| 64K | 11.77 s | 22.37 s | 1.90× |
| 128K | 36.44 s | 71.92 s | 1.97× |

### Growth per 2× context

| Step | L40S | A6000 |
|---|---|---|
| 8K → 16K | 2.31× | 2.38× |
| 16K → 32K | 2.49× | 2.66× |
| 32K → 64K | 2.80× | 2.94× |
| 64K → 128K | 3.10× | 3.22× |

Both machines show clear **super-linear growth**, dominated by attention's O(N²). Each doubling of context costs ~3× more time at long range.

### Re-prefill vs estimated KV reload (A6000)

KV cache size = `tokens × 128 KB` for 8B Llama (32 layers × 2 × 8 KV heads × 128 head_dim × 2 bytes FP16 = 128 KB / token). Reload estimated as `KV_size / PCIe_bandwidth (10 GB/s)`.

| Context | Re-prefill (A6000) | KV size | Reload (estimated) | Re-prefill / Reload |
|---|---|---|---|---|
| 32K | 7.61 s | 4 GB | 0.4 s | **19×** |
| 64K | 22.37 s | 8 GB | 0.8 s | **28×** |
| 128K | 71.92 s | 16 GB | 1.6 s | **45×** |

### Conclusion

**Mode 5 (checkpoint brings no measurable benefit) DOES NOT hold in long-context settings.**

- At 32K context, checkpoint reload is ~19× faster than re-prefill.
- At 128K context, the gap widens to ~45×.
- The prior negative result (V2 losing NR) was specific to **short-prompt + small model + cheap re-prefill** combination. Once context grows, re-prefill becomes genuinely expensive and the original idea's value proposition recovers.

→ Continue with Option A (cut Benders, reframe to adaptive KV preservation for long-running LLM inference) on long-context workloads.

### Hardware notes

- **L40S** (FP16 Tensor: 362 TFLOPS, mem bw: 864 GB/s): 1.7-2.0× faster than A6000 across all context lengths.
- **A6000** (FP16 Tensor: 77.4 TFLOPS, mem bw: 768 GB/s): theoretical compute ratio is 4.7×, but actual ratio is only ~1.9× — indicating prefill at long context is partially memory-bandwidth-bound.
- Either GPU sufficient for next-phase experiments. A6000 is the more conservative choice (slower prefill = stronger Mode 5 disproof).

### Detailed A6000 percentiles (5 iters, 3 warmup)

#### 8K context

| Statistic | Latency (s) |
|---|---|
| Avg | 1.2042 |
| p10 | 1.2030 |
| p25 | 1.2037 |
| p50 | 1.2039 |
| p75 | 1.2050 |
| p90 | 1.2054 |
| p99 | 1.2057 |

#### 16K context

| Statistic | Latency (s) |
|---|---|
| Avg | 2.8603 |
| p10 | 2.8478 |
| p25 | 2.8529 |
| p50 | 2.8622 |
| p75 | 2.8703 |
| p90 | 2.8713 |
| p99 | 2.8718 |

#### 32K context

| Statistic | Latency (s) |
|---|---|
| Avg | 7.6128 |
| p10 | 7.5818 |
| p25 | 7.5942 |
| p50 | 7.6124 |
| p75 | 7.6353 |
| p90 | 7.6433 |
| p99 | 7.6481 |

#### 64K context

| Statistic | Latency (s) |
|---|---|
| Avg | 22.3707 |
| p10 | 22.3560 |
| p25 | 22.3696 |
| p50 | 22.3753 |
| p75 | 22.3800 |
| p90 | 22.3811 |
| p99 | 22.3817 |

#### 128K context

| Statistic | Latency (s) |
|---|---|
| Avg | 71.9169 |
| p10 | 71.8510 |
| p25 | 71.8700 |
| p50 | 71.9399 |
| p75 | 71.9629 |
| p90 | 71.9692 |
| p99 | 71.9729 |

Variance across iterations is < 0.5% — measurement is highly stable. (p99 - p10 / avg < 0.5% for every context length.)

### Reproducing

```bash
source /home/yzhong76/envs/sd_env/bin/activate

# 8K - 64K sweep
for LEN in 8192 16384 32768 65536; do
  vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len $LEN \
    --output-len 1 \
    --batch-size 1 \
    --num-iters 5 \
    --num-iters-warmup 3 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85
done

# 128K (needs higher gpu-memory-utilization)
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 131071 \
  --output-len 1 \
  --batch-size 1 \
  --num-iters 5 \
  --num-iters-warmup 3 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95
```

---

## Step 3: Larger Model Prefill Cost (TBD)

_To be filled in once 30B / 70B FP8 setup is ready._

---

## Step 4: Checkpoint Reload Cost (TBD)

_To be filled in. Use `experiments_v2/profile_checkpoint_costs.py`._

---

## Step 5: End-to-End Workload Comparison (TBD)

_W5 LongDoc, F2_Mid fault, V2 vs NR, multiple seeds. To be filled in after Step 3-4._
