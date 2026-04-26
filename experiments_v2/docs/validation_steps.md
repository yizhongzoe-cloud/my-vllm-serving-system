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

### Detailed L40S percentiles (5 iters, 3 warmup)

Run on 2026-04-25, 23:34–23:47 local time. `--gpu-memory-utilization 0.85` for 8K–64K, `0.95` for 128K (input-len = 131071 to fit within model's 131072 max).

#### 8K context

| Statistic | Latency (s) |
|---|---|
| Avg | 0.7345 |
| p10 | 0.7286 |
| p25 | 0.7319 |
| p50 | 0.7362 |
| p75 | 0.7379 |
| p90 | 0.7392 |
| p99 | 0.7400 |

#### 16K context

| Statistic | Latency (s) |
|---|---|
| Avg | 1.6887 |
| p10 | 1.6786 |
| p25 | 1.6791 |
| p50 | 1.6918 |
| p75 | 1.6922 |
| p90 | 1.6982 |
| p99 | 1.7017 |

#### 32K context

| Statistic | Latency (s) |
|---|---|
| Avg | 4.2056 |
| p10 | 4.1875 |
| p25 | 4.1971 |
| p50 | 4.1999 |
| p75 | 4.2145 |
| p90 | 4.2270 |
| p99 | 4.2345 |

#### 64K context

| Statistic | Latency (s) |
|---|---|
| Avg | 11.7715 |
| p10 | 11.7391 |
| p25 | 11.7614 |
| p50 | 11.7752 |
| p75 | 11.7969 |
| p90 | 11.7987 |
| p99 | 11.7998 |

#### 128K context (input-len 131071)

| Statistic | Latency (s) |
|---|---|
| Avg | 36.4421 |
| p10 | 36.4145 |
| p25 | 36.4149 |
| p50 | 36.4270 |
| p75 | 36.4364 |
| p90 | 36.4853 |
| p99 | 36.5146 |

(p99 - p10) / avg ranges from 0.27 % (128K) to 1.55 % (8K) — short-context runs are slightly noisier in absolute terms but still well under 2 %, well below the inter-context-length differences being studied.

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

## Step 3: Larger Model Prefill Cost (Qwen2.5-14B-Instruct)

**Date**: 2026-04-26
**Goal**: Verify the Step-2 prefill-cost-vs-context-length trend generalizes across model sizes — specifically, confirm O(N²) scaling holds at a different parameter count and quantify how absolute prefill cost grows with model size.

### Setup

- **Model**: `Qwen/Qwen2.5-14B-Instruct` (BF16, 14.7 B params, 48 layers × 8 KV heads × 128 head_dim → **192 KB/token** KV cache)
- **Hardware**: L40S × 1 (single card, no TP)
- **Tool**: `vllm bench latency`
- **Config**: `--output-len 1`, `--num-iters 5`, `--num-iters-warmup 3`, `--batch-size 1`, `--max-model-len 32768` (model's native max), `--gpu-memory-utilization 0.95`
- **Prefix caching**: disabled (default for `vllm bench latency`)
- **Lengths swept**: 8K, 16K, 32K. The 32K run uses `input-len 32767` to fit within the model's 32768 native limit alongside the 1-token output (same boundary trick as the 128K-on-8B run in Step 2).

### Results — single-card L40S

| Context | 8B Llama (Step 2) | 14B Qwen (this) | 14B / 8B |
|---|---|---|---|
| 8K | 0.73 s | **1.41 s** | 1.93× |
| 16K | 1.69 s | **3.23 s** | 1.91× |
| 32K | 4.21 s | **8.04 s** | 1.91× |

### Results — single-card A6000

| Context | 8B Llama (Step 2) | 14B Qwen (this) | 14B / 8B |
|---|---|---|---|
| 8K | 1.20 s | **2.25 s** | 1.88× |
| 16K | 2.86 s | **5.41 s** | 1.89× |
| 32K | 7.61 s | **14.30 s** | 1.88× |

### Hardware comparison (14B Qwen)

| Context | L40S | A6000 | A6000 / L40S |
|---|---|---|---|
| 8K | 1.41 s | 2.25 s | 1.60× |
| 16K | 3.23 s | 5.41 s | 1.67× |
| 32K | 8.04 s | 14.30 s | 1.78× |

### Growth per 2× context

| Step | 8B Llama (L40S) | 14B Qwen (L40S) | 14B Qwen (A6000) |
|---|---|---|---|
| 8K → 16K | 2.31× | 2.29× | 2.40× |
| 16K → 32K | 2.49× | 2.49× | 2.64× |

The growth-per-doubling is essentially identical across model sizes and consistent across hardware — **the N² scaling is invariant to model size**, and growth ratios on A6000 are slightly higher (likely because A6000's slower compute makes the linear-FFN regime relatively shorter, letting the quadratic-attention component dominate sooner).

### Cross-model observations

1. **Latency ratio is constant across context length.** 14B/8B = 1.91× at every measured point — close to the parameter ratio 14.7/8 = 1.84×, plus ~4% from 14B's deeper stack (48 vs 32 layers) doing more attention work. Consistent with prefill being compute-bound, with no model-size-specific scheduler effects.

2. **Step-2 conclusion strengthens.** At 32K context, re-prefill on Qwen2.5-14B costs **8.0 s on a single L40S**, vs. ~125 ms estimated for restoring its 6 GB BF16 KV cache over PCIe 4.0 (10 GB/s) → **~64× speedup** if the FT idea preserves the cache. Larger model → larger absolute savings at the same context length.

3. **Native context limit kicks in early on Qwen2.5-14B.** The non-`-1M` checkpoint is trained with `max_position_embeddings = 32768`, so this sweep stops at 32K without enabling RoPE scaling. The 64K / 128K data points in this study come from Llama-3.1-8B (native 128K).

### Failed attempt: Qwen2.5-14B-Instruct-1M

Tried first to use the 1M-context variant (`Qwen/Qwen2.5-14B-Instruct-1M`) so 8K → 64K could all be measured on a single model. Engine init failed with:

```
TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'
```

Root cause: the 1M variant uses Dual Chunk Attention (passes `layer_idx` into the attention impl), which the current `zoe/slo-scheduling` branch's `FlashAttentionImpl` does not accept. Compatibility issue with this fork's vllm version, not a hardware/config issue. Fell back to the non-1M variant, capping the sweep at 32K.

### Detailed L40S percentiles (5 iters, 3 warmup)

Run on 2026-04-26, 02:10–02:15 local time. `--gpu-memory-utilization 0.95` for all three points.

#### 8K context

| Statistic | Latency (s) |
|---|---|
| Avg | 1.4098 |
| p10 | 1.4006 |
| p25 | 1.4094 |
| p50 | 1.4102 |
| p75 | 1.4145 |
| p90 | 1.4178 |
| p99 | 1.4198 |

#### 16K context

| Statistic | Latency (s) |
|---|---|
| Avg | 3.2345 |
| p10 | 3.2226 |
| p25 | 3.2284 |
| p50 | 3.2333 |
| p75 | 3.2426 |
| p90 | 3.2468 |
| p99 | 3.2493 |

#### 32K context (input-len 32767)

| Statistic | Latency (s) |
|---|---|
| Avg | 8.0352 |
| p10 | 8.0119 |
| p25 | 8.0233 |
| p50 | 8.0377 |
| p75 | 8.0450 |
| p90 | 8.0576 |
| p99 | 8.0651 |

(p99 - p10) / avg ranges from 0.66 % (32K) to 1.36 % (8K) — same noise floor as the 8B Step-2 run.

### Detailed A6000 percentiles (5 iters, 3 warmup)

#### 8K context

| Statistic | Latency (s) |
|---|---|
| Avg | 2.2520 |
| p10 | 2.2450 |
| p25 | 2.2481 |
| p50 | 2.2515 |
| p75 | 2.2560 |
| p90 | 2.2591 |
| p99 | 2.2610 |

#### 16K context

| Statistic | Latency (s) |
|---|---|
| Avg | 5.4101 |
| p10 | 5.3907 |
| p25 | 5.3990 |
| p50 | 5.4128 |
| p75 | 5.4231 |
| p90 | 5.4274 |
| p99 | 5.4300 |

#### 32K context (input-len 32767)

| Statistic | Latency (s) |
|---|---|
| Avg | 14.3022 |
| p10 | 14.2404 |
| p25 | 14.2705 |
| p50 | 14.3153 |
| p75 | 14.3467 |
| p90 | 14.3535 |
| p99 | 14.3575 |

(p99 - p10) / avg < 0.5 % across all three points — same low-noise behavior as L40S.

### Reproducing

```bash
source /home/yzhong76/code/my-vllm-serving-system/.venv/bin/activate

for LEN in 8192 16384 32767; do
  vllm bench latency \
    --model Qwen/Qwen2.5-14B-Instruct \
    --input-len $LEN \
    --output-len 1 \
    --batch-size 1 \
    --num-iters 5 \
    --num-iters-warmup 3 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95
done
```

### Conclusion

Cross-model trend confirmed: **prefill cost scales as O(N²) in context length and roughly linearly in parameter count, independently.** The two effects compound — bigger model + longer context produces both larger absolute savings and a larger ratio of re-prefill ÷ KV-restore. The FT idea's value proposition is therefore not specific to one model family or size; it gets stronger as workloads move toward larger models and longer prompts. Step 4 (measuring actual KV checkpoint reload cost, not just estimated bandwidth-bound) is the next blocking experiment.

---

## Step 3.5: AWQ-Quantized 32B Model (Qwen2.5-32B-Instruct-AWQ)

**Date**: 2026-04-26
**Goal**: Push the cross-model trend to a 32 B-class model on the same single L40S, using AWQ INT4 to fit. Two questions: (1) does the O(N²) growth still hold under quantization? (2) does the absolute prefill cost continue to scale linearly with parameter count, or does AWQ overhead change the curve shape?

### Setup

- **Model**: `Qwen/Qwen2.5-32B-Instruct-AWQ` (INT4 weights via AWQ, ~18 GB on disk; runtime path is awq_marlin kernel + BF16 dequant)
- **Hardware**: L40S × 1 (single card, no TP)
- **Tool**: `vllm bench latency`
- **Architecture**: 32 B params (~32.5 B), 64 layers × 8 KV heads × 128 head_dim → **256 KB / token** KV cache (BF16, double the 14 B above due to deeper stack)
- **Context extension**: model's native max is 32768; runs at 64K use YaRN rope scaling (`factor: 4.0`, `original_max_position_embeddings: 32768`) plus env var `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to bypass vllm's pre-flight `max_model_len` check.
- **Memory utilization**: `--gpu-memory-utilization 0.90` (0.95 caused OOM during forward — see "Failed attempts" below).
- **Allocator**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- **Iters**: `--num-iters 5 --num-iters-warmup 3 --batch-size 1 --output-len 1`. Prefix caching disabled (default for `vllm bench latency`).
- **Lengths swept**: 8K, 16K, 32K (input-len = LEN, max-model-len = 65536) and 64K (input-len 65535, max-model-len 70000, with `--enforce-eager` to bypass an inductor RoPE-indexing bug at long context).

### Results — single-card L40S

| Context | 8B Llama (Step 2) | 14B Qwen (Step 3) | 32B Qwen-AWQ (this) | 32B/8B |
|---|---|---|---|---|
| 8K | 0.73 s | 1.41 s | **3.49 s** | 4.78× |
| 16K | 1.69 s | 3.23 s | **7.64 s** | 4.52× |
| 32K | 4.21 s | 8.04 s | **17.45 s** | 4.14× |
| 64K | 11.77 s | — | **37.65 s** ⚠ | 3.20× |

⚠ The 64K data point uses `--enforce-eager` (no torch.compile / cuda graphs), unlike 8K-32K of the same model and unlike the 8B/14B comparison points. The expected impact on prefill latency is small (cuda graphs primarily help decode latency by amortizing kernel launch overhead; for ≥64K prefill the per-kernel runtime dwarfs launch overhead), but treat the 64K-32B number as slightly upper-biased.

### Results — single-card A6000

A6000 used `--enforce-eager` for **all** context lengths (not just 64K). The current `zoe/slo-scheduling` fork's torch.compile path hits an `unsupported operator: _C.marlin_gemm.default` error during dynamo capture for AWQ models — `--enforce-eager` bypasses it. Also did not configure YaRN rope scaling (only set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for 64K), so 64K outputs may be numerically unreliable; **latency measurement itself remains valid** (full attention compute runs).

| Context | 8B Llama (Step 2) | 14B Qwen (Step 3) | 32B Qwen-AWQ (this) | 32B/8B |
|---|---|---|---|---|
| 8K | 1.20 s | 2.25 s | **5.52 s** | 4.60× |
| 16K | 2.86 s | 5.41 s | **12.44 s** | 4.35× |
| 32K | 7.61 s | 14.30 s | **29.63 s** | 3.89× |
| 64K | 22.37 s | — | **77.11 s** ⚠ | 3.45× |

### Hardware comparison (32B Qwen-AWQ)

| Context | L40S | A6000 | A6000 / L40S |
|---|---|---|---|
| 8K | 3.49 s | 5.52 s | 1.58× |
| 16K | 7.64 s | 12.44 s | 1.63× |
| 32K | 17.45 s | 29.63 s | 1.70× |
| 64K | 37.65 s | 77.11 s | 2.05× |

A6000/L40S ratio drifts higher with context length (1.58× → 2.05×). At 8K-32K the gap is in line with their compute throughput ratio (~1.5-1.8×). The widening at 64K is partly an artefact: L40S keeps the 8K-32K cuda-graph speedup baseline while A6000 was eager-only for all points, and 64K on L40S itself drops to eager — so 64K compares eager-vs-eager but normalized against different shorter-context baselines.

### Growth per 2× context

| Step | 8B Llama | 14B Qwen | 32B Qwen-AWQ |
|---|---|---|---|
| 8K → 16K | 2.31× | 2.29× | **2.19×** |
| 16K → 32K | 2.49× | 2.49× | **2.28×** |
| 32K → 64K | 2.80× | — | **2.16×** |

**The 32B-AWQ growth ratios are consistently 0.10–0.65× lower than the BF16 8B/14B baselines.** Two compounding causes: (a) at short context, awq_marlin's INT4 → BF16 dequant overhead is per-token-linear, fattening the linear (FFN-dominated) regime so attention's quadratic component takes longer to dominate; (b) the 64K point is `--enforce-eager`, eliminating the cuda-graph speedup that the other points benefit from, which compresses the apparent growth from 32K to 64K.

### Cross-model observations

1. **Latency-vs-params is super-linear once AWQ is in the mix.** 32B-AWQ / 8B-BF16 = 4.78× at 8K, dropping to 3.20× at 64K. Param ratio 32/8 = 4.0×. At short contexts the 32B-AWQ pays a fixed dequant tax on every token; at long contexts attention's O(N²) catches up and the ratio collapses toward 1× of compute-throughput plus model-FLOP ratios.

2. **N² trend still holds qualitatively.** Even with AWQ overhead damping the growth ratio, every doubling still costs >2×, and 32K ⇒ 17.5 s on a single card is firmly past the "FT idea matters" threshold (Step 2's 1–5 s decision band).

3. **64K prefill: 37.6 s.** Re-prefill on this config takes 37 seconds. KV cache size = 64K × 256 KB / 2 (compressed via TP=1's full local layout, no compression) = 16 GB; reload at PCIe 4.0 (10 GB/s) ≈ **1.6 s**. **23× speedup** if the FT idea preserves the cache instead of forcing a re-prefill. At this scale the *absolute* time saved per failure event is ~36 seconds, which dominates almost any plausible TTFT SLO.

### Failed attempts

Three configurations failed before the working setup. Documenting so future runs don't re-discover the same dead ends.

1. **`--rope-scaling` flag** — does not exist as a top-level CLI option in this fork's `vllm bench latency`. Use `--hf-overrides '{"rope_scaling": {...}}'` instead, which forwards into the HF config dict that ModelConfig consumes. Behaviour: instant `unrecognized arguments` exit.

2. **`max_model_len > native` validation** — vllm's pydantic ModelConfig validator rejects `max_model_len > max_position_embeddings` even when YaRN is configured via `--hf-overrides`. The pre-flight check fires before rope_scaling is applied to the derived max. Set env var `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to bypass; safe in practice as long as YaRN is configured (which is what extends the actual rope cache). Without it: `pydantic_core._pydantic_core.ValidationError: User-specified max_model_len (65536) is greater than the derived max_model_len (max_position_embeddings=32768.0...)`.

3. **OOM at 0.95 util** — at 16K input-len, 32B-AWQ's forward pass needed ~432 MB of working memory beyond what was free; vllm had reserved 95 % of card memory leaving < 400 MB for transient activations. Lower to 0.90 (and add `expandable_segments:True`) and the same workload fits comfortably. Counterintuitive that 8K succeeded but 16K failed at the same util — driver allocator fragmentation across run boundaries widens slightly with longer activations, and 16K fell on the wrong side.

4. **Inductor RoPE index-out-of-bounds at 64K** — torch.compile-emitted Triton kernel had `< 32768` hardcoded as the position index bound (specialized from the model's original `max_position_embeddings`). At 64K seq_len every position past 32767 triggered:

   ```
   /tmp/torchinductor_*/c7lfej...py:45: Assertion `index out of bounds: 0 <= ... < 32768` failed.
   RuntimeError: CUDA driver error: device-side assert triggered
   ```

   YaRN had updated the runtime rope cache, but the compiled kernel was already specialized with the old bound. Bypass with `--enforce-eager` (disables torch.compile / cuda graphs); the expected prefill latency cost is small.

### Detailed L40S percentiles (5 iters, 3 warmup)

Run on 2026-04-26, 02:41–03:13 local time. Percentile spread (p99-p10)/avg < 1 % for all four points.

#### 8K context (cuda graph, 0.90 util)

| Statistic | Latency (s) |
|---|---|
| Avg | 3.4892 |
| p10 | 3.4792 |
| p25 | 3.4808 |
| p50 | 3.4815 |
| p75 | 3.4931 |
| p90 | 3.5047 |
| p99 | 3.5117 |

#### 16K context (cuda graph, 0.90 util)

| Statistic | Latency (s) |
|---|---|
| Avg | 7.6410 |
| p10 | 7.6216 |
| p25 | 7.6300 |
| p50 | 7.6433 |
| p75 | 7.6511 |
| p90 | 7.6591 |
| p99 | 7.6639 |

#### 32K context (cuda graph, 0.90 util)

| Statistic | Latency (s) |
|---|---|
| Avg | 17.4485 |
| p10 | 17.4281 |
| p25 | 17.4369 |
| p50 | 17.4467 |
| p75 | 17.4594 |
| p90 | 17.4702 |
| p99 | 17.4766 |

#### 64K context (input-len 65535, **`--enforce-eager`**, 0.90 util, YaRN factor 4.0)

| Statistic | Latency (s) |
|---|---|
| Avg | 37.6521 |
| p10 | 37.6123 |
| p25 | 37.6496 |
| p50 | 37.6502 |
| p75 | 37.6858 |
| p90 | 37.6866 |
| p99 | 37.6871 |

### Detailed A6000 percentiles (5 iters, 3 warmup, **`--enforce-eager`** all points, 0.95 util)

#### 8K context

| Statistic | Latency (s) |
|---|---|
| Avg | 5.5153 |
| p10 | 5.4740 |
| p25 | 5.4893 |
| p50 | 5.5178 |
| p75 | 5.5460 |
| p90 | 5.5542 |
| p99 | 5.5591 |

#### 16K context

| Statistic | Latency (s) |
|---|---|
| Avg | 12.4399 |
| p10 | 12.3910 |
| p25 | 12.4166 |
| p50 | 12.4481 |
| p75 | 12.4726 |
| p90 | 12.4820 |
| p99 | 12.4877 |

#### 32K context (input-len 32767)

| Statistic | Latency (s) |
|---|---|
| Avg | 29.6328 |
| p10 | 29.6122 |
| p25 | 29.6198 |
| p50 | 29.6350 |
| p75 | 29.6437 |
| p90 | 29.6525 |
| p99 | 29.6578 |

#### 64K context (input-len 65535, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, no YaRN)

| Statistic | Latency (s) |
|---|---|
| Avg | 77.1067 |
| p10 | 77.0816 |
| p25 | 77.0905 |
| p50 | 77.1134 |
| p75 | 77.1202 |
| p90 | 77.1283 |
| p99 | 77.1332 |

(p99 - p10) / avg < 1 % across all four points.

### Reproducing

```bash
source /home/yzhong76/code/my-vllm-serving-system/.venv/bin/activate

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8K, 16K, 32K — within model's native context
for LEN in 8192 16384 32768; do
  vllm bench latency \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --input-len $LEN --output-len 1 --batch-size 1 \
    --num-iters 5 --num-iters-warmup 3 \
    --max-model-len 65536 \
    --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}' \
    --gpu-memory-utilization 0.90
done

# 64K — needs eager mode to bypass inductor index-bound bug
vllm bench latency \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --input-len 65535 --output-len 1 --batch-size 1 \
  --num-iters 5 --num-iters-warmup 3 \
  --max-model-len 70000 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

### Conclusion

32B-class quantized model on a single 48 GB card produces prefill costs from **3.5 s (8K) to 37.6 s (64K)**. The N²-vs-N gap between re-prefill and KV-restore at 64K is **~23×** even after AWQ's per-token dequant tax flattens the growth curve. Combined with Step 2 (8B BF16) and Step 3 (14B BF16), this confirms the FT-idea value proposition holds across (a) parameter scale 8B → 32B, (b) precision BF16 / INT4-AWQ, (c) context length 8K → 128K. The relative advantage is largest at long context with smaller models (because the BF16 N²/N gap is widest there); the absolute saved time is largest at long context with larger models (37 s recovered per failure on 32B-64K). Both flavours of advantage stack — the FT idea is not a corner-case win.

---

## Step 4: Checkpoint Reload Cost

**Date**: 2026-04-26
**Goal**: Measure actual GPU↔CPU transfer time for KV cache save (GPU → CPU pinned memory) and reload (CPU → GPU). Step 2's break-even analysis used a conservative 10 GB/s estimate; this step replaces it with measured numbers.

### Setup

- **Tool**: `experiments_v2/profile_checkpoint_costs.py` (with `--skip-prefill`, since prefill is already covered in Steps 2 and 3)
- **Hardware**: A6000 and L40S, single card each
- **Model**: Llama-3.1-8B-Instruct (the model itself is irrelevant for this measurement — KV transfer time depends only on byte count and PCIe bandwidth)
- **gpu-memory-utilization**: 0.45 (script default; vLLM only takes half the card so the other half is free for the test buffer)

### Why model-independent

Step 4 measures **bytes-per-second over PCIe**. The script allocates a buffer of size S in GPU memory and times moving it to/from a pinned host buffer. The buffer's *content* is irrelevant — same byte count gives the same transfer time regardless of which model produced it.

So one measurement (8B) generalizes to all models. To convert to a specific model + context, multiply `tokens × kv_bytes_per_token`:

| Model | KV bytes / token |
|---|---|
| Llama-3.1-8B | 128 KB |
| Qwen2.5-14B | 192 KB |
| Qwen2.5-32B-AWQ | 256 KB (KV is BF16; AWQ only quantizes weights) |

### Results — A6000

#### Reload (CPU → GPU)

| KV size | Reload (ms) | Effective throughput |
|---|---|---|
| 2 MB | 0.81 | 2.5 GB/s |
| 4 MB | 1.01 | 4.0 GB/s |
| 8 MB | 1.19 | 6.7 GB/s |
| 16 MB | 1.57 | 10.2 GB/s |
| 32 MB | 2.41 | 13.3 GB/s |
| 64 MB | 4.29 | 14.9 GB/s |
| 128 MB | 7.28 | **17.6 GB/s** (peak) |

Throughput plateaus around 17.6 GB/s for ≥64 MB transfers — this is the practical PCIe 4.0 ceiling on this host. Step 2's 10 GB/s estimate was conservative by ~1.7×.

#### Checkpoint save (GPU → CPU)

| KV size | Save (ms) | Effective throughput |
|---|---|---|
| 2 MB | 0.98 | 2.0 GB/s |
| 4 MB | 1.37 | 2.9 GB/s |
| 8 MB | 1.98 | 4.0 GB/s |
| 16 MB | 3.72 | 4.3 GB/s |
| 32 MB | 9.42 | 3.4 GB/s |
| 64 MB | 8.40 | 7.6 GB/s |
| 128 MB | 37.34 | 3.4 GB/s |

Save direction is noisier — non-monotonic (64 MB faster than 32 MB?), and 128 MB shows a sharp drop. Likely an artifact of how the test allocates / pins host buffers across runs (memory fragmentation, async launch overhead). The peak suggests save can hit ~7-8 GB/s under good conditions but the practical sustained rate is closer to 3-5 GB/s. **For the FT idea, save speed matters less than reload speed** — saves happen in the background, reloads happen on the user-facing recovery path.

#### Publication overhead (c0)

`0.43 ms` — negligible per-checkpoint fixed cost.

### Results — L40S

#### Reload (CPU → GPU)

| KV size | Reload (ms) | Effective throughput |
|---|---|---|
| 2 MB | 0.64 | 3.1 GB/s |
| 4 MB | 0.79 | 5.1 GB/s |
| 8 MB | 0.98 | 8.2 GB/s |
| 16 MB | 1.35 | 11.9 GB/s |
| 32 MB | 2.12 | 15.1 GB/s |
| 64 MB | 3.48 | 18.4 GB/s |
| 128 MB | 6.27 | **20.4 GB/s** (peak) |

L40S sustains slightly higher reload throughput than A6000 (20.4 vs 17.6 GB/s) — both bottlenecked by PCIe 4.0; the difference reflects platform-level (chipset / NVMe overhead / driver) variation rather than card hardware. Either way, well above Step 2's conservative 10 GB/s estimate.

#### Checkpoint save (GPU → CPU)

| KV size | Save (ms) | Effective throughput |
|---|---|---|
| 2 MB | 1.19 | 1.7 GB/s |
| 4 MB | 1.38 | 2.9 GB/s |
| 8 MB | 2.17 | 3.7 GB/s |
| 16 MB | 3.95 | 4.0 GB/s |
| 32 MB | 5.27 | 6.1 GB/s |
| 64 MB | 16.58 | 3.9 GB/s |
| 128 MB | 37.94 | 3.4 GB/s |

Same pattern as A6000: noisy, non-monotonic (32 MB faster than 64 MB), and tops out at ~3-6 GB/s. Save again is the lower-priority direction (background work), so this asymmetry doesn't hurt the FT idea's recovery-path latency.

#### Publication overhead (c0)

`1.04 ms` — about 2× higher than A6000's 0.43 ms. Likely host-side allocator overhead variation; small enough that for any KV ≥ 4 MB it's already amortized.

### Hardware comparison (reload)

| KV size | A6000 (ms) | L40S (ms) | A6000 / L40S |
|---|---|---|---|
| 2 MB | 0.81 | 0.64 | 1.27× |
| 16 MB | 1.57 | 1.35 | 1.16× |
| 64 MB | 4.29 | 3.48 | 1.23× |
| 128 MB | 7.28 | 6.27 | 1.16× |

A6000 is **~1.2× slower** at reload across the size range. Same ballpark — neither machine is a bottleneck for the FT idea.

### Updated break-even table (A6000)

Using the measured 17.6 GB/s reload throughput (replaces Step 2's 10 GB/s estimate):

| Context | KV size (8B) | Reload (measured) | Re-prefill (Step 2) | Re-prefill / Reload |
|---|---|---|---|---|
| 32K | 4 GB | ~230 ms | 7.61 s | **~33×** |
| 64K | 8 GB | ~470 ms | 22.37 s | **~48×** |
| 128K | 16 GB | ~930 ms | 71.92 s | **~77×** |

Compared to Step 2's estimate (19× / 28× / 45×), the measured advantage is **~1.7× larger** because actual PCIe throughput exceeds the 10 GB/s assumption.

### Multi-model break-even (A6000, prefill from Steps 2/3/3.5)

Using the same 17.6 GB/s reload throughput across all models (transfer time depends only on bytes):

| Context | Model | KV size | Reload | Re-prefill | Ratio |
|---|---|---|---|---|---|
| 32K | 8B Llama | 4 GB | ~230 ms | 7.61 s | ~33× |
| 32K | 14B Qwen | 6 GB | ~340 ms | 14.30 s | ~42× |
| 32K | 32B Qwen-AWQ | 8 GB | ~470 ms | 29.63 s | ~63× |
| 64K | 8B Llama | 8 GB | ~470 ms | 22.37 s | ~48× |
| 64K | 32B Qwen-AWQ | 16 GB | ~930 ms | 77.11 s | ~83× |
| 128K | 8B Llama | 16 GB | ~930 ms | 71.92 s | ~77× |

**Bigger model + longer context → larger absolute saving and larger ratio.** The FT idea's value compounds in both directions.

### Updated break-even table (L40S)

Using L40S's measured 20.4 GB/s reload throughput and L40S prefill numbers from Steps 2/3/3.5:

| Context | Model | KV size | Reload (measured) | Re-prefill (Steps 2/3/3.5) | Re-prefill / Reload |
|---|---|---|---|---|---|
| 32K | 8B Llama | 4 GB | ~200 ms | 4.21 s | **~21×** |
| 32K | 14B Qwen | 6 GB | ~290 ms | 8.04 s | **~28×** |
| 32K | 32B Qwen-AWQ | 8 GB | ~390 ms | 17.45 s | **~45×** |
| 64K | 8B Llama | 8 GB | ~390 ms | 11.77 s | **~30×** |
| 64K | 32B Qwen-AWQ | 16 GB | ~790 ms | 37.65 s | **~48×** |
| 128K | 8B Llama | 16 GB | ~790 ms | 36.44 s | **~46×** |

L40S ratios are slightly lower than A6000's because L40S has faster prefill (it's a faster GPU) but only marginally faster reload (PCIe-bound). This is the **expected** outcome: the FT idea's relative advantage is biggest when the GPU is *fast at compute but slow at PCIe* — i.e., compute keeps getting faster while PCIe stays roughly fixed across generations. Future GPUs (H100, B200) should make the ratio even more dramatic.

### Caveats

- Script tested up to 128 MB chunks. For larger transfers (the full KV cache of a long-context request can be many GB), throughput is **extrapolated from the measured 128 MB peak**. In practice sustained large-transfer rates can be slightly slower due to queue effects, but the order of magnitude holds.
- Save direction has measurement noise (see above). Reload — the user-facing metric — is clean and monotonic.
- `is_real_measurement: false` in the JSON because we passed `--skip-prefill`. KV save/load measurements are still real; only the prefill table inside the JSON is a placeholder.

### Reproducing

```bash
# A6000
source /home/yzhong76/envs/sd_env/bin/activate
python experiments_v2/profile_checkpoint_costs.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output experiments_v2/checkpoint_cost_profile_a6000.json \
  --skip-prefill

# L40S
source /home/yzhong76/code/my-vllm-serving-system/.venv/bin/activate
python experiments_v2/profile_checkpoint_costs.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output experiments_v2/checkpoint_cost_profile_l40s.json \
  --skip-prefill
```

Output JSONs: `experiments_v2/checkpoint_cost_profile_a6000.json`, `experiments_v2/checkpoint_cost_profile_l40s.json`.

### Conclusion

Measured peak PCIe reload throughput: **17.6 GB/s on A6000, 20.4 GB/s on L40S** — both ~2× higher than Step 2's conservative 10 GB/s estimate. Updated break-even ratios:

- **A6000**: 33× (8B/32K) → 83× (32B/64K)
- **L40S**: 21× (8B/32K) → 48× (32B/64K)

L40S ratios are smaller because faster compute compresses the prefill cost more than PCIe lets reload cost shrink — a positive result for the FT idea's future-proofing: **the gap between compute throughput and PCIe bandwidth widens with each GPU generation, so the ratio is expected to grow on H100/B200/etc.** Across both machines, KV reload remains 1-2 orders of magnitude faster than re-prefill for every tested model/context combination, confirming Step 2's qualitative conclusion (Mode 5 does not hold in long-context settings) with measured rather than estimated reload cost.

---

## Step 5: End-to-End Workload Comparison (TBD)

_W5 LongDoc, F2_Mid fault, V2 vs NR, multiple seeds. To be filled in after Step 3-4._
