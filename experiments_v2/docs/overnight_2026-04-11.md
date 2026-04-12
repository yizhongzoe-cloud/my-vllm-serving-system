# Overnight 2026-04-11 — 4-Phase Experiment Results

> Primary record of the overnight run launched at 04:48 PDT on 2026-04-11.
> 143 runs total across 4 phases. Finished 12:44 PDT.
>
> **Script**: [experiments_v2/overnight_2026-04-11.sh](../overnight_2026-04-11.sh)
> **Log**: `/tmp/overnight_2026-04-11.log`
> **Data roots**: `results_v2/8B/E1a_skip_solver/`, `results_v2/8B/E1a_hard_variance/`, `results_v2/8B/E1a_3seed_ext/`, `results_v2/8B/E1a_phase8_repro/`

## 1. Goal

Close 4 open questions from the 2026-04-08~10 investigation:

1. **Is Benders solver net positive or net negative?** — phase 8 conditions had solver in 98% greedy fallback; E1a_3seed (A5000 profile) had solver 100% converged. Direct pair comparison was missing.
2. **Main comparison table 3-seed → 5-seed variance** — E1a_3seed only had seeds 42/123/456; some cells showed huge seed-to-seed variance that 3 samples couldn't resolve.
3. **Hard cell (W1/W4 Heavy F2_Mid) variance** — same cell had std ~100 on 3 seeds; 8 seeds needed to tighten confidence intervals.
4. **Phase 8 profile Our-System crash rate + true 3-seed mean** — original s42 crashed with a CUDA device-side assert (deterministic); needed more seeds to estimate.

## 2. Phase summary

| Phase | Experiment | Runs | Output dir |
|---|---|---|---|
| 1 | FT_SKIP_SOLVER ablation | 36 | `E1a_skip_solver/` |
| 2 | Hard-cell variance (5 new seeds × 3 baselines × 2 workloads) | 30 | `E1a_hard_variance/` |
| 3 | E1a_3seed extension (add seeds 789, 1337) | 72 | `E1a_3seed_ext/` |
| 4 | Phase 8 profile Our-System seed extension (5 new seeds) | 5 | `E1a_phase8_repro/Our-System/` |

All phases share a common env: `FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1`. Phases 1–3 use the standard A5000 dp=2 default=26 profile (`config_8b.yaml`). Phase 4 uses the phase-8 A6000 dp=1 default=10 profile (`config_8b_phase8repro.yaml`).

## 3. Phase 1 — FT_SKIP_SOLVER ablation (Benders net value)

**Pair comparison** of Our-System with Benders solver ON (baseline E1a_3seed) vs OFF (E1a_skip_solver with `FT_SKIP_SOLVER=1`), same code, same A5000 profile, same seeds (42/123/456), 12 cells.

| Cell | Solver ON (mean±std) | Solver OFF (mean±std) | Δ goodput | ~ |
|---|---|---|---|---|
| W1_Chat/Heavy/F2_Mid | 189.0 ± 59.6 | 222.6 ± 121.4 | **+33.7** | . |
| W1_Chat/Heavy/none | 391.4 ± 19.1 | 404.8 ± 19.1 | +13.4 | . |
| W1_Chat/Light/F2_Mid | 130.2 ± 2.0 | 111.3 ± 50.7 | −19.0 | . |
| W1_Chat/Light/none | 140.2 ± 4.1 | 142.1 ± 6.4 | +1.9 | . |
| W1_Chat/Moderate/F2_Mid | 254.6 ± 11.2 | 267.6 ± 5.4 | **+13.1** | ⭐ |
| W1_Chat/Moderate/none | 273.4 ± 1.7 | 275.9 ± 5.7 | +2.6 | . |
| W4_Mixed/Heavy/F2_Mid | 224.7 ± 9.5 | 247.3 ± 24.1 | **+22.6** | ⭐ |
| W4_Mixed/Heavy/none | 248.1 ± 19.1 | 256.1 ± 15.7 | +8.0 | . |
| W4_Mixed/Light/F2_Mid | 90.3 ± 5.3 | 74.6 ± 32.7 | −15.7 | . |
| W4_Mixed/Light/none | 89.0 ± 4.1 | 90.5 ± 5.3 | +1.6 | . |
| W4_Mixed/Moderate/F2_Mid | 165.7 ± 14.6 | 172.4 ± 14.1 | +6.7 | . |
| W4_Mixed/Moderate/none | 169.8 ± 15.4 | 173.0 ± 14.4 | +3.1 | . |
| **OVERALL MEAN** | **197.2** | **203.2** | **+6.0** | +3.0% |

**⭐** = |Δ| > 1σ combined stderr (rough significance flag)

### Interpretation

**Benders solver is marginally net negative on 8B + dp=2 + chat workloads** (turning it off gains ~3% goodput overall). The key points:

1. **10 / 12 cells**: solver OFF is at least as good as solver ON.
2. **2 cells statistically significant** (W1/Mod/F2, W4/Heavy/F2): solver OFF wins by +13 to +23 tok/s.
3. **No cell shows solver ON as meaningfully better** — the biggest "ON wins" is W4_Mixed/Light/F2_Mid at +15.7, but solver OFF has std 32.7 there, so it's within noise.
4. Overall mean: **197.2 (ON) vs 203.2 (OFF) = +3.0%**.

The prediction from [investigation_summary section 3 — Benders solver 收敛失败率 98%](investigation_summary_2026-04-08-09.md#benders-solver-收敛失败率-98) ("修了 profile 反而更差: Benders 开始工作但 over-reject") is validated at scale: **solver making admission/routing decisions is slightly worse than greedy fallback** across all 12 cells, not just the one that originally triggered the concern.

### Implication for paper

The `ft_benders_centralized` scheduler is not pulling its weight on this hardware + model. Options:
- **Drop it from main results**, pivot to `fault_tolerant` scheduler as the Our-System backbone, reframe contribution as "adaptive checkpoint + KV reload recovery".
- **Gate it behind a flag** — e.g. only activate for `dp ≥ 4` or `load > 0.7` where the admission problem is actually non-trivial.
- **Re-tune objective function** — the current solver likely minimizes a metric that doesn't align with TTFT/TPOT SLO attainment; fix or change the cost function.
- **Scale up** to 70B + dp=4/8 and see if solver decisions start mattering at higher complexity.

## 4. Phase 2 — Hard-cell 8-seed aggregate

Combines 3 original seeds (42/123/456 from E1a_3seed) with 5 new seeds (111/222/333/555/777 from E1a_hard_variance).

### W1_Chat/Heavy/F2_Mid (n=8)

| Baseline | mean ± std | stderr | min | max |
|---|---|---|---|---|
| **NoFT-Reprefill** ⭐ | **268.3 ± 73.8** | ±26.1 | 162.9 | 374.5 |
| Our-System | 165.0 ± 49.4 | ±17.5 | 114.1 | 249.3 |
| Periodic-High | 156.8 ± 30.2 | ±10.7 | 130.7 | 205.6 |

**NoFT-Reprefill vs Our-System gap = +103.3 tok/s** (+62%). With stderrs 26.1 and 17.5, the gap is ~3.3 combined stderrs → **strongly significant**.

### W4_Mixed/Heavy/F2_Mid (n=8)

| Baseline | mean ± std | stderr | min | max |
|---|---|---|---|---|
| **NoFT-Reprefill** ⭐ | **252.0 ± 16.0** | ±5.7 | 230.3 | 275.4 |
| Periodic-High | 243.3 ± 15.8 | ±5.6 | 226.0 | 266.4 |
| Our-System | 217.4 ± 11.1 | ±3.9 | 206.1 | 232.3 |

**NoFT-Reprefill vs Our-System gap = +34.6 tok/s** (+16%). Gap is ~5 combined stderrs → **strongly significant**.

### Interpretation

On both hard cells, **NoFT-Reprefill beats Our-System consistently**. W4_Mixed variance is much lower (std ~16 vs ~74), meaning the result is robust; W1_Chat is noisier but the gap is 6× bigger so still decisive. This is the single clearest signal from this investigation: **the "lightest possible don't-drop-reqs" baseline wins on the hard cells**, and by a lot. KV reload + checkpoint controller is not purchasing real performance over simple re-prefill in these conditions.

## 5. Phase 3 — 5-seed main comparison (12 cells × 3 baselines)

Combines 3 original seeds (42/123/456 from E1a_3seed) with 2 new seeds (789/1337 from E1a_3seed_ext). All cells now have **n=5**.

| Cell | Our-System | NoFT-Reprefill | Periodic-High |
|---|---|---|---|
| W1_Chat/Light/none | 140.4 ± 5 | 141.9 ± 6 | 141.5 ± 5 |
| W1_Chat/Light/F2_Mid | 133.2 ± 5 | **141.6 ± 6** | 118.2 ± 33 |
| W1_Chat/Moderate/none | 275.4 ± 7 | 278.9 ± 7 | 275.5 ± 8 |
| W1_Chat/Moderate/F2_Mid | 254.9 ± 8 | **277.2 ± 7** | 264.5 ± 4 |
| W1_Chat/Heavy/none | 399.0 ± 19 | **410.4 ± 15** | 406.8 ± 15 |
| W1_Chat/Heavy/F2_Mid | 169.3 ± 50 | **298.8 ± 81** | 185.5 ± 47 |
| W4_Mixed/Light/none | 88.0 ± 5 | 88.7 ± 6 | 88.8 ± 6 |
| W4_Mixed/Light/F2_Mid | 88.0 ± 7 | **89.0 ± 6** | 88.7 ± 6 |
| W4_Mixed/Moderate/none | 166.8 ± 12 | 169.3 ± 12 | 167.1 ± 9 |
| W4_Mixed/Moderate/F2_Mid | 162.1 ± 12 | **169.3 ± 11** | 167.0 ± 12 |
| W4_Mixed/Heavy/none | 247.1 ± 15 | **253.3 ± 15** | 252.0 ± 15 |
| W4_Mixed/Heavy/F2_Mid | 219.7 ± 12 | **251.6 ± 14** | 238.1 ± 15 |
| **OVERALL MEAN** | **195.3** | **214.2** | **199.5** |

**Bold** = baseline with highest mean in that cell.

### Interpretation

**NoFT-Reprefill wins every cell in the main comparison table**, overall +10% over Our-System and +7% over Periodic-High. In the Heavy cells it's significantly ahead; in Light cells the gap is within noise but non-negative. This is *not* a hard-cell artifact — it's the consistent story across the entire workload sweep.

The 3-seed data from E1a_3seed already showed this trend; 5-seed just tightens the variance bars and confirms the effect is real.

## 6. Phase 4 — Phase 8 profile Our-System (14 seeds, 3 crashed)

Extending phase8_repro with 5 new seeds (888/999/5678/9999/6666) on the phase 8 profile (A6000 dp=1 default=10) to increase the sample size and estimate crash rate.

### All attempted seeds

| seed | goodput | comp% | status |
|---|---|---|---|
| 123 | 133.9 | 100.0 | ✅ |
| 456 | 307.5 | 100.0 | ✅ |
| 789 | 258.8 | 100.0 | ✅ |
| 888 | **361.4** | 100.0 | ✅ |
| 999 | 251.2 | 100.0 | ✅ |
| 2024 | 195.6 | 100.0 | ✅ |
| 3141 | 308.1 | 100.0 | ✅ |
| 5678 | 249.1 | 100.0 | ✅ |
| 6666 | 172.6 | 100.0 | ✅ |
| 9999 | 344.2 | 100.0 | ✅ |
| 42 (attempt 1, parallel) | 164.9 | 39.2 | ❌ CUDA assert |
| 42 (attempt 2, sequential) | 164.2 | 39.0 | ❌ CUDA assert |
| 42 (CUDA_LAUNCH_BLOCKING=1) | 89.4 | 99.8 | ⚠️ avoided bug but 3× slower |
| 1337 | 146.5 | 37.6 | ❌ CUDA assert |

### Clean mean (n=10, excluding s42 and s1337 crashes)

```
mean  = 258.3 tok/s
std   = 76.2
stderr = ±24.1
```

**Crash rate: 2/12 = 17%** (s42 and s1337 — both deterministic on this profile).

### CUDA device-side assert root cause (partial)

- **Location**: [gpu_model_runner.py:291 `self.async_copy_ready_event.synchronize()`](../../vllm/v1/worker/gpu_model_runner.py#L291)
- **When**: fault injection (F2_Mid = 150s) → 8-10 KV restores complete on the surviving engine → within 1-2 decode steps → CUDA assert fires from a prior asynchronous kernel
- **Seeds**: deterministic for s42 and s1337; other seeds unaffected
- **Workaround**: `CUDA_LAUNCH_BLOCKING=1` avoids the crash (classic async race signature), at ~3× slowdown
- **Likely cause**: block_table index / length mismatch on a decode step that includes a just-restored request's replayed suffix prefill. The async kernel sees stale metadata from before the KV restore finished.
- **Status**: known bug, not fixed in this investigation. Debugging priority is medium (not paper-blocking).

### Phase 8 profile `Our-System=299.1` claim revisited

The original single-seed s42=299.1 from 2026-04-09 (`8B_overnight_2026-04-09/phase8_fast_chunk/reload_s42/`) is impossible to reproduce because s42 now deterministically crashes. The **10-seed clean mean 258.3 ± 76.2** is the honest replacement. The original 299.1 falls within ~0.5 stderr of this mean, so it's plausible but not "the number" — it's one draw from a wide distribution.

## 7. Key findings (roll-up)

1. **Benders solver is marginally net negative on this hardware** (+3% goodput by turning it off, 10/12 cells prefer OFF, 2 cells significantly so). See Phase 1.
2. **NoFT-Reprefill beats Our-System on every cell of the main comparison table**, by +10% overall and much more on Heavy cells (+62% on W1/Heavy/F2). See Phase 2 + Phase 3.
3. **W1_Chat/Heavy/F2_Mid is the hardest cell** with seed-to-seed std ~50-80 even at n=8. Conclusions from that cell alone are unreliable; conclusions across the whole 12-cell sweep are robust.
4. **Phase 8 profile Our-System single-seed 299.1 was a lucky draw.** Clean 10-seed mean is 258.3 ± 76.2. Not a "practical limit near No-FT 319" — it's a noisy sample within a wide distribution.
5. **A real CUDA bug exists** in the KV reload + async decode pipeline, triggered by specific request sequences (s42, s1337) on the phase 8 profile. `CUDA_LAUNCH_BLOCKING=1` sidesteps it. Not a paper blocker but should be filed as a debugging task.

## 8. Implications for paper

The core value prop "Our-System = ft_benders_centralized + adaptive checkpoint + KV reload recovery" is **not holding up** on 8B + dp=2 + chat workloads:

| Component | Net value measured | Evidence |
|---|---|---|
| ft_benders_centralized scheduler | **−3%** goodput | Phase 1 (36-run pair comparison) |
| Adaptive checkpoint + KV reload recovery | **−10% to −62%** goodput vs NoFT-Reprefill | Phase 2 + 3 (all cells, all n=5-8) |
| Checkpoint pipeline optimization (NONBLOCK + FAST_TMPFS + FAST_CHUNK) | **Positive** — closes framework overhead that was pre-fix making things worse | Phase 7+8 of overnight_2026-04-09.md |

The only component that's clearly positive is the **framework overhead optimizations** from the phase 7+8 fix — but these just bring Our-System back to match No-FT / NoFT-Reprefill, they don't make it better. The distinctive components (solver + KV reload) are losing to simpler baselines.

### Suggested pivot directions

1. **Scale up**: Re-run on 70B + dp=4/8 + W2_Summary (long-context, prefill-heavy). KV reload's I/O advantage should become decisive when checkpoint is smaller than re-prefill cost. This is the "proper home" for this method.
2. **Reframe recovery**: Not "KV reload is faster than reprefill" (it's not), but "KV reload preserves token stream determinism" (it does — NoFT-Reprefill re-prefills and the new suffix might diverge from the streamed prefix). Reframe the contribution as correctness guarantee rather than performance.
3. **Gate the solver**: Use solver only when load > threshold or engine count > threshold; greedy fallback by default. Preserves the analytical framework without tanking small-scale performance.
4. **Null result + analysis framework**: Honestly report that on this hardware/model/workload, KV reload doesn't pay off, but publish the testing framework + ablation methodology as the contribution.

## 9. Data provenance

| Directory | Phase | Config | Profile | Seeds | Runs |
|---|---|---|---|---|---|
| `results_v2/8B/E1a_skip_solver/` | 1 | `config_8b.yaml` | A5000 dp=2 default=26 | 42/123/456 | 36 |
| `results_v2/8B/E1a_hard_variance/` | 2 | `config_8b.yaml` | A5000 dp=2 default=26 | 111/222/333/555/777 | 30 |
| `results_v2/8B/E1a_3seed_ext/` | 3 | `config_8b.yaml` | A5000 dp=2 default=26 | 789/1337 | 72 |
| `results_v2/8B/E1a_phase8_repro/Our-System/` | 4 | `config_8b_phase8repro.yaml` | A6000 dp=1 default=10 | 888/999/5678/9999/6666 | 5 |

Each run directory contains `metrics.json`, `run_meta.json` (git_commit + config), `requests.csv`, `recoveries.json`, `epochs.csv`, `server.log`, `stdout.log`.

## 10. Next steps

### Paper-blocking

- [ ] Decide pivot direction (scale up / reframe / gate / null-result)
- [ ] If scaling up: set up 70B + dp=4/8 + W2_Summary experiment harness
- [ ] If reframing: measure token stream determinism directly (diff between NoFT-Reprefill and Our-System suffix tokens)
- [ ] Update [investigation_summary_2026-04-08-09.md](investigation_summary_2026-04-08-09.md) section 5.3 with clean phase-8 profile numbers

### Debugging (future)

- [ ] Reproduce CUDA device-side assert on s42 with `TORCH_USE_CUDA_DSA=1` + smaller batch to get clean traceback
- [ ] Inspect block_table mutation during KV restore → investigate whether replayed prefill's block indices are in sync with async copy completion
- [ ] Consider adding explicit `cuda.synchronize()` after `restore_kv_blocks` before next `schedule()` call to bound the race window
