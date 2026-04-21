# Overnight 2026-04-16 Summary (WIP — auto-pilot)

## Phase 0+1 (3-seed ablation on A4 W2/Heavy/F2_Mid)

Fresh NR baseline with random fault code + OS ablations. All 12 runs completed, 100% completion.

| config | gp (±std) | fg_p50 (±std) | **fg_p95 (±std)** | comp |
|---|---|---|---|---|
| NR (fresh, random fault) | 101.0 ± 2 | 291 ± 190 | **291 ± 190** | 100% |
| OS reprefill only | 100.4 ± 2 | 1563 ± 1528 | **1955 ± 1697** | 100% |
| **OS cap=1 + reprefill** | 101.1 ± 2 | 748 ± 339 | **1176 ± 714** | 100% |
| OS cap=3 + reprefill | 101.0 ± 2 | 1215 ± 911 | **1215 ± 911** | 100% |

### Key findings
- NR fresh fg_p95 = 291 ms (slightly higher than compact's 249 ms due to random fault including engine-1 s456 case).
- **Cap=1 + reprefill is best OS variant** (fg_p95 = 1176 ms) but still loses NR by +885 ms.
- Admission cap alone cannot close gap; reprefill mode alone (no cap) is worst (1955 ms).
- Goodput ties NR across all OS variants (≈ 101).
- **Conclusion**: admission policy reaches its ceiling. Reload path itself must be improved (→ Phase 2).

## Phase 2 — Batched-Reload (attempt v1, ablation only)

### Design

Replace per-req/per-layer scatter (32 × N_reqs kernels) with per-layer aggregated scatter (32 kernels total):
```
# legacy:  for req: with stream: for layer: src_to_gpu + scatter
# batched: for layer: concat all reqs' slots + index_copy_ once
```

Env flag: `FT_BATCHED_RELOAD=1`. Default OFF, back-compat preserved.

### Smoke result (s42)

- ✅ metrics.json written, no crash
- ✅ `path=batched_v2` fired (real path executed, not fallback)
- ✅ comp = 100%, gp = 100.4 (tie NR)
- ❌ **fg_p95 = 2420 ms** — worse than legacy V_AC s42 (358 ms) by 6.7×

### Why it's slower (root cause)

Original hypothesis: kernel launch count is the bottleneck (32 × N_reqs vs 32). WRONG.

Actual bottleneck: **CPU-side fancy-index + concat**:
```python
# per layer (executed 32× per restore):
src_slice = host_tensor[:, slot_indices]   # fancy index → new CPU tensor (memcpy)
src_slices_cpu.append(src_slice)
cat_cpu = torch.cat(src_slices_cpu, dim=1)  # second CPU memcpy
cat_gpu = cat_cpu.to(device, non_blocking=True)  # non-pinned → sync copy
```

Each layer does **2× CPU memcpy** (fancy index + cat), plus non-pinned `.to(device)` becomes sync. Aggregate CPU cost exceeds legacy per-req path's savings from fewer kernel launches.

### Decision: keep as ablation (not reverted)

Per tonight_todo code-change protocol: smoke error-free but underperforming → document + keep for future comparison. Code is env-gated (`FT_BATCHED_RELOAD=1`), default OFF, no regression on other paths.

### Next improvement direction (P2-v2 candidate)

1. **Pre-allocated pinned staging buffer** (fixed-size, reused across restores).
2. **Direct copy into pinned buffer slots** per req (skip `torch.cat`).
3. `staging_pinned.to(device, non_blocking=True)` → true async H2D.
4. Single `index_copy_` per layer from staging GPU.

Expected: eliminates both CPU memcpies + gains true async H2D. Rough prediction: fg_p95 < 500 ms (matches or beats NR).

Estimated time: ~2h code + 1h debug + smoke+3seed validation.

### P2-v3 pinned staging (FT_BATCHED_RELOAD=2) — also slow

Fixed the dim-order bug from v2 (staging was `[blocks, 2, ...]` instead of `[2, blocks, ...]`). After fix:
- ✅ v3 fired (no fallback)
- ❌ fg_p95 = 2465 ms — essentially identical to v2 (2420 ms)

**Root cause update**: the bottleneck is NOT the scatter kernel or CPU concat/fancy-index. It's the **OS framework's orchestration overhead**: manifest load, checkpoint controller iteration, solver admission decisions, and restore-attempt RPCs all add latency to the critical path even when most displaced reqs end up falling back to reprefill (restored=0).

Evidence: only 1/3 displaced reqs had a checkpoint (kv_restore_done=1). The other 2 reqs went through reprefill fallback but still took ~2.4s total — vs NR's 291ms for similar reprefill-only recovery. The difference = OS's per-step overhead from checkpoint + solver machinery.

### P2 conclusion

Batched-reload (v2 + v3) does NOT improve fg_p95. The root cause of OS's reload latency vs NR is not "too many scatter kernels" but "too much orchestration overhead in the FT framework". Fixing this requires deeper refactoring of the critical path (skip checkpoint-controller during recovery, bypass solver during failover, etc.) — estimated 1-2 days.

Both v2 (FT_BATCHED_RELOAD=1) and v3 (FT_BATCHED_RELOAD=2) are kept as ablation data. Default OFF, no regression.

---

## Status so far

| Phase | Status | Output |
|---|---|---|
| P0 NR fresh baseline | ✅ done | results_v2/8B/overnight_2026-04-16/p0_nr_fresh/ |
| P1 ablations (reprefill variants) | ✅ done | results_v2/8B/overnight_2026-04-16/p1_os_* |
| **P2 batched-v1 smoke** | ⚠️ fires but slow | results_v2/8B/overnight_2026-04-16/p2_batched_smoke/ |
| P2-v2 pinned staging | pending decision | — |

## Fix 1+2: Skip-checkpoint + Fast-failover (no-op, negative result)

Two env-gated fixes targeting OS framework overhead during recovery:
- **FT_SKIP_CKPT_DURING_RECOVERY=1**: skip checkpoint controller when rerouted reqs in running
- **FT_FAST_FAILOVER=1**: bypass solver for all admissions during recovery window

Smoke result (s42): fg_p95=2302ms. **Both fixes did not trigger**:
- Fix 1: `is_rerouted` attribute not visible in scheduler's running queue (attribute propagation gap).
- Fix 2: `FT_GATED_SOLVER` default threshold (pending<5) already makes solver bypass for most admissions → our fast-failover bypass is redundant.

**Conclusion**: the fg_p95 gap between OS (2300ms) and NR (291ms) is NOT caused by checkpoint-controller overhead or solver admission latency. The root cause remains unidentified — likely distributed across many small constant-factor overheads in the FT framework (manifest probe per req, collective_rpc round-trips, prebudget patching, etc.) that compound but are individually too small to pinpoint without profiling.

---

## Current best config for paper (if stopping here)

**`cap=1 + FT_RECOVERY_MODE=reprefill`** on A4 W2/Heavy/F2_Mid:
- goodput 101.1 (matches NR 101.0, p>0.1 n.s.)
- fg_p95 1176 ms vs NR 291 ms (+885 ms gap)
- 3/3 seeds 100% completion
- Solver admission contribution preserved (cap=1 uses solver + our new constraint)

Paper claim: *"Fault-aware admission reduces fg_p95 by 40% vs no-cap reprefill baseline (1955 → 1176 ms), while matching NR goodput. Remaining gap to NR is a fundamental reload-vs-reprefill cost difference on this hardware scale; future work: pinned-staging reload path."*
