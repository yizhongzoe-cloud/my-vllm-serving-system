# Tonight TODO (2026-04-16 overnight, ~01:30 → 08:30 target, 7h budget)

## Context carry-over from 2026-04-15

**What we proved**:
- **V_AC** (FT_SOLVER_RECOVERY_PENALTY=0.01 + FT_SOLVER_RUNNING_CAP=3) improves OS fg_p95 by 31% vs greedy (1577→1090 ms) on A4 W2/Heavy. Goodput preserved.
- **T4** (FT_SOLVER_RUNNING_CAP=1) achieves fg_p95=79ms on s42 single seed (≈ NR's 74ms). But 3-seed mean = 1015ms (s123/s456 still slow).

**Root cause of remaining gap**:
- s42 fast because cap=1 delayed admission → reqs hadn't committed checkpoints yet → fault triggered reprefill fallback (vLLM optimized path).
- s123/s456 slow because checkpoints had committed → reload path fires (per-req sync + scatter + serial = ~1400ms).
- **Counter-intuitive finding**: vLLM's batched prefill is **faster** than our custom KV-reload on 8B/A5000. Conventional "reload > reprefill" wisdom doesn't hold at this scale.

**Implication**: To truly beat NR, we must either:
1. Force reprefill path consistently (admission-level trick), OR
2. Fix the reload path itself to be genuinely faster than reprefill, OR
3. Sidestep both via background shadow prefetch.

All results under `results_v2/8B/overnight_2026-04-16/`.
Decision rule: every cell crash → skip + log, max 2 retries. Hard stop: 08:30 AM.

**GPU allocation**: all runs tonight use `CUDA_VISIBLE_DEVICES=4,5` (dp_size=2, single pair).
- No parallel OS + NR on 4-5 / 6-7 split tonight — NR 3-seed baselines already exist in `results_v2/8B/overnight_2026-04-15/compact/.../nr/` and can be reused for comparison.
- Keeps GPU 6-7 free (e.g., for other users or emergency runs).

---

## Phase 0 — Fresh NR baseline (01:30–01:50, 20 min) ⚠️ **新增，必做**

**Why**: fault engine selection was randomized by seed post-compact-run. NR baseline from `compact/` always killed engine 0; today's s456 kills engine 1. Rerun NR with current code for fair comparison.

- [ ] **P0-01** Launch NR 3-seed on A4 W2/Heavy/F2_Mid with current random fault code. Output: `results_v2/8B/overnight_2026-04-16/nr_baseline/`.

## Phase 1 — Quick Validation (01:50–03:00, 1h10) ⚠️ **加 ablation**

**Goal**: Test if admission cap is *necessary* or `reprefill mode` alone is enough.

- [ ] **P1-01** **`reprefill only` × 3 seeds** (ablation, no cap). Test if forcing reprefill path alone matches NR. Output: `results_v2/8B/overnight_2026-04-16/p1_reprefill_only/`.
- [ ] **P1-02** **`cap=1 + reprefill` × 3 seeds** (hypothesis from T4). Output: `results_v2/8B/overnight_2026-04-16/p1_cap1_reprefill/`.
- [ ] **P1-03** **`cap=3 + reprefill` × 3 seeds** (mid-cap comparison). Output: `results_v2/8B/overnight_2026-04-16/p1_cap3_reprefill/`.
- [ ] **P1-04** Analyze: (a) is reprefill alone enough? (b) does cap add value on top? Record fg_p50/p95 stddev.
- [ ] **P1-05** If any variant 3/3 matches NR → flag as **Candidate A**.
- [ ] **P1-06** Fault timing mini-ablation: winning config × {F1_Early, F3_Late} × s42 only (~12 min).

**Config A full env**:
```
FT_SOLVER_RUNNING_CAP=1 FT_RECOVERY_MODE=reprefill FT_RESTORE_PER_REQ_SYNC=1 
FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RECOVERY_PREBUDGET=1
```

---

## Phase 2 — Batched-Reload Implementation (03:00–08:00, **5h budget**) ⚠️ 时间扩到 5h

**Goal**: Make reload path genuinely faster than reprefill, eliminating our core bottleneck.

**Design**: pre-allocated staging buffer + single batched H2D copy + single scatter kernel, replacing per-req/per-layer serial loop. Avoids OOM by bounding GPU temp to fixed staging size.

- [ ] **P2-01** Read current `_process_ft_pending_restores` + `restore_kv_blocks_batch` to understand current data flow (15 min).
- [ ] **P2-02** Implement `StagingBuffer` class in `gpu_model_runner.py`:
  - Pre-allocate `staging = torch.empty(MAX_SLOTS, 2, kv_heads, head_dim)` on GPU at init, ~100 MB fixed.
  - Reused across restore batches.
  - Env-gated by `FT_BATCHED_RELOAD=1`, default OFF.
- [ ] **P2-03** Rewrite `restore_kv_blocks_batch` to use staging:
  - Phase 1: parallel host-side pin + layout to contiguous pinned buffer (CPU loop, OK to parallel).
  - Phase 2: single `staging.copy_(pinned_src, non_blocking=True)` H2D.
  - Phase 3: single `gpu_cache.index_put_(..., staging[:n])` or custom scatter.
  - Phase 4: single `stream.synchronize()` at end.
- [ ] **P2-04** **Run code-change protocol** (see bottom of this doc): syntax check, import sanity, relevant unit tests (`tests/ft/test_async_ft_checkpoint_flow.py`).
- [ ] **P2-05** Smoke test `FT_BATCHED_RELOAD=1` on A4/s42. Expect fg_p95 <500ms with actual kv_restore_done firing (not fallback). **Then inspect server.log for Traceback/OOM/assert**.
- [ ] **P2-06** If smoke crashes (OOM/scatter shape mismatch) OR log has new errors → **revert immediately** (not fix-in-place), triage with 1h timebox. If not resolved, rollback all Phase 2 code + pivot to Phase 4.
- [ ] **P2-07** Only if smoke is clean: 3-seed validation on A4 W2/Heavy. Target: **fg_p95 < 500ms AND comp=100% AND gp≥100 across all seeds**.
- [ ] **P2-08** If P2-07 wins → this is the real paper winner (not the admission trick). Flag as **Candidate B**.

**Risk controls**:
- Keep old `restore_kv_blocks_batch` path as fallback when `FT_BATCHED_RELOAD` unset.
- Add logging on staging buffer fill/scatter size.
- If scatter kernel produces wrong KV layout → corruption may look like CUDA assert; roll back immediately.

---

## Phase 3 — Cross-Workload Ablation (if Phase 2 wraps early, optional, 1h)

Validate winner config(s) generalize beyond W2_Summary/Heavy.

- [ ] **P3-01** Best config from Phase 1/2 × **W1_Chat / Heavy / F2_Mid** × 3 seeds (18 min).
  - Expected challenge: W1 short prompts + high RPS, running batch needs to be larger → cap=1 may bottleneck throughput.
- [ ] **P3-02** Best config × **W4_Mixed / Moderate / F2_Mid** × 3 seeds (18 min). Production-realistic mix.
- [ ] **P3-03** Compile per-cell table. Flag where OS beats NR on (fg_p95, gp) pair.

---

## Phase 4 — Shadow Prefetch Foundation (**pure doc-only, can do anytime**, 1h)

**Goal**: Start the high-impact feature. Full implementation is 2-3 days; tonight we scaffold.

- [ ] **P4-01** Sketch `ShadowCopyManager` class (paper-targeted design):
  - `shadow_state: dict[req_id → (peer_replica, coverage_ratio)]`
  - Async CUDA stream for copies.
  - Pairwise mapping for dp=2 (uniform prior, no fault predictor needed).
- [ ] **P4-02** Design fault-handler fast path:
  - At failover, check shadow_state per displaced req.
  - `req.num_computed_tokens = shadow_coverage × req.prompt_len`.
  - Remaining tail goes through reprefill (automatic via existing scheduler).
- [ ] **P4-03** Document design in `experiments_v2/docs/shadow_prefetch_design_2026-04-16.md` (paper-ready).
- [ ] **P4-04** (Stretch) Prototype a minimal shadow stream loop if time permits.

---

## Phase 5 — Paper-ready consolidation (if time left)

- [ ] **P5-01** Generate `results_v2/8B/overnight_2026-04-16/paper_table.csv` with all winners.
- [ ] **P5-02** Update `overnight_2026-04-16_summary.md` with:
  - Exec summary (which configs beat NR, under what conditions).
  - Trade-off curves (cap size vs fg_p95 vs goodput).
  - Batched-reload vs reprefill mode comparison (if both winning).
  - Shadow prefetch design (forward-looking).

---

## Decision tree

```
Phase 1 result?
├── 3/3 seeds match NR → Candidate A wins. 
│   Phase 2 can still improve but is optional.
│   Go directly to Phase 3 cross-workload.
│
├── s42 matches but s123/s456 don't → 
│   Admission alone insufficient; need Phase 2 batched-reload.
│
└── All 3 seeds still slow → 
    Admission cap has no effect; investigate why (reload path issue).
    Go directly to Phase 2.

Phase 2 result?
├── fg_p95 < 500ms AND 3/3 stable → Candidate B wins (real paper contribution). 
│   Update summary as "batched-reload makes OS faster than NR".
│
├── fg_p95 slightly improved but still > NR → Combine with Candidate A (cap+reprefill)
│   for final config.
│
└── fg_p95 unchanged or worse → Rollback. Document as negative result.
    Pivot to Phase 4 shadow prefetch design as future work.
```

---

## Monitoring cadence

- Every ~30 min auto-wakeup via ScheduleWakeup
- Each wakeup: check log tail, update this md's checkboxes, schedule next
- Hard stop 08:30: even if Phase 2 mid-flight, commit current state + roll back any unfinished code changes

## ⚠️ Code-change protocol (MANDATORY for every edit)

**Every code modification in Phase 2/4 MUST follow this checklist before the change is considered "applied"**:

1. **Syntax check**
   ```
   python3 -c "import ast; ast.parse(open('<path>').read())"
   ```
2. **Import sanity** (catch missing imports, circular deps, broken module)
   ```
   python3 -c "from vllm.v1.worker.gpu_model_runner import <Class>; print('OK')"
   ```
3. **Existing unit tests** (if path affects `tests/ft/test_*.py`)
   ```
   python3 -m pytest tests/ft/test_async_ft_checkpoint_flow.py -x --tb=short
   python3 -m pytest tests/ft/test_benders_solver.py -x --tb=short
   ```
4. **Single-seed smoke test** (must succeed before 3-seed extension)
   - W2_Summary/Heavy/F2_Mid/s42 with the new flag enabled
   - Verify: engine starts, fault injection fires, metrics.json written
   - Expected duration: ~6 min
5. **Log inspection after smoke**
   - `grep -E "Traceback|Error|OutOfMemory|assert|CUDA error"` on server.log
   - Check `kv_restore_done` count matches expectation
   - Verify `reroute_plan` events match `failover_complete recovered=N`
6. **If any step fails** → **revert immediately** (`git diff` + manual undo), do not "push through" with hacks

**Rollback trigger**: smoke fg_p95 > 2× baseline OR comp < 95% OR any new `ERROR`/`Traceback` in server.log.

**Revert template**:
```
git diff <file> > /tmp/rollback_<tag>.patch   # save for later if needed
git checkout <file>                            # undo
# or use Edit tool to reverse the specific block
```

## Env flags reference (new ones for tonight)

| flag | default | Phase | description |
|---|---|---|---|
| `FT_SOLVER_RECOVERY_PENALTY` | 0 | P1 (reuse) | admit penalty for high-recovery reqs |
| `FT_SOLVER_RUNNING_CAP` | 0 | P1 (reuse) | per-replica admit cap |
| `FT_RECOVERY_MODE` | reload | P1 | `reprefill` forces NR-style recovery |
| `FT_BATCHED_RELOAD` | 0 | P2 (new) | single-kernel scatter restore |
| `FT_SHADOW_PREFETCH` | 0 | P4 (new, design only) | pairwise KV shadow |

## Files touched tonight

- `vllm/v1/core/sched/benders/master.py` — already has A/C extensions, no new changes in Phase 1.
- `vllm/v1/worker/gpu_model_runner.py` — Phase 2 adds StagingBuffer + batched restore path.
- `vllm/v1/engine/ft_client.py` — Phase 4 scaffolding for shadow (optional).
- `experiments_v2/run.py` — already has seed-random fault, no changes needed.

All code changes behind env flags; default behavior unchanged.

---

## Priorities if time runs short

**Must-do** (core contribution validation):
- Phase 1: **Exp A** (cap=1 + reprefill × 3-seed) — 30 min, answers if admission alone is enough.

**Should-do** (technical contribution):
- Phase 2: Batched-reload — 4h, this is the path to beating NR legitimately.

**Nice-to-have** (paper narrative):
- Phase 3 cross-workload ablation — 1h for paper table breadth.
- Phase 4 shadow prefetch design doc — 1h for paper novelty claim.

**Skip if crunched**:
- Phase 5 consolidation (can be done in morning light).
