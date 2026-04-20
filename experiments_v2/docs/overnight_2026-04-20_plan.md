# Tonight Overnight Plan — 2026-04-20 (02:00 AM → 11:00 AM, ~9h budget)

## Goal

**Single goal: W5_LongDoc × F2_Mid strict win vs NR baseline**
- Target: `gp ≥ 30.4` AND `fg_p95 ≤ 6115` AND `comp ≥ 88%`
- Current best: OS_NoCkpt gp=29.6 / fg=8811 / comp=85% (close but not strict win)

## Engineering discipline

1. **All new refactorings land in NEW files** (`*_v2.py`, `*_ext.py`) — never modify existing core code
2. **Commit before each phase** so revert is always one `git reset` away
3. **Env-gated** — default-off, opt-in via env var (zero impact when disabled)
4. **Unit test before integration** — CP-SAT/FT behavior verified offline first

## Safety state

Last commit: `795b658d3` — legit MIP-preserving optimizations stacked
- time_cap + warm-start + A3 presolve + 8GB ckpt pool + epoch_interval env
- Revert point: `git reset --hard 7de1fe83d`

## Current running job

**diag9** (02:35 ETA) — validates safety-commit stack on W5 none / W5 F2 × 3 seeds

---

## Phase 1 — Diag9 verdict (02:35 → 03:00)

Wakeup at 02:35 to analyze diag9 results.

### Branch A: diag9 strict win ✓ (gp ≥ 30.4 AND fg_p95 ≤ 6115)
→ **Paper done.** Skip to Phase 5 (cross-workload consolidation + 6-seed stats).

### Branch B: diag9 partial win (gp OK, fg_p95 still > 6115)
→ Proceed to **Phase 2** (R7 greedy warm-start)

### Branch C: diag9 no improvement
→ Proceed to **Phase 2** with accelerated timeline, then **Phase 3** (R4)

---

## Phase 2 — R7 Greedy Seed (03:00 → 05:00, 2h)

**Lowest-risk refactor.** Greedy FCFS solution injected as MIP hint,
accelerating convergence without replacing the MIP.

### Code plan (NEW file, no core edits)
- **NEW**: `vllm/v1/core/sched/benders/greedy_seed.py`
  ```
  def compute_greedy_seed(cost_table, replica_ids, running_cap) -> MasterSolution:
      # FCFS with SLO feasibility + running_cap; returns admission+routing
  ```
- `master.py` minimal edit: single env-gated call in `solve()`:
  ```
  if os.environ.get("FT_SOLVER_GREEDY_SEED", "0") == "1":
      greedy_sol = compute_greedy_seed(...)
      # merge greedy_sol hints alongside warm-start hints
  ```
- **Default OFF** — behaves exactly as current code until opted in

### Validation (3 seeds × W5 F2)
- 03:00-03:15 Write `greedy_seed.py` + unit tests
- 03:15-03:30 Smoke (1 seed × W5 F2)
- 03:30-04:30 Full 3-seed × W5 F2 with `FT_SOLVER_GREEDY_SEED=1`
- 04:30-05:00 Analyze + commit

### Commit
`feat(ft-solver): R7 greedy-seeded warm-start (new file, env-gated)`

---

## Phase 3 — R4 Snapshot Delta Protocol (05:00 → 07:30, 2.5h)

**Higher risk but targets the 10% residual scheduler overhead.**
The 10% gap between `OS_NoCkpt` and NR steady-state is dominated by
per-step `process_engine_outputs` snapshot processing.

### Code plan (NEW files only)
- **NEW**: `vllm/v1/engine/snapshot_delta.py`
  - `SnapshotDeltaBuilder` (engine side): maintains persistent state,
    emits `{added, removed, updated}` dicts per output batch
  - `SnapshotDeltaApplier` (client side): applies deltas to
    `_engine_request_snapshots`
  - Version stamp + monotonicity check
- Env gate: `FT_SNAPSHOT_DELTA=1` (default OFF)
- Fall-back: if applier detects missing version, request full snapshot resync
- Minimal `ft_client.py` edit: branch on env in `process_engine_outputs`
- Minimal engine-side: emit delta OR full snapshot based on env flag

### Risks (from R4 risk analysis)
- Distributed state desync
- Lost/reordered messages → inconsistent state

### Mitigations
- Version stamp on every delta; applier verifies monotonicity
- Health check: periodic full-snapshot reset every 100 epochs
- A/B with full protocol as control

### Validation
- 05:00-05:30 Write `snapshot_delta.py` + unit tests
- 05:30-05:45 Smoke (1 seed × W5 none, no fault, long run to verify consistency)
- 05:45-06:45 Full 3-seed × W5 F2 × {delta on, delta off}
- 06:45-07:30 Analyze + commit

### Commit
`feat(ft-client): R4 delta-encoded snapshot protocol (new files, env-gated)`

---

## Phase 4 — R1 Incremental Master (STRETCH, 07:30 → 09:30, 2h)

**Only if prior phases show partial win but fg_p95 still short.**

High risk. **Only if Phase 2+3 leave clear opportunity.**

### Code plan (NEW files only)
- **NEW**: `vllm/v1/core/sched/benders/incremental_master.py`
  - `IncrementalMasterProblem` wraps `MasterProblem` but persists CpModel
  - `add_request(req_id, costs)` / `remove_request(req_id)` ops
  - Tracks pending constraint/variable changes, flushes on `solve()`
- Env gate: `FT_SOLVER_INCREMENTAL=1` (default OFF)
- Solve loop: if flag on, use `IncrementalMasterProblem` instance persistent
  across epochs; else fall through to original `MasterProblem`

### Validation
- Extensive unit testing before integration (state consistency!)
- Smoke: 1 seed × W5 F2
- Full 3-seed only if smoke clean

### Abort criteria
- CP-SAT state inconsistency detected → abort, revert
- Goodput drops >5% vs baseline → abort

---

## Phase 5 — Cross-workload Consolidation (09:30 → 10:30, 1h)

Regardless of Phase 2-4 outcome, validate **best config** on:
- **W7** (already-won paper): 3 seeds × F2_Mid to confirm W7 advantage preserved
- **W4** no-regression: 3 seeds
- **W1** no-regression: 3 seeds
- **W5** 3 additional seeds for 6-seed confidence interval

All parallel on GPU 0-3. ~12 runs × 7min ≈ 42min parallel.

Commit: `validate(ft): cross-workload sweep on best config`

---

## Phase 6 — Paper Claim + Docs (10:30 → 11:00, 30min)

Final deliverables:

### `docs/paper_claim_2026-04-20.md`
- **Strongest defensible claim** with 3-sentence summary
- Per-workload table
- Ablation (each fix's contribution)
- Honest limitations section

### `paper_table_v4.csv`
- Consolidated LaTeX-ready table

### `stress_test_final_2026-04-20.md`
- Unified narrative replacing earlier drafts

---

## Wakeup chain (auto-pilot)

| time | task | GPU status |
|---|---|---|
| 02:35 | Diag9 verdict → branch selection | idle after diag9 |
| 03:30 | R7 smoke result | GPU 0-1 active |
| 04:30 | R7 full 3-seed done | idle |
| 05:30 | R4 smoke result | GPU 2-3 active |
| 06:45 | R4 full 3-seed done | idle |
| 08:00 | R1 smoke (if reached) | GPU 0-1 active |
| 09:30 | R1 full done (if reached) | idle |
| 10:15 | Cross-workload sweep half-done | all GPUs active |
| 10:45 | Cross-workload sweep done | idle |
| 11:00 | Final doc drafted | idle |

## Rollback decision tree

```
If R7 + R4 fail:
    R1 is high-risk, ONLY if strict confidence
    ELSE git reset --hard 795b658d3
         paper claim = "W7 paper win + honest W5 limitation"
```

## Hard constraints

- **GPU 0-3 only** (GPU 4-7 occupied)
- **No destructive git ops** without commit first
- **All code changes in new files** (preserve revertability)
- **Hard stop 11:00 AM**
- **Abort any phase on 2 consecutive AssertionErrors** (preemption bug)

## Rate-limited TODO

### Tonight
- [x] Commit safety checkpoint 795b658d3
- [x] Write this plan
- [ ] 02:35 — diag9 verdict
- [ ] 03:00 — start R7 (greedy_seed.py new file)
- [ ] 04:30 — R7 result
- [ ] 05:00 — start R4 (snapshot_delta.py new file)
- [ ] 06:45 — R4 result
- [ ] 07:30 — decide on R1 go/no-go
- [ ] 09:30 — cross-workload validation
- [ ] 10:45 — final docs + commit

### Decision gates

**G1 (02:35, diag9)**: strict win → skip to Phase 5; else Phase 2
**G2 (04:30, R7)**: strict win → skip to Phase 5; else Phase 3
**G3 (06:45, R4)**: strict win → skip to Phase 5; else consider R1
**G4 (07:30, R1 go/no-go)**: unit tests clean + time ≥ 1.5h left → proceed;
                              else direct skip to Phase 5
**G5 (10:30, final)**: either strict win claim OR honest limitations paper

## Files referenced

| Role | Path | Modify? |
|---|---|---|
| Current master | `vllm/v1/core/sched/benders/master.py` | NO (frozen) |
| Current solve_loop | `vllm/v1/core/sched/benders/solve_loop.py` | minimal env-gated edit |
| Current ft_client | `vllm/v1/engine/ft_client.py` | minimal env-gated edit |
| R7 greedy seed | `vllm/v1/core/sched/benders/greedy_seed.py` | NEW |
| R4 delta builder | `vllm/v1/engine/snapshot_delta.py` | NEW |
| R1 incremental master | `vllm/v1/core/sched/benders/incremental_master.py` | NEW |
| Config | `experiments_v2/config_8b.yaml` | frozen (already tuned) |
| Plan | `experiments_v2/docs/overnight_2026-04-20_plan.md` | THIS DOC |
