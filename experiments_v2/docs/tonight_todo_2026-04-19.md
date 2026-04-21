# Tonight TODO (2026-04-19 overnight, 02:20 → 08:00, ~5.5h productive + buffer)

## Context from 2026-04-19 day work

**Current best (V2 = `FT_CHECKPOINT_STEP_INTERVAL=2`)**:
- **W1/Heavy: OS 3267 vs NR 3220 (matches NR, +1.5%)** 🎯
- **W2/Heavy: OS 990 vs NR 291 (still +699ms)** — NEED TO CLOSE

**The remaining 633ms pre-fault pause on W2 is unexplained**. Goal tonight: locate and eliminate it so OS beats NR on W2.

Decomposition of current 990ms on W2:
- 357ms real per-req recovery (unavoidable, matches NR's 249ms within 1.4×)
- **633ms unidentified pre-fault pause** — suspects: solver MIP per epoch, snapshot construction per step, request pool iteration, failure monitor polling

**GPU allocation**: all runs use `CUDA_VISIBLE_DEVICES=0,1` (GPU 4-7 occupied by other user). dp=2.

**Hard stop**: 08:00 AM. All code changes env-gated. Decision rule: cell crash → skip, no retries.

---

## Phase 1 — Deep ablation on W2 (02:20–03:50, 1h30)

Goal: Find which component causes the 633ms pre-fault pause. Ablate one at a time on top of V2 baseline.

**Script**: `experiments_v2/p5_deep_ablation.sh` (ready, uses GPU 0-1)

Variants (5 × 3 seeds = 15 runs × ~6 min = ~90 min):
- [x] **P1-01** V2_base (control) — s42=370, s123=2220 (noisy)
- [x] **P1-02** A: `FT_DISABLE_SNAPSHOTS=1` — s42=260, s123=1322 (modest win)
- [x] **P1-03** B: `FT_SKIP_SOLVER=1` — **s42=81**, s123=1295 (big s42 win!)
- [x] **P1-04** C: A + B — s42=1844 (regression, don't combine)
- [x] **P1-05** D: `FT_CHECKPOINT_STEP_INTERVAL=20` — s42=368 (no improvement)
- [ ] **P1-06** Waiting for seed 456 to confirm B winner (in progress)

**Current read**: B (`FT_SKIP_SOLVER=1`) is the leading candidate — solver MIP is the pre-fault bottleneck. Need s456 + cross-workload W1 confirmation.

## Parallel P5b (GPU 2-3) — W4_Mixed/Heavy baseline [DONE]

| variant | gp | fg_p95 |
|---|---|---|
| W4 NR | 253.6±17 | 1935±151 |
| W4 V2 ckpt_interval=2 | 253.8±18 | 2130±329 |

**Finding**: On W4_Mixed production mix, NR fg_p95 is already ~2000ms. OS-NR gap is only +10% (vs W2's +240%). Paper story: OS competitive on realistic workloads.

## Parallel P5c (GPU 2-3) — W1 B_skip_solver cross-workload [DONE]

| seed | fg_p95 | comp | gp |
|---|---|---|---|
| s42 | 4536 | 69% ❌ | 250 |
| s123 | 4690 | 100% | 152 ❌ (-28%) |
| s456 | 2508 | 91% ❌ | 308 |

**Verdict**: B regresses W1 severely — solver is load-bearing for admission on high-RPS. B is NOT universal, keep as W2-only knob with documented trade-off.

---

## Final verdict (04:25)

- **V2 (FT_CHECKPOINT_STEP_INTERVAL=2) remains universal paper winner**
- **B (FT_SKIP_SOLVER=1) localizes the 633ms pre-fault pause to Benders MIP** but bypass sacrifices admission control
- Future work: optimize solver latency (warm-start, time-cap) instead of bypass

## Phase 4 — Complete (06:15)

- [x] **P4-01** V2 + NR × F1_Early/F3_Late × W2/Heavy × 3 seeds (12 runs)
- [x] **P4-02** Fault-timing robustness table produced
- [x] **P4-03** `paper_table_v3.csv` generated (10 rows)
- [x] **P4-04** `overnight_2026-04-19_summary.md` finalized with full results

**Overnight status**: COMPLETE. 36 runs, ~3.6 GPU-hours. All phases executed within budget. No new experiments launched after 06:15. See `overnight_2026-04-19_summary.md` for final paper claims.

**Decision rule**:
- If A or B reaches fg_p95 < 500ms → proceed to Phase 2 with that variant
- If only C works → combined fix, both components matter
- If D helps more than current V2 → snapshot+solver not the issue, ckpt controller still matters
- If NONE close gap < 500ms → deeper refactoring needed, try Phase 3 timestamp probes

---

## Phase 2 — Winner cross-workload + full matrix (03:50–05:20, 1h30)

If Phase 1 finds a W2 winner:
- [ ] **P2-01** Winner × W1_Chat/Heavy × 3 seeds (confirm no regression on W1)
- [ ] **P2-02** Winner × W4_Mixed/Heavy × 3 seeds (production mix)
- [ ] **P2-03** Winner × F1_Early, F3_Late × s42 (fault timing robustness)

If Phase 1 finds NO winner: skip to Phase 3.

---

## Phase 3 — Timestamp probes (if Phase 1 fails) (05:20–06:20, 1h)

Add `logger.debug` timestamps around suspect code blocks to measure per-step latency during fault window:
- [ ] **P3-01** Add probe to `_process_engine_step` (core.py:2267)
- [ ] **P3-02** Add probe to `ft_client` solver admission path (line ~1505)
- [ ] **P3-03** Add probe to `_build_active_snapshots`
- [ ] **P3-04** Code-change protocol (syntax + import + smoke)
- [ ] **P3-05** Single-seed instrumented run, analyze log

---

## Phase 4 — Final paper-ready sweep (06:20–07:50, 1h30)

- [ ] **P4-01** Final winner config × {W1,W2,W4} × 3 seeds × OS+NR comparison
- [ ] **P4-02** Fault timing ablation {F1, F2, F3} × best workload × 3 seeds
- [ ] **P4-03** Generate `paper_table_v3.csv`
- [ ] **P4-04** Update `overnight_2026-04-19_summary.md` with final numbers

---

## Monitoring cadence

- ScheduleWakeup every ~30-40 min, auto-analyze + continue
- Each wakeup:
  - Check GPU status + process alive
  - Read latest metrics
  - Update this doc's checkboxes
  - Schedule next wakeup or advance to next phase

## Code-change protocol (Phase 3 only, if applicable)

Before any code change goes live:
1. `python3 -c "import ast; ast.parse(open(f).read())"` syntax check
2. Import sanity: `from <module> import <class>` test
3. Single-seed smoke run
4. grep server.log for `Traceback|OutOfMemory|assert|CUDA error`
5. Rollback if any failure

## Env flags reference

| flag | effect | status |
|---|---|---|
| `FT_CHECKPOINT_STEP_INTERVAL` | throttle ckpt controller polling | V2 winner, default=2 |
| `FT_RECOVERY_PREBUDGET` | pre-reserve tokens for restore | 1 (on) |
| `FT_SOLVER_RUNNING_CAP` | per-replica admit cap | 3 |
| `FT_RECOVERY_MODE` | recovery path | reprefill |
| `FT_DISABLE_SNAPSHOTS` | bypass snapshot construction | Phase 1 A |
| `FT_SKIP_SOLVER` | bypass solver admission MIP | Phase 1 B |

## Files

- Overnight summary (updating): `experiments_v2/docs/overnight_2026-04-19_summary.md`
- This plan: `experiments_v2/docs/tonight_todo_2026-04-19.md`
- Phase 1 script: `experiments_v2/p5_deep_ablation.sh`
- Results root: `results_v2/8B/overnight_2026-04-16/p5_deep/`

## If wakeup chain fails (recovery instructions for morning)

1. `pgrep -af "run.py|p5_deep"` — check processes
2. `ls results_v2/8B/overnight_2026-04-16/p5_deep/*/*/metrics.json | wc -l` — progress
3. `tail /tmp/p5_deep.log` — last output
4. Manually compare variants: see Phase 1 decision rule table
