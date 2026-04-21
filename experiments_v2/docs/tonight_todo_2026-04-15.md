# Tonight TODO (2026-04-15 overnight, 01:30 → 10:00 target)

**Status** ✅ **COMPLETED** (compressed to compact run 10:45-14:09) — see [overnight_2026-04-15_summary.md](overnight_2026-04-15_summary.md)

**Outcome**: No winner cell. OS systematically loses fg_p95 to NR (+180 ms to +1328 ms) on all 5 tested cells. Root cause: vLLM's prefill engine is faster than OS KV-reload on 8B/A5000. Paper story pivoted to scaling regime (70B / long-context / H100).

**Goal**: OS v4 goodput AND fg_p95 both beat NR on Heavy fault-tolerance scenarios.
Find the workload/fault setup + solver variant where this holds for 3/3 seeds.

**Decision rule** for Phase 1 winner: `3-seed mean OS goodput > NR × 1.05 AND OS fg_p95 < NR`.
If NO cell qualifies after Phase 1, pivot: Moderate load instead of Heavy + Phase 3 solver work.

**Auto-pilot mode**: every cell crash → log + skip → next cell. Do NOT loop-retry failing runs.

All results under `results_v2/8B/overnight_2026-04-15/`.

---

## Phase 1 — Discovery (01:30 – 03:30, 2h)

6 discovery cells. For each: **3 seeds × OS v4 + NR baseline**, paired GPUs 4-5 / 6-7 parallel.

- [x] **P1-01** Verify dataset files exist (alpaca_full / cnndm_3000 / sharegpt_5000) — all present
- [x] **P1-02** Cell A1: `W2_Summary / Moderate / F3_Late` → tied goodput, OS fg +269 ms (loses)
- [x] **P1-03** Cell A2: `W2_Summary / Moderate / F2_Mid` → tied goodput, OS fg +182 ms (loses)
- [x] **P1-04** Cell A4: `W2_Summary / Heavy / F2_Mid` (replaced F3_Late as system-capped probe) → tied, OS fg +1328 ms (loses)
- [x] **P1-05** Cell A5: `W4_Mixed / Heavy / F2_Mid` (replaced Moderate as system-capped) → OS gp +5.8 but comp 81%, fg +892 ms (loses)
- [x] **P1-06** Cell A3: `W1_Chat / Moderate / F3_Late` → tied, OS fg +973 ms, comp 89%
- [ ] **P1-07** ~~W3_Instruct cell~~ — skipped (dropped from compact run, short-prompt control redundant given A3 result)
- [x] **P1-08** Compile Phase 1 results table — [paper_table.csv](../../results_v2/8B/overnight_2026-04-15/compact/paper_table.csv)
- [x] **P1-09** Winner selection: **no cell qualifies**. fg_p95 lost in 5/5.

**OS config (v4)**: `gpu_util=0.9, FT_RESTORE_PARALLEL_LOAD=1, FT_RESTORE_PER_REQ_SYNC=1, FT_RESTORE_BATCH_RPC=1, FT_RECOVERY_PREBUDGET=1`

---

## Phase 2 — Winner Ablation (03:30 – 05:30, 2h)

With winner setup from Phase 1, sweep fault timing + load level.

- [~] **P2-01** Fault timing partial coverage via A1/A2 (F3_Late vs F2_Mid) — both tied, same verdict
- [~] **P2-02** Load level partial via A1 vs A4 (Moderate vs Heavy) — Heavy A4 shows OS fg much worse; no cross-load winner
- [ ] **P2-03** Full ablation skipped — Phase 1 showed no winner cell worth ablating
- [x] **P2-04** Crash patterns flagged: A3 89% (s123 save race), A5 81% (similar save race); fg_p95 loss universal across all 5 cells

---

## Phase 3 — Solver Enhancement + Engineering (05:30 – 08:00, 2.5h)

**Gated by env flag `FT_BENDERS_STAGGERED=1`. All changes isolated to benders solver + ft_client.**

- [x] **P3-01** Design: pivoted to reuse existing `FT_LAZY_RELOAD` (engine-layer staggered admission) as proxy. Full solver MIP + API change (~1 day) skipped because root cause is vLLM prefill ≈ reload speed on 8B, not admission ordering.
- [ ] **P3-02** ~~Implement solver MIP changes~~ — skipped; FT_LAZY_RELOAD already provides equivalent behavior at engine layer.
- [ ] **P3-03** ~~Modify ft_client batch-dispatch~~ — LAZY_RELOAD is engine-side; ft_client-side rate-limit previously failed with CUBLAS_INTERNAL.
- [ ] **P3-04** ~~Dry-run solver~~ — N/A.
- [x] **P3-05** Smoke test: A4+LAZY/s42 fg_p95 dropped 1616→239 ms (−85%) — promising signal on long-prompt high-load.
- [x] **P3-06** 3-seed validation on A1/A4/A5 cells — completed.
- [x] **P3-07** Head-to-head v4 vs v4+LAZY_RELOAD vs NR (see phase3 section of [summary.md](overnight_2026-04-15_summary.md)).
- [x] **P3-08** Final pick: **v4 (per-req sync, no LAZY_RELOAD)**. LAZY_RELOAD only helps on A4 (W2/Heavy, 1577→1133 ms); worsens A1 (+4.7×) and A5 (+3.2×). Not universally a winner.

**Fallback applied**: v4 is final combo for paper. LAZY_RELOAD documented as ablation with mixed results.

---

## Phase 4 — Paper-ready (08:00 – 10:00, 2h)

With final best combo selected, produce paper-ready data.

- [x] **P4-01** Final sweep covered by compact run (5 cells × 3 seeds × OS+NR = 30 runs)
- [ ] **P4-02** Extended matrix not needed — Phase 1 already covered {W1,W2,W4} × {Moderate,Heavy} × {F2,F3}
- [x] **P4-03** Paper-ready markdown table + CSV: [paper_table.csv](../../results_v2/8B/overnight_2026-04-15/compact/paper_table.csv), [summary.md](overnight_2026-04-15_summary.md)
- [x] **P4-04** Summary written: [overnight_2026-04-15_summary.md](overnight_2026-04-15_summary.md)
  - Executive summary: **no winner, OS systematically loses fg_p95 on A5000/8B**
  - Journey notes: W2 Moderate throughput-capped; Heavy fg_p95 gap largest (+1.3 s)
  - Final config: v4 (per-req sync) stays as most-stable baseline
  - Limitations: A5000 headroom + 8B model too small for reload to beat vLLM prefill; paper pivot to 70B+ / long-context / H100

---

## Monitoring cadence

- Every 20-30 min auto-wakeup check: read relevant log, update this md checklist
- On crash: extract traceback, classify, record under each task's sub-notes, continue
- Hard stop: if wall clock > 10:00 AM, freeze Phase 4 and commit whatever's done

## Risk control

- **No infinite retry**: max 2 attempts per cell before skip
- **GPU lock**: use GPU 4-5 (OS) + 6-7 (NR) exclusively; kill zombie VLLM processes between cells
- **/dev/shm cleanup**: `rm -rf /dev/shm/vllm_ft_checkpoints` between cells to avoid state pollution
- **Safety nets**:
  - Commit nothing to main branch without review
  - All new files under `results_v2/8B/overnight_2026-04-15/` or `experiments_v2/overnight_*`
  - Solver changes behind env flag (`FT_BENDERS_STAGGERED`) — default OFF

## Escape hatches

- If Phase 1 shows zero winner (unlikely): move to Moderate load only, Phase 2 still proceeds
- If Phase 3 solver breaks scheduler invariants: revert immediately, fall back to v4+LAZY_RELOAD
- If GPU 0-3 (other user) resource contention changes: no action needed, we only use 4-7
