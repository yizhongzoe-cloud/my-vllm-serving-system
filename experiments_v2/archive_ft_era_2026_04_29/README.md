# Archived: FT-era experiments (pre-2026-04-29)

## Why archived

All experiment data here was produced under the **FT (fault-tolerance) paper
framing**, before the project pivoted to **disruption-aware long-context
serving**. None of these numbers can be reused in the current paper.

Three reasons:

1. **Profile-calibration bug**. The checkpoint cost profile only covered up to
   2048 tokens; everything at 16K+ context was extrapolated and economic-policy
   decisions made on it are wrong. Any long-context experiment here is
   suspect (see `project_profile_calibration_bug.md` in MEMORY).

2. **Framing changed**. FT-era main metric was `failover_gap_p95`
   (engine-death → first new token). Disruption-era main metric is `goodput`
   (SLO-met tokens / sec) + SLO attainment rate. Different metrics, different
   baselines, not comparable.

3. **Code changed**. FT solver (Benders dp=2) and cross-replica shm publish
   have been removed in the migration to `zoe/disruption`. SLO retain was
   added in round 2. Old baselines can no longer be reproduced exactly.

## What's in here

### Main 4-cell comparison (FT paper main figure)
- `results_main_4cell/` — rps03 + rps05, final clean run
- `results_main_4cell.bak.20260507_031414/` — earlier snapshot, has rps06 / rps10
  plus per-cell layout (A_vanilla / B_capacity_release / C_slo_retain /
  D_routed)
- `results_main_4cell.bak.20260507_133939/` — intermediate snapshot, rps06 / rps10

Cells:
- A_vanilla = NoFT-Reprefill
- B_capacity_release = CkptReload (corresponds to current V3 capacity-preempt
  reload path)
- C_slo_retain = SLO retain (corresponds to current round-2 SLO retain path)
- D_routed = CkptReload + Benders solver routing (solver since removed)

### Targeted ablations
- `results_ckpt_overhead/` — W_Ruler16K_CkptOnly, measures checkpoint save
  overhead in isolation. 3 seeds.
- `results_reload/` — W_Ruler16K_CkptReload, measures reload-path recovery
  latency (failover_gap_p95). 3 seeds.
- `results_reload_v2/` — same as above, rerun.
- `results_extension/` — context-length sweep (Ruler1K alongside 16K) to
  test whether reload is worth the cost at short context.
- `results_slo_priority/` — sanity test before SLO retain went live.

### V3 crash debugging
- `debug_v3/` — single-cell repro (didn't crash).
- `debug_v3_seq/` — minimal A→B sequence repro.
- `debug_v3_loop/` — 5-iter loop to find crash point.

Root cause was OOB index triggering device-side assert. Fixed; these dirs
are the post-mortem artifacts.

## What can still be useful

- **yaml config field names** — when building new experiment configs in the
  disruption era, you can look up the FT-era yaml structure here as a
  starting template.
- **metric column names in CSV** — `paired_recovery.csv`, `paired_diff.csv`
  have the metric naming convention from the FT era; some columns are still
  relevant (TTFT, TPOT, completion_rate).
- **directory layout** — A/B/C/D per-cell, rps× per-load layout is a
  reasonable template for new sweep runs.

## What CANNOT be reused

- Any numerical result. None of these numbers should appear in the
  disruption-era paper, even as "preliminary data".

---

Archived 2026-05-11 at the start of round-3 migration (cross-engine
shm publish + restore framework + minimal smoke).
