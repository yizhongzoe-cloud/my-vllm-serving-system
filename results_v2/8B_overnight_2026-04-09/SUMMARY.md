# Overnight 2026-04-09 — Final Summary

_Generated at 2026-04-09T07:01:43_

Investigation goal: localize Our-System framework baseline overhead (the ~200 tok/s gap to No-FT on W1_Chat/Heavy that today's drop-mode test showed is NOT from recovery).

---

## Phase 1 — Baseline reload (1 run)

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `01_reload_baseline/Our-System/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 120.9 | 17356 | 53631 | 75.8 | 108.2 | 66.5 | 100.0 | ? | 4922 |

## Phase 1 — Baseline No-FT (1 run, upper bound)

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `02_noft_baseline/No-FT/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 319.5 | 398 | 10436 | 43.0 | 63.3 | 20.3 | 98.5 | ? | 0 |

## Phase 1 — Our-System-NoCkpt (1 run, ckpt-off ablation)

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `03_nockpt/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 312.6 | 490 | 10241 | 44.7 | 63.7 | 24.0 | 98.5 | ? | 1864 |

## Phase 1 — cProfile WITH ckpt + fault

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `04_cprofile/Our-System/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 135.0 | 14238 | 46995 | 73.2 | 111.3 | 64.1 | 100.0 | ? | 3671 |

## Phase 1 — cProfile WITHOUT ckpt + fault

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `05_nockpt_cprofile/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 316.4 | 478 | 9469 | 44.1 | 63.4 | 23.4 | 97.6 | ? | 2063 |

## Phase 1 — A5000 profile swap

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `06_reload_a5000_profile/Our-System/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 79.1 | 1191 | 7721 | 72.9 | 132.5 | 57.6 | 29.9 | ? | 0 |

## Phase 1 — cProfile no fault (steady state)

_1 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `07_cprofile_nofault/Our-System/W1_Chat/Heavy/none/42/metrics.json` | 402.7 | 422 | 1737 | 53.0 | 82.5 | 5.4 | 100.0 | ? | 0 |

## Phase 2 — Variance: Our-System reload (3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase2_variance/reload_s123/Our-System/W1_Chat/Heavy/F2_Mid/123/metrics.json` | 103.5 | 34135 | 63415 | 67.2 | 103.9 | 67.5 | 96.1 | ? | 7529 |
| `phase2_variance/reload_s42/Our-System/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 120.4 | 16581 | 52337 | 73.6 | 114.8 | 66.5 | 100.0 | ? | 7035 |
| `phase2_variance/reload_s456/Our-System/W1_Chat/Heavy/F2_Mid/456/metrics.json` | 128.0 | 3905 | 43948 | 70.7 | 104.8 | 60.3 | 98.0 | ? | 7554 |

**goodput mean ± stdev**: 117.3 ± 12.6 tok/s (min 103.5, max 128.0)

## Phase 2 — Variance: Our-System-NoCkpt (3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase2_variance/nockpt_s123/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/123/metrics.json` | 196.6 | 1914 | 9119 | 53.4 | 66.9 | 49.6 | 98.7 | ? | 2357 |
| `phase2_variance/nockpt_s42/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 315.3 | 500 | 10175 | 44.6 | 63.7 | 23.6 | 98.7 | ? | 1509 |
| `phase2_variance/nockpt_s456/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/456/metrics.json` | 363.1 | 393 | 1651 | 43.1 | 59.8 | 5.9 | 97.5 | ? | 2249 |

**goodput mean ± stdev**: 291.7 ± 85.7 tok/s (min 196.6, max 363.1)

## Phase 3 — Steady state: Our-System (no fault, 3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase3_steady/reload_s123/Our-System/W1_Chat/Heavy/none/123/metrics.json` | 395.5 | 406 | 1609 | 50.7 | 80.4 | 5.8 | 96.1 | ? | 0 |
| `phase3_steady/reload_s42/Our-System/W1_Chat/Heavy/none/42/metrics.json` | 390.5 | 413 | 1928 | 52.2 | 83.7 | 7.8 | 100.0 | ? | 0 |
| `phase3_steady/reload_s456/Our-System/W1_Chat/Heavy/none/456/metrics.json` | 376.2 | 411 | 1564 | 48.8 | 83.1 | 5.0 | 99.5 | ? | 0 |

**goodput mean ± stdev**: 387.4 ± 10.0 tok/s (min 376.2, max 395.5)

## Phase 3 — Steady state: Our-System-NoCkpt (no fault, 3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase3_steady/nockpt_s123/Our-System-NoCkpt/W1_Chat/Heavy/none/123/metrics.json` | 410.5 | 307 | 773 | 33.3 | 42.8 | 1.3 | 98.7 | ? | 0 |
| `phase3_steady/nockpt_s42/Our-System-NoCkpt/W1_Chat/Heavy/none/42/metrics.json` | 413.3 | 310 | 774 | 34.2 | 44.6 | 0.4 | 99.6 | ? | 0 |
| `phase3_steady/nockpt_s456/Our-System-NoCkpt/W1_Chat/Heavy/none/456/metrics.json` | 381.8 | 291 | 755 | 32.9 | 39.0 | 1.1 | 99.1 | ? | 0 |

**goodput mean ± stdev**: 401.9 ± 17.4 tok/s (min 381.8, max 413.3)

## Phase 4 — Our-System + FT_DISABLE_SNAPSHOTS=1 (3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase4_no_snapshots/reload_s123/Our-System/W1_Chat/Heavy/F2_Mid/123/metrics.json` | 118.3 | 31850 | 65109 | 70.0 | 109.0 | 63.9 | 100.0 | ? | 4138 |
| `phase4_no_snapshots/reload_s42/Our-System/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 125.1 | 15215 | 50097 | 74.0 | 109.0 | 65.2 | 100.0 | ? | 9004 |
| `phase4_no_snapshots/reload_s456/Our-System/W1_Chat/Heavy/F2_Mid/456/metrics.json` | 155.7 | 1919 | 36112 | 68.0 | 104.4 | 54.6 | 100.0 | ? | 3961 |

**goodput mean ± stdev**: 133.1 ± 19.9 tok/s (min 118.3, max 155.7)

## Phase 4 — Our-System-NoCkpt + FT_DISABLE_SNAPSHOTS=1 (3 seeds)

_3 rows_

| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |
|---|---|---|---|---|---|---|---|---|---|
| `phase4_combined/nockpt_nosnap_s123/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/123/metrics.json` | 201.3 | 1985 | 8759 | 53.5 | 67.4 | 50.0 | 100.0 | ? | 2027 |
| `phase4_combined/nockpt_nosnap_s42/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/42/metrics.json` | 309.6 | 472 | 10325 | 44.5 | 64.0 | 24.7 | 100.0 | ? | 1862 |
| `phase4_combined/nockpt_nosnap_s456/Our-System-NoCkpt/W1_Chat/Heavy/F2_Mid/456/metrics.json` | 374.5 | 382 | 1625 | 42.8 | 59.6 | 3.4 | 100.0 | ? | 2086 |

**goodput mean ± stdev**: 295.1 ± 87.5 tok/s (min 201.3, max 374.5)


## cProfile dump locations

- `phase3_steady/nockpt_s123/ft_profile.txt`
- `phase3_steady/nockpt_s42/ft_profile.txt`
- `phase3_steady/nockpt_s456/ft_profile.txt`
- `phase3_steady/reload_s123/ft_profile.txt`
- `phase3_steady/reload_s42/ft_profile.txt`
- `phase3_steady/reload_s456/ft_profile.txt`
- `04_cprofile/ft_schedule_profile.txt`
- `05_nockpt_cprofile/ft_schedule_profile.txt`
- `07_cprofile_nofault/ft_schedule_profile.txt`
