# Overnight investigation — 2026-04-09 night

## Goal

Find the actual source of Our-System's framework baseline overhead so we can attack it tomorrow. Today's session established that the gap between Our-System (124 tok/s) and No-FT (321 tok/s) on W1_Chat/Heavy is **~200 tok/s**, and that this gap is **NOT** in the recovery path (drop mode showed ~117 tok/s — almost identical to reload). The cost is somewhere in the everyday FT machinery. Tonight's job is to localize *which part* of that machinery.

## TL;DR for the morning

1. **Read first**: [results_v2/8B_overnight_2026-04-09/SUMMARY.md](../../results_v2/8B_overnight_2026-04-09/SUMMARY.md) — auto-generated table of every run
2. **Check phase 4** (snapshot bypass): Compare `phase4_no_snapshots/reload_s*` against `phase2_variance/reload_s*`. Same 3 seeds, only difference is `FT_DISABLE_SNAPSHOTS=1`. If goodput jump > 1 stdev, snapshot construction was a meaningful overhead source.
3. **Check phase 1 #3** (`Our-System-NoCkpt`): if its goodput is > 200 tok/s, CheckpointController is the dominant overhead.
4. **Read cProfile dumps**: the multi-seed dumps are at `phase3_steady/{reload,nockpt}_s{42,123,456}/ft_profile.txt`. Top 20 cumulative-time entries show where GIL time goes.
5. **Diff cProfile dumps**: the diff of `reload_s42/ft_profile.txt` vs `nockpt_s42/ft_profile.txt` isolates checkpoint controller's cost.

## Autonomous execution plan (running tonight)

The user gave 8 hours of autonomous time. Plan:

| Phase | What | How | Expected runtime |
|---|---|---|---|
| 1 | 7-experiment investigation suite | `overnight_2026-04-09.sh` (already running when master launched) | ~60 min |
| 2 | Multi-seed variance baselines (3 seeds × {Our-System reload, Our-System-NoCkpt}) | `overnight_master.sh` Phase 2 | ~45 min |
| 3 | Multi-seed cProfile dumps no-fault (3 seeds × {Our-System, Our-System-NoCkpt}, fault=none) | `overnight_master.sh` Phase 3 | ~45 min |
| 4 | Snapshot bypass ablation: 3 seeds × Our-System with `FT_DISABLE_SNAPSHOTS=1`, plus 3 seeds × Our-System-NoCkpt + bypass | `overnight_master.sh` Phase 4 | ~45 min |
| 5 | Generate `SUMMARY.md` + cProfile dump locations | `overnight_master.sh` Phase 5 (Python aggregation) | ~1 min |

Total: ~3.5 hours of runs. Buffer is ample for retries / hangs.

## Code change committed tonight

**`8a7a98c59 feat(ft): add FT_DISABLE_SNAPSHOTS env var to bypass solver snapshot build`**

Adds opt-in `FT_DISABLE_SNAPSHOTS=1` env var. When set, `_build_active_snapshots()` and `_build_replica_snapshot()` in `vllm/v1/engine/core.py` return `None` immediately. Default is OFF (no behavior change). Phase 4 toggles this on to measure the snapshot path cost.

Justification: snapshots are constructed every ~100ms and shipped via msgpack/ZMQ to the API server, but they're only consumed by the centralized Benders solver which is in greedy fallback ~98% of the time on W1_Chat/Heavy (per today's diagnosis — see follow-up #6). When the solver is bypassed, the snapshots are wasted CPU + network.

## How to launch (history; this was already done before sleep)

```bash
cd /home/jlpang/my-vllm-serving-system

# Phase 1 (started first)
nohup bash experiments_v2/overnight_2026-04-09.sh > /tmp/overnight.log 2>&1 &
disown

# Master orchestrator (started after Phase 1, runs Phases 2-5)
nohup bash experiments_v2/overnight_master.sh > /tmp/overnight_master.log 2>&1 &
disown
```

To monitor:
```bash
tail -f /tmp/overnight.log
ls results_v2/8B_overnight_2026-04-09/
```

Expected runtime: ~50–60 minutes (7 sequential runs × ~7 min each).

> **Why sequential, not parallel?** `/dev/shm/vllm_ft_checkpoints` is hardcoded in `vllm/v1/engine/ft_client.py:53`. Two parallel `ft_benders_centralized` runs collide on `os.replace()` of checkpoint chunks (we hit this today — `active_requests_at_fault=0` and goodput≈0).

## What it runs

Held constant across all runs:
- workload `W1_Chat`, load `Heavy`, seed `42`
- `experiments_v2/config_8b.yaml`
- GPU `4,5` (CUDA_VISIBLE_DEVICES)
- port `8400`

| # | Tag | Baseline | Fault | Extra env | Purpose |
|---|---|---|---|---|---|
| 1 | `01_reload_baseline` | Our-System | F2_Mid | – | Fresh reload data point — variance bound for comparisons |
| 2 | `02_noft_baseline` | No-FT | F2_Mid | – | Fresh No-FT data point — upper-bound goal |
| 3 | `03_nockpt` | **Our-System-NoCkpt** | F2_Mid | – | **Key experiment**: ft_benders_centralized WITH checkpoint controller OFF |
| 4 | `04_cprofile` | Our-System | F2_Mid | `FT_PROFILE_MAX_CALLS=300` | cProfile of schedule() with checkpointing |
| 5 | `05_nockpt_cprofile` | Our-System-NoCkpt | F2_Mid | `FT_PROFILE_MAX_CALLS=300` | cProfile of schedule() WITHOUT checkpointing |
| 6 | `06_reload_a5000_profile` | Our-System | F2_Mid | swap profile | Test new A5000-rough decode capacity profile (default=26) |
| 7 | `07_cprofile_nofault` | Our-System | none | `FT_PROFILE_MAX_CALLS=300` | Steady-state cProfile (no recovery noise) |

## What each experiment tells us

### 1+2: Baselines

These re-establish the variance bound. We saw goodput swing 116–142 across runs today depending on `active_requests_at_fault` (12–17). New runs will tell us where in the noise band each subsequent test sits.

### 3: `Our-System-NoCkpt` (the most informative single test)

`Our-System-NoCkpt` is `ft_benders_centralized` policy + `enable_checkpointing: false` (added to config_8b.yaml in this session).

**Decision rule:**

| If goodput jumps to | Means | Next attack target |
|---|---|---|
| ~200+ tok/s | CheckpointController is the dominant overhead | Refactor checkpoint policy iteration / KV pool tracking |
| ~140-180 tok/s | CheckpointController is *part* of the overhead but not all | Need profile (exp 4-7) to find the rest |
| ~120 tok/s (no improvement) | CheckpointController is NOT the bottleneck | Snapshot collection / ft_client output processing / scheduler wrapper |

### 4: cProfile WITH fault + checkpointing

`FT_PROFILE_MAX_CALLS=300` enables the existing cProfile instrumentation in `ft_scheduler_impl.py:50-55`. Captures the first 300 `schedule()` calls and dumps cumulative stats to `04_cprofile/.../ft_schedule_profile.txt`.

**To analyze tomorrow:**
```bash
head -50 results_v2/8B_overnight_2026-04-09/04_cprofile/ft_schedule_profile.txt
```
Top 20 cumulative-time entries point at the actual hot functions on the FT scheduler path. Anything that's not in upstream `vllm/v1/core/sched/scheduler.py` is FT-specific overhead.

### 5: cProfile WITHOUT checkpointing

The diff between exp 4 and exp 5 isolates checkpoint controller's cost in the cProfile output. Subtract function-level cumulative times to see exactly which functions disappear when ckpt is off.

### 6: A5000 profile (sanity check the trade-off)

Today we found that `decode_capacity_profile_8b.json` is configured for A6000 dp=1 max_model_len=4096 (default=10), causing Benders to declare admission infeasible on every epoch (active running ~25 > capacity 10) → 98% greedy fallback. We tested an arbitrary fix (default=64) and it made goodput WORSE (124→43) because Benders started actively rejecting requests.

This run uses `decode_capacity_profile_8b_a5000.json` (rough estimate from observed runtime stats: default=26, measured-from-runtime). Expected: similar goodput regression to the v4 test but with a more honest profile. The point is to confirm the trade-off curve, not to "fix" anything.

> The original profile is restored to `decode_capacity_profile_8b.json` after this run.

### 7: cProfile WITHOUT fault

Same as exp 4 but `fault=none`. Removes the recovery-path noise from the cProfile aggregates. The 2 cProfile dumps (4 vs 7) together tell us:
- exp 7: pure steady-state per-step overhead
- exp 4 minus exp 7: failover-related per-step overhead spike

## What to look at in the morning

1. **Summary table** at the end of `/tmp/overnight.log` — quick goodput / TTFT comparison across all 7 runs.
2. **Exp 3 metrics.json** — does removing checkpointing recover meaningful goodput?
3. **Exp 7 cProfile dump** (`results_v2/8B_overnight_2026-04-09/07_cprofile_nofault/ft_schedule_profile.txt`) — top 20 by cumulative time. This is the real "where is time going" answer.
4. **Diff exp 4 vs exp 5 cProfile dumps** — find the functions that disappear when ckpt is off.

## Files added in this session

- `experiments_v2/config_8b.yaml` — added `Our-System-NoCkpt` baseline (lines after `Our-System`)
- `experiments_v2/decode_capacity_profile_8b_a5000.json` — new (rough A5000 dp=2 estimate)
- `experiments_v2/overnight_2026-04-09.sh` — new (sequential test runner)
- `experiments_v2/docs/overnight_2026-04-09.md` — this file

## Files NOT touched

- `experiments_v2/decode_capacity_profile_8b.json` — restored to original A6000 dp=1 version
- `vllm/v1/engine/ft_client.py` — committed logfix (`ec533cd72`), no other changes since
- All other source files

## Decision point after overnight

Based on exp 3 + exp 4/5/7 cProfile results, pick the next attack:

- **CheckpointController-dominated** → refactor `ft_scheduler.py:248-315` `run_checkpoint_step()` to skip iteration when no req is eligible (cache last-decision based on token count delta)
- **ft_client output processing-dominated** → batch `process_engine_outputs()` per-output dict updates
- **Scheduler wrapper-dominated** → add a fast path that bypasses ft_scheduler wrapping when no failover state is in flight
- **Snapshot collection-dominated** → add an env var to disable `active_request_snapshots`/`replica_snapshots` collection when Benders is in greedy fallback mode (which is currently 98% of the time)
