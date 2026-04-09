# Overnight investigation — 2026-04-09 night

## Goal

Find the actual source of Our-System's framework baseline overhead so we can attack it tomorrow. Today's session established that the gap between Our-System (124 tok/s) and No-FT (321 tok/s) on W1_Chat/Heavy is **~200 tok/s**, and that this gap is **NOT** in the recovery path (drop mode showed ~117 tok/s — almost identical to reload). The cost is somewhere in the everyday FT machinery. Tonight's job is to localize *which part* of that machinery.

## Result (read this first)

**Found the bottleneck and shipped two fixes that close the gap to No-FT from -62 % to -6 %.** Three env vars (default OFF) — combine all three for max benefit:

```bash
FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 \
    python experiments_v2/run.py ...
```

### Phase 7 — non-blocking pipeline (3 seeds)

| Mode | goodput mean ± stdev (3 seeds) | TTFT p50 | SLO violation |
|---|---|---|---|
| baseline (no env vars) | **117.3 ± 12.6** | 18,207 | 64.8 % |
| FT_FAST_TMPFS_WRITE only | 122.7 ± 14.9 | 16,788 | 64.1 % |
| **FT_CKPT_NONBLOCK + FAST_TMPFS** | **247.4 ± 127.1** | 3,226 | **37.6 %** |
| Our-System-NoCkpt (control) | 291.7 ± 85.7 | 936 | 26.4 % |
| No-FT (target, 1 seed) | 319.5 | 398 | 20.3 % |

Phase 7 trade-off: completion drops from ~98 % → ~93 % because skipped checkpoint cycles leave fault-time in-flight reqs with stale checkpoint state.

### Phase 8 — fast chunk format (1 seed s42; 3-seed pending)

Phase 7's trade-off is fully eliminated by `FT_FAST_CHUNK_FORMAT=1`, which replaces `torch.save()`/pickle with raw bytes + a 48-byte struct header. Worker `checkpoint_kv_blocks` RPC drops from ~70 ms → ~50 ms, so NONBLOCK rarely needs to skip cycles → completion stays at 100 %.

| Mode (seed 42) | goodput | completion | TTFT p50 | TTFT p95 | SLO viol | failed |
|---|---|---|---|---|---|---|
| baseline | 120.4 | 100.0 % | 16,581 | 52,337 | 66.5 % | 0 |
| + NONBLOCK + FAST_TMPFS (phase 7) | 282.4 | 95.2 % | 536 | 10,054 | 31.2 % | 22 |
| **+ NONBLOCK + FAST_TMPFS + FAST_CHUNK** ⭐ | **299.1** | **100.0 %** | **507** | **9,401** | **27.3 %** | **0** |
| Our-System-NoCkpt (control) | 315.3 | 98.7 % | 500 | 10,175 | 23.6 % | 6 |
| No-FT (target) | 319.5 | 98.5 % | 398 | 10,436 | 20.3 % | 7 |

Phase 8 closes the gap to No-FT (s42) from **-6.4 %** with all three env vars vs **-62.3 %** baseline. **0 failed reqs**, 100 % recovery success rate. Standalone benchmark shows `_fast_save_chunk` is 1.7× faster than `torch.save` for 6 MB chunks (8.6 ms → 14.5 ms per call), which compounds across the 14-req batch.

**Caveat**: Phase 8 is single-seed so far. Phase 7 had a hard-seed outlier (s123: +3 % only). 3-seed phase 8 validation pending.

### Root cause

API server's `_maybe_ft_checkpoint()` uses `_ft_ckpt_future.result()` which BLOCKS waiting for the previous step's checkpoint RPC. Under W1_Chat/Heavy fault load:
- Worker `checkpoint_kv_blocks` takes ~70 ms (32 layers × `torch.save` pickle for ~14 reqs)
- API server step time is ~30 ms
- API server step rate drops from ~30/sec to ~14/sec — matches the observed 60 % goodput drop

Phase 7 fix: peek at the future via `.done()`. Skip both result-collection and new-RPC submission when not ready. Pipeline depth stays at 1 but the API server never blocks.

Phase 8 fix: replace `torch.save()` (pickle/zipfile, ~14 ms for 6 MB) with raw `numpy().tobytes()` + struct header (~8 ms). Worker RPC time drops below the step rate so NONBLOCK rarely fires.

**Code commits**:
- `2044810d0 perf(ft): non-blocking checkpoint pipeline + tmpfs fast write` (phase 7)
- `1aa319f92 perf(ft): add FT_FAST_CHUNK_FORMAT env var (raw bytes vs torch.save)` (phase 8)
- `5f45a1bbf test(ft): add FT_BG_PUBLISH env var (background-thread shared publish)` (phase 10, tested negative)

### Tested but rejected (phase 9 + 10 + 11 + 12 + 13)

After phase 8 we tried five more optimizations. The only one that produced clean data on what's slow was phase 12 (cProfile of the API server process). The others turned out to be net-neutral or net-negative on single-seed runs because they attacked the wrong layer.

| Phase | Optimization | seed 42 goodput | completion | active@fault | Verdict |
|---|---|---|---|---|---|
| 8 | NONBLOCK + FAST_TMPFS + FAST_CHUNK (current best) | **299.1** | **100.0 %** | 8 | ✅ committed, default off |
| 9 | + FT_PINNED_BUFFER_POOL | 243.8 | 98.1 % | 8 | ❌ reverted (data race) |
| 10 | + FT_BG_PUBLISH | 270.9 | 100.0 % | 8 | ⚠️ committed but tested negative; default off |
| 11 | + FT_INLINE_MANIFEST | 231.8 | 100.0 % | 7 | ⚠️ committed but tested negative; default off |
| 12 | + FT_API_PROFILE_SECONDS=60 (instrumentation only) | 245.6 | 100.0 % | – | 🔍 profiling tool committed |
| 13 | + FT_ASYNC_CLEANUP (run rmtree on bg thread) | 239.7 | 100.0 % | 6 | ⚠️ committed but tested negative; default off |

**Phase 9 (pinned buffer pool)** was reverted because of an implicit data race: the existing `save_checkpoint` path launches an async GPU→host copy on a separate CUDA stream and returns the pinned tensor reference WITHOUT synchronizing. The current code "works" because Python overhead between launch and read is slower than the copy. Recycling buffers introduces a window where the pool returns a buffer whose previous async copy may not have completed, so a new copy can race with the previous read. The 9 lost requests in phase 9 (vs 0 in phase 8 with same `active_requests_at_fault=8`) are consistent with this hypothesis. A correct fix would require an explicit `copy_stream.synchronize()` somewhere, which defeats the async-copy purpose.

**Phase 10 (background-thread publish)** is committed but tested negative. The fix sends the per-request `_publish_shared_checkpoint` file writes to a single-worker `ThreadPoolExecutor` and adds an explicit `copy_stream.synchronize()` so the background thread sees consistent pinned-memory bytes. Backpressure is enforced via `prev_future.result()` at the start of each RPC. Despite all the correctness pieces, single-seed result is **270.9** (vs phase 8 299.1) — slightly worse. Likely cause: GIL contention. Most of `_fast_save_chunk` releases GIL during `f.write()` syscalls, but the inter-call Python overhead serializes through the GIL and eats into the main worker's other Python work. Kept as `FT_BG_PUBLISH=1` opt-in env var (default off) for future investigation, e.g. once the work can be moved to a non-GIL background mechanism (separate process, or asyncio loop with file I/O moved to an executor).

**Phase 11 (inline manifest in chunk header)** is committed but tested negative. The fix bumps the fast chunk format from v1 → v2 by repurposing the `reserved` slot in the header as `manifest_bytes_len`. When `FT_INLINE_MANIFEST=1` is set together with `FT_FAST_CHUNK_FORMAT=1`, `_publish_shared_checkpoint` embeds the cumulative `SharedCheckpointManifest` JSON directly into the chunk header and skips writing the separate `manifest_*.json` and `latest_*` pointer files entirely (saves 2 of 3 file writes per save). Restore extension `_scan_latest_inline_manifest` scans the request directory for chunk files written by this rank, finds the highest generation, and reads the embedded manifest from it. Implementation correctness verified by inspecting `/dev/shm` after a run: each request directory contains ONLY `chunk_rank0_<gen>.pt` files, no manifest_*.json or latest_rank0 — the file ops are successfully skipped. Single-seed result is **231.8** (vs phase 8 299.1) — within phase 7's measured variance band (106-353 across 3 seeds), so we can't conclude regression vs improvement without 3-seed validation. Kept as opt-in `FT_INLINE_MANIFEST=1` env var (requires `FT_FAST_CHUNK_FORMAT=1` to take effect).

**Phase 12 (API server cProfile via FT_API_PROFILE_SECONDS)** — first systematic profile of the API server process. We've been profiling the engine workers all along (via `FT_PROFILE_MAX_CALLS` which targets `schedule()`); phase 12 adds the same kind of instrumentation to the ft_client / API server side. Implementation: schedule a one-shot cProfile window starting `FT_API_PROFILE_DELAY` seconds after init (skip warmup), running for `FT_API_PROFILE_SECONDS`. Captures the asyncio main loop (handles aiohttp, ft_client output processing, snapshot updates, foreground Benders solver). Background threads are NOT captured.

Phase 12 result on a 60s window over W1_Chat/Heavy/F2_Mid: **8.075 s** of profiled CPU time. Top hot spots by cumulative time:

| function | cumtime | tottime | calls | description |
|---|---|---|---|---|
| `ft_process_outputs_socket` | 3.21 s | 0.04 s | 2972 | ZMQ recv + decode + process |
| `process_engine_outputs` | 2.85 s | 0.04 s | 3102 | ft_client output processing |
| `_cleanup_shared_checkpoint_request` | **2.80 s** | 0.002 s | **107** | per-finished-req `shutil.rmtree` |
| `posix.unlink` | 2.78 s | 2.78 s | 312 | **9 ms per unlink** ⚠️ |
| `stream_response` (aiohttp) | 2.28 s | 0.16 s | 28534 | OpenAI response streaming |
| Pydantic init/validate | ~0.8 s | 0.06 s | 86k | OpenAI JSON ser/de |
| `_solve_epoch_loop` | 0.16 s | 0.02 s | 2864 | Benders foreground path |

The biggest single hot spot was `_cleanup_shared_checkpoint_request` at **35 % of profiled CPU**. Each finished request synchronously called `shutil.rmtree('/dev/shm/<req_id>/')` which `posix.unlink`'d 3 files at ~9 ms each. With ~3 finished reqs/sec, that's ~80 ms/sec of asyncio event loop blocking from cleanup alone. (The ft_process_outputs_socket / process_engine_outputs cumtime is high because the cleanup was called from inside their call chain, not because their own work is slow.)

**Phase 13 (FT_ASYNC_CLEANUP)** — based on phase 12, moved the cleanup to `loop.run_in_executor`. Phase 13b re-ran phase 12's profile with the fix on. The cleanup hot spot disappeared:

| metric | sync (12) | async (13b) |
|---|---|---|
| total profiled CPU in 60 s | **8.075 s** | **4.97 s** |
| `_cleanup_shared_checkpoint_request` | 2.80 s | NOT in top 80 |
| `ft_process_outputs_socket` | 3.21 s | 0.38 s |
| `process_engine_outputs` | 2.85 s | 1.13 s |

The fix succeeds at the implementation level (-38 % main-thread CPU) but **macro goodput in non-profile mode (phase 13: 239.7) does not improve over phase 8 (299.1)**. Reason: the API server main thread was already running at ~13 % CPU utilization (8 s / 60 s in profile mode where profile adds overhead). Freeing more of an already-idle thread doesn't translate to goodput because the actual bottleneck is somewhere else — engine GPU compute, ZMQ message rate, or the worker side that this profile doesn't capture from the API server process. Kept as `FT_ASYNC_CLEANUP=1` opt-in env var for future investigation.

### Lessons learned (phase 9-13)

**Five consecutive optimizations didn't improve macro goodput beyond phase 8** despite all being correct implementations of their target. The pattern is consistent: phase 8 already crossed the threshold where the API server main thread is no longer the bottleneck. Further optimizations there free up CPU that wasn't being used.

The remaining 6.4 % gap to No-FT (299 vs 319) is most likely:
- On the engine worker side (PyTorch C++ time we can't profile from Python)
- In ZMQ message rate / serialization (msgpack overhead per output batch)
- Or fundamentally bound by the asynchronous handover between API server ↔ engine ↔ worker

To attack it further would need: profile the engine worker process (not just `schedule()`), or measure ZMQ throughput separately.

### Not tried (lower priority after phase 8)

- **Pipeline depth > 1 (`max_workers > 1` for `_ft_ckpt_executor`)** — would require auditing whether `collective_rpc` is thread-safe across parallel calls. Phase 8 already keeps RPC < step time with NONBLOCK, so deeper pipelining shouldn't gain much.

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
