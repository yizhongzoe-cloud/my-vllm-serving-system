# Ckpt Overhead Experiment — Status

Started: 2026-05-02 02:38 PT

## What's running

- **Main sweep**: `experiments_v2/run_ckpt_overhead_sweep.sh`
- **PID**: see `ps aux | grep run_ckpt_overhead_sweep`
- **Detached** with `nohup` + `disown` → independent of Claude Code session
- **Live log**: `experiments_v2/results_ckpt_overhead/sweep.log`
- **30 cells**: 5 contexts (1K/4K/8K/16K/32K) × 3 seeds × 2 baselines (No-FT, CkptOnly)
- **Estimated wall time**: 6-8 hours

## Sanity verification (passed before launch)

- Ckpt fire pattern: **1506 delta : 201 full** (88% delta) — incremental confirmed
- Per-fire delta size: **2 MB** (1 KV block × 32 layers, every 16 new tokens)
- First-save full size: **134 MB** (prompt KV after prefill, expected)
- Forward time at 1K context, saturation RPS: No-FT 265ms vs CkptOnly 319ms (+20%) — short-context single-cell signal; main sweep produces accurate paired diff across contexts.

## Where to look when you wake up

```
cd /home/yzhong76/code/my-vllm-serving-system
tail -200 experiments_v2/results_ckpt_overhead/sweep.log
ls experiments_v2/results_ckpt_overhead/W_Ruler*/                      # per-cell dirs
cat experiments_v2/results_ckpt_overhead/FAILED.txt                    # any cell failures
cat experiments_v2/results_ckpt_overhead/summary.txt                   # written when sweep completes
```

## Key bug fixed during sanity

`checkpoint_pool_bytes` raised from 16 GB → 128 GB.
Old value caused KVCheckpointPool eviction under high in-flight req
counts → evicted entries forced subsequent saves down the "full save"
path → broke incremental ckpt semantics. Fixed in
`experiments_v2/config_8b_ckpt_overhead.yaml`.

## Known risk: 32K RPS may trigger capacity-driven preempt

At RPS 0.3 with 32K context, expected in-flight ≈ 8 reqs vs A6000
KV capacity ≈ 6 reqs. vLLM may auto-trigger RECOMPUTE preempt to
free KV. Both No-FT and CkptOnly will preempt similarly (M3 is
disabled, so neither uses reload), so paired diff should still
isolate ckpt overhead — but absolute forward times for 32K cells
will be noisier than shorter contexts.

**To verify after sweep**:
```bash
grep -i "preempt" experiments_v2/results_ckpt_overhead/W_Ruler32K_*/server.log | wc -l
```
If preempt count is high (>20 per cell), consider rerunning W_Ruler32K
cells only with RPS lowered to 0.15:
```bash
# Edit config_8b_ckpt_overhead.yaml: Sat_32K rps 0.3 -> 0.15
# Then rerun just the 32K cells
```

16K is borderline; 1K/4K/8K are safe.

## Phase 4 (analysis) — when sweep completes

```bash
python experiments_v2/analysis/ckpt_overhead.py
```

Produces:
- `experiments_v2/results_ckpt_overhead/paired_diff.csv` — per (workload, seed) raw diffs
- `experiments_v2/results_ckpt_overhead/summary.md` — aggregated overhead per context
- `experiments_v2/results_ckpt_overhead/figures/scaling_overhead.png` — main figure

## Per-cell directory layout

```
W_Ruler<N>K_<Baseline>_seed<seed>/
├── forward_times_pid<N>.csv      # CUDA-event-timed forward kernel time
├── ckpt_stats_pid<N>.csv         # Per-fire ckpt log (CkptOnly only)
├── metrics.json                  # vLLM run aggregate
├── requests.csv                  # per-req TTFT/TPOT
├── server.log                    # vLLM server stdout
└── run.log                       # launcher log
```

## Env vars used (set in sweep script)

```
FT_CUDA_EVENT_PROFILE=1     # Enable CUDA event timing in _model_forward
FT_CKPT_STATS_LOG=1         # Enable per-fire ckpt logging
FT_DELTA_CHECKPOINT=1       # Enable incremental delta save
FT_SLO_PREEMPT=0            # Disable M3
FT_USE_FCFS_BASE_QUEUE=1    # Disable SLO-aware queue
FT_SKIP_SOLVER=1            # Disable Benders
FT_SLO_AWARE_OBJECTIVE=0    # Disable SLO objective
CUDA_VISIBLE_DEVICES=0      # Single GPU only
```
