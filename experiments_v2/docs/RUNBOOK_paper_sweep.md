# Paper sweep runbook (2026-05-16 v2 — mixed short+long)

## What this runs

Single master script `experiments_v2/eval/scripts/run_paper_sweep_master.sh`
that does, in sequence:

1. **PHASE 1: E_M1 mixed_short_long** — main per-class SLO attainment sweep.
   - Workload: 70% ShareGPT (short) + 30% ArXiv-Summarization (long), interleaved Poisson arrival.
   - Per-class SLO (JITServe-style): short tier from ShareGPT P95×2, long tier from ArXiv P95×2.
   - 4 baselines (vllm_fcfs / reroute_no_ckpt / ours_no_picker / ours) × 5 QPS × 3 seeds = 60 runs.

2. **PHASE 2: E_M2 mixed_short_long tightness** — SLO sensitivity at fixed QPS.
   - Same mixed workload; fixed QPS=1.5; SLO tightness factor ∈ {1.5, 2, 3, 4}× baseline P95.
   - 4 baselines × 4 tightness × 2 seeds = 32 runs (seed 100/101 to avoid collision with PHASE 1).

3. **PHASE 3: E_M3 sharegpt** — no-load overhead microbench.
   - ShareGPT, 40 req at 5s fixed inter-arrival (~0.2 QPS, single-request batches).
   - 3 baselines (vllm_fcfs / reroute_no_ckpt / ours) × 3 seeds = 9 runs.

**E_M4** (picker ablation) reuses PHASE 1's `ours` and `ours_no_picker` columns; no separate run.
**E_D1** (failover) already done in earlier sweep (RULER 16K); not repeated.

Total time on A6000: ~4.5h. On L40S: ~3-4h (faster prefill).

---

## Prerequisites

### 1. Per-GPU calibration must already exist

For each GPU box, run **once** before any sweep:

```bash
cd ~/code/my-vllm-serving-system
PYTHONPATH=. python -m experiments_v2.eval.scripts.slo_calibration \
  --dataset sharegpt --num-requests 30 --arrival-rate-qps 0.1 --seed 0
PYTHONPATH=. python -m experiments_v2.eval.scripts.slo_calibration \
  --dataset arxivsumm --num-requests 30 --arrival-rate-qps 0.05 --seed 0
```

Writes `experiments_v2/eval/results/<gpu>/slo_calib_<ds>_n30_qps0.<x>_seed0_metrics.json`.

Read `ttft_ms.p95` and `tpot_ms.p95` from each calibration JSON. These are
the `SHORT_P95_*` (from sharegpt calib) and `LONG_P95_*` (from arxivsumm calib)
env vars passed to the master sweep.

### 2. ArXiv-Summ dataset must be cached locally

`experiments_v2/datasets/cached/arxivsumm.jsonl` is **NOT in git** (198MB > GitHub limit). Each box must build it once:

```bash
python3 experiments_v2/datasets/make_arxivsumm.py
```

Downloads ccdv/arxiv-summarization test split (~5 min, ~200MB cached), filters
to prompt_tokens ∈ [1K, 30K], shuffles seed=42.

### 3. GPU must be clean

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Both GPUs should show <2GB. If higher, kill zombies first:

```bash
# 1. Kill api_server frontends
pkill -9 -f "api_server" 2>/dev/null

# 2. Find leftover VLLM::EngineCore workers (these are the real memory hogs)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 3. Kill each by PID
kill -9 <PID1> <PID2>

# 4. Recheck
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

The master script also does a `check_gpu_free` guard on startup and aborts
if >2GB used. **Do not skip this guard**; running with leftover memory was
the cause of the previous all-fail sweep.

### 4. Clean shm

```bash
rm -rf /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_preempt_queue
```

---

## Launching the master sweep

### A6000

```bash
cd ~/code/my-vllm-serving-system

nohup setsid env HARDWARE_TAG=a6000 \
  SHORT_P95_TTFT_MS=456 SHORT_P95_TPOT_MS=22 \
  LONG_P95_TTFT_MS=1951 LONG_P95_TPOT_MS=26 \
  bash experiments_v2/eval/scripts/run_paper_sweep_master.sh \
  > /tmp/master_a6000.log 2>&1 &
disown
echo "PID=$!"
```

### L40S

Read your own per-GPU calibration P95 values first by inspecting the
calibration JSONs in `experiments_v2/eval/results/l40s/slo_calib_*.json`.

```bash
cd ~/code/my-vllm-serving-system

nohup setsid env HARDWARE_TAG=l40s \
  SHORT_P95_TTFT_MS=268 SHORT_P95_TPOT_MS=21 \
  LONG_P95_TTFT_MS=<your_L40S_arxivsumm_p95_ttft> \
  LONG_P95_TPOT_MS=<your_L40S_arxivsumm_p95_tpot> \
  bash experiments_v2/eval/scripts/run_paper_sweep_master.sh \
  > /tmp/master_l40s.log 2>&1 &
disown
```

---

## Monitoring

```bash
# Master log (high-level phase progression)
tail -f /tmp/master_a6000.log

# Or via the in-results-dir copy (preserves across /tmp clears)
tail -f experiments_v2/eval/results/a6000/master_sweep_*.log

# Watch GPU
watch -n 5 nvidia-smi
```

### Check progress mid-run

```bash
# Count completed metrics.json by phase
ls experiments_v2/eval/results/a6000/e_m1_*_mixed_short_long_*_metrics.json | wc -l
# expected: phase 1 = 60, phase 2 = 32 (seeds 100/101)

ls experiments_v2/eval/results/a6000/e_m3_*_sharegpt_*_metrics.json | wc -l
# expected: phase 3 = 9
```

### What "healthy" looks like in the master log

```
[master] GPU pre-check OK: max used 15 MiB
[master] PHASE 1: E_M1 mixed_short_long — per-class SLO
[master] E_M1 mixed baseline=vllm_fcfs qps=0.5 seed=0
[E_M1] schedule: {'n': 60, ...}
[E_M1]   composition: short=42 long=18
[E_M1]   short: TTFT≤912ms, TPOT≤44.0ms
[E_M1]   long: TTFT≤3902ms, TPOT≤52.0ms
[E_M1] both engines ready
[E_M1] fired 60 requests; waiting completion
[E_M1] outcomes: 60/60 200 OK, 0 errored
[E_M1] SLO_met overall: 93.3% (56/60)
[E_M1]   short: SLO_met=95.2% (n=42)
[E_M1]   long: SLO_met=88.9% (n=18)
```

### Red flags

- `[master] FAIL: GPU has XXXX MiB used at startup` — zombie processes. Kill and restart (see Prerequisites #3).
- `[E_M1] FAIL: an engine never became healthy` — engine OOM or import error. Check `engine0.log` / `engine1.log` for stack trace.
- `[E_M1] WARN: ... FAILED` repeatedly — same as above. Stop and diagnose; don't let it grind through 100 failed runs.
- `composition: short=0 long=N` or `short=N long=0` — workload builder broke; check `mixed_short_long` registry in `workload_builder.py`.

---

## Recovery

If sweep dies mid-way, the master script is not resumable from a checkpoint;
just re-launch. Existing metrics.json files will be **overwritten** without
warning, so back up or move them aside first if you care about partial data.

To resume only the remaining work, manually edit the `SEEDS` / `E_M1_QPS` /
`E_M2_TIGHTNESS_FACTORS` env vars in your launch command to skip already-
completed combinations.

---

## Where output ends up

```
experiments_v2/eval/results/<HARDWARE_TAG>/
  master_sweep_<timestamp>.log                                  ← master log
  e_m1_<baseline>_mixed_short_long_qps<q>_n60_seed<s>_metrics.json
  e_m1_<baseline>_mixed_short_long_qps<q>_n60_seed<s>_engine0.log
  e_m1_<baseline>_mixed_short_long_qps<q>_n60_seed<s>_engine1.log
  e_m1_<baseline>_mixed_short_long_qps<q>_n60_seed<s>_router.log (if applicable)
  e_m3_<baseline>_sharegpt_n40_seed<s>_metrics.json
```

For analysis, key fields in metrics.json:

```python
{
  "baseline": "...",
  "dataset": "mixed_short_long",
  "slo_mode": "mixed",
  "slo": {
    "tier_ttft_ms": {"short": ..., "long": ...},  # short/long, not tight/normal/loose
    "tier_tpot_ms": {"short": ..., "long": ...},
  },
  "slo_met_pct": float,         # overall
  "per_class": {
    "short": {"n": ..., "slo_met_pct": ..., "ttft_p50": ..., "ttft_p95": ..., "tpot_p95": ...},
    "long":  {"n": ..., "slo_met_pct": ..., "ttft_p50": ..., "ttft_p95": ..., "tpot_p95": ...},
  },
  "per_request": [...],          # per-request details with class tag
}
```
