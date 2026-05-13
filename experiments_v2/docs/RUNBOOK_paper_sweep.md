# Paper sweep runbook

How to run the 4-experiment (E_M1/M2/M3/M4) paper sweep on each
hardware platform, and how the results are kept separate so we can
compare A6000 vs L40S and pick the one that gives the better story.

---

## 1. Output layout

Every eval script reads an `EVAL_RESULTS_DIR` environment variable
that overrides the default results directory.

| `HARDWARE_TAG` | Output goes to |
|---|---|
| (unset, default) | `experiments_v2/eval/results/` |
| `a6000` | `experiments_v2/eval/results/a6000/` |
| `l40s` | `experiments_v2/eval/results/l40s/` |

Pre-existing data (historical runs before we introduced the env var)
stays in the root `results/` directory. Calibration `read_calibration()`
in `e_m2_slo_tightness.py` looks at the hardware-specific dir first,
then falls back to root so the A6000-era data still wires up.

---

## 2. A6000 (this machine)

### 2a. Calibration

Already done historically. The files

```
experiments_v2/eval/results/slo_calib_sharegpt_n30_qps0.1_seed0_metrics.json
experiments_v2/eval/results/slo_calib_ruler_16k_n30_qps0.02_seed0_metrics.json
experiments_v2/eval/results/slo_calib_ruler_64k_n30_qps0.04_seed0_metrics.json
```

live at the project default `results/` path. E_M2's fallback picks
them up automatically when sweeping on A6000 with
`HARDWARE_TAG=a6000`.

### 2b. Full sweep

```bash
cd /home/yzhong76/code/my-vllm-serving-system

# Cleanup leftover shm from any previous run
rm -rf /dev/shm/vllm_ft_preempt_queue \
       /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

# Kick off — ~13-15 hours for 3 seeds × 4 experiments × ShareGPT.
# Use nohup if running over SSH so disconnects don't kill it.
HARDWARE_TAG=a6000 nohup \
  bash experiments_v2/eval/scripts/run_paper_sweep_sharegpt.sh \
  > /tmp/a6000_sweep.log 2>&1 &
echo "PID: $!"
```

### 2c. Monitor

```bash
# Master driver log (tee'd into both stdout and a file in results/)
tail -f experiments_v2/eval/results/a6000/paper_sweep_*.log

# What's running on the GPUs right now
watch -n 5 nvidia-smi

# How many metrics.json files have been written
ls experiments_v2/eval/results/a6000/*_metrics.json | wc -l
```

---

## 3. L40S (other machine)

### 3a. Sync code first

```bash
cd <repo path on L40S>
git checkout zoe/disruption
git pull origin zoe/disruption
```

### 3b. Calibration (must run before sweep)

L40S has never been profiled on this branch. Without a fresh
calibration, `E_M2` would fall back to the A6000 calibration in
`results/` root — which produces SLO thresholds that don't match
L40S's actual baseline P95. Run this once per dataset you intend to
sweep.

```bash
# ShareGPT calibration — required for the current sweep
rm -rf /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

EVAL_RESULTS_DIR="$(pwd)/experiments_v2/eval/results/l40s" \
PYTHONPATH="$(pwd)" \
python -m experiments_v2.eval.scripts.slo_calibration \
  --dataset sharegpt \
  --num-requests 30 \
  --arrival-rate-qps 0.1 \
  --seed 0

# Verify the output landed in the right place
ls experiments_v2/eval/results/l40s/slo_calib_sharegpt_*.json
```

Takes about 10 minutes (engine startup + 30 requests at QPS 0.1).

Eyeball the P95 numbers to make sure they look sane (TTFT P95 in the
few-hundred ms range, TPOT P95 ~20-30 ms on Qwen2.5-7B):

```bash
python -c "
import json
d = json.load(open('experiments_v2/eval/results/l40s/slo_calib_sharegpt_n30_qps0.1_seed0_metrics.json'))
print('TTFT p95:', d['ttft_ms']['p95'], 'ms')
print('TPOT p95:', d['tpot_ms']['p95'], 'ms')
"
```

### 3c. Full sweep

```bash
cd <repo path on L40S>

rm -rf /dev/shm/vllm_ft_preempt_queue \
       /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

HARDWARE_TAG=l40s nohup \
  bash experiments_v2/eval/scripts/run_paper_sweep_sharegpt.sh \
  > /tmp/l40s_sweep.log 2>&1 &
echo "PID: $!"
```

### 3d. Monitor

```bash
tail -f experiments_v2/eval/results/l40s/paper_sweep_*.log
```

---

## 4. RULER_16K sweep (both machines)

The ShareGPT sweep above uses 500-token prompts where prefill is
one-shot — picker has no mid-prefill window to fire in. RULER_16K
(~13K-token NIAH prompts) gives the picker a real workload: chunked
prefill splits each prompt into ~6 chunks, and picker can preempt a
mid-prefill victim once its host checkpoint has caught up.

Scope: E_M1 only (4 baselines × 3 QPS × 3 seeds = 36 runs, ~1.5h
per machine). The driver is
`experiments_v2/eval/scripts/run_paper_sweep_ruler16k.sh`.

### 4a. A6000

Calibration already exists at
`experiments_v2/eval/results/slo_calib_ruler_16k_n30_qps0.02_seed0_metrics.json`
(baseline TTFT P95 = 2736 ms, TPOT P95 = 24.3 ms). The default SLO
values in the sweep script (TTFT 5472/8208/16416 ms, TPOT 49/73/146 ms,
i.e. baseline × {2, 3, 6}) are computed from this.

```bash
cd /home/yzhong76/code/my-vllm-serving-system

rm -rf /dev/shm/vllm_ft_preempt_queue \
       /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

HARDWARE_TAG=a6000 nohup \
  bash experiments_v2/eval/scripts/run_paper_sweep_ruler16k.sh \
  > /tmp/a6000_ruler16k_sweep.log 2>&1 &
echo "PID: $!"
```

Monitor:
```bash
tail -f experiments_v2/eval/results/a6000/ruler16k_sweep_*.log
ls experiments_v2/eval/results/a6000/e_m1_*_ruler_16k_*_metrics.json | wc -l
```

### 4b. L40S calibration (first time only)

L40S has never been profiled on RULER_16K. Run calibration first so
the SLO thresholds reflect L40S's own baseline, not A6000's. Takes
~30 minutes (30 requests at QPS 0.02).

```bash
cd <repo path on L40S>
git checkout zoe/disruption
git pull origin zoe/disruption

rm -rf /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

EVAL_RESULTS_DIR="$(pwd)/experiments_v2/eval/results/l40s" \
PYTHONPATH="$(pwd)" \
python -m experiments_v2.eval.scripts.slo_calibration \
  --dataset ruler_16k \
  --num-requests 30 \
  --arrival-rate-qps 0.02 \
  --seed 0

# Verify file landed in the right place with the expected name
ls experiments_v2/eval/results/l40s/slo_calib_ruler_16k_n30_qps0.02_seed0_metrics.json
```

The QPS arg is mandatory: it must be `0.02` exactly, because the
downstream E_M2 driver looks for the file by that name
(see `e_m2_slo_tightness.py` `_CALIB_FILES`).

### 4c. L40S — derive SLO numbers from L40S calibration

```bash
python -c "
import json
m = json.load(open('experiments_v2/eval/results/l40s/slo_calib_ruler_16k_n30_qps0.02_seed0_metrics.json'))
ttft = m['ttft_ms']['p95']
tpot = m['tpot_ms']['p95']
print(f'# L40S baseline: TTFT P95 {ttft:.0f}ms, TPOT P95 {tpot:.1f}ms')
print(f'export E_M1_TTFT_TIGHT_MS={int(ttft*2)}')
print(f'export E_M1_TTFT_NORMAL_MS={int(ttft*3)}')
print(f'export E_M1_TTFT_LOOSE_MS={int(ttft*6)}')
print(f'export E_M1_TPOT_TIGHT_MS={int(round(tpot*2))}')
print(f'export E_M1_TPOT_NORMAL_MS={int(round(tpot*3))}')
print(f'export E_M1_TPOT_LOOSE_MS={int(round(tpot*6))}')
"
```

Copy-paste the `export` lines into your shell. They override the
defaults in the sweep script (which are A6000-derived).

### 4d. L40S full sweep

```bash
rm -rf /dev/shm/vllm_ft_preempt_queue \
       /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map \
       /dev/shm/vllm_ft_checkpoints

HARDWARE_TAG=l40s nohup \
  bash experiments_v2/eval/scripts/run_paper_sweep_ruler16k.sh \
  > /tmp/l40s_ruler16k_sweep.log 2>&1 &
echo "PID: $!"
```

Monitor:
```bash
tail -f experiments_v2/eval/results/l40s/ruler16k_sweep_*.log
```

### 4e. Sanity check — picker is actually firing

Unlike ShareGPT (where the picker correctly fired 0 times after the
manifest guard), RULER_16K runs should show picker fires. Look at
the engine logs of an `ours` run:

```bash
grep "PICKER_DIAG" experiments_v2/eval/results/<hw>/e_m1_ours_ruler_16k_qps2.0_n60_seed0_engine*.log | head
```

`host_manifest_exists=True` should be the dominant case. If you see
many fires with `host_manifest_exists=False`, the manifest guard is
broken. If you see zero fires across all 3 seeds, the workload is
not stressful enough — bump QPS or tighten SLO.

---

## 5. Which SLO each experiment uses

Heads-up because this is non-obvious: only E_M2 reads the calibration
file. The others use hardcoded SLO numbers. The multiplier is baseline
P95 × {2, 3, 6} for tight/normal/loose (tight was 1.5× before
2026-05-13; bumped to 2× because 1.5× was inside the batch-size
jitter on ShareGPT).

| Experiment | Reads calibration? | SLO source |
|---|---|---|
| E_M1 ShareGPT | No | Hardcoded in `run_paper_sweep_sharegpt.sh` |
| E_M1 RULER_16K | No | Hardcoded defaults in `run_paper_sweep_ruler16k.sh` (overridable via env vars — see section 4c for L40S) |
| E_M2 | Yes | Per-hardware calibration × tightness factor × {2, 3, 6} |
| E_M3 | No | Just measures TTFT/TPOT, no SLO threshold |
| E_M4 | No | Same hardcoded numbers as E_M1 |
| E_D1 | No | Disruption demo — not SLO-bound |

Reasoning: the paper main figure (E_M1) needs both machines plotted
against the same SLO bar so the curves are comparable. E_M2 is
specifically a tightness sweep where each machine's curve is plotted
against its own baseline — that's the only experiment where per-
hardware calibration matters.

---

## 6. After both machines finish

```bash
# Compare A6000 vs L40S tight-tier attainment at each QPS
python -c "
import json
for hw in ('a6000', 'l40s'):
    print(f'\n=== {hw} ===')
    for qps in (1.0, 2.0, 4.0, 6.0, 8.0):
        for b in ('vllm_fcfs', 'reroute_no_ckpt', 'ours'):
            f = f'experiments_v2/eval/results/{hw}/e_m1_{b}_sharegpt_qps{qps}_n60_seed0_metrics.json'
            try:
                d = json.load(open(f))
                t = d['per_class']['tight']['slo_met_pct']
                print(f'  qps={qps} {b:18s} tight={t:.0f}%')
            except FileNotFoundError:
                pass
"
```

Pick the hardware where ours' margin over baselines is biggest (or
the only one where it wins) and submit that to the paper. The other
becomes a portability check in the discussion or appendix.

---

## 7. Cleanup / restart

If a sweep gets interrupted and you want to retry, master shell is
fail-tolerant (it doesn't `set -e`) — already-completed runs left
their `metrics.json` on disk and the new run will overwrite. If you
want a clean slate:

```bash
# Wipe one hardware's results
rm -rf experiments_v2/eval/results/a6000/
# or l40s/

# Then rerun the calibration (L40S) and sweep
```

---

## 8. Where things live

```
experiments_v2/
├── eval/
│   ├── scripts/
│   │   ├── e_m1_slo_sweep.py            # main sweep (used by all of these)
│   │   ├── e_m2_slo_tightness.py         # driver — wraps e_m1 across tightness levels
│   │   ├── e_m3_overhead.py             # standalone (no e_m1 wrap)
│   │   ├── e_m4_picker_ablation.py       # driver — wraps e_m1 ours vs ours_no_picker
│   │   ├── e_d1_disruption_demo.py       # standalone (single-kill demo)
│   │   ├── slo_calibration.py            # standalone, run once per (hardware, dataset)
│   │   ├── run_paper_sweep_sharegpt.sh   # ShareGPT master driver
│   │   └── run_paper_sweep_ruler16k.sh   # RULER_16K master driver (E_M1 only)
│   └── results/
│       ├── *.json / *.log                # historical A6000-era files
│       ├── a6000/                        # output of HARDWARE_TAG=a6000 runs
│       └── l40s/                         # output of HARDWARE_TAG=l40s runs
└── docs/
    └── RUNBOOK_paper_sweep.md            # this file
```
