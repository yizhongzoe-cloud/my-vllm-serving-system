# Machine Setup Runbook — Long-Context FT Serving

**Use this any time you're setting up a new machine** for the FT serving experiments. Applies to:
- Renting a cloud cluster (L40S, H100, A100, etc.)
- Configuring a local machine (e.g., your local 2× L40S)
- Coming back to a machine after a long break (treat it like a new machine)

**Last updated:** 2026-04-29 (after a full debug session at 64K on local A6000)

---

## Two axes you must distinguish

| Type | Frequency | Examples |
|---|---|---|
| **One-time** (per machine) | Once after rental / first setup | env install, dataset transfer, profile generation |
| **Every session** | Each time you start working on this machine | GPU sanity, /dev/shm cleanup, smoke test |
| **Per cell** | Inside the shell script that loops cells | tmpfs cleanup, port advance |

Each step below is tagged with one of these.

---

## Phase 0 — BEFORE you connect to the machine

### [ONE-TIME — if rented] Confirm hardware matches the plan

What you need depends on workload:

```
Per-card requirement for 64K Llama-3.1-8B:
  Memory: 48+ GB (16 GB weights + 8 GB KV per req + headroom)
  Per-card prefill speed: aim for ≤10s, otherwise queue blowup at fault
  
Tested:
  A6000 (38 TFLOPS):  prefill 64K = 22s.  dp=2 INSUFFICIENT for 0.05 RPS.
  L40S  (91 TFLOPS):  prefill 64K ≈ 10s estimated. dp=2 likely OK at 0.05 RPS.
  H100/H200:           prefill 64K ≈ 5-8s. dp=2 fine, dp=4 comfortable.

Per-fault tolerance:
  dp=2 → loses 50% capacity on fault. Need ρ_post < 80%.
  dp=4 → loses 25% capacity on fault. Almost always fine.
  dp=8 → loses 12.5%. Comfortable.

Recommended minimums:
  64K experiments:  L40S × 2 (best effort) OR  ≥4× any modern GPU
  128K experiments: ≥4× L40S/H100, OR fewer H200/B200
```

Don't pay for under-spec hardware.

### [ONE-TIME] Have these files ready to scp/upload

From the original development machine (where this code was written):
- The repo at branch `zoe/slo-scheduling` (with all code patches)
- `experiments_v2/datasets/cached/ruler_64k_niah.jsonl` (~48 MB, 200 records)
- `experiments_v2/datasets/cached/ruler_128k_niah.jsonl` (if applicable, generate locally first)
- Reference profile `experiments_v2/checkpoint_cost_profile_8b_a6000.json` (just for diff/comparison; generate fresh per hardware)
- Reference config `experiments_v2/config_8b_ruler64k.yaml` (template; copy + tweak per hardware)

---

## Phase 1 — First connection to the machine

### [ONE-TIME] 1.1 Repo + env

```bash
# Clone the right branch
git clone --branch zoe/slo-scheduling <repo-url> ~/code/vllm-serving
cd ~/code/vllm-serving

# Verify the patches are present (these are critical bug fixes)
git log --oneline -20
grep -n "read_bufsize" experiments_v2/run.py    # MUST find 3 hits with 4 * 1024 * 1024
grep -n "_ft_ckpt_inst" vllm/v1/engine/core.py  # MUST find ~10 hits
grep -n "no-enable-prefix-caching" experiments_v2/profile_checkpoint_costs.py  # MUST find 1 hit
grep -n "max-num-batched-tokens" experiments_v2/profile_checkpoint_costs.py    # MUST find 1 hit

# Set up venv
python3 -m venv ~/envs/vllm
source ~/envs/vllm/bin/activate
pip install -e .  # vllm in dev mode

# RULER deps (only if planning to regenerate datasets)
pip install nltk wonderwords html2text
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

If `grep` doesn't find expected patches, **STOP**. Either you're on the wrong branch or the code state regressed. Don't proceed.

### [ONE-TIME] 1.2 GPU + driver sanity

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Required:
- `torch.cuda.is_available() == True`
- `device_count` matches what you rented
- Each GPU has at least 48 GB (for 8B at 64K)

### [ONE-TIME] 1.3 Cross-host networking (only if multi-node)

```bash
# Test ping latency between nodes (should be <1ms for same DC)
for h in node1 node2 node3 node4; do ping -c 3 $h; done

# Test NCCL
python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl')
print('NCCL OK', dist.get_rank(), dist.get_world_size())
"
```

If NCCL fails, fix BEFORE running any vLLM. Common gotchas:
- `NCCL_IB_DISABLE=1` if Infiniband isn't available
- `NCCL_SOCKET_IFNAME=eth0` (or whatever the real iface is)
- `MASTER_ADDR=node1`, `MASTER_PORT=29500`

### [ONE-TIME] 1.4 Disk / tmpfs

```bash
df -h /dev/shm  # MUST have enough space for KV checkpoint pool
# 64K KV per req = 8GB. With 50 reqs/cell + leakage between cells = up to 400 GB
# If /dev/shm < 400 GB, increase tmpfs size:
sudo mount -o remount,size=500G /dev/shm
```

### [ONE-TIME] 1.5 Transfer datasets

```bash
# scp from origin machine
scp ruler_64k_niah.jsonl <new-host>:~/code/vllm-serving/experiments_v2/datasets/cached/

# Verify after transfer
python -c "
import json
toks = []
with open('experiments_v2/datasets/cached/ruler_64k_niah.jsonl') as f:
    for line in f: toks.append(json.loads(line)['prompt_tokens'])
print('count:', len(toks), 'p50:', sorted(toks)[100], 'p95:', sorted(toks)[190])
"
# Expect: count 200, p50 ~65389, p95 ~65393
```

### [ONE-TIME — if generating new] 1.5b Regenerate RULER for new model/length

Only needed if going to 128K or switching model. Skip if dataset transferred from another machine.

```bash
git clone https://github.com/NVIDIA/RULER /tmp/RULER
cd /tmp/RULER/scripts/data
python prepare.py \
  --save_dir /tmp/ruler_out \
  --benchmark synthetic \
  --task niah_single_1 \
  --tokenizer_path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer_type hf \
  --max_seq_length 131072 \
  --num_samples 200

cd ~/code/vllm-serving
python experiments_v2/datasets/convert_ruler.py \
  --src /tmp/ruler_out/niah_single_1/validation.jsonl \
  --dst experiments_v2/datasets/cached/ruler_128k_niah.jsonl \
  --dataset-name ruler_128k_niah \
  --subtask niah_single_1 \
  --expected-output-tokens 64
```

---

## Phase 2 — Profile generation (per hardware × max_model_len combo)

A profile from A6000 is NOT valid for L40S. A profile for max_model_len=32K is NOT valid for max_model_len=64K. Re-profile for each (hardware, max_model_len) pair.

### [ONE-TIME PER HARDWARE+MAXLEN] 2.1 Configure profile script

For 64K experiments, the defaults from Phase 1 patches are correct. For 128K, edit:

```python
# experiments_v2/profile_checkpoint_costs.py

# In _start_server signature:
def _start_server(model: str, port: int, max_model_len: int = 132000) -> ...:
    # ...
    "--max-model-len", str(max_model_len),
    "--gpu-memory-utilization", "0.85",  # bump from 0.80 for tighter fit at 128K
    # ...

# In main(), token_lengths list:
token_lengths=[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
               4096, 8192, 16384, 32768, 65536, 131072],

# In KV benchmark, block_counts list (block_size=16, KV=2MB/block):
# For 16 GB peak (128K req KV), need 8192 blocks
block_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
```

### [ONE-TIME PER HARDWARE+MAXLEN] 2.2 Run profile

```bash
CUDA_VISIBLE_DEVICES=0 python experiments_v2/profile_checkpoint_costs.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output experiments_v2/checkpoint_cost_profile_8b_<HW>.json \
  --port 8500 \
  > /tmp/reprofile.log 2>&1 &

# Expected runtime: 30-60 min (more for 128K — single 65K prefill measurement is ~22s × 7 trials)
```

### [ONE-TIME PER HARDWARE+MAXLEN] 2.3 ⚠️ MANDATORY profile sanity check

**This is the step that, if skipped, costs 4+ hours of debugging. We learned the hard way.**

After profile finishes:

```bash
python -c "
import json
with open('experiments_v2/checkpoint_cost_profile_8b_<HW>.json') as f:
    p = json.load(f)
print('Profile prefill at 65536:', p['prefill_ms_by_tokens'].get('65536'), 'ms')
print('Profile load at 4GB:', p['load_ms_by_bytes'].get('4294967296'), 'ms')
print('c0:', p.get('publication_overhead_ms'))
"
```

Expected ranges (8B Llama):
| Hardware | t_prefill(64K) range | t_load(4GB) range |
|---|---|---|
| A6000  | 18000-25000 ms | 200-300 ms |
| L40S   | 8000-12000 ms  | 200-300 ms |
| H100   | 5000-9000 ms   | 200-300 ms |
| H200   | 3500-7000 ms   | 200-300 ms |

**If t_prefill(64K) is 100ms or 1000ms, the profile is broken**. Likely causes (in order of likelihood):
1. `--no-enable-prefix-caching` missing in `_start_server` (most common)
2. `--max-num-batched-tokens` set too high (must be 2048 to match production chunking)
3. Server didn't actually start with the flags you wanted (check `/tmp/ckpt_profile_server.log`)

Don't proceed to experiments until profile passes sanity check.

### [ONE-TIME] 2.4 Backup the profile

```bash
cp experiments_v2/checkpoint_cost_profile_8b_<HW>.json{,.snapshot_$(date +%Y%m%d)}
```

### [ONE-TIME] 2.5 Decode capacity profile (separate file)

```bash
CUDA_VISIBLE_DEVICES=0 python experiments_v2/profile_decode_capacity.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-len 65536 \
  --tpot-ms 200 \
  --output experiments_v2/decode_capacity_profile_8b_<HW>.json
```

This rarely changes results, but missing it makes Benders solver fall back to legacy.

---

## Phase 3 — Config calibration (per experiment campaign)

### [ONE-TIME PER CAMPAIGN] 3.1 Copy + edit config

```bash
cp experiments_v2/config_8b_ruler64k.yaml experiments_v2/config_8b_<HW>.yaml
```

Edit hardware-dependent fields:

```yaml
model: "meta-llama/Llama-3.1-8B-Instruct"
max_model_len: 65536               # match prompt length + small buffer
gpu_memory_utilization: 0.80       # 0.85 if model+KV is tight (e.g., 128K)
dtype: "float16"                   # bfloat16 on H100/H200 if preferred
enforce_eager: true                # keep ON; cuda graphs add startup time
dp_size: <your dp>                 # match GPU count

ft_checkpoint_cost_profile: "experiments_v2/checkpoint_cost_profile_8b_<HW>.json"
ft_decode_capacity_profile: "experiments_v2/decode_capacity_profile_8b_<HW>.json"

checkpoint_pool_bytes: 68719476736  # 64GB at 64K, 137GB at 128K. Per-req KV × 8 reqs = pool
```

### [ONE-TIME PER HARDWARE] 3.2 Boot test for SLO calibration

Run a one-cell boot test (no fault, Light load) to measure clean prefill TTFT and clean decode TPOT for THIS hardware:

```bash
cp config_8b_<HW>.yaml /tmp/config_boot.yaml
# Edit: run_duration_sec=240, warmup_sec=30 (shorter for boot test)

mkdir -p /tmp/boot_test
python experiments_v2/run.py \
  --config /tmp/config_boot.yaml \
  --baseline No-FT \
  --workload W9_Ruler64K \
  --load Light \
  --fault none \
  --seed 42 \
  --output-dir /tmp/boot_test \
  --port 8400 \
  > /tmp/boot.log 2>&1
```

Extract clean metrics:

```bash
python -c "
import csv
with open('/tmp/boot_test/requests.csv') as f:
    rows = list(csv.DictReader(f))
clean = [r for r in rows if r['success'] == 'True' and float(r['ttft_ms']) < 30000]
import statistics
print('Clean count:', len(clean))
print('Clean TTFT median:', statistics.median(float(r['ttft_ms']) for r in clean))
print('Clean TPOT median:', statistics.median(float(r['tpot_ms']) for r in clean if float(r['tpot_ms']) > 0))
"
```

Use these to set SLOs:

```yaml
slo:
  ttft_ms: <2x clean_TTFT_median>          # passes p50, violates p95 under queue
  failure_gap_ms: 5000.0                   # tight enough that reprefill physically fails

workloads:
  W9_Ruler64K:
    tpot_slo_ms: <halfway between clean_TPOT and congested_TPOT>
```

### [ONE-TIME PER HARDWARE] 3.3 Load levels (Little's Law)

```
Capacity = (1 / clean_prefill_seconds) × dp_size

Set Light to ~50% of capacity, Moderate to ~75%, Heavy to ~95%.
NEVER exceed 100% — queue blowup invalidates everything.

Also check post-fault: capacity_after = (dp_size - 1) / dp_size × capacity
Make sure RPS / capacity_after < 80% even at Light load. If not, pick fewer cells / more dp.
```

```yaml
load_levels:
  Light:    {pct: 0.25, rps: <0.5 × capacity>}
  Moderate: {pct: 0.40, rps: <0.75 × capacity>}
  Heavy:    {pct: 0.55, rps: <0.95 × capacity>}
```

### [ONE-TIME PER HARDWARE] 3.4 Fault timing

```yaml
fault_timing:
  none: null
  F1_Early: <0.2 × run_duration>      # after warmup, while pipe fills
  F2_Mid:   <0.5 × run_duration>      # pipe full, steady state
  F3_Late:  <0.8 × run_duration>      # near end, leaves recovery window
```

For run_duration=900s: F1=200, F2=500, F3=750.

### [ONE-TIME PER HARDWARE] 3.5 Run parameters

```yaml
run_duration_sec: 900.0           # ~15 min per cell, ~45 reqs at Light
warmup_sec: 60.0                  # vLLM scheduler unstable for first 30-60s
seeds: [42, 123, 456]             # add more if results have high variance
request_timeout_sec: 300.0        # 64K e2e can hit 100s+; give 3x buffer
server_startup_timeout: 600.0     # large model + dp=2 startup ~30-60s; give 10x

force_output_len: true            # vital — variable output length adds noise
```

---

## Phase 4 — Smoke test (run BEFORE every experiment session)

### [EVERY SESSION] 4.1 Verify no zombie state

```bash
pgrep -af "vllm" 2>&1            # should be empty
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should all be ~15 MiB
df -h /dev/shm                   # should have plenty free
ls /dev/shm/vllm_ft_checkpoints/ 2>/dev/null | wc -l       # should be 0
```

If any of these is dirty, clean up:
```bash
# Kill zombie vllm
pkill -9 -f "vllm"
# Wipe leftover checkpoints
rm -rf /dev/shm/vllm_ft_checkpoints/
```

### [EVERY SESSION] 4.2 Single Our-System smoke cell

This catches profile drift, env-var regression, code regression. Costs ~12 min, saves hours.

```bash
mkdir -p /tmp/smoke
FT_CKPT_TRUE_ASYNC=1 FT_CKPT_NONBLOCK=1 FT_WORKLOAD_EXHAUST=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python -m experiments_v2.run \
  --config experiments_v2/config_8b_<HW>.yaml \
  --baseline Our-System \
  --workload W9_Ruler64K \
  --load Light \
  --fault none \
  --seed 42 \
  --output-dir /tmp/smoke \
  --port 8400 \
  > /tmp/smoke.log 2>&1
```

### [EVERY SESSION] 4.3 ⚠️ Verify checkpointing IS firing

```bash
grep "FT_CKPT_INST FINAL" /tmp/smoke/server.log
```

Expected output (for ~30-50 reqs in a 600s cell):
```
FT_CKPT_INST FINAL fires=N blocks=M (prefill: P fires/B blocks; decode: D fires/E blocks) per_req_count=R
```

Sanity checks:
- `fires_total / per_req_count` should be **8-30 fires per request** (limited by NONBLOCK back-pressure; actual count depends on hardware speed)
- **If fires_per_req ≤ 2: profile is broken or env is wrong. STOP. Re-do Phase 2.**
- `blocks_total / fires_total` should be in the hundreds (each fire publishes hundreds of blocks)

**Also verify completion rate**: if smoke test (no fault, Light load) is < 90% completion or TTFT p95 > 60s, hardware is undersized for the workload. Either drop RPS or swap hardware before continuing.

### [EVERY SESSION] 4.4 Verify no /dev/shm leak

```bash
df -h /dev/shm
ls /dev/shm/vllm_ft_checkpoints/ 2>/dev/null | wc -l
```

After smoke test, /dev/shm should have at most a few hundred MB used (warmup leftovers). If GBs leaked, the cleanup hook isn't working. Investigate before main run.

---

## Phase 5 — Main experiment

### [EVERY SESSION] 5.1 Shell script template (with /dev/shm cleanup)

```bash
#!/bin/bash
set -u
cd ~/code/vllm-serving

OUTDIR=/tmp/v2full_main
mkdir -p "$OUTDIR"

CONFIG=experiments_v2/config_8b_<HW>.yaml
PYTHON=~/envs/vllm/bin/python
COMMON_ENV="FT_CKPT_TRUE_ASYNC=1 FT_CKPT_NONBLOCK=1 FT_WORKLOAD_EXHAUST=1 CUDA_VISIBLE_DEVICES=0,1"

declare -a LABELS=("No-FT" "Our-System" "NoFT-Reprefill")
declare -a SEEDS=(42 123 456)

PORT=9800

for seed in "${SEEDS[@]}"; do
  for label in "${LABELS[@]}"; do
    cell_dir="$OUTDIR/${label}_seed${seed}"
    mkdir -p "$cell_dir"

    # ===== [PER CELL] /dev/shm cleanup BEFORE each cell =====
    rm -rf /dev/shm/vllm_ft_checkpoints/

    while pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; do
      sleep 5
    done

    eval "$COMMON_ENV $PYTHON -m experiments_v2.run \
      --config '$CONFIG' \
      --baseline '$label' \
      --workload W9_Ruler64K \
      --load Light \
      --fault F2_Mid \
      --seed $seed \
      --output-dir '$cell_dir' \
      --port $PORT" \
      >> "$cell_dir/run.log" 2>&1

    PORT=$((PORT + 1))
    sleep 5
  done
done
```

### [EVERY SESSION] 5.2 Monitor

In a second terminal:

```bash
watch -n 60 'tail -3 /tmp/v2full_main/main.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h /dev/shm'
```

Watch for:
- A cell that stalls > 30 min → check server.log for OOM/CUDA errors
- /dev/shm > 80% → cleanup didn't fire; manually `rm -rf` and continue
- nvidia-smi util at 0% for > 5 min during a cell → server died silently

### [PER CELL] 5.3 Quick check after each cell

```bash
python -c "
import json
m = json.load(open('/tmp/v2full_main/${label}_seed${seed}/metrics.json'))
print(f\"goodput={m['goodput']:.2f} comp={m['completion_rate']*100:.0f}% gap_p95={m.get('failover_gap_p95_ms', 0):.0f}ms\")
"
```

Expected order of magnitude for 64K + F2_Mid (assuming hardware is sufficient):
- NoFT-Reprefill gap_p95: ~10-25s (one engine's reprefill cost; hardware-dependent)
- Our-System gap_p95: ~2-5s (KV reload + small replay)
- No-FT gap_p95: 0 (drops failed reqs entirely; goodput drops instead)

If Our-System gap_p95 ≈ NoFT-Reprefill gap_p95: **mechanism is being throttled by post-fault saturation**. Either dp is too low or RPS is too high. Drop RPS or add more cards.

---

## Common failure modes — symptoms and fixes

| Symptom | When it happens | Likely cause | Fix |
|---|---|---|---|
| `OSError: No space left on device` | mid-cell | /dev/shm tmpfs full from prior cell leakage | Phase 5.1 — `rm -rf /dev/shm/vllm_ft_checkpoints/` between cells |
| HTTP 400 on requests, 0% completion | every cell | prompt + max_tokens > max_model_len | Bump max_model_len in config to 70000 OR drop expected_output_tokens in jsonl |
| `aiohttp Chunk too big`; 0% completion | every cell | Default 64KB read buffer overflows on long-prompt SSE streams | Phase 1.1 — verify `read_bufsize=4 * 1024 * 1024` patches present in run.py |
| `FT_CKPT_INST FINAL fires=1 per_req` | smoke test | Cost profile too narrow OR profile measured wrong conditions | Phase 2.3 — re-profile with `--no-enable-prefix-caching` and extended ranges |
| Profile shows 32K prefill = 100ms (too fast) | profile generation | Prefix caching enabled in profile server | Re-check `_start_server` in profile_checkpoint_costs.py has `--no-enable-prefix-caching` |
| TTFT p50 > 60s under Light load (no fault) | smoke test or main | RPS > capacity, queue blowup | Phase 3.3 — drop RPS to 50% of capacity; verify Little's Law math |
| Server takes > 90s to start | every session | Model loading at high gpu_memory_utilization | Bump `server_startup_timeout` to 600s in config |
| NCCL timeout / hang at startup | first session, multi-node | Multi-node networking not configured | Phase 1.3 — run NCCL test BEFORE running vLLM |
| Goodput is 0 across all cells | main experiment | Mass timeout (queue + long requests) | Lower RPS, raise request_timeout_sec |
| Single fault wipes ALL in-flight reqs | main experiment | dp=1 effectively after fault, capacity halves, all reqs queue past timeout | Add cards for higher dp, OR drop RPS to ensure fault-survivable |
| Our-System gap_p95 ≈ NoFT-Reprefill gap_p95 | main experiment | Recovery work blocked by surviving engine's queue (post-fault saturation) | Hardware undersized — add cards for dp=4+ |

---

## Cross-machine / cross-session continuity (handoff)

When migrating between machines, copy these:

```
experiments_v2/checkpoint_cost_profile_8b_<HW>.json   # profile per HW
experiments_v2/decode_capacity_profile_8b_<HW>.json   # profile per HW
experiments_v2/datasets/cached/ruler_*_niah.jsonl     # dataset (model-specific)
experiments_v2/config_8b_<HW>.yaml                    # hardware-specific config
/tmp/v2full_main/                                      # raw experiment output
experiments_v2/docs/CURRENT_STATUS_<DATE>.md          # status / findings
```

Don't bother copying `/dev/shm/vllm_ft_checkpoints/` — runtime artifacts.

---

## Self-calibration follow-up (NOT for this run, but worth knowing)

The "profile mismatch" problem (Phase 2 sanity check) goes away if the cost model uses live experiment data instead of an offline profile. Mechanism: each request's prefill steps already produce real `(num_computed_tokens, wall_clock_time)` pairs. Aggregate these into `t_prefill(N)` directly; cost model queries this instead of an interpolation table.

Implementation outline:
- Maintain a per-process moving-average of `(N, t)` samples in `core.py` `_maybe_ft_checkpoint`
- First few prefill chunks bootstrap; subsequent decisions use measurements from the same workload's recent past
- Replaces hardcoded profile JSON values with self-tuning that always matches production conditions

Estimated effort: 200-300 LOC, 1-2 days. Eliminates the entire Phase 2 calibration step. Save for follow-up paper or post-deadline polish.

---

## What you should NOT spend time on (especially on rented machines)

- **Generating RULER datasets on rented machine** — generate locally on free CPU before renting
- **Editing schema-level code on rented machine** — debug locally first
- **Trying new model architectures on rented machine** — stick to Llama-3.1-8B-Instruct unless explicitly testing scaling
- **Re-running an experiment "to be sure"** — if smoke test passed, trust the data and move on
- **Tuning Benders solver** — known broken at dp=2, not worth fixing for cloud experiments at dp=4+
- **Running goodput-only sweeps** — failover_gap_p95 is the metric we care about; goodput is secondary
- **Tuning c0 in profile** — has no effect when FT_CKPT_NONBLOCK back-pressure is the limiter (which it is in our experiments)

---

## Final pre-experiment checklist (5 min, run BEFORE every paid 9-cell run)

| # | Check | Phase |
|---|---|---|
| 1 | `git status` clean (or only known modifications) | every session |
| 2 | `git log -5` shows the patches from Phase 1.1 | every session |
| 3 | `nvidia-smi` shows GPUs idle, full memory available | every session |
| 4 | `df -h /dev/shm` has at least 400 GB free | every session |
| 5 | `experiments_v2/checkpoint_cost_profile_8b_<HW>.json` exists, sanity-checked | one-time per hardware |
| 6 | `experiments_v2/decode_capacity_profile_8b_<HW>.json` exists | one-time per hardware |
| 7 | `experiments_v2/datasets/cached/ruler_*_niah.jsonl` exists, validated | one-time |
| 8 | `experiments_v2/config_8b_<HW>.yaml` exists, SLOs match boot test | one-time per hardware |
| 9 | Smoke test (Phase 4) passed: FT_CKPT_INST shows ≥ 5 fires per request | every session |
| 10 | Smoke test completion rate > 90% (else hardware is undersized) | every session |
| 11 | Run script has `rm -rf /dev/shm/vllm_ft_checkpoints/` before each cell | every session |
| 12 | `request_timeout_sec` ≥ 300 in config | one-time per hardware |
| 13 | `force_output_len: true` in config | one-time per hardware |

If any item is unchecked, **fix it before starting the paid 3-hour run.**
