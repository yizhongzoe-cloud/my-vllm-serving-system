#!/usr/bin/env bash
# Diag12: cross-workload validation lock-down.
# Confirms V2_FIXED (legit optimizations only, no bypass) on:
#   - W7_Saturated (should win)
#   - W1_Chat (should not regress)
#   - W4_Mixed (should not regress)
#   - W5_LongDoc (additional seeds for 6-seed CI)

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag12"
mkdir -p "$ROOT"

# All env flags are legitimate MIP-preserving optimizations.
# Solver runs every epoch, checkpoint controller runs on every Kth engine step.
V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'
# FT_SOLVER_TRIVIAL_SKIP is default off (no bypass)
# FT_SOLVER_GREEDY_SEED stays off (R7 didn't help)

run_cell() {
    local tag="$1" gpus="$2" port="$3" baseline="$4" wl="$5" load="$6" seed="$7"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" != "NoFT-Reprefill" ]; then
        env_base="$V2_BASE"
    else
        env_base="FT_RECOVERY_MODE=reprefill"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} ${baseline}/${wl}/${load}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag12: cross-workload lock-down $(date) ########"

# GPU 0-1: W7 confirm + W1 no-regression
(
    for s in 42 123 456; do
        run_cell "W7_V2_confirm"   "0,1" "8500" "Our-System"     "W7_Saturated" "Heavy" "$s" || true
        run_cell "W1_V2_check"     "0,1" "8500" "Our-System"     "W1_Chat"      "Heavy" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag12_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: W4 no-regression + W5 extra seeds
(
    for s in 42 123 456; do
        run_cell "W4_V2_check"     "2,3" "8501" "Our-System"     "W4_Mixed"     "Heavy" "$s" || true
    done
    # Extra W5 seeds for 6-seed CI
    for s in 789 1234 5678; do
        run_cell "W5_V2_extra_seed" "2,3" "8501" "Our-System"     "W5_LongDoc"   "Moderate" "$s" || true
        run_cell "W5_NR_extra_seed" "2,3" "8501" "NoFT-Reprefill" "W5_LongDoc"   "Moderate" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag12_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag12 cross-workload summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    # W7 confirm
    ("W7 NR F2 (from stress)",       f"{ROOT}/w7/W7_NR_Heavy"),
    ("W7 V2_FIXED F2 (NEW confirm)",  f"{ROOT}/diag12/W7_V2_confirm"),
    # W1 no-regression
    ("W1 NR F2 (ref from stress)",    f"{ROOT}/w7_ext/W7_NR_F1"),  # placeholder, W1 NR from overnight
    ("W1 V2_FIXED F2 (NEW)",          f"{ROOT}/diag12/W1_V2_check"),
    # W4 no-regression
    ("W4 V2_FIXED F2 (NEW)",          f"{ROOT}/diag12/W4_V2_check"),
    # W5 6-seed
    ("W5 NR (3 orig + 3 new, 6 seeds combined)", None),  # special
    ("W5 V2 extra seeds (3 new)",     f"{ROOT}/diag12/W5_V2_extra_seed"),
    ("W5 NR extra seeds (3 new)",     f"{ROOT}/diag12/W5_NR_extra_seed"),
]
print(f"{'variant':<42} {'gp':>12} {'comp':>6} {'fg_p95':>9}")
for name, base in CFGS:
    if base is None: continue
    gps, c, f95 = [], [], []
    for s in [42,123,456,789,1234,5678]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        print(f"{name:<42} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")

# 6-seed combined W5
print("\n### W5 F2 6-seed combined (3 orig + 3 new) ###")
for name, bases in [
    ("W5 NR 6 seeds", [f"{ROOT}/w5/W5_NR", f"{ROOT}/diag12/W5_NR_extra_seed"]),
    ("W5 V2_FIXED 6 seeds", [f"{ROOT}/w5/W5_V2", f"{ROOT}/diag12/W5_V2_extra_seed"]),
]:
    gps, f95 = [], []
    for base in bases:
        for s in [42,123,456,789,1234,5678]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        fs = st.stdev(f95) if len(f95)>1 else 0
        print(f"  {name} (n={len(gps)}): gp={st.mean(gps):.1f}±{gs:.0f}  fg_p95={st.mean(f95):.0f}±{fs:.0f}")
PYEOF
echo "[$(date +%H:%M:%S)] diag12 complete"
