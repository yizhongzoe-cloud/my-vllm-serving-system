#!/usr/bin/env bash
# Diag11: validate R7 (greedy warm-start seed) on W5 F2.
# Solver ALWAYS runs — greedy is only a hint for CP-SAT warm-start.
# Launched after diag10 completes (by wakeup logic or manually).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag11"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# Wait for diag10 to finish so GPUs are free
echo "[$(date +%H:%M:%S)] diag11: waiting for diag10 to clear GPUs..."
while pgrep -f "diag10" > /dev/null 2>&1; do
    sleep 30
done
sleep 15
pkill -9 -f "api_server.*port 8500" 2>/dev/null
pkill -9 -f "api_server.*port 8501" 2>/dev/null
sleep 5

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" seed="$5"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} $V2_BASE ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W5_LongDoc --load Moderate --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag11: R7 greedy-seed validation $(date) ########"

# GPU 0-1: R7 alone + R7 + interval=10 (assuming diag10 picks 10 as best)
(
    for s in 42 123 456; do
        run_cell "R7_alone" "0,1" "8500" "FT_SOLVER_GREEDY_SEED=1" "$s" || true
        run_cell "R7_I10"   "0,1" "8500" "FT_SOLVER_GREEDY_SEED=1 FT_CHECKPOINT_STEP_INTERVAL=10" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag11_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: R7 + warm-start/cap combo
(
    for s in 42 123 456; do
        run_cell "R7_tightcap" "2,3" "8501" "FT_SOLVER_GREEDY_SEED=1 FT_SOLVER_TIME_CAP_MS=50" "$s" || true
        run_cell "R7_I20"      "2,3" "8501" "FT_SOLVER_GREEDY_SEED=1 FT_CHECKPOINT_STEP_INTERVAL=20" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag11_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag11 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("NR F2 target",           f"{ROOT}/w5/W5_NR"),
    ("V2 baseline",            f"{ROOT}/w5/W5_V2"),
    ("OS_NoCkpt (best V2)",    f"{ROOT}/diag3/OS_NoCkpt_F2"),
    ("R7 alone",               f"{ROOT}/diag11/R7_alone"),
    ("R7 + ckpt I=10",         f"{ROOT}/diag11/R7_I10"),
    ("R7 + ckpt I=20",         f"{ROOT}/diag11/R7_I20"),
    ("R7 + tight cap",         f"{ROOT}/diag11/R7_tightcap"),
]
print(f"{'variant':<26} {'gp':>12} {'comp':>5} {'fg_p95':>9}")
for name, base in CFGS:
    gps, c, f95 = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        print(f"{name:<26} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")

print("\n=== Beat-NR ===")
for name, base in CFGS:
    if "target" in name or "V2 baseline" in name or "NoCkpt" in name: continue
    gps, f95 = [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        mg = st.mean(gps); mf = st.mean(f95)
        w = "🎯 WIN" if (mg>=30.4 and mf<=6115) else ""
        print(f"  {name}: gp={mg:.1f} fg={mf:.0f} {w}")
PYEOF
