#!/usr/bin/env bash
# Diag10: FT_CHECKPOINT_STEP_INTERVAL sweep on W5 F2.
# Tests if reducing checkpoint controller polling frequency closes the 17% gap
# WITHOUT disabling checkpointing (enable_checkpointing stays True).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag10"
mkdir -p "$ROOT"

# Same V2_BASE but stripped FT_CHECKPOINT_STEP_INTERVAL=2; we override per variant
V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1'

run_cell() {
    local tag="$1" gpus="$2" port="$3" interval="$4" seed="$5"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (interval=${interval})" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} $V2_BASE FT_CHECKPOINT_STEP_INTERVAL=${interval} python experiments_v2/run.py \
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

echo "######## Diag10: ckpt_step_interval sweep $(date) ########"

# GPU 0-1: interval=10, 50
(
    for s in 42 123 456; do
        run_cell "I10" "0,1" "8500" "10" "$s" || true
        run_cell "I50" "0,1" "8500" "50" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag10_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: interval=20, 100
(
    for s in 42 123 456; do
        run_cell "I20" "2,3" "8501" "20" "$s" || true
        run_cell "I100" "2,3" "8501" "100" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag10_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag10 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("NR F2 (target)",              f"{ROOT}/w5/W5_NR"),
    ("V2 I=2 (baseline)",            f"{ROOT}/w5/W5_V2"),
    ("OS_NoCkpt (best V2 so far)",   f"{ROOT}/diag3/OS_NoCkpt_F2"),
    ("I=10",                         f"{ROOT}/diag10/I10"),
    ("I=20",                         f"{ROOT}/diag10/I20"),
    ("I=50",                         f"{ROOT}/diag10/I50"),
    ("I=100",                        f"{ROOT}/diag10/I100"),
]
print(f"{'variant':<30} {'gp':>12} {'comp':>5} {'fg_p95':>9}")
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
        print(f"{name:<30} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")
    else: print(f"{name:<30}  NO DATA")

# Scorecard
print("\n=== Beat-NR scorecard ===")
for name, base in CFGS:
    if "target" in name or "NoCkpt" in name: continue
    gps, f95 = [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        mg = st.mean(gps); mf = st.mean(f95)
        gp_w = "✓" if mg >= 30.4 else "✗"
        fg_w = "✓" if mf <= 6115 else "✗"
        print(f"  {name:<26} gp={mg:5.1f}[{gp_w}] fg={mf:5.0f}[{fg_w}]{'🎯 WIN' if mg>=30.4 and mf<=6115 else ''}")
PYEOF
