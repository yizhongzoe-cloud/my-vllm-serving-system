#!/usr/bin/env bash
# Diag5: validate 3 solver fixes (trivial_skip + time_cap + warm_start).
# Waits for diag4 to clear GPUs, then runs V2 with the new code on W5.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag5"
mkdir -p "$ROOT"

# Wait for diag4 to finish (all its run.py processes to exit)
echo "[$(date +%H:%M:%S)] diag5: waiting for diag4 to clear GPUs..."
while pgrep -f "baseline Our-System-NoCkpt.*diag4" > /dev/null 2>&1; do
    sleep 30
done
sleep 15  # extra buffer for server shutdown

# Kill any lingering vllm servers
pkill -9 -f "api_server.*port 8500" 2>/dev/null
pkill -9 -f "api_server.*port 8501" 2>/dev/null
sleep 5

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# Fixes enabled by default (trivial_skip=1 by default; time_cap 100ms; warm_start always)
FIX_ENV='FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" wl="$6" fault="$7" seed="$8"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" != "NoFT-Reprefill" ]; then
        env_base="$V2_BASE $FIX_ENV"
    else
        env_base="FT_RECOVERY_MODE=reprefill"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline}, ${wl}, ${fault})" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load Moderate --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} ttft_p50={m.get(\"ttft_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f} comp={m[\"completion_rate\"]*100:.0f}%')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
        grep -E "AssertionError|CUDA out of memory|Error" "${out}/server.log" 2>/dev/null | head -3
    fi
}

echo "######## Diag5: solver fixes validation $(date) ########"

# GPU 0-1: V2_FIXED on W5 no-fault (compare to V2_no_fix gp=87)
(
    for s in 42 123 456; do
        run_cell "V2_fixed_none" "0,1" "8500" "" "Our-System" "W5_LongDoc" "none" "$s" || true
    done
    # Also F2 with fixes
    for s in 42 123 456; do
        run_cell "V2_fixed_F2" "0,1" "8500" "" "Our-System" "W5_LongDoc" "F2_Mid" "$s" || true
    done
) > /tmp/diag5_gpu01.log 2>&1 &

# GPU 2-3: also V2_FIXED on W7 (sanity — should still be ~NR)
(
    for s in 42 123 456; do
        run_cell "V2_fixed_W7_F2" "2,3" "8501" "" "Our-System" "W7_Saturated" "F2_Mid" "$s" || true
    done
) > /tmp/diag5_gpu23.log 2>&1 &

wait
echo ""
echo "=== Diag5 summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("W5 NR none (target)",                 f"{ROOT}/w5_ext/W5_NR_none"),
    ("W5 V2_old none (baseline -27%)",       f"{ROOT}/w5_ext/W5_V2_none"),
    ("W5 OS_NoCkpt none (ref)",              f"{ROOT}/diag2/OS_NoCkpt"),
    ("W5 V2_FIXED none (trivial+cap+warm)", f"{ROOT}/diag5/V2_fixed_none"),
    ("W5 NR F2 (target)",                    f"{ROOT}/w5/W5_NR"),
    ("W5 V2_old F2 (baseline)",              f"{ROOT}/w5/W5_V2"),
    ("W5 OS_NoCkpt F2 (ref)",                f"{ROOT}/diag3/OS_NoCkpt_F2"),
    ("W5 V2_FIXED F2",                       f"{ROOT}/diag5/V2_fixed_F2"),
    ("W7 NR F2 (sanity)",                    f"{ROOT}/w7/W7_NR_Heavy"),
    ("W7 V2_FIXED F2 (sanity)",              f"{ROOT}/diag5/V2_fixed_W7_F2"),
]
print(f"{'variant':<42} {'gp':>14} {'comp':>6} {'ttft_p50':>10} {'fg_p95':>10}")
for name, base in CFGS:
    gps, c, t50, f95 = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            t50.append(m.get('ttft_p50_ms',0)); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        print(f"{name:<42} {st.mean(gps):6.1f}±{st.stdev(gps) if len(gps)>1 else 0:3.0f} {st.mean(c):4.0f}% {st.mean(t50):5.0f} {st.mean(f95):5.0f}")
    else:
        print(f"{name:<42}   NO DATA")
PYEOF
