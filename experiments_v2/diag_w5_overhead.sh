#!/usr/bin/env bash
# Diagnostic: isolate the 27% W5 no-fault overhead source.
# Tests V2 with solver/checkpoint components turned off one by one.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/stress_2026-04-20/diag"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" seed="$6"
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
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline})" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload W5_LongDoc --load Moderate --fault none \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} ttft_p50={m.get(\"ttft_p50_ms\",0):.0f} ttft_p95={m.get(\"ttft_p95_ms\",0):.0f} comp={m[\"completion_rate\"]*100:.0f}%')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

echo "######## W5 overhead diagnostic $(date) ########" >&2
echo "Testing which V2 component causes 27% goodput loss on no-fault long-context" >&2

# GPU 0-1: V2 with solver off (should recover if solver is culprit)
(
    for s in 42 123 456; do
        run_cell "V2_no_solver"       "0,1" "8500" "FT_SKIP_SOLVER=1" "Our-System" "$s" || true
        run_cell "V2_no_ckpt"          "0,1" "8500" "FT_CHECKPOINT_STEP_INTERVAL=999" "Our-System" "$s" || true
    done
) > /tmp/diag_gpu01.log 2>&1 &
GPU01_PID=$!

# GPU 2-3 in parallel: V2 minimal (both off) + NR sanity
(
    for s in 42 123 456; do
        run_cell "V2_minimal"          "2,3" "8501" "FT_SKIP_SOLVER=1 FT_CHECKPOINT_STEP_INTERVAL=999" "Our-System" "$s" || true
        run_cell "NR_sanity"           "2,3" "8501" "" "NoFT-Reprefill" "$s" || true
    done
) > /tmp/diag_gpu23.log 2>&1 &
GPU23_PID=$!

wait $GPU01_PID $GPU23_PID 2>/dev/null || true

echo "" >&2
echo "######## Diagnostic summary $(date) ########" >&2
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("W5 NR (stress_2026-04-20)",            f"{ROOT}/w5_ext/W5_NR_none"),
    ("W5 V2 (stress_2026-04-20)",            f"{ROOT}/w5_ext/W5_V2_none"),
    ("W5 NR_sanity (diag)",                   f"{ROOT}/diag/NR_sanity"),
    ("W5 V2_no_solver (diag)",                f"{ROOT}/diag/V2_no_solver"),
    ("W5 V2_no_ckpt (diag)",                  f"{ROOT}/diag/V2_no_ckpt"),
    ("W5 V2_minimal (no solver, no ckpt)",   f"{ROOT}/diag/V2_minimal"),
]
print(f"{'variant':<40} {'gp':>12} {'ttft_p50':>10} {'ttft_p95':>10}")
for name, base in CFGS:
    gps, ttft50, ttft95 = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); ttft50.append(m.get('ttft_p50_ms',0)); ttft95.append(m.get('ttft_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        t50s = st.stdev(ttft50) if len(ttft50)>1 else 0
        t95s = st.stdev(ttft95) if len(ttft95)>1 else 0
        print(f"{name:<40} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(ttft50):5.0f}±{t50s:3.0f} {st.mean(ttft95):5.0f}±{t95s:3.0f}")
PYEOF
