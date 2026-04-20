#!/usr/bin/env bash
# Extended stress pipeline — runs AFTER main pipeline (stress_pipeline_2026-04-20.sh)
# completes. Fills the 4h window until 19:00 with fault timing, no-fault baselines,
# and admission-ablation runs.
#
# Auto-gated: polls for metrics.json from the main pipeline before starting.
# Uses same run_cell helper logic. Idempotent via metrics.json checkpoint.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/stress_2026-04-20"
mkdir -p "$ROOT/w5_ext" "$ROOT/w7_ext"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# Read the MAIN_LOAD used by primary pipeline (set in smoke gate; default Moderate)
MAIN_LOAD=Moderate
[ -f "${ROOT}/smoke/W5_NR_RPS0p5/42/metrics.json" ] && MAIN_LOAD=Light

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" wl="$6" load="$7" fault="$8" seed="$9" root_tag="${10}"
    local out="${ROOT}/${root_tag}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" != "NoFT-Reprefill" ]; then
        env_base="$V2_BASE"
    else
        env_base="FT_RECOVERY_MODE=reprefill"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline}, ${wl}, ${load}, ${fault}) GPU=${gpus}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

gate_gpu_free() {
    # $1=port  Wait until the given port is not bound by vllm server
    local port="$1"
    while pgrep -af "api_server.*port ${port}" > /dev/null 2>&1; do
        sleep 30
    done
}

# ---- Gate: wait until main pipeline's primary cells are done ----
echo "######## Extended pipeline started $(date) ########" >&2
echo "Gate: waiting for main Phase B (W7) to clear GPU 2-3 ..." >&2
# W7 on port 8501: wait until last main W7 cell done
while [ ! -f "${ROOT}/w7/W7_V2_Heavy/456/metrics.json" ] && [ ! -f "${ROOT}/w7/W7_V2_Heavy/456/stdout.log" ]; do
    sleep 60
done
# Also wait for any in-progress port 8501 server to exit
gate_gpu_free 8501

# ---- GPU 2-3 extended: W7 skip_solver + fault timing + no-fault, W5 skip_solver ----
echo "GPU 2-3 extended: W7 skip_solver + fault timing + no-fault + W5 skip_solver" >&2
(
    # W7 skip_solver × F2_Mid
    for s in 42 123 456; do
        run_cell "W7_V2_skip_solver" "2,3" "8501" "FT_SKIP_SOLVER=1" "Our-System" "W7_Saturated" "Heavy" "F2_Mid" "$s" "w7_ext" || true
    done
    # W7 fault timing F1 / F3
    for s in 42 123 456; do
        run_cell "W7_NR_F1" "2,3" "8501" "" "NoFT-Reprefill" "W7_Saturated" "Heavy" "F1_Early" "$s" "w7_ext" || true
        run_cell "W7_V2_F1" "2,3" "8501" "" "Our-System"     "W7_Saturated" "Heavy" "F1_Early" "$s" "w7_ext" || true
        run_cell "W7_NR_F3" "2,3" "8501" "" "NoFT-Reprefill" "W7_Saturated" "Heavy" "F3_Late"  "$s" "w7_ext" || true
        run_cell "W7_V2_F3" "2,3" "8501" "" "Our-System"     "W7_Saturated" "Heavy" "F3_Late"  "$s" "w7_ext" || true
    done
    # W7 no-fault reference
    for s in 42 123 456; do
        run_cell "W7_NR_none" "2,3" "8501" "" "NoFT-Reprefill" "W7_Saturated" "Heavy" "none" "$s" "w7_ext" || true
        run_cell "W7_V2_none" "2,3" "8501" "" "Our-System"     "W7_Saturated" "Heavy" "none" "$s" "w7_ext" || true
    done
    # If time left: W5 skip_solver × F1/F3 (GPU 2-3, different port)
    for s in 42 123 456; do
        run_cell "W5_V2_skip_solver_F1" "2,3" "8503" "FT_SKIP_SOLVER=1" "Our-System" "W5_LongDoc" "${MAIN_LOAD}" "F1_Early" "$s" "w5_ext" || true
        run_cell "W5_V2_skip_solver_F3" "2,3" "8503" "FT_SKIP_SOLVER=1" "Our-System" "W5_LongDoc" "${MAIN_LOAD}" "F3_Late"  "$s" "w5_ext" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 extensions done" >&2
) > /tmp/stress_ext_gpu23.log 2>&1 &
GPU23_PID=$!

# ---- Gate: wait until main Phase C (W5) clears GPU 0-1 ----
echo "Gate: waiting for main Phase C (W5) to clear GPU 0-1 ..." >&2
while [ ! -f "${ROOT}/w5/W5_V2_skip_solver/456/metrics.json" ] && [ ! -f "${ROOT}/w5/W5_V2_skip_solver/456/stdout.log" ]; do
    sleep 60
done
gate_gpu_free 8500

# ---- GPU 0-1 extended: W5 fault timing + W5 no-fault ----
echo "GPU 0-1 extended: W5 fault timing (F1, F3) + no-fault" >&2
for s in 42 123 456; do
    run_cell "W5_NR_F1" "0,1" "8500" "" "NoFT-Reprefill" "W5_LongDoc" "${MAIN_LOAD}" "F1_Early" "$s" "w5_ext" || true
    run_cell "W5_V2_F1" "0,1" "8500" "" "Our-System"     "W5_LongDoc" "${MAIN_LOAD}" "F1_Early" "$s" "w5_ext" || true
    run_cell "W5_NR_F3" "0,1" "8500" "" "NoFT-Reprefill" "W5_LongDoc" "${MAIN_LOAD}" "F3_Late"  "$s" "w5_ext" || true
    run_cell "W5_V2_F3" "0,1" "8500" "" "Our-System"     "W5_LongDoc" "${MAIN_LOAD}" "F3_Late"  "$s" "w5_ext" || true
done
for s in 42 123 456; do
    run_cell "W5_NR_none" "0,1" "8500" "" "NoFT-Reprefill" "W5_LongDoc" "${MAIN_LOAD}" "none" "$s" "w5_ext" || true
    run_cell "W5_V2_none" "0,1" "8500" "" "Our-System"     "W5_LongDoc" "${MAIN_LOAD}" "none" "$s" "w5_ext" || true
done
echo "[$(date +%H:%M:%S)] GPU 0-1 extensions done" >&2

wait $GPU23_PID 2>/dev/null || true

# ---- Final consolidated summary ----
echo "" >&2
echo "######## Extended pipeline complete $(date) ########" >&2
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"

CONFIGS = [
    # (name, path)
    ("W5 NR F2",                   f"{ROOT}/w5/W5_NR"),
    ("W5 V2 F2",                   f"{ROOT}/w5/W5_V2"),
    ("W5 V2+skip_solver F2",       f"{ROOT}/w5/W5_V2_skip_solver"),
    ("W5 NR F1",                   f"{ROOT}/w5_ext/W5_NR_F1"),
    ("W5 V2 F1",                   f"{ROOT}/w5_ext/W5_V2_F1"),
    ("W5 NR F3",                   f"{ROOT}/w5_ext/W5_NR_F3"),
    ("W5 V2 F3",                   f"{ROOT}/w5_ext/W5_V2_F3"),
    ("W5 NR none",                 f"{ROOT}/w5_ext/W5_NR_none"),
    ("W5 V2 none",                 f"{ROOT}/w5_ext/W5_V2_none"),
    ("W5 V2+skip_solver F1",       f"{ROOT}/w5_ext/W5_V2_skip_solver_F1"),
    ("W5 V2+skip_solver F3",       f"{ROOT}/w5_ext/W5_V2_skip_solver_F3"),
    ("W7 NR F2",                   f"{ROOT}/w7/W7_NR_Heavy"),
    ("W7 V2 F2",                   f"{ROOT}/w7/W7_V2_Heavy"),
    ("W7 V2+skip_solver F2",       f"{ROOT}/w7_ext/W7_V2_skip_solver"),
    ("W7 NR F1",                   f"{ROOT}/w7_ext/W7_NR_F1"),
    ("W7 V2 F1",                   f"{ROOT}/w7_ext/W7_V2_F1"),
    ("W7 NR F3",                   f"{ROOT}/w7_ext/W7_NR_F3"),
    ("W7 V2 F3",                   f"{ROOT}/w7_ext/W7_V2_F3"),
    ("W7 NR none",                 f"{ROOT}/w7_ext/W7_NR_none"),
    ("W7 V2 none",                 f"{ROOT}/w7_ext/W7_V2_none"),
]

print(f"{'variant':<30} {'gp':>12} {'fg_p50':>12} {'fg_p95':>12} {'comp':>6}")
for name, base in CONFIGS:
    gps, f50, f95, comps = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            comps.append(m.get('completion_rate',0)*100)
            if m.get('completion_rate',0) >= 0.9:
                gps.append(m['goodput']); f50.append(m.get('failover_gap_p50_ms',0))
                f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        f50s = st.stdev(f50) if len(f50)>1 else 0
        f95s = st.stdev(f95) if len(f95)>1 else 0
        cm = st.mean(comps) if comps else 0
        print(f"{name:<30} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(f50):7.0f}±{f50s:4.0f} {st.mean(f95):7.0f}±{f95s:4.0f} {cm:4.0f}%")
    else:
        print(f"{name:<30}   NO VALID RUNS")

# Write CSV
import csv
with open(f"{ROOT}/stress_table.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['variant','n_ok','gp_mean','gp_std','fg_p50_mean','fg_p50_std','fg_p95_mean','fg_p95_std','comp_mean'])
    for name, base in CONFIGS:
        gps, f50, f95, comps = [], [], [], []
        for s in [42,123,456]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                comps.append(m.get('completion_rate',0)*100)
                if m.get('completion_rate',0) >= 0.9:
                    gps.append(m['goodput']); f50.append(m.get('failover_gap_p50_ms',0)); f95.append(m.get('failover_gap_p95_ms',0))
        if gps:
            w.writerow([name, len(gps),
                        f"{st.mean(gps):.1f}", f"{st.stdev(gps) if len(gps)>1 else 0:.1f}",
                        f"{st.mean(f50):.0f}", f"{st.stdev(f50) if len(f50)>1 else 0:.0f}",
                        f"{st.mean(f95):.0f}", f"{st.stdev(f95) if len(f95)>1 else 0:.0f}",
                        f"{st.mean(comps) if comps else 0:.0f}"])
print(f"\nCSV written to {ROOT}/stress_table.csv")
PYEOF
