#!/usr/bin/env bash
# ============================================================================
# overnight_3seed_2026-04-10.sh — Run seeds 123 + 456 for all 5 baselines
# across 12 cells (2 workloads × 3 loads × 2 faults).
#
# Seed 42 already done in E1a_Quick_v2_HalfA/B + E1a_NoFT_Reprefill.
# This script fills in seeds 123 + 456 to get 3-seed variance bounds.
#
# Splits into two GPU groups:
#   GPU 4-5 (port 8400): No-FT + Periodic-Low + NoFT-Reprefill
#   GPU 6-7 (port 8500): Periodic-High + Our-System
#
# Each group: 3 baselines × 12 cells × 2 seeds = 72 runs (GPU 4-5)
#             2 baselines × 12 cells × 2 seeds = 48 runs (GPU 6-7)
# Total: 120 runs × ~7 min = ~14 hours (but parallel → ~8.5 hours)
#
# How to run:
#   cd /home/jlpang/my-vllm-serving-system
#   nohup bash experiments_v2/overnight_3seed_2026-04-10.sh > /tmp/overnight_3seed.log 2>&1 &
#   disown
# ============================================================================

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1

OUT_BASE="results_v2/8B/E1a_3seed"
CONFIG="experiments_v2/config_8b.yaml"

WORKLOADS=("W1_Chat" "W4_Mixed")
LOADS=("Light" "Moderate" "Heavy")
FAULTS=("none" "F2_Mid")
SEEDS=(123 456)

run_cell() {
    local gpu="$1"
    local port="$2"
    local baseline="$3"
    local workload="$4"
    local load="$5"
    local fault="$6"
    local seed="$7"
    local extra_env="${8:-}"

    local out_dir="${OUT_BASE}/${baseline}/${workload}/${load}/${fault}/${seed}"

    # Skip if already done
    if [ -f "${out_dir}/metrics.json" ]; then
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] START ${baseline}/${workload}/${load}/${fault}/s${seed} (GPU ${gpu}, port ${port})" >&2

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config $CONFIG \
        --baseline $baseline \
        --workload $workload \
        --load $load \
        --fault $fault \
        --seed $seed \
        --port $port \
        --output-dir $out_dir" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%')
" 2>/dev/null
    else
        echo "  FAILED (rc=$rc)" >&2
    fi
}

run_group_a() {
    # GPU 4-5, port 8400: No-FT + Periodic-Low + NoFT-Reprefill
    local gpu="4,5"
    local port="8400"

    for seed in "${SEEDS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            for load in "${LOADS[@]}"; do
                for fault in "${FAULTS[@]}"; do
                    run_cell "$gpu" "$port" "No-FT" "$workload" "$load" "$fault" "$seed"
                    run_cell "$gpu" "$port" "Periodic-Low" "$workload" "$load" "$fault" "$seed"
                    run_cell "$gpu" "$port" "NoFT-Reprefill" "$workload" "$load" "$fault" "$seed" "FT_RECOVERY_MODE=reprefill"
                done
            done
        done
    done
}

run_group_b() {
    # GPU 6-7, port 8500: Periodic-High + Our-System
    local gpu="6,7"
    local port="8500"

    for seed in "${SEEDS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            for load in "${LOADS[@]}"; do
                for fault in "${FAULTS[@]}"; do
                    run_cell "$gpu" "$port" "Periodic-High" "$workload" "$load" "$fault" "$seed"
                    run_cell "$gpu" "$port" "Our-System" "$workload" "$load" "$fault" "$seed"
                done
            done
        done
    done
}

echo "########################################################################"
echo "# Overnight 3-seed suite — $(date)"
echo "# Seeds: 123, 456 (seed 42 already done)"
echo "# GPU 4-5: No-FT + Periodic-Low + NoFT-Reprefill (72 runs)"
echo "# GPU 6-7: Periodic-High + Our-System (48 runs)"
echo "########################################################################"

# Also copy seed=42 results from existing runs into E1a_3seed dir
echo "Linking seed=42 results from E1a_Quick_v2_Half{A,B} + E1a_NoFT_Reprefill..."
for baseline_dir in results_v2/8B/E1a_Quick_v2_HalfA/* results_v2/8B/E1a_Quick_v2_HalfB/* results_v2/8B/E1a_NoFT_Reprefill/*; do
    bl=$(basename "$baseline_dir")
    for mf in $(find "$baseline_dir" -name metrics.json 2>/dev/null); do
        rel=$(echo "$mf" | sed "s|.*${bl}/||")
        target="${OUT_BASE}/${bl}/${rel}"
        target_dir=$(dirname "$target")
        if [ ! -f "$target" ]; then
            mkdir -p "$target_dir"
            cp -r "$(dirname "$mf")/"* "$target_dir/" 2>/dev/null
        fi
    done
done
echo "Done linking."

# Launch both groups in parallel
run_group_a &
PID_A=$!
run_group_b &
PID_B=$!

echo "Group A (GPU 4-5) pid=$PID_A"
echo "Group B (GPU 6-7) pid=$PID_B"
echo "Waiting for both groups..."

wait $PID_A
echo "Group A finished (rc=$?)"
wait $PID_B
echo "Group B finished (rc=$?)"

echo ""
echo "########################################################################"
echo "# Overnight 3-seed suite finished at $(date)"
echo "########################################################################"

# Summary
echo ""
echo "=== Results count ==="
for bl in No-FT Periodic-Low Periodic-High Our-System NoFT-Reprefill; do
    n=$(find "${OUT_BASE}/${bl}" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${bl}: ${n}/36 cells (12 cells × 3 seeds)"
done
