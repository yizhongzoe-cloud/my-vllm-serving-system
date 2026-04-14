#!/usr/bin/env bash
# ============================================================================
# phase8_repro_3seed.sh — Reproduce investigation_summary 2026-04-08-09 final
# comparison table on W1_Chat/Heavy/F2_Mid with 3 seeds.
#
# Restores phase 8 conditions:
#   - decode_capacity_profile = A6000 dp=1, default=10 (the OLD profile)
#     → Benders solver becomes 98% infeasible → greedy fallback (= phase 8)
#   - env vars = FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1
#   - workload = W1_Chat / Heavy / F2_Mid
#
# 6 logical baselines x 3 seeds = 18 runs, split into two GPU groups:
#   GPU 4-5 (port 8400): No-FT, NoFT-Restart, NoFT-Reprefill           (9 runs)
#   GPU 6-7 (port 8500): Our-System-Restart, Our-System-Reprefill,
#                        Our-System (KV reload)                         (9 runs)
#
# How to run:
#   cd /home/jlpang/my-vllm-serving-system
#   nohup bash experiments_v2/phase8_repro_3seed.sh \
#       > /tmp/phase8_repro.log 2>&1 &
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

OUT_BASE="results_v2/8B/E1a_phase8_repro"
CONFIG="experiments_v2/config_8b_phase8repro.yaml"

WORKLOAD="W1_Chat"
LOAD="Heavy"
FAULT="F2_Mid"
SEEDS=(42 123 456)

run_cell() {
    local gpu="$1"
    local port="$2"
    local logical_name="$3"     # display name (output dir)
    local config_baseline="$4"  # actual baseline name in config_8b.yaml
    local recovery_mode="$5"    # "" | reload | restart | reprefill
    local seed="$6"

    local out_dir="${OUT_BASE}/${logical_name}/${WORKLOAD}/${LOAD}/${FAULT}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${logical_name}/s${seed} (already done)" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local mode_str="${recovery_mode:-default}"
    echo "[$(date +%H:%M:%S)] START ${logical_name}/s${seed} (mode=${mode_str}, GPU ${gpu}, port ${port})" >&2

    local extra_env=""
    if [ -n "$recovery_mode" ]; then
        extra_env="FT_RECOVERY_MODE=${recovery_mode}"
    fi

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline ${config_baseline} \
        --workload ${WORKLOAD} \
        --load ${LOAD} \
        --fault ${FAULT} \
        --seed ${seed} \
        --port ${port} \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${logical_name}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  ttft_p50={m.get(\"ttft_p50_ms\",-1):.0f}  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  ${logical_name}/s${seed}: FAILED (rc=$rc)" >&2
    fi
}

run_group_a() {
    # GPU 4-5, port 8400: No-FT, NoFT-Restart, NoFT-Reprefill
    local gpu="4,5"
    local port="8400"

    for seed in "${SEEDS[@]}"; do
        run_cell "$gpu" "$port" "No-FT"          "No-FT"          ""          "$seed"
        run_cell "$gpu" "$port" "NoFT-Restart"   "NoFT-Reprefill" "reload"    "$seed"
        run_cell "$gpu" "$port" "NoFT-Reprefill" "NoFT-Reprefill" "reprefill" "$seed"
    done
}

run_group_b() {
    # GPU 6-7, port 8500: Our-System-Restart, Our-System-Reprefill, Our-System
    local gpu="6,7"
    local port="8500"

    for seed in "${SEEDS[@]}"; do
        run_cell "$gpu" "$port" "Our-System-Restart"   "Our-System" "restart"   "$seed"
        run_cell "$gpu" "$port" "Our-System-Reprefill" "Our-System" "reprefill" "$seed"
        run_cell "$gpu" "$port" "Our-System"           "Our-System" "reload"    "$seed"
    done
}

echo "########################################################################"
echo "# Phase 8 reproduction — 3 seeds × 6 baselines on W1_Chat/Heavy/F2_Mid"
echo "# Started: $(date)"
echo "# Config: ${CONFIG} (uses old A6000 dp=1 profile, default=10)"
echo "# Env: FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1"
echo "########################################################################"

# Launch both groups in parallel
run_group_a &
PID_A=$!
run_group_b &
PID_B=$!

echo "Group A (GPU 4-5) pid=$PID_A — No-FT + NoFT-Restart + NoFT-Reprefill"
echo "Group B (GPU 6-7) pid=$PID_B — Our-System-Restart + Our-System-Reprefill + Our-System"
echo "Waiting for both groups..."

wait $PID_A
echo "Group A finished (rc=$?)"
wait $PID_B
echo "Group B finished (rc=$?)"

echo ""
echo "########################################################################"
echo "# Phase 8 reproduction finished at $(date)"
echo "########################################################################"

# Summary
echo ""
echo "=== Results count ==="
for bl in No-FT NoFT-Restart NoFT-Reprefill Our-System-Restart Our-System-Reprefill Our-System; do
    n=$(find "${OUT_BASE}/${bl}" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${bl}: ${n}/3 seeds"
done
