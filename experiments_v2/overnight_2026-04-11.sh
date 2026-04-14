#!/usr/bin/env bash
# ============================================================================
# overnight_2026-04-11.sh — Overnight experiment plan
#
# Waits for 2 current tasks (pid 2910750, 2912353), then launches 4 phases:
#
# Phase 1 (parallel, ~2h):  FT_SKIP_SOLVER ablation
#     Our-System + FT_SKIP_SOLVER=1 × 12 cells × 3 seeds (A5000 profile)
#     → pair with E1a_3seed Our-System to measure Benders solver's net value
#     36 runs, split W1 on GPU 4-5 + W4 on GPU 6-7
#
# Phase 2 (parallel, ~70 min): Hard-cell variance enhancement
#     W1_Chat/Heavy/F2_Mid + W4_Mixed/Heavy/F2_Mid × 5 new seeds ×
#     {Our-System, NoFT-Reprefill, Periodic-High}
#     → tighter mean estimate on the highest-variance cells
#     30 runs
#
# Phase 3 (parallel, ~2h):   E1a_3seed extension to 5 seeds
#     Our-System + Periodic-High + NoFT-Reprefill × 12 cells × 2 new seeds
#     (789, 1337) [skip the already-done 42/123/456]
#     → bump 3-seed → 5-seed confidence on the main comparison table
#     72 runs
#
# Phase 4 (sequential on GPU 4-5, ~45 min): Phase 8 profile Our-System seeds
#     Our-System × W1/Heavy/F2_Mid × 5 new seeds (888/999/5678/9999/6666)
#     phase 8 profile (A6000 dp=1 default=10)
#     → estimate crash rate + more data for "phase 8 conditions" comparison
#     5 runs sequential (crash tolerance)
#
# Total: ~6 hours. Buffer ~2 hours.
#
# How to run:
#   nohup bash experiments_v2/overnight_2026-04-11.sh \
#       > /tmp/overnight_2026-04-11.log 2>&1 &
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

CONFIG_STD="experiments_v2/config_8b.yaml"             # A5000 dp=2 default=26
CONFIG_PHASE8="experiments_v2/config_8b_phase8repro.yaml"  # A6000 dp=1 default=10

# ---------------------------------------------------------------------------
# Generic run_cell helper
# ---------------------------------------------------------------------------
run_cell() {
    local out_base="$1"
    local logical_name="$2"
    local config_baseline="$3"
    local workload="$4"
    local load="$5"
    local fault="$6"
    local seed="$7"
    local gpu="$8"
    local port="$9"
    local config="${10}"
    local extra_env="${11:-}"

    local out_dir="${out_base}/${logical_name}/${workload}/${load}/${fault}/${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] START ${logical_name}/${workload}/${load}/${fault}/s${seed} GPU=${gpu}" >&2

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config ${config} \
        --baseline ${config_baseline} \
        --workload ${workload} \
        --load ${load} \
        --fault ${fault} \
        --seed ${seed} \
        --port ${port} \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${logical_name}/${workload}/${load}/${fault}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%')
" 2>/dev/null
    else
        echo "  s${seed}: FAILED (rc=$rc)" >&2
    fi
}

# ---------------------------------------------------------------------------
# Wait for current running tasks
# ---------------------------------------------------------------------------
echo "########################################################################"
echo "# Overnight 2026-04-11 — Started at $(date)"
echo "########################################################################"

echo "[$(date +%H:%M:%S)] Waiting for current tasks to finish..."
for pid in 2910750 2912353; do
    while kill -0 $pid 2>/dev/null; do
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] pid=$pid done"
done

# Also wait for any leftover run.py processes from those tasks
while pgrep -f "run.py.*8[45]00" > /dev/null 2>&1; do
    sleep 10
done
sleep 5

# ---------------------------------------------------------------------------
# Phase 1: FT_SKIP_SOLVER ablation (A5000 profile)
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "# Phase 1: FT_SKIP_SOLVER ablation — $(date)"
echo "########################################################################"

PHASE1_OUT="results_v2/8B/E1a_skip_solver"

phase1_group_a() {
    local gpu="4,5"; local port="8400"
    for seed in 42 123 456; do
        for load in Light Moderate Heavy; do
            for fault in none F2_Mid; do
                run_cell "$PHASE1_OUT" "Our-System" "Our-System" \
                    "W1_Chat" "$load" "$fault" "$seed" "$gpu" "$port" \
                    "$CONFIG_STD" "FT_SKIP_SOLVER=1"
            done
        done
    done
}

phase1_group_b() {
    local gpu="6,7"; local port="8500"
    for seed in 42 123 456; do
        for load in Light Moderate Heavy; do
            for fault in none F2_Mid; do
                run_cell "$PHASE1_OUT" "Our-System" "Our-System" \
                    "W4_Mixed" "$load" "$fault" "$seed" "$gpu" "$port" \
                    "$CONFIG_STD" "FT_SKIP_SOLVER=1"
            done
        done
    done
}

phase1_group_a &
P1A=$!
phase1_group_b &
P1B=$!
echo "Phase 1 Group A (W1 on GPU 4-5): pid=$P1A"
echo "Phase 1 Group B (W4 on GPU 6-7): pid=$P1B"
wait $P1A; echo "[$(date +%H:%M:%S)] Phase 1 Group A done (rc=$?)"
wait $P1B; echo "[$(date +%H:%M:%S)] Phase 1 Group B done (rc=$?)"

sleep 5

# ---------------------------------------------------------------------------
# Phase 2: Hard-cell variance enhancement (A5000 profile)
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "# Phase 2: Hard-cell 5 new seeds × 3 baselines × 2 workloads — $(date)"
echo "########################################################################"

PHASE2_OUT="results_v2/8B/E1a_hard_variance"
NEW_SEEDS_PHASE2=(111 222 333 555 777)

phase2_group_a() {
    local gpu="4,5"; local port="8400"
    # Our-System + NoFT-Reprefill × W1/Heavy/F2_Mid × 5 new seeds = 10 runs
    for seed in "${NEW_SEEDS_PHASE2[@]}"; do
        run_cell "$PHASE2_OUT" "Our-System" "Our-System" \
            "W1_Chat" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD"
        run_cell "$PHASE2_OUT" "NoFT-Reprefill" "NoFT-Reprefill" \
            "W1_Chat" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD" \
            "FT_RECOVERY_MODE=reprefill"
        run_cell "$PHASE2_OUT" "Periodic-High" "Periodic-High" \
            "W1_Chat" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD"
    done
}

phase2_group_b() {
    local gpu="6,7"; local port="8500"
    # Same 3 baselines × W4/Heavy/F2_Mid × 5 new seeds = 15 runs
    for seed in "${NEW_SEEDS_PHASE2[@]}"; do
        run_cell "$PHASE2_OUT" "Our-System" "Our-System" \
            "W4_Mixed" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD"
        run_cell "$PHASE2_OUT" "NoFT-Reprefill" "NoFT-Reprefill" \
            "W4_Mixed" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD" \
            "FT_RECOVERY_MODE=reprefill"
        run_cell "$PHASE2_OUT" "Periodic-High" "Periodic-High" \
            "W4_Mixed" "Heavy" "F2_Mid" "$seed" "$gpu" "$port" "$CONFIG_STD"
    done
}

phase2_group_a &
P2A=$!
phase2_group_b &
P2B=$!
echo "Phase 2 Group A (W1 hard on GPU 4-5): pid=$P2A"
echo "Phase 2 Group B (W4 hard on GPU 6-7): pid=$P2B"
wait $P2A; echo "[$(date +%H:%M:%S)] Phase 2 Group A done (rc=$?)"
wait $P2B; echo "[$(date +%H:%M:%S)] Phase 2 Group B done (rc=$?)"

sleep 5

# ---------------------------------------------------------------------------
# Phase 3: E1a_3seed extension — bump from 3 seeds to 5 seeds
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "# Phase 3: E1a_3seed extension (add seeds 789, 1337) — $(date)"
echo "########################################################################"

PHASE3_OUT="results_v2/8B/E1a_3seed_ext"
NEW_SEEDS_PHASE3=(789 1337)

phase3_group_a() {
    local gpu="4,5"; local port="8400"
    # Our-System + NoFT-Reprefill + Periodic-High × W1 × 6 cells × 2 seeds = 36 runs
    for seed in "${NEW_SEEDS_PHASE3[@]}"; do
        for load in Light Moderate Heavy; do
            for fault in none F2_Mid; do
                run_cell "$PHASE3_OUT" "Our-System" "Our-System" \
                    "W1_Chat" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD"
                run_cell "$PHASE3_OUT" "NoFT-Reprefill" "NoFT-Reprefill" \
                    "W1_Chat" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD" \
                    "FT_RECOVERY_MODE=reprefill"
                run_cell "$PHASE3_OUT" "Periodic-High" "Periodic-High" \
                    "W1_Chat" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD"
            done
        done
    done
}

phase3_group_b() {
    local gpu="6,7"; local port="8500"
    # Same × W4_Mixed × 6 cells × 2 seeds = 36 runs
    for seed in "${NEW_SEEDS_PHASE3[@]}"; do
        for load in Light Moderate Heavy; do
            for fault in none F2_Mid; do
                run_cell "$PHASE3_OUT" "Our-System" "Our-System" \
                    "W4_Mixed" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD"
                run_cell "$PHASE3_OUT" "NoFT-Reprefill" "NoFT-Reprefill" \
                    "W4_Mixed" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD" \
                    "FT_RECOVERY_MODE=reprefill"
                run_cell "$PHASE3_OUT" "Periodic-High" "Periodic-High" \
                    "W4_Mixed" "$load" "$fault" "$seed" "$gpu" "$port" "$CONFIG_STD"
            done
        done
    done
}

phase3_group_a &
P3A=$!
phase3_group_b &
P3B=$!
echo "Phase 3 Group A (W1 on GPU 4-5): pid=$P3A"
echo "Phase 3 Group B (W4 on GPU 6-7): pid=$P3B"
wait $P3A; echo "[$(date +%H:%M:%S)] Phase 3 Group A done (rc=$?)"
wait $P3B; echo "[$(date +%H:%M:%S)] Phase 3 Group B done (rc=$?)"

sleep 5

# ---------------------------------------------------------------------------
# Phase 4: Phase 8 profile Our-System 5 new seeds (sequential, crash tolerance)
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "# Phase 4: Phase 8 profile Our-System seed extension — $(date)"
echo "########################################################################"

PHASE4_OUT="results_v2/8B/E1a_phase8_repro"
PHASE4_SEEDS=(888 999 5678 9999 6666)

for seed in "${PHASE4_SEEDS[@]}"; do
    run_cell "$PHASE4_OUT" "Our-System" "Our-System" \
        "W1_Chat" "Heavy" "F2_Mid" "$seed" "4,5" "8400" "$CONFIG_PHASE8"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "# Overnight 2026-04-11 finished at $(date)"
echo "########################################################################"

echo ""
echo "=== Phase 1 (FT_SKIP_SOLVER ablation) ==="
find results_v2/8B/E1a_skip_solver -name metrics.json 2>/dev/null | wc -l | xargs -I{} echo "  total cells: {} / 36"

echo ""
echo "=== Phase 2 (hard-cell variance) ==="
for bl in Our-System NoFT-Reprefill Periodic-High; do
    n=$(find "results_v2/8B/E1a_hard_variance/${bl}" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${bl}: ${n} / 10 (2 workloads × 5 seeds)"
done

echo ""
echo "=== Phase 3 (E1a_3seed extension) ==="
for bl in Our-System NoFT-Reprefill Periodic-High; do
    n=$(find "results_v2/8B/E1a_3seed_ext/${bl}" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${bl}: ${n} / 24 (2 workloads × 6 cells × 2 seeds)"
done

echo ""
echo "=== Phase 4 (phase 8 profile Our-System extension) ==="
n=$(find results_v2/8B/E1a_phase8_repro/Our-System -name metrics.json 2>/dev/null | wc -l)
echo "  Our-System total seeds (including 123/456/789/2024/3141 + 5 new): ${n}"
