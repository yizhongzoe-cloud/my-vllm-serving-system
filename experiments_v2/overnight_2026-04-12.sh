#!/usr/bin/env bash
# ============================================================================
# overnight_2026-04-12.sh — Fill gaps for paper, runs until ~10 AM
#
# GPU 4-5: waits for ckpt_guards to finish (~03:40), then runs checkpointing
#          baselines (Our-System, Periodic-High) sequentially.
# GPU 6-7: starts immediately with NoFT-Reprefill (no /dev/shm, safe parallel).
#
# /dev/shm safety: Our-System and Periodic-High both write /dev/shm checkpoints.
# They must NEVER run in parallel on different GPU pairs. This script ensures:
#   - GPU 6-7 only runs NoFT-Reprefill (enable_checkpointing=false) until GPU 4-5
#     finishes all checkpoint-writing runs.
#   - GPU 4-5 runs Our-System and Periodic-High sequentially.
#
# Total: ~130 runs across 6 phases, ~10 hours.
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

CONFIG="experiments_v2/config_8b.yaml"

WORKLOADS_W1=("W1_Chat")
WORKLOADS_W2=("W2_Summary")
LOADS=("Light" "Moderate" "Heavy")
FAULTS=("none" "F2_Mid")
SEEDS=(42 123 456)

run_cell() {
    local out_base="$1"
    local baseline="$2"
    local workload="$3"
    local load="$4"
    local fault="$5"
    local seed="$6"
    local gpu="$7"
    local port="$8"
    local extra_env="${9:-}"

    local out_dir="${out_base}/${baseline}/${workload}/${load}/${fault}/${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] START ${baseline}/${workload}/${load}/${fault}/s${seed} GPU=${gpu}" >&2

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline ${baseline} \
        --workload ${workload} \
        --load ${load} \
        --fault ${fault} \
        --seed ${seed} \
        --port ${port} \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json; m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${baseline}/${workload}/${load}/${fault}/s${seed}: gp={m.get(\"goodput\",-1):.1f} comp={m.get(\"completion_rate\",-1)*100:.0f}% slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%')
" 2>/dev/null
    else
        echo "  FAILED" >&2
    fi
}

echo "########################################################################"
echo "# Overnight 2026-04-12 — $(date)"
echo "# Filling paper gaps: W2_Summary + Phase 1 best Our-System full sweep"
echo "########################################################################"

# =====================================================================
# GPU 6-7 THREAD: NoFT-Reprefill only (no /dev/shm, safe parallel)
# =====================================================================
gpu67_thread() {
    local gpu="6,7"
    local port="8500"
    local out="results_v2/8B/overnight_2026-04-12"

    # ── Phase 1b: NoFT-Reprefill × W2_Summary × 12 cells × 3 seeds ──
    echo "[$(date +%H:%M:%S)] GPU 6-7: Phase 1b — NoFT-Reprefill W2_Summary"
    for seed in "${SEEDS[@]}"; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "NoFT-Reprefill" "W2_Summary" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_RECOVERY_MODE=reprefill"
            done
        done
    done

    # ── Phase 3b: NoFT-Reprefill × W1_Chat Phase 1 best extended seeds ──
    echo "[$(date +%H:%M:%S)] GPU 6-7: Phase 3b — NoFT-Reprefill W1_Chat extra seeds"
    for seed in 789 1337 777 888 999; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "NoFT-Reprefill" "W1_Chat" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_RECOVERY_MODE=reprefill"
            done
        done
    done

    # ── Phase 5b: NoFT-Reprefill × W2_Summary extended seeds ──
    echo "[$(date +%H:%M:%S)] GPU 6-7: Phase 5b — NoFT-Reprefill W2_Summary extra seeds"
    for seed in 789 1337; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "NoFT-Reprefill" "W2_Summary" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_RECOVERY_MODE=reprefill"
            done
        done
    done

    echo "[$(date +%H:%M:%S)] GPU 6-7 thread done."
}

# =====================================================================
# GPU 4-5 THREAD: waits for ckpt_guards, then checkpoint-writing baselines
# =====================================================================
gpu45_thread() {
    local gpu="4,5"
    local port="8400"
    local out="results_v2/8B/overnight_2026-04-12"

    # Wait for ckpt_guards to finish
    echo "[$(date +%H:%M:%S)] GPU 4-5: Waiting for ckpt_guards to finish..."
    while pgrep -f "ckpt_guards_ab.sh" > /dev/null 2>&1; do
        sleep 60
    done
    sleep 10
    echo "[$(date +%H:%M:%S)] GPU 4-5: ckpt_guards done. Starting experiments."

    # ── Phase 2: Our-System (C1+A3 best) × W1_Chat × 12 cells × 3 seeds ──
    echo "[$(date +%H:%M:%S)] GPU 4-5: Phase 2 — Our-System (Phase 1 best) W1_Chat full sweep"
    for seed in "${SEEDS[@]}"; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "Our-System" "W1_Chat" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1"
            done
        done
    done

    # ── Phase 3: Periodic-High × W2_Summary × 12 cells × 3 seeds ──
    echo "[$(date +%H:%M:%S)] GPU 4-5: Phase 3 — Periodic-High W2_Summary"
    for seed in "${SEEDS[@]}"; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "Periodic-High" "W2_Summary" \
                    "$load" "$fault" "$seed" "$gpu" "$port"
            done
        done
    done

    # ── Phase 4: Our-System (C1+A3) × W2_Summary × 12 cells × 3 seeds ──
    echo "[$(date +%H:%M:%S)] GPU 4-5: Phase 4 — Our-System (Phase 1 best) W2_Summary"
    for seed in "${SEEDS[@]}"; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "Our-System" "W2_Summary" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1"
            done
        done
    done

    # ── Phase 5: Our-System W1_Chat extended seeds ──
    echo "[$(date +%H:%M:%S)] GPU 4-5: Phase 5 — Our-System W1_Chat extra seeds"
    for seed in 789 1337; do
        for load in "${LOADS[@]}"; do
            for fault in "${FAULTS[@]}"; do
                run_cell "$out" "Our-System" "W1_Chat" \
                    "$load" "$fault" "$seed" "$gpu" "$port" \
                    "FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1"
            done
        done
    done

    echo "[$(date +%H:%M:%S)] GPU 4-5 thread done."
}

# Launch both threads
gpu67_thread &
PID_67=$!
gpu45_thread &
PID_45=$!

echo "GPU 6-7 thread pid=$PID_67 (NoFT-Reprefill, starts immediately)"
echo "GPU 4-5 thread pid=$PID_45 (Our-System+Periodic-High, waits for ckpt_guards)"
echo "Waiting for both threads..."

wait $PID_67
echo "[$(date +%H:%M:%S)] GPU 6-7 thread finished (rc=$?)"
wait $PID_45
echo "[$(date +%H:%M:%S)] GPU 4-5 thread finished (rc=$?)"

echo ""
echo "########################################################################"
echo "# Overnight 2026-04-12 finished at $(date)"
echo "########################################################################"

# Summary
echo ""
echo "=== Results count ==="
out="results_v2/8B/overnight_2026-04-12"
for bl in Our-System NoFT-Reprefill Periodic-High; do
    for wl in W1_Chat W2_Summary; do
        n=$(find "${out}/${bl}/${wl}" -name metrics.json 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && echo "  ${bl}/${wl}: ${n} cells"
    done
done
