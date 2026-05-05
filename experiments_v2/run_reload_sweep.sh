#!/usr/bin/env bash
# Phase 2: Reload-cost sweep launcher.
#
# Pairs:
#   NoFT-Reprefill vs CkptReload
# across:
#   5 contexts (1K/4K/8K/16K/32K) x 3 seeds = 30 cells
#
# Triggers capacity-driven preempt by running with elevated RPS so
# vLLM naturally evicts requests under KV pressure. NoFT-Reprefill
# routes preempt -> RECOMPUTE; CkptReload routes preempt -> our
# host-resident KV restore (FT_CAPACITY_PREEMPT_RELOAD=1).
#
# Usage:
#   nohup bash experiments_v2/run_reload_sweep.sh \
#       > experiments_v2/results_reload/sweep.log 2>&1 &
#
# Per-cell output_dir: experiments_v2/results_reload/<cell>/
# Per-cell artifacts:
#   forward_times_pid<N>.csv
#   reload_times_pid<N>.csv      (CkptReload only)
#   ckpt_stats_pid<N>.csv        (CkptReload only)
#   server.log, run.log, metrics.json, requests.csv

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
RESULTS_DIR="${ROOT}/experiments_v2/results_reload_v2"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
SHM_THRESHOLD=70  # abort sweep if /dev/shm above this %

mkdir -p "${RESULTS_DIR}"

WORKLOADS=(W_Ruler1K W_Ruler4K W_Ruler8K W_Ruler16K W_Ruler32K)
declare -A WORKLOAD_LOAD=(
    [W_Ruler1K]=Pressure_1K
    [W_Ruler4K]=Pressure_4K
    [W_Ruler8K]=Pressure_8K
    [W_Ruler16K]=Pressure_16K
    [W_Ruler32K]=Pressure_32K
)
SEEDS=(42 123 456)
BASELINES=(NoFT-Reprefill CkptReload)

PORT_BASE=8500
TOTAL_CELLS=$(( ${#WORKLOADS[@]} * ${#SEEDS[@]} * ${#BASELINES[@]} ))

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${RESULTS_DIR}/sweep.log"; }

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${SHM_DIR}"
}

shm_pct() {
    df --output=pcent /dev/shm 2>/dev/null \
        | tail -1 | tr -d '% '
}

check_shm() {
    local pct
    pct=$(shm_pct)
    if [ -z "$pct" ]; then return 0; fi
    if [ "$pct" -gt "$SHM_THRESHOLD" ]; then
        log "ABORT: /dev/shm at ${pct}% > ${SHM_THRESHOLD}%"
        return 1
    fi
    return 0
}

CELL_IDX=0
CONSECUTIVE_FAILS=0
declare -a FAILED=()

log "=========================================="
log "Reload sweep starting"
log "Total cells: ${TOTAL_CELLS}"
log "Results dir: ${RESULTS_DIR}"
log "Config: ${CONFIG}"
log "=========================================="

for workload in "${WORKLOADS[@]}"; do
    load_level="${WORKLOAD_LOAD[$workload]}"
    for seed in "${SEEDS[@]}"; do
        for baseline in "${BASELINES[@]}"; do
            CELL_IDX=$((CELL_IDX + 1))
            cell_name="${workload}_${baseline}_seed${seed}"
            cell_out="${RESULTS_DIR}/${cell_name}"
            port=$(( PORT_BASE + CELL_IDX ))

            log "----------------------------------------"
            log "[${CELL_IDX}/${TOTAL_CELLS}] Cell: ${cell_name}"
            log "  workload=${workload} load=${load_level} baseline=${baseline} seed=${seed} port=${port}"

            if ! check_shm; then
                log "  /dev/shm full -> abort sweep"
                break 3
            fi
            log "  /dev/shm usage: $(shm_pct)%"

            cleanup_shm
            mkdir -p "${cell_out}"

            export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
            export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
            export FT_CUDA_EVENT_PROFILE=1
            export FT_CKPT_STATS_LOG=1
            export FT_DELTA_CHECKPOINT=1
            # NEW for Phase 2: route capacity preempt -> reload (only
            # takes effect when checkpoint exists; NoFT-Reprefill
            # baselines won't have ckpt so this is a no-op there).
            export FT_CAPACITY_PREEMPT_RELOAD=1
            # Phase 2 v2 (C-mode): true async overlap. Reload runs on
            # copy stream while OTHER reqs' forward runs on default
            # stream. Recovering req's forward is deferred to the next
            # step (when reload completes via query_restore_done).
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
            # Disable upper-layer logic that could pollute timing:
            export FT_SLO_PREEMPT=0
            export FT_USE_FCFS_BASE_QUEUE=1
            export FT_SKIP_SOLVER=1
            export FT_SLO_AWARE_OBJECTIVE=0
            # Lock to single GPU.
            export CUDA_VISIBLE_DEVICES=0

            # Background nvidia-smi sampler. Started for EVERY cell
            # (both baselines) so the CPU overhead of sampling is
            # symmetric across NoFT-Reprefill and CkptReload — keeps
            # paired comparison fair. ~1Hz sampling, output to
            # gpu_util.csv. Killed after the cell finishes.
            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free \
                --format=csv,nounits \
                -lms 1000 \
                > "${cell_out}/gpu_util.csv" 2>/dev/null &
            NVSMI_PID=$!

            python "${ROOT}/experiments_v2/run.py" \
                --config "${CONFIG}" \
                --baseline "${baseline}" \
                --workload "${workload}" \
                --load "${load_level}" \
                --fault none \
                --seed "${seed}" \
                --port "${port}" \
                --output-dir "${cell_out}" \
                > "${cell_out}/run.log" 2>&1
            rc=$?

            # Stop the GPU sampler (after capturing the python rc).
            kill "${NVSMI_PID}" 2>/dev/null
            wait "${NVSMI_PID}" 2>/dev/null

            if [ $rc -eq 0 ]; then
                log "  OK"
                CONSECUTIVE_FAILS=0
            else
                log "  FAILED (rc=$rc)"
                FAILED+=("${cell_name}")
                echo "${cell_name} rc=$rc" >> "${RESULTS_DIR}/FAILED.txt"
                CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))
                if [ $CONSECUTIVE_FAILS -ge 3 ]; then
                    log "ABORT: 3 consecutive failures"
                    break 3
                fi
            fi

            cleanup_shm
        done
    done
done

cleanup_shm

log "=========================================="
log "Sweep done. Cells run: ${CELL_IDX}/${TOTAL_CELLS}"
log "Failures: ${#FAILED[@]}"
log "=========================================="

{
    echo "Sweep finished at $(ts)"
    echo "Cells attempted: ${CELL_IDX}/${TOTAL_CELLS}"
    echo "Failures:"
    for f in "${FAILED[@]:-}"; do
        echo "  $f"
    done
    echo ""
    echo "Per-cell directories:"
    ls -1 "${RESULTS_DIR}" | grep -E '^W_Ruler' | sort
} > "${RESULTS_DIR}/summary.txt"

log "Summary written to ${RESULTS_DIR}/summary.txt"
