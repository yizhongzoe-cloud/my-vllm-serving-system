#!/usr/bin/env bash
# Extension sweep:
#   - 1K and 4K context @ low RPS (vs main V3 sweep's high RPS)
#     to isolate whether short-context throughput drop is caused by
#     high RPS or by ckpt mechanism overhead.
#   - 64K context to extend long-context scaling data.
#
# 18 cells = 3 contexts x 2 baselines x 3 seeds.
# nvidia-smi sampled per cell (symmetric across baselines).

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_extension.yaml"
RESULTS_DIR="${ROOT}/experiments_v2/results_extension"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
SHM_THRESHOLD=70

mkdir -p "${RESULTS_DIR}"

WORKLOADS=(W_Ruler1K_Low W_Ruler4K_Low W_Ruler64K)
declare -A WORKLOAD_LOAD=(
    [W_Ruler1K_Low]=Pressure_1K_Low
    [W_Ruler4K_Low]=Pressure_4K_Low
    [W_Ruler64K]=Pressure_64K
)
SEEDS=(42 123 456)
BASELINES=(NoFT-Reprefill CkptReload)

PORT_BASE=8600
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
log "Extension sweep starting"
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
            export FT_CAPACITY_PREEMPT_RELOAD=1
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
            export FT_SLO_PREEMPT=0
            export FT_USE_FCFS_BASE_QUEUE=1
            export FT_SKIP_SOLVER=1
            export FT_SLO_AWARE_OBJECTIVE=0
            export CUDA_VISIBLE_DEVICES=0

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
} > "${RESULTS_DIR}/summary.txt"

log "Summary written to ${RESULTS_DIR}/summary.txt"
