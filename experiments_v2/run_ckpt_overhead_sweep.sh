#!/usr/bin/env bash
# Background-runnable sweep for the ckpt overhead experiment.
# Pairs (No-FT vs CkptOnly) at 5 context lengths x 3 seeds.
#
# Designed to run under nohup, independent of the Claude Code session:
#     nohup bash experiments_v2/run_ckpt_overhead_sweep.sh \
#         > /home/yzhong76/code/my-vllm-serving-system/experiments_v2/results_ckpt_overhead/sweep.log 2>&1 &
#
# Per-cell isolation:
#   - Each cell launches its own vLLM server (run.py spawns + SIGTERMs it)
#   - /dev/shm/vllm_ft_checkpoints/ cleaned BEFORE and AFTER each cell
#   - /dev/shm capacity checked before each cell; aborts if >70% full
#   - Per-cell output_dir: experiments_v2/results_ckpt_overhead/<workload>_<baseline>_seed<seed>/
#   - Forward-time CSVs and ckpt-stats CSVs written to per-cell out dir
#
# Failure policy:
#   - Single cell failure: skip, log to FAILED, continue
#   - 3 consecutive failures: abort sweep (likely systematic)
#   - SHM full: abort sweep

set -u  # error on undefined vars (do NOT use -e; we want to continue on cell failure)

ROOT=/home/yzhong76/code/my-vllm-serving-system
RESULTS_DIR="${ROOT}/experiments_v2/results_ckpt_overhead"
LOG_FILE="${RESULTS_DIR}/sweep.log"
FAILED_FILE="${RESULTS_DIR}/FAILED.txt"
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"

CONFIG="${ROOT}/experiments_v2/config_8b_ckpt_overhead.yaml"
SHM_DIR=/dev/shm/vllm_ft_checkpoints
SHM_THRESHOLD_PCT=70

mkdir -p "${RESULTS_DIR}"
mkdir -p "${SHM_DIR}"
: > "${FAILED_FILE}"
: > "${SUMMARY_FILE}"

cd "${ROOT}"

# Activate venv if present (otherwise rely on system python).
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${LOG_FILE}"
}

shm_full_pct() {
    df --output=pcent /dev/shm | tail -1 | tr -d ' %'
}

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
}

# Map workload -> matching saturation load level.
declare -A WORKLOAD_LOAD=(
    [W_Ruler1K]=Sat_1K
    [W_Ruler4K]=Sat_4K
    [W_Ruler8K]=Sat_8K
    [W_Ruler16K]=Sat_16K
    [W_Ruler32K]=Sat_32K
)

# Cell ordering: workload outer (so all 6 paired cells for one context
# run consecutively), seed -> baseline interleaved (so paired diff at
# same seed is temporally closest -> minimizes drift).
#
# 30 cells total: 5 contexts x 3 seeds x 2 baselines.
WORKLOADS=(W_Ruler1K W_Ruler4K W_Ruler8K W_Ruler16K W_Ruler32K)
SEEDS=(42 123 456)
BASELINES=(No-FT CkptOnly)

PORT_BASE=8400
CELL_IDX=0
CONSECUTIVE_FAILURES=0
TOTAL_CELLS=$((${#WORKLOADS[@]} * ${#SEEDS[@]} * ${#BASELINES[@]}))

log "=========================================="
log "Ckpt overhead sweep starting"
log "Total cells: ${TOTAL_CELLS}"
log "Results dir: ${RESULTS_DIR}"
log "Config: ${CONFIG}"
log "=========================================="

for workload in "${WORKLOADS[@]}"; do
    load_level="${WORKLOAD_LOAD[$workload]}"
    for seed in "${SEEDS[@]}"; do
        for baseline in "${BASELINES[@]}"; do
            CELL_IDX=$((CELL_IDX + 1))
            cell_id="${workload}_${baseline}_seed${seed}"
            cell_out="${RESULTS_DIR}/${cell_id}"
            port=$((PORT_BASE + CELL_IDX))

            log "----------------------------------------"
            log "[$CELL_IDX/$TOTAL_CELLS] Cell: ${cell_id}"
            log "  workload=${workload} load=${load_level} baseline=${baseline} seed=${seed} port=${port}"

            # SHM capacity guard
            shm_pct=$(shm_full_pct)
            log "  /dev/shm usage: ${shm_pct}%"
            if [ "${shm_pct}" -gt "${SHM_THRESHOLD_PCT}" ]; then
                log "  ABORT: /dev/shm > ${SHM_THRESHOLD_PCT}% before cell"
                echo "${cell_id}: aborted (shm full ${shm_pct}%)" >> "${FAILED_FILE}"
                break 3
            fi

            cleanup_shm
            mkdir -p "${cell_out}"

            # Profiling output dirs: each cell gets its own to avoid
            # mixing forward-times and ckpt-stats across cells.
            export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
            export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"

            # Always-on env for this experiment:
            export FT_CUDA_EVENT_PROFILE=1
            # Ckpt stats and delta only matter when ckpt is on, but
            # set them globally for safety; they are no-ops in No-FT.
            export FT_CKPT_STATS_LOG=1
            export FT_DELTA_CHECKPOINT=1

            # Disable upper-layer logic that could pollute timing:
            export FT_SLO_PREEMPT=0
            export FT_USE_FCFS_BASE_QUEUE=1
            export FT_SKIP_SOLVER=1
            export FT_SLO_AWARE_OBJECTIVE=0
            # Lock to single GPU to avoid TP/multi-GPU complications.
            export CUDA_VISIBLE_DEVICES=0

            # Run the cell (run.py manages server lifecycle).
            python "${ROOT}/experiments_v2/run.py" \
                --config "${CONFIG}" \
                --baseline "${baseline}" \
                --workload "${workload}" \
                --load "${load_level}" \
                --fault none \
                --seed "${seed}" \
                --output-dir "${cell_out}" \
                --port "${port}" \
                >> "${cell_out}/run.log" 2>&1

            rc=$?
            if [ $rc -ne 0 ]; then
                log "  FAIL rc=${rc}"
                echo "${cell_id}: rc=${rc}" >> "${FAILED_FILE}"
                CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                if [ $CONSECUTIVE_FAILURES -ge 3 ]; then
                    log "  ABORT: 3 consecutive failures (likely systematic)"
                    break 3
                fi
            else
                log "  OK"
                CONSECUTIVE_FAILURES=0
            fi

            # Always clean shm after cell to prevent inter-cell leaks.
            cleanup_shm
        done
    done
done

log "=========================================="
log "Sweep done. Cells run: ${CELL_IDX}/${TOTAL_CELLS}"
log "Failures: $(wc -l < "${FAILED_FILE}")"
log "=========================================="

{
    echo "Sweep finished at $(date -Iseconds)"
    echo "Cells attempted: ${CELL_IDX}/${TOTAL_CELLS}"
    echo "Failures:"
    cat "${FAILED_FILE}"
    echo ""
    echo "Per-cell directories:"
    ls -1 "${RESULTS_DIR}" | grep -E '^W_Ruler' | sort
} > "${SUMMARY_FILE}"

cleanup_shm

log "Summary written to ${SUMMARY_FILE}"
exit 0
