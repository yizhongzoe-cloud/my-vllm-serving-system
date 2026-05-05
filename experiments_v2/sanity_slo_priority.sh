#!/usr/bin/env bash
# Sanity for the SLO_PRIORITY_PREEMPT path (independent of fault).
#
# Goal: verify the new path
#   1. fires (trigger condition met under high-RPS pressure)
#   2. retains blocks correctly
#   3. drained requests resume without crashing vLLM
#
# Setup:
#   - W_Ruler16K @ Pressure_16K (RPS 1.2) — same load as V3 sanity
#   - Single CkptReload-style cell with SLO_PRIORITY_PREEMPT=1
#   - Capacity-preempt routing (FT_CAPACITY_PREEMPT_RELOAD*) is OFF so
#     we don't mix mechanisms.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
SANITY_DIR="${ROOT}/experiments_v2/results_slo_priority/SANITY"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"

mkdir -p "${SANITY_DIR}"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${SANITY_DIR}/sanity.log"; }

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${SHM_DIR}"
}

run_cell() {
    local baseline="$1"
    local workload="W_Ruler16K"
    local load_level="Pressure_16K"
    local seed=42
    local port="$2"

    local cell_out="${SANITY_DIR}/${baseline}_${workload}_seed${seed}"
    mkdir -p "${cell_out}"

    log "===== Sanity cell: ${baseline} ====="

    cleanup_shm

    local sanity_cfg="${cell_out}/sanity_config.yaml"
    sed -e 's/^run_duration_sec:.*/run_duration_sec: 90.0/' \
        -e 's/^warmup_sec:.*/warmup_sec: 20.0/' \
        "${CONFIG}" > "${sanity_cfg}"

    export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
    export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
    export FT_CUDA_EVENT_PROFILE=1
    export FT_CKPT_STATS_LOG=1
    export FT_DELTA_CHECKPOINT=1
    # Capacity-preempt routing OFF so capacity preempt goes through
    # vLLM's vanilla path (RECOMPUTE), not our reload path. We're
    # testing only SLO priority retain here.
    export FT_CAPACITY_PREEMPT_RELOAD=0
    export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=0
    # Fault recovery SLO path OFF.
    export FT_SLO_PREEMPT=0
    # NEW path enabled.
    export SLO_PRIORITY_PREEMPT=1
    # Lower the slack-gap gate so trigger fires more easily under our
    # synthetic 90s sanity window.
    export SLO_PRIORITY_PREEMPT_MIN_GAP_MS=1000
    # Shorter retain window so we observe full retain → resume cycle
    # within 90 seconds.
    export SLO_PRIORITY_PREEMPT_RETAIN_STEPS=3

    export FT_USE_FCFS_BASE_QUEUE=1
    export FT_SKIP_SOLVER=1
    export FT_SLO_AWARE_OBJECTIVE=0
    export CUDA_VISIBLE_DEVICES=0

    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free \
        --format=csv,nounits \
        -lms 1000 \
        > "${cell_out}/gpu_util.csv" 2>/dev/null &
    local nvsmi_pid=$!

    python "${ROOT}/experiments_v2/run.py" \
        --config "${sanity_cfg}" \
        --baseline "${baseline}" \
        --workload "${workload}" \
        --load "${load_level}" \
        --fault none \
        --seed "${seed}" \
        --port "${port}" \
        --output-dir "${cell_out}" \
        > "${cell_out}/run.log" 2>&1
    local rc=$?
    log "${baseline} cell rc=${rc}"

    kill "${nvsmi_pid}" 2>/dev/null
    wait "${nvsmi_pid}" 2>/dev/null

    cleanup_shm
}

run_cell CkptReload 8492

log "===== Validation ====="

CELL_DIR="${SANITY_DIR}/CkptReload_W_Ruler16K_seed42"

log "[1] cell completion status..."
if [ -f "${CELL_DIR}/run.log" ]; then
    log "  run.log exists"
fi

log "[2] forward_times CSV..."
fcsv=$(ls $CELL_DIR/forward_times_pid*.csv 2>/dev/null | head -1)
if [ -n "$fcsv" ]; then
    n=$(($(wc -l < "$fcsv") - 1))
    log "  forward steps recorded: $n"
else
    log "  NO forward_times CSV — engine may have crashed"
fi

log "[3] SLO priority trigger fires..."
if [ -f "${CELL_DIR}/server.log" ]; then
    fired=$(grep -c "SLO_PRIORITY_PREEMPT #" "${CELL_DIR}/server.log" || echo 0)
    log "  SLO_PRIORITY_PREEMPT trigger fires: $fired"
    if [ "$fired" -gt 0 ]; then
        log "  Sample fires:"
        grep "SLO_PRIORITY_PREEMPT #" "${CELL_DIR}/server.log" | head -3 | sed 's/^/    /' | tee -a "${SANITY_DIR}/sanity.log"
    fi
fi

log "[4] retain → resume completions..."
if [ -f "${CELL_DIR}/server.log" ]; then
    resumed=$(grep -c "FT SLO retain.*resumed" "${CELL_DIR}/server.log" || echo 0)
    log "  retained reqs resumed: $resumed"
    if [ "$resumed" -gt 0 ]; then
        log "  Sample resumes:"
        grep "FT SLO retain.*resumed" "${CELL_DIR}/server.log" | head -3 | sed 's/^/    /' | tee -a "${SANITY_DIR}/sanity.log"
    fi
fi

log "[5] crashes / fatal errors..."
if [ -f "${CELL_DIR}/server.log" ]; then
    fatal=$(grep -c "fatal error\|EngineDeadError\|RuntimeError\|AssertionError" "${CELL_DIR}/server.log" || echo 0)
    log "  fatal/error mentions in server.log: $fatal"
    if [ "$fatal" -gt 0 ]; then
        log "  First 5 error lines:"
        grep "fatal error\|EngineDeadError\|RuntimeError\|AssertionError" "${CELL_DIR}/server.log" | head -5 | sed 's/^/    /' | tee -a "${SANITY_DIR}/sanity.log"
    fi
fi

log "===== Sanity done. Review ${SANITY_DIR}/sanity.log ====="
