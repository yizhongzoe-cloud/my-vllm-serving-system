#!/usr/bin/env bash
# Phase 2 sanity: launch ONE NoFT-Reprefill cell and ONE CkptReload
# cell on W_Ruler16K (capacity-pressure point), each ~90s, then
# validate:
#   - reload_times CSV exists for CkptReload, has rows
#   - reload count > 0 (preempt+reload actually fired)
#   - reload count ≈ "preempted" mentions in server.log
#   - per-reload bytes scale with context (16K KV ≈ 16-32 MB)

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
SANITY_DIR="${ROOT}/experiments_v2/results_reload_v2/SANITY"
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

    # Cap run_duration_sec via a temp config copy.
    local sanity_cfg="${cell_out}/sanity_config.yaml"
    sed -e 's/^run_duration_sec:.*/run_duration_sec: 90.0/' \
        -e 's/^warmup_sec:.*/warmup_sec: 20.0/' \
        "${CONFIG}" > "${sanity_cfg}"

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

    # Background nvidia-smi sampler (symmetric across baselines).
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

run_cell NoFT-Reprefill 8490
run_cell CkptReload     8491

log "===== Validation ====="

CK_DIR="${SANITY_DIR}/CkptReload_W_Ruler16K_seed42"
NF_DIR="${SANITY_DIR}/NoFT-Reprefill_W_Ruler16K_seed42"

log "[1] forward_times CSVs..."
for d in "${NF_DIR}" "${CK_DIR}"; do
    csv=$(ls $d/forward_times_pid*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
        n=$(wc -l < "$csv")
        log "  $(basename $d): $n forward rows"
    else
        log "  $(basename $d): NO forward_times CSV"
    fi
done

log "[2] reload_times CSV (CkptReload only)..."
rcsv=$(ls $CK_DIR/reload_times_pid*.csv 2>/dev/null | head -1)
if [ -n "$rcsv" ]; then
    rcount=$(($(wc -l < "$rcsv") - 1))
    log "  Reload events recorded: $rcount"
    if [ "$rcount" -gt 0 ]; then
        log "  Sample:"
        head -4 "$rcsv" | sed 's/^/    /' | tee -a "${SANITY_DIR}/sanity.log"
        log "  Per-reload stats:"
        awk -F, 'NR>1 {sum_ms+=$3; sum_b+=$6; n++} END{
            if (n>0) printf "    n=%d mean_ms=%.2f mean_MB=%.1f\n", n, sum_ms/n, (sum_b/n)/1048576
        }' "$rcsv" | tee -a "${SANITY_DIR}/sanity.log"
    fi
else
    log "  NO reload_times CSV — capacity preempt did not route to reload!"
fi

log "[3] preempt count (server.log) vs reload count (CSV)..."
if [ -f "$CK_DIR/server.log" ]; then
    pcount=$(grep -ci "preempt" $CK_DIR/server.log || echo 0)
    log "  CkptReload server.log preempt mentions: $pcount"
fi
if [ -f "$NF_DIR/server.log" ]; then
    npcount=$(grep -ci "preempt" $NF_DIR/server.log || echo 0)
    log "  NoFT-Reprefill server.log preempt mentions: $npcount"
fi

log "[4] ckpt_stats CSV (sanity check that ckpt is also writing)..."
csv=$(ls $CK_DIR/ckpt_stats_pid*.csv 2>/dev/null | head -1)
if [ -n "$csv" ]; then
    n=$(($(wc -l < "$csv") - 1))
    log "  ckpt fires: $n"
fi

log "===== Sanity done. Review ${SANITY_DIR}/sanity.log ====="
