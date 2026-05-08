#!/usr/bin/env bash
# Reproduce V3 crash in cell-sequence context. The single-cell repro
# (B seed=42 RPS=1.0 600s) doesn't crash, but main run (24-cell sequence)
# crashes B at the same config. Hypothesis: state from previous cells
# (GPU driver, CUDA context, /dev/shm, kernel page cache) is poisoning
# subsequent cells.
#
# Minimum sequence to test: A_vanilla then B_capacity_release at seed=42
# RPS=1.0 — mimicking the main-run crash position (B was 2nd cell after
# A in the RPS=1.0 batch).

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
OUT_ROOT="${ROOT}/experiments_v2/debug_v3_seq"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
PORT=8492

mkdir -p "${OUT_ROOT}"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${OUT_ROOT}/run.log"; }

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${SHM_DIR}"
}

reset_env() {
    unset FT_CAPACITY_PREEMPT_RELOAD
    unset FT_CAPACITY_PREEMPT_RELOAD_OVERLAP
    unset FT_SLO_PREEMPT
    unset SLO_PRIORITY_PREEMPT
}

# Build config: 600s run, 60s warmup, RPS=1.0
DBG_CFG="${OUT_ROOT}/seq_config.yaml"
python3 - <<EOF
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['run_duration_sec'] = 600.0
cfg['warmup_sec'] = 60.0
cfg['load_levels']['Pressure_16K']['rps'] = 1.0
yaml.safe_dump(cfg, open('${DBG_CFG}', 'w'), sort_keys=False)
EOF

run_one() {
    local cell="$1"
    local baseline="$2"
    local cell_out="${OUT_ROOT}/${cell}"
    mkdir -p "${cell_out}"

    cleanup_shm
    reset_env

    export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
    export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
    export FT_CUDA_EVENT_PROFILE=1
    export FT_CKPT_STATS_LOG=1
    export FT_DELTA_CHECKPOINT=1
    export FT_USE_FCFS_BASE_QUEUE=1
    export FT_SKIP_SOLVER=1
    export FT_SLO_AWARE_OBJECTIVE=0
    export FT_SLO_PREEMPT=0
    export CUDA_VISIBLE_DEVICES=0

    case "${cell}" in
        A_vanilla)
            export FT_CAPACITY_PREEMPT_RELOAD=0
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=0
            ;;
        B_capacity_release)
            export FT_CAPACITY_PREEMPT_RELOAD=1
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
            ;;
    esac

    log "===== Cell ${cell} starting ====="
    python "${ROOT}/experiments_v2/run.py" \
        --config "${DBG_CFG}" \
        --baseline "${baseline}" \
        --workload W_Ruler16K \
        --load Pressure_16K \
        --fault none \
        --seed 42 \
        --port "${PORT}" \
        --output-dir "${cell_out}" \
        > "${cell_out}/run.log" 2>&1
    local rc=$?
    log "  ${cell} rc=${rc}"

    cleanup_shm
}

run_one A_vanilla NoFT-Reprefill
run_one B_capacity_release CkptReload

log "===== Sequence done. Validation ====="
for cell in A_vanilla B_capacity_release; do
    log "--- ${cell} ---"
    SRV="${OUT_ROOT}/${cell}/server.log"
    if [ -f "${SRV}" ]; then
        log "  fatal: $(grep -cE 'fatal error|EngineDeadError|IndexKernel|device-side assert' ${SRV})"
        log "  V3 queued: $(grep -c 'FT overlap V3:.*queued' ${SRV})"
        log "  V3 done: $(grep -c 'FT overlap V3:.*done' ${SRV})"
        log "  V3 alloc validation:"
        grep "FT_V3_DEBUG.*alloc'd" ${SRV} | sed 's/^/    /' | tee -a "${OUT_ROOT}/run.log"
    fi
done

log "===== Done ====="
