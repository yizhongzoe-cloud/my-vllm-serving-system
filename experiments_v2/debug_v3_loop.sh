#!/usr/bin/env bash
# V3 crash bug — repeated seed=42 runs to hit probabilistic race.
#
# Single-cell repro doesn't reliably reproduce. Hypothesis: race fires
# ~5-10% per V3 trigger; we need 15-20 triggers to hit it with 80%+
# confidence. Each cell fires ~2 V3 triggers, so loop 10 times.
#
# When crash occurs, FT_V3_DEBUG_SNAPSHOT will dump full state at the
# V3 alloc moment (running reqs' block_ids, pool state) so we have
# the moment-of-crash data to identify the corruption.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
OUT_ROOT="${ROOT}/experiments_v2/debug_v3_loop"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
PORT=8492
N_RUNS=5

mkdir -p "${OUT_ROOT}"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${OUT_ROOT}/run.log"; }

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${SHM_DIR}"
}

DBG_CFG="${OUT_ROOT}/loop_config.yaml"
python3 - <<EOF
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['run_duration_sec'] = 600.0
cfg['warmup_sec'] = 60.0
cfg['load_levels']['Pressure_16K']['rps'] = 1.0
yaml.safe_dump(cfg, open('${DBG_CFG}', 'w'), sort_keys=False)
EOF

for i in $(seq 1 ${N_RUNS}); do
    cell_out="${OUT_ROOT}/run${i}"
    mkdir -p "${cell_out}"
    cleanup_shm

    export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
    export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
    export FT_CUDA_EVENT_PROFILE=1
    export FT_CKPT_STATS_LOG=1
    export FT_DELTA_CHECKPOINT=1
    export FT_USE_FCFS_BASE_QUEUE=1
    export FT_SKIP_SOLVER=1
    export FT_SLO_AWARE_OBJECTIVE=0
    export FT_SLO_PREEMPT=0
    export FT_CAPACITY_PREEMPT_RELOAD=1
    export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
    unset SLO_PRIORITY_PREEMPT
    export CUDA_VISIBLE_DEVICES=0

    log "===== Run ${i}/${N_RUNS} starting ====="
    python "${ROOT}/experiments_v2/run.py" \
        --config "${DBG_CFG}" \
        --baseline CkptReload \
        --workload W_Ruler16K \
        --load Pressure_16K \
        --fault none \
        --seed 42 \
        --port "${PORT}" \
        --output-dir "${cell_out}" \
        > "${cell_out}/run_full.log" 2>&1
    local_rc=$?

    SRV="${cell_out}/server.log"
    fatal=$(grep -cE "fatal error|EngineDeadError|IndexKernel|device-side assert" $SRV 2>/dev/null)
    v3=$(grep -c "FT overlap V3:.*queued" $SRV 2>/dev/null)
    snap=$(grep -c "FT_V3_DEBUG_SNAPSHOT" $SRV 2>/dev/null)
    log "  Run ${i} rc=${local_rc} fatal=${fatal} V3_queued=${v3} snapshots=${snap}"

    if [ "${fatal}" -gt 0 ]; then
        log "===== CRASH on run ${i} — preserving snapshot data ====="
        log "  See ${SRV}"
        # Extract V3 alloc + snapshot lines together for analysis
        grep -E "FT overlap V3|FT_V3_DEBUG" $SRV > "${cell_out}/v3_diagnostic.log" 2>/dev/null
        log "  V3 diagnostic extracted to ${cell_out}/v3_diagnostic.log"
        break
    fi

    cleanup_shm
done

log "===== Loop done ====="
