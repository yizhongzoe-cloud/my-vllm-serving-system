#!/usr/bin/env bash
# Minimal V3 crash repro. Runs the exact configuration that crashed in
# the 4-cell main run: B_capacity_release, W_Ruler16K @ RPS=1.0, seed=42.
#
# CUDA_LAUNCH_BLOCKING=1 makes the CUDA index assert report at the
# precise kernel that triggered it (synchronous), not at a later sync
# point. FT_V3_DEBUG=1 turns on our diagnostic logging for V3 alloc
# block_id validation and pre-forward block_table OOB scan.
#
# Run for 90s to reproduce — previous crash hit ~30s in.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
OUT_DIR="${ROOT}/experiments_v2/debug_v3"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
PORT=8492

mkdir -p "${OUT_DIR}"
rm -rf "${SHM_DIR}"/* 2>/dev/null || true
mkdir -p "${SHM_DIR}"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${OUT_DIR}/run.log"; }

# Build a temp config matching the main run that crashed: 600s run,
# 60s warmup, RPS=1.0. 90s sanity didn't reproduce — bug seems to need
# longer load buildup before triggering.
DBG_CFG="${OUT_DIR}/debug_config.yaml"
python3 - <<EOF
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['run_duration_sec'] = 600.0
cfg['warmup_sec'] = 60.0
cfg['load_levels']['Pressure_16K']['rps'] = 1.0
yaml.safe_dump(cfg, open('${DBG_CFG}', 'w'), sort_keys=False)
EOF

log "Built debug config at ${DBG_CFG}"

# B cell env: V3 (capacity release+reload) on, SLO retain off.
export FT_CUDA_EVENT_OUTPUT_DIR="${OUT_DIR}"
export FT_CKPT_STATS_OUTPUT_DIR="${OUT_DIR}"
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

# Diagnostic flags. The light-weight V3 alloc validation log in
# engine/core.py runs unconditionally. FT_V3_DEBUG=1 also enables a
# heavier per-step block_table scan in model_runner _update_states —
# but that adds ~ms overhead per step which may shift timing enough
# to mask a race-condition bug. Disable for now to match the original
# crashed run's timing as closely as possible.
unset CUDA_LAUNCH_BLOCKING
unset FT_V3_DEBUG
export TORCH_USE_CUDA_DSA=1

log "Launching B_capacity_release / seed=42 / RPS=1.0 / 90s ..."

python "${ROOT}/experiments_v2/run.py" \
    --config "${DBG_CFG}" \
    --baseline CkptReload \
    --workload W_Ruler16K \
    --load Pressure_16K \
    --fault none \
    --seed 42 \
    --port "${PORT}" \
    --output-dir "${OUT_DIR}" \
    > "${OUT_DIR}/run_full.log" 2>&1
rc=$?
log "rc=${rc}"

log "===== Diagnostic findings ====="

SRV="${OUT_DIR}/server.log"
if [ ! -f "${SRV}" ]; then
    log "NO server.log found at ${SRV}"
    exit 1
fi

log "[1] V3 alloc events:"
grep -c "FT overlap V3:.*queued" "${SRV}" 2>/dev/null
grep -c "FT_V3_DEBUG: .* alloc'd" "${SRV}" 2>/dev/null

log "[2] V3 OOB block_ids logged:"
grep "FT_V3_DEBUG:.*OOB block_ids" "${SRV}" 2>/dev/null | head -5

log "[3] block_table OOB pre-forward:"
grep "FT_V3_DEBUG: OOB block_id detected" "${SRV}" 2>/dev/null | head -5

log "[4] CUDA assertion details:"
grep "IndexKernel.cu" "${SRV}" 2>/dev/null | head -2
grep "CUDA error:" "${SRV}" 2>/dev/null | head -2
grep "device-side assert" "${SRV}" 2>/dev/null | head -2

log "[5] fatal error count:"
grep -cE "fatal error|EngineDeadError|RuntimeError|AssertionError" "${SRV}" 2>/dev/null

log "===== Done. Full server log at ${SRV} ====="
