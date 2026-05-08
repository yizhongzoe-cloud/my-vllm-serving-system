#!/usr/bin/env bash
# Wait for current 5x loop (PID 1623398) to finish, then launch
# 24-cell main_4cell.sh. Run in background so user can sleep.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
LOG=/tmp/auto_launch_24cell.log

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" >> "${LOG}"; }

log "Waiting for 5x loop (PID 1623398) to finish..."
while kill -0 1623398 2>/dev/null; do
    sleep 30
done
log "5x loop finished. Verifying GPU is free..."

# Verify GPU free
nvidia-smi --query-gpu=memory.used --format=csv,nounits | tail -2 >> "${LOG}"

# Back up old 24-cell results so script doesn't overwrite
if [ -d "${ROOT}/experiments_v2/results_main_4cell" ]; then
    backup="${ROOT}/experiments_v2/results_main_4cell.bak.$(date +%Y%m%d_%H%M%S)"
    mv "${ROOT}/experiments_v2/results_main_4cell" "${backup}"
    log "Backed up old results to ${backup}"
fi

log "Launching 24-cell main run..."
cd "${ROOT}"
nohup bash experiments_v2/run_main_4cell.sh > /tmp/main_4cell_v3.log 2>&1 &
LAUNCHED_PID=$!
log "Launched main_4cell.sh as PID ${LAUNCHED_PID}"
echo "${LAUNCHED_PID}" > /tmp/main_4cell_pid

log "Done. 24-cell run will take ~8 hours."
