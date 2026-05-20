#!/bin/bash
# Failover concurrency sweep: vary survivor load. Lower concurrency = the
# surviving engine has spare capacity (models a cluster where a failed
# engine's load spreads over many survivors), so the recovery gap is
# dominated by per-request reload-vs-reprefill (not survivor contention).
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
export FT_ROUTER_STALE_THRESHOLD_S=0.5
export FT_ROUTER_STATUS_POLL_INTERVAL_S=0.2
export FT_REDISPATCH_TIMEOUT_S=0.5
export FT_STATUS_WRITE_INTERVAL_MS=100
LOG_DIR="${EVAL_RESULTS_DIR}/logs/failover_conc_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f "experiments_v2.router.router" 2>/dev/null
  pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
PMIN=8000; PMAX=12000; S=0
echo "[failover_conc] start $(date)"
for CONC in 2 4; do
  for bl in ours reroute_no_ckpt; do
    OUT="${EVAL_RESULTS_DIR}/e_d1cl_${bl}_n${CONC}_seed${S}_metrics.json"
    [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; continue; }
    cleanup; echo "[run $(date '+%H:%M:%S')] $bl conc=$CONC seed=$S"
    python -u experiments_v2/eval/scripts/e_d1_disruption_demo.py \
      --baseline "$bl" --closed-loop --concurrency "$CONC" --seed "$S" \
      --min-prompt-tokens "$PMIN" --max-prompt-tokens "$PMAX" \
      > "${LOG_DIR}/${bl}_c${CONC}.log" 2>&1
    echo "  exit=$?"
  done
done
echo "[failover_conc] done $(date)"
