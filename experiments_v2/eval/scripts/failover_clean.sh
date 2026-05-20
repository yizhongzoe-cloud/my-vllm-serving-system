#!/bin/bash
# Clean failover: LOW survivor load (conc 2 -> survivor handles its 1 own +
# 1 rerouted) + LONG prompts (20-26K -> reprefill ~10s >> reload <1s). This
# isolates the recovery mechanism (reload vs reprefill) from survivor
# contention. Models a cluster where a dead engine's load lands on a
# survivor with spare capacity.
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
export FT_ROUTER_STALE_THRESHOLD_S=0.5
export FT_ROUTER_STATUS_POLL_INTERVAL_S=0.2
export FT_REDISPATCH_TIMEOUT_S=0.5
export FT_STATUS_WRITE_INTERVAL_MS=100
LOG_DIR="${EVAL_RESULTS_DIR}/logs/failover_clean_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f "experiments_v2.router.router" 2>/dev/null
  pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
CONC=2; PMIN=20000; PMAX=26000
echo "[failover_clean] start $(date) conc=$CONC prompt=[$PMIN,$PMAX]"
for S in 0 1 2; do
  for bl in ours reroute_no_ckpt; do
    OUT="${EVAL_RESULTS_DIR}/e_d1cl_${bl}_n${CONC}_long_seed${S}_metrics.json"
    [ -f "$OUT" ] && { echo "[skip]"; continue; }
    cleanup; echo "[run $(date '+%H:%M:%S')] $bl conc=$CONC seed=$S"
    # tag override: append _long via out file rename trick -> use a distinct
    # concurrency-derived tag by symlinking is messy; instead rely on the
    # script's own tag (e_d1cl_<bl>_n<conc>_seed<s>) but we changed prompts,
    # so move the result to a _long name afterward.
    python -u experiments_v2/eval/scripts/e_d1_disruption_demo.py \
      --baseline "$bl" --closed-loop --concurrency "$CONC" --seed "$S" \
      --min-prompt-tokens "$PMIN" --max-prompt-tokens "$PMAX" \
      > "${LOG_DIR}/${bl}_s${S}.log" 2>&1
    rc=$?
    SRC="${EVAL_RESULTS_DIR}/e_d1cl_${bl}_n${CONC}_seed${S}_metrics.json"
    [ -f "$SRC" ] && mv "$SRC" "$OUT"
    echo "  exit=$rc -> $(basename "$OUT")"
  done
done
echo "[failover_clean] done $(date)"
