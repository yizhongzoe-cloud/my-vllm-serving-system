#!/bin/bash
# Long-prompt failover (disruption core): 2 engines, kill engine0 mid-decode,
# survivor recovers the rerouted requests. ours (reload from host checkpoint)
# vs reroute_no_ckpt (reprefill on survivor). Detection tuned FAST (stale
# 0.5s, status 100ms, redispatch 0.5s) so the recovery gap is dominated by
# the KV-restore mechanism, not detection (which both pay). Long prompts so
# reprefill (~5s) dwarfs detection. concurrency 6 -> 3 rerouted, survivor
# holds 6x~10K=60K < KV budget (no OOM/thrash). Records detection latency +
# net (mechanism) gap separately.
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
# fast failure detection (common to both arms; isolates the mechanism)
export FT_ROUTER_STALE_THRESHOLD_S=0.5
export FT_ROUTER_STATUS_POLL_INTERVAL_S=0.2
export FT_REDISPATCH_TIMEOUT_S=0.5
export FT_STATUS_WRITE_INTERVAL_MS=100
LOG_DIR="${EVAL_RESULTS_DIR}/logs/failover_long_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f "experiments_v2.router.router" 2>/dev/null
  pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
CONC=6; PMIN=8000; PMAX=12000
echo "[failover_long] start $(date) conc=$CONC prompt=[$PMIN,$PMAX] (stale=0.5s)"
for S in 0 1 2; do
  for bl in ours reroute_no_ckpt; do
    OUT="${EVAL_RESULTS_DIR}/e_d1cl_${bl}_n${CONC}_seed${S}_metrics.json"
    [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; continue; }
    cleanup; echo "[run $(date '+%H:%M:%S')] $bl seed=$S"
    python -u experiments_v2/eval/scripts/e_d1_disruption_demo.py \
      --baseline "$bl" --closed-loop --concurrency "$CONC" --seed "$S" \
      --min-prompt-tokens "$PMIN" --max-prompt-tokens "$PMAX" \
      > "${LOG_DIR}/${bl}_s${S}.log" 2>&1
    echo "  exit=$?"
  done
done
echo "[failover_long] done $(date)"
