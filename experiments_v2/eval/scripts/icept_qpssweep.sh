#!/bin/bash
# #2b QPS sweep (extends v1's 0.3/0.5 to a curve): ours vs recompute,
# same v1 config (2-8K prompts, pause 2-7s, n40, seed0). Shows throughput
# diverging as load rises (recompute's reprefill clogs the engine).
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_qps_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
echo "[qps] start $(date)"
for Q in 0.2 0.4 0.6; do
  for bl in ours vllm_fcfs; do
    OUT="${EVAL_RESULTS_DIR}/icept_${bl}_qps${Q}_n40_seed0_v1_metrics.json"
    [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; continue; }
    cleanup; echo "[run $(date '+%H:%M:%S')] $bl qps=$Q"
    python -u experiments_v2/eval/scripts/icept_microbench.py \
      --baseline "$bl" --arrival-rate-qps "$Q" --num-requests 40 \
      --icept-ratio 0.5 --icept-at 30 --total-output 80 \
      --pause-min 2 --pause-max 7 --seed 0 --out-tag v1 \
      > "${LOG_DIR}/${bl}_q${Q}.log" 2>&1; echo "  exit=$?"
  done
done
echo "[qps] done $(date)"
