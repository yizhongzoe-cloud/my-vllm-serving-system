#!/bin/bash
# APC control: vllm_fcfs WITH GPU prefix caching, same interception workload
# as the ours/vllm sweep. Tests whether the baseline's continuation reuses
# the cached prompt prefix (cheap, ours loses) or it gets LRU-evicted under
# pressure (recompute, ours wins).
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_apc_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
echo "[icept_apc] start $(date)"
for q in 0.3 0.5; do
  OUT="${EVAL_RESULTS_DIR}/icept_vllm_fcfs_qps${q}_n40_seed0_apc_metrics.json"
  [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; continue; }
  cleanup
  echo "[run $(date '+%H:%M:%S')] vllm_fcfs+APC qps=$q seed=0"
  python -u experiments_v2/eval/scripts/icept_microbench.py \
    --baseline vllm_fcfs --prefix-caching --arrival-rate-qps "$q" \
    --num-requests 40 --icept-ratio 0.5 --icept-at 30 --total-output 80 \
    --pause-min 2 --pause-max 7 --seed 0 --out-tag apc \
    > "${LOG_DIR}/vllm_apc_q${q}_s0.log" 2>&1
  echo "  exit=$?"
done
echo "[icept_apc] done $(date)"
