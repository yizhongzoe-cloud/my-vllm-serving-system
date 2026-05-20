#!/bin/bash
# Interception (augmented-LLM) experiment: ours (cheap host-RAM reload on
# resume) vs vllm_fcfs (reprefill the accumulated context on resume).
# Single engine. Same workload per (qps,seed) across both arms.
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f "api_server" 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
echo "[icept_sweep] start $(date)"
# (qps seed) outer, baseline inner -> first comparison = qps0.3 s0 ours vs vllm
for qss in "0.3 0" "0.5 0" "0.3 1" "0.5 1"; do
  set -- $qss; q=$1; s=$2
  for bl in ours vllm_fcfs; do
    OUT="${EVAL_RESULTS_DIR}/icept_${bl}_qps${q}_n40_seed${s}_v1_metrics.json"
    if [ -f "$OUT" ]; then echo "[skip] $(basename "$OUT")"; continue; fi
    cleanup
    echo "[run $(date '+%H:%M:%S')] $bl qps=$q seed=$s"
    python -u experiments_v2/eval/scripts/icept_microbench.py \
      --baseline "$bl" --arrival-rate-qps "$q" --num-requests 40 \
      --icept-ratio 0.5 --icept-at 30 --total-output 80 \
      --pause-min 2 --pause-max 7 --seed "$s" --out-tag v1 \
      > "${LOG_DIR}/${bl}_q${q}_s${s}.log" 2>&1
    echo "  exit=$?"
  done
done
echo "[icept_sweep] done $(date)"
