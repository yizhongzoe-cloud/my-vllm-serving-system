#!/bin/bash
# Long-prompt high-pressure three-way: ours vs vllm-recompute vs vllm-APC.
# Long prompts (12-24K) make each request's KV large, so the GPU can hold
# only a few; under load + pauses, APC's cached prefixes get LRU-evicted ->
# recompute on resume, while ours' host checkpoint survives -> cheap reload.
# Distinct prompt per request (no APC cross-request sharing).
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_long_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
Q=0.5; N=20; PMIN=12000; PMAX=24000; TOUT=60
echo "[icept_long] start $(date)  qps=$Q n=$N prompt=[$PMIN,$PMAX]"
run(){ # $1 baseline  $2 extra-flags  $3 out-tag
  OUT="${EVAL_RESULTS_DIR}/icept_$1_qps${Q}_n${N}_seed0_$3_metrics.json"
  [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; return; }
  cleanup; echo "[run $(date '+%H:%M:%S')] $1 $2 -> $3"
  python -u experiments_v2/eval/scripts/icept_microbench.py \
    --baseline "$1" $2 --arrival-rate-qps "$Q" --num-requests "$N" \
    --icept-ratio 0.5 --icept-at 30 --total-output "$TOUT" \
    --pause-min 2 --pause-max 7 --seed 0 \
    --prompt-min-tokens "$PMIN" --prompt-max-tokens "$PMAX" \
    --out-tag "$3" > "${LOG_DIR}/$1_$3.log" 2>&1
  echo "  exit=$?"
}
run ours        ""                  long
run vllm_fcfs   ""                  long
run vllm_fcfs   "--prefix-caching"  long_apc
echo "[icept_long] done $(date)"
