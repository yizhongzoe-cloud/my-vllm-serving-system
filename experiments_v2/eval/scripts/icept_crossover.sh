#!/bin/bash
# #1 Pause-duration crossover: seg2 resume cost vs pause length, 3 arms.
# Fixed 6-12K prompts, qps0.3, n16, seed0; pause in {2,30,60}s.
# Expect: ours flat-low; recompute flat-high; APC low at short pause
# (cache survives) rising to ~recompute at long pause (LRU-evicted).
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_cross_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
Q=0.3; N=16; PMIN=6000; PMAX=12000; TOUT=60; SEED=${SEED:-0}
echo "[cross] start $(date) seed=$SEED"
run(){ # $1 baseline  $2 extra  $3 out-tag  $4 pause
  OUT="${EVAL_RESULTS_DIR}/icept_$1_qps${Q}_n${N}_seed${SEED}_$3_metrics.json"
  [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; return; }
  cleanup; echo "[run $(date '+%H:%M:%S')] $1 $2 pause=$4 -> $3"
  python -u experiments_v2/eval/scripts/icept_microbench.py \
    --baseline "$1" $2 --arrival-rate-qps "$Q" --num-requests "$N" \
    --icept-ratio 0.5 --icept-at 30 --total-output "$TOUT" \
    --pause-min "$4" --pause-max "$4" --seed "$SEED" \
    --prompt-min-tokens "$PMIN" --prompt-max-tokens "$PMAX" \
    --out-tag "$3" > "${LOG_DIR}/$1_seed${SEED}_$3.log" 2>&1; echo "  exit=$?"; }
for P in 2 5 10 15 20 30 45 60; do
  run ours        ""                 "cross_p${P}" "$P"
  run vllm_fcfs   ""                 "cross_p${P}" "$P"
  run vllm_fcfs   "--prefix-caching" "cross_apc_p${P}" "$P"
done
echo "[cross] done $(date)"
