#!/bin/bash
# Eviction regime via LONG PAUSES (not high qps -> avoids thrash).
# During a 20-40s tool pause, many requests cycle through and LRU-evict the
# paused request's cached prefix -> APC must recompute on resume, while ours'
# host checkpoint survives -> cheap reload. Moderate prompts (6-12K) + qps0.3
# keep the engine making progress. Three-way: ours / recompute / APC.
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/icept_pressure_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }
Q=0.3; N=16; PMIN=6000; PMAX=12000; TOUT=60; PAUSEMIN=20; PAUSEMAX=40
echo "[icept_pressure] start $(date) qps=$Q n=$N prompt=[$PMIN,$PMAX] pause=${PAUSEMIN}-${PAUSEMAX}s"
run(){ OUT="${EVAL_RESULTS_DIR}/icept_$1_qps${Q}_n${N}_seed0_$3_metrics.json"
  [ -f "$OUT" ] && { echo "[skip] $(basename "$OUT")"; return; }
  cleanup; echo "[run $(date '+%H:%M:%S')] $1 $2 -> $3"
  python -u experiments_v2/eval/scripts/icept_microbench.py \
    --baseline "$1" $2 --arrival-rate-qps "$Q" --num-requests "$N" \
    --icept-ratio 0.5 --icept-at 30 --total-output "$TOUT" \
    --pause-min "$PAUSEMIN" --pause-max "$PAUSEMAX" --seed 0 \
    --prompt-min-tokens "$PMIN" --prompt-max-tokens "$PMAX" \
    --out-tag "$3" > "${LOG_DIR}/$1_$3.log" 2>&1
  echo "  exit=$?"; }
run ours        ""                  press
run vllm_fcfs   ""                  press
run vllm_fcfs   "--prefix-caching"  press_apc
echo "[icept_pressure] done $(date)"
