#!/usr/bin/env bash
# Picker home-turf test: heterogeneous (tiered) SLO + bursty arrivals +
# load-balancing router OFF (round_robin), so engines drift imbalanced and
# the picker has in-flight rebalancing to do.
#
# Three configs, identical bursty-tiered schedule (same seed), scored
# against the same completion-deadline SLO:
#   ours        : picker (laxity victim) + checkpoint (reload resume),
#                 router policy = round_robin (balancing off)
#   vllm_fcfs   : native preempt = kick newest + recompute (no ckpt)
#   vllm_random : native preempt = kick RANDOM + recompute (no ckpt)
#
# "test one version" first: seed 0 only. Expand after we see signal.

set -uo pipefail
cd "$(dirname "$0")/../../.."

export FT_GPU_MEMORY_UTILIZATION="${FT_GPU_MEMORY_UTILIZATION:-0.9}"
export EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-experiments_v2/eval/results/a6000}"
mkdir -p "$EVAL_RESULTS_DIR"

TAG="burst_rr_s0"
SEED=0
QPS=0.5            # mean ~0.37 empirical after burst clustering (~74% of sat)
BURST_SIZE=4
BURST_SPREAD=1.0
LOG_DIR="$EVAL_RESULTS_DIR/burst_runlogs"
mkdir -p "$LOG_DIR"

cleanup() {
  local gpu_pids
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
             | tr -d ' ' | grep -v '^$' || true)
  [ -n "$gpu_pids" ] && kill -KILL $gpu_pids 2>/dev/null || true
  for port in 8400 8401 8402; do
    local pids; pids=$(lsof -ti:"$port" 2>/dev/null || true)
    [ -n "$pids" ] && kill -KILL $pids 2>/dev/null || true
  done
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

run_one() {
  local baseline="$1" extra_env="${2-}"
  cleanup
  local logf="$LOG_DIR/dual_${baseline}_${TAG}.log"
  echo "[run $(date '+%H:%M:%S')] baseline=$baseline tiered burst=$BURST_SIZE/$BURST_SPREAD qps=$QPS seed=$SEED"
  env $extra_env python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$baseline" --dataset arxivsumm \
    --arrival-rate-qps "$QPS" --num-requests 60 --seed "$SEED" \
    --tiered --tpot-slo-ms 200 --tight-ratio 0.3 --loose-mult 2.0 \
    --burst-size "$BURST_SIZE" --burst-spread "$BURST_SPREAD" \
    --out-tag "$TAG" > "$logf" 2>&1
  local rc=$?
  [ "$rc" -ne 0 ] && echo "  FAILED rc=$rc — see $logf"
}

echo "========== burst picker test started $(date) =========="
cleanup
run_one ours "FT_ROUTER_POLICY=round_robin"
run_one vllm_fcfs
run_one vllm_random
cleanup
echo "========== burst picker test done $(date) =========="
