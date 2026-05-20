#!/usr/bin/env bash
# Re-measure two-engine arxivsumm sweep with the CURRENT working-tree code
# (new completion-deadline laxity + redesigned picker), seeds 0,1.
#
# Fresh tag `recheck_0520` so existing arxivsumm_v2_canonical data is NOT
# overwritten — lets us diff old-vs-new design on the same (qps,seed,baseline).
#
# Grid: 4 baselines × qps {0.5,0.4,0.6,0.3} × seeds {0,1} = 32 runs.
# qps=0.5 runs first (headline/ablation point) for early signal.
# Idempotent: skips any combination whose _metrics.json already exists.

set -uo pipefail
cd "$(dirname "$0")/../../.."

export FT_GPU_MEMORY_UTILIZATION="${FT_GPU_MEMORY_UTILIZATION:-0.9}"
export EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-experiments_v2/eval/results/a6000}"
mkdir -p "$EVAL_RESULTS_DIR"

TAG="recheck_0520"
SEEDS=(0 1)
QPS_ORDER=(0.5 0.4 0.6 0.3)
BASELINES=(vllm_fcfs reroute_no_ckpt ours_no_picker ours)

LOG_DIR="$EVAL_RESULTS_DIR/recheck_runlogs"
mkdir -p "$LOG_DIR"

cleanup_stale_state() {
  local gpu_pids
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
             | tr -d ' ' | grep -v '^$' || true)
  if [ -n "$gpu_pids" ]; then
    echo "[cleanup] killing GPU-holding pids: $gpu_pids"
    kill -KILL $gpu_pids 2>/dev/null || true
  fi
  for port in 8400 8401 8402; do
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "[cleanup] killing stale process(es) on port $port: $pids"
      kill -KILL $pids 2>/dev/null || true
    fi
  done
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

run_dual() {
  local baseline="$1" qps="$2" seed="$3"
  local out_file="$EVAL_RESULTS_DIR/dual_${baseline}_arxivsumm_qps${qps}_n60_seed${seed}_${TAG}_metrics.json"
  if [ -f "$out_file" ]; then
    echo "[skip exists] $out_file"
    return 0
  fi
  cleanup_stale_state
  local logf="$LOG_DIR/dual_${baseline}_qps${qps}_seed${seed}_${TAG}.log"
  echo "[run $(date '+%H:%M:%S')] dual baseline=$baseline qps=$qps seed=$seed tag=$TAG"
  python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$baseline" --dataset arxivsumm \
    --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
    --ttft-slo-ms 8348 --tpot-slo-ms 200 \
    --out-tag "$TAG" > "$logf" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAILED rc=$rc — log at $logf"
  fi
}

START_TS=$(date '+%s')
echo "========== recheck sweep started $(date) =========="
echo "tag=$TAG  EVAL_RESULTS_DIR=$EVAL_RESULTS_DIR"
cleanup_stale_state
echo

for qps in "${QPS_ORDER[@]}"; do
  echo "=========== qps=$qps × seeds {0,1} × 4 baselines ==========="
  for seed in "${SEEDS[@]}"; do
    for baseline in "${BASELINES[@]}"; do
      run_dual "$baseline" "$qps" "$seed"
    done
  done
done

cleanup_stale_state
END_TS=$(date '+%s')
ELAPSED=$(( (END_TS - START_TS) / 60 ))
echo
echo "========== recheck sweep done $(date) — total ${ELAPSED} min =========="
