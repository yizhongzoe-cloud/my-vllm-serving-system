#!/bin/bash
# Backfill reroute_no_ckpt at QPS=0.3, 0.4, 0.6 to fill the gap in
# fig_qps_sweep (existing data only covers QPS=0.5, 0.7, 0.8).
# Seeds 0 and 1 to match canonical setup. ~30-50 min total runtime.
#
# Usage:
#   bash experiments_v2/eval/scripts/backfill_reroute_no_ckpt_qps346.sh

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"

HARDWARE_TAG="${HARDWARE_TAG:-a6000}"
EVAL_RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/backfill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cleanup_state() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

echo "[backfill] starting at $(date), logs in $LOG_DIR"

for qps in 0.3 0.4 0.6; do
  for seed in 0 1; do
    OUT_FILE="${EVAL_RESULTS_DIR}/dual_reroute_no_ckpt_arxivsumm_qps${qps}_n60_seed${seed}_arxivsumm_v2_canonical_metrics.json"
    if [ -f "$OUT_FILE" ]; then
      echo "[skip exists] $OUT_FILE"
      continue
    fi
    cleanup_state
    LOG="${LOG_DIR}/dual_reroute_no_ckpt_qps${qps}_seed${seed}.log"
    echo "[run $(date '+%H:%M:%S')] qps=$qps seed=$seed"
    python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
      --baseline reroute_no_ckpt --dataset arxivsumm \
      --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
      --ttft-slo-ms 8348 --tpot-slo-ms 200 \
      --out-tag arxivsumm_v2_canonical \
      > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "  FAILED — log at $LOG"
    fi
  done
done

echo "[backfill] done at $(date)"
