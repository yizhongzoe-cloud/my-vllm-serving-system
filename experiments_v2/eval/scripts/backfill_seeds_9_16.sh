#!/bin/bash
# Backfill all 4 baselines on dual engines for the arxivsumm QPS sweep,
# seeds 9 and 16, so Figures 4/5 can be recomputed over a 4-seed set
# {0,1,9,16}. Skips any run whose metrics already exist.
#
# 4 baselines x 6 QPS x 2 seeds = 48 runs (~3h). Matches the SLO and
# out-tag of the existing canonical data exactly.
#
# Usage:
#   bash experiments_v2/eval/scripts/backfill_seeds_9_16.sh

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"

HARDWARE_TAG="${HARDWARE_TAG:-a6000}"
EVAL_RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/backfill_seeds_9_16_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
export EVAL_RESULTS_DIR="$REPO_ROOT/$EVAL_RESULTS_DIR"

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
n_run=0; n_skip=0; n_fail=0
for seed in 9 16; do
  for qps in 0.3 0.4 0.5 0.6 0.7 0.8; do
    for baseline in vllm_fcfs reroute_no_ckpt ours_no_picker ours; do
      OUT_FILE="${EVAL_RESULTS_DIR}/dual_${baseline}_arxivsumm_qps${qps}_n60_seed${seed}_arxivsumm_v2_canonical_metrics.json"
      if [ -f "$OUT_FILE" ]; then
        echo "[skip exists] $(basename "$OUT_FILE")"; n_skip=$((n_skip+1)); continue
      fi
      cleanup_state
      LOG="${LOG_DIR}/dual_${baseline}_qps${qps}_seed${seed}.log"
      echo "[run $(date '+%H:%M:%S')] baseline=$baseline qps=$qps seed=$seed"
      python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
        --baseline "$baseline" --dataset arxivsumm \
        --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
        --ttft-slo-ms 8348 --tpot-slo-ms 200 \
        --out-tag arxivsumm_v2_canonical \
        > "$LOG" 2>&1
      if [ $? -ne 0 ]; then
        echo "  FAILED — log at $LOG"; n_fail=$((n_fail+1))
      else
        n_run=$((n_run+1))
      fi
    done
  done
done
echo "[backfill] done at $(date) — ran=$n_run skipped=$n_skip failed=$n_fail"
