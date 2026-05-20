#!/bin/bash
# Re-run ONLY the 'ours' points after fixing the picker head-danger
# config (removed FT_PICKER_IN_DANGER_MS=300 and HEAD_TOO_LATE=9999999
# from start_engine, so the laxity ratio gate + 200ms too-late gate now
# apply). out-tag tiered2 so the buggy-config 'tiered' ours runs are
# preserved for before/after. Other baselines are picker-independent;
# their 'tiered' numbers still stand.
set -u
REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
export EVAL_RESULTS_DIR="$REPO_ROOT/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/tiered_ab_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cleanup_state() {
  rm -rf /dev/shm/vllm_ft_preempt_queue /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map /dev/shm/vllm_ft_checkpoints 2>/dev/null
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

echo "[tiered_ab_fix] start $(date), logs in $LOG_DIR"
for cfg in "ours 0.5 9" "ours 0.5 0" "ours 0.4 9" "ours 0.4 0"; do
  set -- $cfg; bl=$1; q=$2; s=$3
  OUT="${EVAL_RESULTS_DIR}/dual_${bl}_arxivsumm_qps${q}_n60_seed${s}_tiered2_metrics.json"
  if [ -f "$OUT" ]; then echo "[skip] $(basename "$OUT")"; continue; fi
  cleanup_state
  echo "[run $(date '+%H:%M:%S')] $bl qps=$q seed=$s"
  python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$bl" --dataset arxivsumm --tiered \
    --arrival-rate-qps "$q" --num-requests 60 --seed "$s" \
    --tpot-slo-ms 200 --tight-ratio 0.3 --loose-mult 2.0 \
    --out-tag tiered2 > "${LOG_DIR}/${bl}_q${q}_s${s}.log" 2>&1
  echo "  exit=$? -> $(basename "$OUT")"
done
echo "[tiered_ab_fix] done $(date)"
