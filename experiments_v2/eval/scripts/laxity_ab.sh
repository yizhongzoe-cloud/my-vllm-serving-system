#!/bin/bash
# A/B for the laxity picker: rerun "ours" (new laxity-based picker +
# deadline metric) on the cases where the OLD picker churned/collapsed,
# plus one known-good case as a regression check. Saved under out-tag
# "laxity" so the old canonical data is preserved for comparison.
#
#   collapse cases: seed16 qps0.4, seed16 qps0.5, seed9 qps0.5
#   regression:     seed0  qps0.5
set -u
REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
EVAL_RESULTS_DIR="experiments_v2/eval/results/a6000"
export EVAL_RESULTS_DIR="$REPO_ROOT/$EVAL_RESULTS_DIR"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/ddl_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cleanup_state() {
  rm -rf /dev/shm/vllm_ft_preempt_queue /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map /dev/shm/vllm_ft_checkpoints 2>/dev/null
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

echo "[laxity_ab] start $(date), logs in $LOG_DIR"
# config: "baseline qps seed"
for cfg in "ours 0.4 16" "ours 0.5 16" "ours 0.5 9" "ours 0.5 0"; do
  set -- $cfg; bl=$1; q=$2; s=$3
  OUT="${EVAL_RESULTS_DIR}/dual_${bl}_arxivsumm_qps${q}_n60_seed${s}_ddl_metrics.json"
  if [ -f "$OUT" ]; then echo "[skip] $(basename "$OUT")"; continue; fi
  cleanup_state
  echo "[run $(date '+%H:%M:%S')] $bl qps=$q seed=$s"
  python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$bl" --dataset arxivsumm --arrival-rate-qps "$q" \
    --num-requests 60 --seed "$s" --ttft-slo-ms 8348 --tpot-slo-ms 200 \
    --out-tag ddl > "${LOG_DIR}/${bl}_q${q}_s${s}.log" 2>&1
  echo "  exit=$? -> $(basename "$OUT")"
done
echo "[laxity_ab] done $(date)"
