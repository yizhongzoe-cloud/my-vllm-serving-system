#!/bin/bash
# Heterogeneous-SLO (tiered) A/B: tight tier (a=0.73*prompt, b=200) vs
# loose tier (2x both -> deadline 2x). K=1 per-request preemption cap
# (engine default). Compares all 4 baselines so we can see whether the
# picker ('ours') now has a genuine beneficiary (delay a slack-rich
# loose request to save a tight one) instead of the zero-sum churn it
# showed under uniform SLO.
#
# Seed order picked for signal:
#   seed 9  — has the 29.7K "monster" prompt IN THE LOOSE TIER (huge
#             laxity -> perennial top victim). Stress-tests the K=1 cap:
#             without it the picker hammers the monster and balloons its
#             e2e. Run FIRST (most informative point).
#   seed 0  — light sample, no monster. Regression: ours must stay
#             competitive when there is nothing to save.
set -u
REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
export EVAL_RESULTS_DIR="$REPO_ROOT/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/tiered_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cleanup_state() {
  rm -rf /dev/shm/vllm_ft_preempt_queue /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map /dev/shm/vllm_ft_checkpoints 2>/dev/null
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 3
}

echo "[tiered_ab] start $(date), logs in $LOG_DIR"
# config: "baseline qps seed"  (seed 9 first = monster/loose stress case)
for cfg in \
    "ours 0.4 9" "ours_no_picker 0.4 9" "reroute_no_ckpt 0.4 9" "vllm_fcfs 0.4 9" \
    "ours 0.5 9" "ours_no_picker 0.5 9" "reroute_no_ckpt 0.5 9" "vllm_fcfs 0.5 9" \
    "ours 0.4 0" "ours_no_picker 0.4 0" "reroute_no_ckpt 0.4 0" "vllm_fcfs 0.4 0" \
    "ours 0.5 0" "ours_no_picker 0.5 0" "reroute_no_ckpt 0.5 0" "vllm_fcfs 0.5 0" ; do
  set -- $cfg; bl=$1; q=$2; s=$3
  OUT="${EVAL_RESULTS_DIR}/dual_${bl}_arxivsumm_qps${q}_n60_seed${s}_tiered_metrics.json"
  if [ -f "$OUT" ]; then echo "[skip] $(basename "$OUT")"; continue; fi
  cleanup_state
  echo "[run $(date '+%H:%M:%S')] $bl qps=$q seed=$s"
  python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$bl" --dataset arxivsumm --tiered \
    --arrival-rate-qps "$q" --num-requests 60 --seed "$s" \
    --tpot-slo-ms 200 --tight-ratio 0.3 --loose-mult 2.0 \
    --out-tag tiered > "${LOG_DIR}/${bl}_q${q}_s${s}.log" 2>&1
  echo "  exit=$? -> $(basename "$OUT")"
done
echo "[tiered_ab] done $(date)"
