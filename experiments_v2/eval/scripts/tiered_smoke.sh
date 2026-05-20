#!/bin/bash
# Smoke test for the heterogeneous-SLO (tiered) pipeline + K=1 cap.
# Small run, baseline=ours so it exercises ALL the new code paths:
# per-request prompt/tier-aware a -> client SLO -> engine laxity ->
# picker (head-danger ratio x a, affordability, K=1 cap) -> tiered
# metric (per-request deadline). Pass iff it completes with a metrics
# JSON and no traceback.
set -u
REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
export EVAL_RESULTS_DIR="$REPO_ROOT/experiments_v2/eval/results/a6000"
LOG_DIR="${EVAL_RESULTS_DIR}/logs/tiered_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

rm -rf /dev/shm/vllm_ft_preempt_queue /dev/shm/vllm_ft_engine_status \
       /dev/shm/vllm_ft_req_map /dev/shm/vllm_ft_checkpoints 2>/dev/null
pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
pkill -KILL -f "EngineCore" 2>/dev/null || true
sleep 3

echo "[tiered_smoke] start $(date), logs in $LOG_DIR"
python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
  --baseline ours --dataset arxivsumm --tiered \
  --arrival-rate-qps 0.6 --num-requests 20 --seed 0 \
  --tpot-slo-ms 200 --tight-ratio 0.3 --loose-mult 2.0 \
  --out-tag tiered_smoke > "${LOG_DIR}/ours_smoke.log" 2>&1
echo "[tiered_smoke] exit=$? done $(date)"
echo "[tiered_smoke] log: ${LOG_DIR}/ours_smoke.log"
