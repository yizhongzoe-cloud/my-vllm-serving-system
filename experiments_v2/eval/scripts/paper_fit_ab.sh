#!/bin/bash
# 2-GPU fit-gate A/B on the paper's uniform-SLO setting (8348/200), corrected
# laxity picker. Only variable = FT_PICKER_FIT_GATE (0=off matches §3.3 picker,
# 1=on adds today's fit gate). K-cap OFF in both. seeds 0,1 at q0.5 and q0.4.
set -u
REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO_ROOT"
export EVAL_RESULTS_DIR="$REPO_ROOT/experiments_v2/eval/results/a6000"
export FT_PICKER_MAX_PREEMPTS_PER_REQ=0   # K-cap off in both arms
LOG_DIR="${EVAL_RESULTS_DIR}/logs/paper_fitab_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
cleanup(){ rm -rf /dev/shm/vllm_ft_preempt_queue /dev/shm/vllm_ft_engine_status \
  /dev/shm/vllm_ft_req_map /dev/shm/vllm_ft_checkpoints 2>/dev/null
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null||true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null||true
  pkill -KILL -f "EngineCore" 2>/dev/null||true; sleep 3; }
echo "[fitab] start $(date)"
# (qps seed) outer, fit inner -> q0.5 s0 nofit/fit first
for qs in "0.5 0" "0.5 1" "0.4 0" "0.4 1"; do
  set -- $qs; q=$1; s=$2
  for fit in 0 1; do
    tag=$([ "$fit" = 1 ] && echo v3_fit || echo v3_nofit)
    export FT_PICKER_FIT_GATE=$fit
    OUT="${EVAL_RESULTS_DIR}/dual_ours_arxivsumm_qps${q}_n60_seed${s}_${tag}_metrics.json"
    if [ -f "$OUT" ]; then echo "[skip] $(basename "$OUT")"; continue; fi
    cleanup
    echo "[run $(date '+%H:%M:%S')] ours q=$q seed=$s FIT_GATE=$fit -> $tag"
    python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
      --baseline ours --dataset arxivsumm --arrival-rate-qps "$q" \
      --num-requests 60 --seed "$s" --ttft-slo-ms 8348 --tpot-slo-ms 200 \
      --out-tag "$tag" > "${LOG_DIR}/ours_q${q}_s${s}_fit${fit}.log" 2>&1
    echo "  exit=$?"
  done
done
echo "[fitab] done $(date)"
