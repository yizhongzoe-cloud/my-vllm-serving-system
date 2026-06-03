#!/usr/bin/env bash
# ONE controlled KV-bound smoke test (codex's config). Staged workload:
# 4 long-context loose holders (ruler_16k, forced 1024 output) sit in decode
# holding ~16K KV each; then 8 short tight heads (ruler_4k, 64 output, tight
# SLO) are injected. Shrunk KV pool (gpu_mem_util=0.75 -> 35,680 tokens) so
# admitting a tight head REQUIRES evicting a loose KV holder = the picker's
# home turf. Peer-load gate OFF so the picker fires for local release+reload.
#
# Compares: ours (slack-aware picker eviction) vs ours_no_picker (native
# FCFS-newest eviction) vs vllm_fcfs (FCFS-newest + recompute, no substrate).
# Run once; do not tune.

set -uo pipefail
cd "$(dirname "$0")/../../.."
export EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-experiments_v2/eval/results/a6000}"
LOG_DIR="$EVAL_RESULTS_DIR/staged_kv_runlogs"; mkdir -p "$LOG_DIR"
TAG="staged_kv_smoke"

cleanup() {
  local pids; pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
  [ -n "$pids" ] && kill -KILL $pids 2>/dev/null || true
  for p in 8400 8401 8402; do local q; q=$(lsof -ti:$p 2>/dev/null || true); [ -n "$q" ] && kill -KILL $q 2>/dev/null || true; done
  pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -KILL -f "experiments_v2.router.router" 2>/dev/null || true
  pkill -KILL -f "EngineCore" 2>/dev/null || true
  sleep 4
}

run() {
  local bl="$1" extra="$2"
  cleanup
  echo "[run $(date '+%H:%M:%S')] baseline=$bl"
  env FT_GPU_MEMORY_UTILIZATION=0.75 $extra python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$bl" --dataset arxivsumm --arrival-rate-qps 0.1 --num-requests 12 --seed 0 \
    --tiered --tpot-slo-ms 200 --loose-mult 2.0 \
    --staged-kv --ignore-eos \
    --out-tag "$TAG" > "$LOG_DIR/dual_${bl}_${TAG}.log" 2>&1
  echo "  done rc=$?"
}

echo "===== staged KV smoke started $(date) ====="
run ours          "FT_ROUTER_POLICY=round_robin FT_PICKER_PEER_LOAD_GATE=0"
run ours_no_picker "FT_ROUTER_POLICY=round_robin"
run vllm_fcfs     ""
cleanup
echo "===== staged KV smoke done $(date) ====="
