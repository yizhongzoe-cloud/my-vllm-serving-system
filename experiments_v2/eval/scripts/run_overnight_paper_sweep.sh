#!/usr/bin/env bash
# Overnight paper sweep: runs all remaining paper experiments serially.
#
# Idempotent: skips any (baseline × QPS × seed × tag) combination whose
# _metrics.json already exists. Safe to interrupt and resume.
#
# Set EVAL_RESULTS_DIR to control output location (default a6000).
# Set DRY_RUN=1 to print commands without executing.

set -uo pipefail

cd "$(dirname "$0")/../../.."

export FT_GPU_MEMORY_UTILIZATION="${FT_GPU_MEMORY_UTILIZATION:-0.9}"
export EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-experiments_v2/eval/results/a6000}"
mkdir -p "$EVAL_RESULTS_DIR"

SEEDS_NEW=(3 4)              # seeds 0,1 already done; skipping 2 (outlier)
SEEDS_ALL=(0 1 3 4)          # full seed list for new experiments
QPS_SWEEP=(0.3 0.4 0.5 0.6 0.7 0.8)
ABLATION_QPS=0.5

LOG_DIR="$EVAL_RESULTS_DIR/overnight_runlogs"
mkdir -p "$LOG_DIR"

# ─────────── helpers ───────────
run_dual() {
  local baseline="$1" qps="$2" seed="$3" tag="$4" extra_env="${5-}"
  local out_file="$EVAL_RESULTS_DIR/dual_${baseline}_arxivsumm_qps${qps}_n60_seed${seed}_${tag}_metrics.json"
  if [ -f "$out_file" ]; then
    echo "[skip exists] $out_file"
    return 0
  fi
  local logf="$LOG_DIR/dual_${baseline}_qps${qps}_seed${seed}_${tag}.log"
  echo "[run $(date '+%H:%M:%S')] dual baseline=$baseline qps=$qps seed=$seed tag=$tag"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  (dry-run; would write $out_file)"
    return 0
  fi
  env $extra_env python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$baseline" --dataset arxivsumm \
    --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
    --ttft-slo-ms 8348 --tpot-slo-ms 200 \
    --out-tag "$tag" > "$logf" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAILED rc=$rc — log at $logf"
  fi
}

run_single() {
  local baseline="$1" qps="$2" seed="$3" tag="$4"
  local out_file="$EVAL_RESULTS_DIR/single_${baseline}_arxivsumm_qps${qps}_n60_seed${seed}_${tag}_metrics.json"
  if [ -f "$out_file" ]; then
    echo "[skip exists] $out_file"
    return 0
  fi
  local logf="$LOG_DIR/single_${baseline}_qps${qps}_seed${seed}_${tag}.log"
  echo "[run $(date '+%H:%M:%S')] single baseline=$baseline qps=$qps seed=$seed tag=$tag"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  (dry-run; would write $out_file)"
    return 0
  fi
  python -u experiments_v2/eval/scripts/single_engine_microbench.py \
    --baseline "$baseline" --dataset arxivsumm \
    --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
    --ttft-slo-ms 8348 --tpot-slo-ms 200 \
    --out-tag "$tag" > "$logf" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAILED rc=$rc — log at $logf"
  fi
}

run_burst() {
  local baseline="$1" qps="$2" seed="$3" tag="$4"
  local out_file="$EVAL_RESULTS_DIR/dual_${baseline}_arxivsumm_qps${qps}_n60_seed${seed}_${tag}_metrics.json"
  if [ -f "$out_file" ]; then
    echo "[skip exists] $out_file"
    return 0
  fi
  local logf="$LOG_DIR/burst_${baseline}_seed${seed}_${tag}.log"
  echo "[run $(date '+%H:%M:%S')] burst baseline=$baseline qps=$qps seed=$seed tag=$tag"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  (dry-run; would write $out_file)"
    return 0
  fi
  python -u experiments_v2/eval/scripts/dual_engine_microbench.py \
    --baseline "$baseline" --dataset arxivsumm \
    --arrival-rate-qps "$qps" --num-requests 60 --seed "$seed" \
    --ttft-slo-ms 8348 --tpot-slo-ms 200 \
    --burst \
    --out-tag "$tag" > "$logf" 2>&1
}

run_e_d1() {
  local baseline="$1" seed="$2"
  local out_file="$EVAL_RESULTS_DIR/e_d1_${baseline}_n20_seed${seed}_metrics.json"
  if [ -f "$out_file" ]; then
    echo "[skip exists] $out_file"
    return 0
  fi
  local logf="$LOG_DIR/e_d1_${baseline}_seed${seed}.log"
  echo "[run $(date '+%H:%M:%S')] e_d1 baseline=$baseline seed=$seed"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  (dry-run; would write $out_file)"
    return 0
  fi
  python -u experiments_v2/eval/scripts/e_d1_disruption_demo.py \
    --baseline "$baseline" --seed "$seed" --num-requests 20 \
    > "$logf" 2>&1
}

START_TS=$(date '+%s')
echo "========== overnight sweep started $(date) =========="
echo "EVAL_RESULTS_DIR=$EVAL_RESULTS_DIR"
echo

# ─────────── A. Gap-fill: dual arxivsumm QPS=0.5 seeds 3,4 ───────────
echo "=========== A. dual arxivsumm QPS=0.5 seeds 3,4 ==========="
for seed in "${SEEDS_NEW[@]}"; do
  for baseline in vllm_fcfs ours; do
    run_dual "$baseline" 0.5 "$seed" arxivsumm_v2_canonical
  done
done

# ─────────── B. Single GPU QPS=0.25 multi-seed ───────────
echo
echo "=========== B. single GPU QPS=0.25 seeds 1,3,4 ==========="
for seed in 1 3 4; do
  for baseline in vllm_fcfs ours; do
    run_single "$baseline" 0.25 "$seed" v1
  done
done

# ─────────── C. QPS sweep dual arxivsumm ───────────
echo
echo "=========== C. dual arxivsumm QPS sweep 0.3-0.8 × seeds 0,1,3,4 ==========="
for qps in "${QPS_SWEEP[@]}"; do
  for seed in "${SEEDS_ALL[@]}"; do
    for baseline in vllm_fcfs ours; do
      run_dual "$baseline" "$qps" "$seed" arxivsumm_v2_canonical
    done
  done
done

# ─────────── D. 4-baseline ablation @ QPS=0.5 ───────────
echo
echo "=========== D. 4-baseline ablation @ QPS=$ABLATION_QPS × seeds 0,1,3,4 ==========="
for seed in "${SEEDS_ALL[@]}"; do
  for baseline in vllm_fcfs reroute_no_ckpt ours_no_picker ours; do
    run_dual "$baseline" "$ABLATION_QPS" "$seed" arxivsumm_v2_canonical
  done
done

# ─────────── E. Router-policy ablation: ours with least_load ───────────
echo
echo "=========== E. router policy ablation (ours + least_load) ==========="
for seed in "${SEEDS_ALL[@]}"; do
  run_dual ours "$ABLATION_QPS" "$seed" arxivsumm_v2_least_load \
    "FT_ROUTER_POLICY=least_load"
done

# ─────────── F. Burst workload (all-at-once arrival) ───────────
echo
echo "=========== F. burst workload @ effective QPS=0.5 × seeds 0,1,3,4 ==========="
for seed in "${SEEDS_ALL[@]}"; do
  for baseline in vllm_fcfs ours; do
    run_burst "$baseline" 0.5 "$seed" arxivsumm_burst
  done
done

# ─────────── G. e_d1 failover demo (arxivsumm) ───────────
echo
echo "=========== G. e_d1 failover (arxivsumm 20 req, kill engine 0) ==========="
for seed in "${SEEDS_ALL[@]}"; do
  for baseline in reroute_no_ckpt ours; do
    run_e_d1 "$baseline" "$seed"
  done
done

END_TS=$(date '+%s')
ELAPSED=$(( (END_TS - START_TS) / 60 ))
echo
echo "========== overnight sweep done $(date) — total ${ELAPSED} min =========="
