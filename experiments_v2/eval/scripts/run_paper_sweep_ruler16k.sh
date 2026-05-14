#!/bin/bash
# RULER_16K paper sweep — E_M1 only (picker validation on long-context).
#
# Usage:
#   HARDWARE_TAG=a6000 bash experiments_v2/eval/scripts/run_paper_sweep_ruler16k.sh

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"

HARDWARE_TAG="${HARDWARE_TAG:-default}"
SEEDS="${SEEDS:-0 1 2}"
E_M1_QPS_SWEEP="${E_M1_QPS_SWEEP:-0.5 1.0 2.0}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"
BASELINES_M1="${BASELINES_M1:-vllm_fcfs reroute_no_ckpt ours_no_picker ours}"

# Tier SLO numbers (baseline P95 × 2/3/6).
# Defaults are from A6000 ruler_16k calibration (TTFT P95 2736ms, TPOT
# P95 24.3ms). Override via env vars when running on a different
# hardware whose calibration produces different baseline P95.
E_M1_TTFT_TIGHT_MS="${E_M1_TTFT_TIGHT_MS:-5472}"
E_M1_TTFT_NORMAL_MS="${E_M1_TTFT_NORMAL_MS:-8208}"
E_M1_TTFT_LOOSE_MS="${E_M1_TTFT_LOOSE_MS:-16416}"
E_M1_TPOT_TIGHT_MS="${E_M1_TPOT_TIGHT_MS:-49}"
E_M1_TPOT_NORMAL_MS="${E_M1_TPOT_NORMAL_MS:-73}"
E_M1_TPOT_LOOSE_MS="${E_M1_TPOT_LOOSE_MS:-146}"

# Long-output variant: when FORCE_OUTPUT_TOKENS is set, every request
# is forced to decode that many tokens (paired with ignore_eos). Output
# files get an "_out<N>" suffix so they don't collide with the natural-
# output runs. Leave unset for the default 128-token NIAH workload.
FORCE_OUTPUT_TOKENS="${FORCE_OUTPUT_TOKENS:-}"

RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
mkdir -p "${RESULTS_DIR}"

export EVAL_RESULTS_DIR="${RESULTS_DIR}"
export PYTHONPATH="${REPO_ROOT}"

LOG_FILE="${RESULTS_DIR}/ruler16k_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[master] writing all output to: ${LOG_FILE}"
echo "[master] hardware tag: ${HARDWARE_TAG}"
echo "[master] results dir:  ${RESULTS_DIR}"
echo "[master] seeds:        ${SEEDS}"
echo "[master] QPS sweep:    ${E_M1_QPS_SWEEP}"
echo "[master] baselines:    ${BASELINES_M1}"
if [ -n "${FORCE_OUTPUT_TOKENS}" ]; then
  echo "[master] FORCE_OUTPUT_TOKENS=${FORCE_OUTPUT_TOKENS} (long-output variant with ignore_eos)"
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup_shm() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
}

echo ""
echo "================================================"
echo "[master] E_M1 — RULER_16K rate sweep"
echo "================================================"
EXTRA_ARGS=""
if [ -n "${FORCE_OUTPUT_TOKENS}" ]; then
  EXTRA_ARGS="--force-max-output-tokens ${FORCE_OUTPUT_TOKENS} --ignore-eos"
fi

for seed in ${SEEDS}; do
  for qps in ${E_M1_QPS_SWEEP}; do
    for baseline in ${BASELINES_M1}; do
      echo "[master] E_M1 ruler_16k baseline=${baseline} qps=${qps} seed=${seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset ruler_16k \
        --arrival-rate-qps "${qps}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${seed}" \
        --slo-mode tiered \
        --ttft-slo-tight-ms ${E_M1_TTFT_TIGHT_MS} \
        --ttft-slo-normal-ms ${E_M1_TTFT_NORMAL_MS} \
        --ttft-slo-loose-ms ${E_M1_TTFT_LOOSE_MS} \
        --tpot-slo-tight-ms ${E_M1_TPOT_TIGHT_MS} \
        --tpot-slo-normal-ms ${E_M1_TPOT_NORMAL_MS} \
        --tpot-slo-loose-ms ${E_M1_TPOT_LOOSE_MS} \
        ${EXTRA_ARGS} \
        2>&1 || echo "[master] WARNING: run failed (baseline=${baseline} qps=${qps} seed=${seed})"
    done
  done
done

echo ""
echo "[master] done. results in ${RESULTS_DIR}"
