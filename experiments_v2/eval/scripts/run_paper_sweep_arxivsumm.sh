#!/bin/bash
# arxivsumm paper sweep — E_M1 only (long-context, uniform SLO mode).
#
# Usage:
#   HARDWARE_TAG=a6000 bash experiments_v2/eval/scripts/run_paper_sweep_arxivsumm.sh
#
# SLO comes from slo_calibration.py on arxivsumm at low load. Override
# E_M1_TTFT_SLO_MS / E_M1_TPOT_SLO_MS via env var when running on a GPU
# whose baseline P95 differs.

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"

HARDWARE_TAG="${HARDWARE_TAG:-default}"
SEEDS="${SEEDS:-0 1 2}"
E_M1_QPS_SWEEP="${E_M1_QPS_SWEEP:-0.1 0.3 0.5 1.0 1.5 2.0}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"
BASELINES_M1="${BASELINES_M1:-vllm_fcfs reroute_no_ckpt ours_no_picker ours}"

# Uniform SLO (JITServe-style: 2x baseline P95). Defaults are from
# A6000 arxivsumm calibration (TTFT P95 1951ms, TPOT P95 25.8ms).
E_M1_TTFT_SLO_MS="${E_M1_TTFT_SLO_MS:-3902}"
E_M1_TPOT_SLO_MS="${E_M1_TPOT_SLO_MS:-52}"

RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
mkdir -p "${RESULTS_DIR}"

export EVAL_RESULTS_DIR="${RESULTS_DIR}"
export PYTHONPATH="${REPO_ROOT}"

LOG_FILE="${RESULTS_DIR}/arxivsumm_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[master] writing all output to: ${LOG_FILE}"
echo "[master] hardware tag: ${HARDWARE_TAG}"
echo "[master] results dir:  ${RESULTS_DIR}"
echo "[master] seeds:        ${SEEDS}"
echo "[master] QPS sweep:    ${E_M1_QPS_SWEEP}"
echo "[master] baselines:    ${BASELINES_M1}"
echo "[master] TTFT SLO:     ${E_M1_TTFT_SLO_MS} ms"
echo "[master] TPOT SLO:     ${E_M1_TPOT_SLO_MS} ms"
exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup_shm() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
}

echo ""
echo "================================================"
echo "[master] E_M1 — arxivsumm rate sweep (uniform SLO)"
echo "================================================"

for seed in ${SEEDS}; do
  for qps in ${E_M1_QPS_SWEEP}; do
    for baseline in ${BASELINES_M1}; do
      echo "[master] E_M1 arxivsumm baseline=${baseline} qps=${qps} seed=${seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset arxivsumm \
        --arrival-rate-qps "${qps}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${seed}" \
        --slo-mode uniform \
        --ttft-slo-ms ${E_M1_TTFT_SLO_MS} \
        --tpot-slo-ms ${E_M1_TPOT_SLO_MS} \
        2>&1 || echo "[master] WARNING: run failed (baseline=${baseline} qps=${qps} seed=${seed})"
    done
  done
done

echo ""
echo "[master] done. results in ${RESULTS_DIR}"
