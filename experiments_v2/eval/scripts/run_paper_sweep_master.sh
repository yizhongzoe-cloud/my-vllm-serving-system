#!/bin/bash
# Master overnight sweep: arxivsumm + sharegpt complete experiment set.
#
# Phases (run sequentially, ~8.2h total):
#   1. E_M1 arxivsumm  (~3.4h)  — main long-context SLO sweep
#   2. E_M2 arxivsumm  (~1.6h)  — SLO tightness sweep at fixed QPS
#   3. E_M3 sharegpt   (~0.75h) — no-load overhead microbench
#   4. E_M1 sharegpt   (~2.4h)  — short-context SLO sweep
#
# Usage:
#   HARDWARE_TAG=a6000 nohup setsid bash run_paper_sweep_master.sh \
#       > /tmp/master_sweep.log 2>&1 &
#
# Env vars (defaults are A6000 calibration values):
#   HARDWARE_TAG               default
#   ARXIVSUMM_P95_TTFT_MS      1951
#   ARXIVSUMM_P95_TPOT_MS      26
#   SHAREGPT_P95_TTFT_MS       456
#   SHAREGPT_P95_TPOT_MS       22
#   SEEDS                      "0 1 2"
#   E_M2_SEEDS                 "0 1"     (E_M2 uses fewer seeds)
#   E_M1_ARXIVSUMM_QPS         "0.3 0.5 1.0 1.5 2.0"
#   E_M2_FIXED_QPS             "1.0"
#   E_M2_TIGHTNESS_FACTORS     "1.5 2 3 4"
#   E_M1_SHAREGPT_QPS          "1.0 2.0 4.0 6.0 8.0 10.0"
#   NUM_REQUESTS               60

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"

HARDWARE_TAG="${HARDWARE_TAG:-default}"
SEEDS="${SEEDS:-0 1 2}"
E_M2_SEEDS="${E_M2_SEEDS:-0 1}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"

E_M1_ARXIVSUMM_QPS="${E_M1_ARXIVSUMM_QPS:-0.3 0.5 1.0 1.5 2.0}"
E_M2_FIXED_QPS="${E_M2_FIXED_QPS:-1.0}"
E_M2_TIGHTNESS_FACTORS="${E_M2_TIGHTNESS_FACTORS:-1.5 2 3 4}"
E_M1_SHAREGPT_QPS="${E_M1_SHAREGPT_QPS:-1.0 2.0 4.0 6.0 8.0 10.0}"

# Calibrated baseline P95 (1× P95). E_M1 SLO uses 2× these.
ARXIVSUMM_P95_TTFT_MS="${ARXIVSUMM_P95_TTFT_MS:-1951}"
ARXIVSUMM_P95_TPOT_MS="${ARXIVSUMM_P95_TPOT_MS:-26}"
SHAREGPT_P95_TTFT_MS="${SHAREGPT_P95_TTFT_MS:-456}"
SHAREGPT_P95_TPOT_MS="${SHAREGPT_P95_TPOT_MS:-22}"

BASELINES_M1="vllm_fcfs reroute_no_ckpt ours_no_picker ours"
BASELINES_M3="vllm_fcfs reroute_no_ckpt ours"

RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
mkdir -p "${RESULTS_DIR}"
export EVAL_RESULTS_DIR="${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/master_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[master] log: ${LOG_FILE}"
echo "[master] HARDWARE_TAG: ${HARDWARE_TAG}"
echo "[master] arxivsumm P95: TTFT=${ARXIVSUMM_P95_TTFT_MS} TPOT=${ARXIVSUMM_P95_TPOT_MS}"
echo "[master] sharegpt P95:  TTFT=${SHAREGPT_P95_TTFT_MS} TPOT=${SHAREGPT_P95_TPOT_MS}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup_shm() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
}

# Compute 2x default SLOs (uniform).
ARXIVSUMM_TTFT_SLO_MS=$((ARXIVSUMM_P95_TTFT_MS * 2))
ARXIVSUMM_TPOT_SLO_MS=$((ARXIVSUMM_P95_TPOT_MS * 2))
SHAREGPT_TTFT_SLO_MS=$((SHAREGPT_P95_TTFT_MS * 2))
SHAREGPT_TPOT_SLO_MS=$((SHAREGPT_P95_TPOT_MS * 2))

echo ""
echo "================================================================"
echo "[master] PHASE 1: E_M1 arxivsumm — uniform SLO ${ARXIVSUMM_TTFT_SLO_MS}ms / ${ARXIVSUMM_TPOT_SLO_MS}ms"
echo "================================================================"
for seed in ${SEEDS}; do
  for qps in ${E_M1_ARXIVSUMM_QPS}; do
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
        --ttft-slo-ms ${ARXIVSUMM_TTFT_SLO_MS} \
        --tpot-slo-ms ${ARXIVSUMM_TPOT_SLO_MS} \
        2>&1 || echo "[master] WARN: E_M1 arxivsumm baseline=${baseline} qps=${qps} seed=${seed} FAILED"
    done
  done
done

echo ""
echo "================================================================"
echo "[master] PHASE 2: E_M2 arxivsumm tightness @ QPS=${E_M2_FIXED_QPS}"
echo "================================================================"
for seed in ${E_M2_SEEDS}; do
  for factor in ${E_M2_TIGHTNESS_FACTORS}; do
    # Bash arithmetic doesn't handle floats; use bc.
    ttft_ms=$(echo "scale=0; ${ARXIVSUMM_P95_TTFT_MS} * ${factor} / 1" | bc)
    tpot_ms=$(echo "scale=0; ${ARXIVSUMM_P95_TPOT_MS} * ${factor} / 1" | bc)
    for baseline in ${BASELINES_M1}; do
      tag_seed=$((seed + 100))  # offset so files don't collide with E_M1
      echo "[master] E_M2 arxivsumm baseline=${baseline} tightness=${factor}x P95 (TTFT=${ttft_ms}ms TPOT=${tpot_ms}ms) seed=${seed}->${tag_seed}"
      cleanup_shm
      # E_M2 output filename embeds the seed; we offset to keep
      # filenames distinct from E_M1 (which uses seed 0/1/2 at QPS=1.0).
      # The tightness factor itself is encoded via TTFT/TPOT values.
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset arxivsumm \
        --arrival-rate-qps "${E_M2_FIXED_QPS}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${tag_seed}" \
        --slo-mode uniform \
        --ttft-slo-ms ${ttft_ms} \
        --tpot-slo-ms ${tpot_ms} \
        2>&1 || echo "[master] WARN: E_M2 arxivsumm baseline=${baseline} factor=${factor} seed=${seed} FAILED"
    done
  done
done

echo ""
echo "================================================================"
echo "[master] PHASE 3: E_M3 sharegpt — no-load overhead"
echo "================================================================"
for seed in ${SEEDS}; do
  for baseline in ${BASELINES_M3}; do
    echo "[master] E_M3 sharegpt baseline=${baseline} seed=${seed}"
    cleanup_shm
    python -m experiments_v2.eval.scripts.e_m3_overhead \
      --baseline "${baseline}" \
      --seed "${seed}" \
      2>&1 || echo "[master] WARN: E_M3 sharegpt baseline=${baseline} seed=${seed} FAILED"
  done
done

echo ""
echo "================================================================"
echo "[master] PHASE 4: E_M1 sharegpt — uniform SLO ${SHAREGPT_TTFT_SLO_MS}ms / ${SHAREGPT_TPOT_SLO_MS}ms"
echo "================================================================"
for seed in ${SEEDS}; do
  for qps in ${E_M1_SHAREGPT_QPS}; do
    for baseline in ${BASELINES_M1}; do
      echo "[master] E_M1 sharegpt baseline=${baseline} qps=${qps} seed=${seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset sharegpt \
        --arrival-rate-qps "${qps}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${seed}" \
        --slo-mode uniform \
        --ttft-slo-ms ${SHAREGPT_TTFT_SLO_MS} \
        --tpot-slo-ms ${SHAREGPT_TPOT_SLO_MS} \
        2>&1 || echo "[master] WARN: E_M1 sharegpt baseline=${baseline} qps=${qps} seed=${seed} FAILED"
    done
  done
done

echo ""
echo "================================================================"
echo "[master] ALL PHASES COMPLETE. Results in ${RESULTS_DIR}"
echo "================================================================"
