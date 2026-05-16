#!/bin/bash
# Master sweep v2: mixed short+long workload (70% sharegpt + 30% arxivsumm),
# per-class SLO (JITServe-style). Replaces the v1 pure-arxivsumm sweep
# which showed no FCFS/ours differentiation (workload too homogeneous;
# see notes.md 2026-05-16 Workload pivot).
#
# Phases (sequential, ~4.5h total on A6000):
#   1. E_M1 mixed_short_long   (~2.5h) — main per-class attainment sweep
#   2. E_M2 mixed_short_long   (~1.3h) — SLO tightness sensitivity at fixed QPS
#   3. E_M3 sharegpt           (~0.7h) — no-load overhead microbench (unchanged)
#
# E_M4 (picker ablation): no separate run; pivot E_M1 ours vs ours_no_picker.
# E_D1 (failover): already done in earlier sweep, not repeated here.
#
# Usage:
#   HARDWARE_TAG=a6000 nohup setsid bash run_paper_sweep_master.sh \
#       > /tmp/master_sweep.log 2>&1 &
#
# Env vars (defaults are A6000 calibration values):
#   HARDWARE_TAG               default
#   SHORT_P95_TTFT_MS          456   (sharegpt low-load P95 on A6000)
#   SHORT_P95_TPOT_MS          22
#   LONG_P95_TTFT_MS           1951  (arxivsumm low-load P95 on A6000)
#   LONG_P95_TPOT_MS           26
#   SEEDS                      "0 1 2"
#   E_M2_SEEDS                 "0 1"
#   E_M1_QPS                   "0.5 1.0 1.5 2.0 3.0"
#   E_M2_FIXED_QPS             "1.5"
#   E_M2_TIGHTNESS_FACTORS     "1.5 2 3 4"
#   MIXED_SHORT_RATIO          0.7
#   NUM_REQUESTS               60
#
# L40S overrides:
#   HARDWARE_TAG=l40s \
#   SHORT_P95_TTFT_MS=268 SHORT_P95_TPOT_MS=21 \
#   LONG_P95_TTFT_MS=1044 LONG_P95_TPOT_MS=...

set -u

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}"

HARDWARE_TAG="${HARDWARE_TAG:-default}"
SEEDS="${SEEDS:-0 1 2}"
E_M2_SEEDS="${E_M2_SEEDS:-0 1}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"
MIXED_SHORT_RATIO="${MIXED_SHORT_RATIO:-0.7}"

E_M1_QPS="${E_M1_QPS:-0.5 1.0 1.5 2.0 3.0}"
E_M2_FIXED_QPS="${E_M2_FIXED_QPS:-1.5}"
E_M2_TIGHTNESS_FACTORS="${E_M2_TIGHTNESS_FACTORS:-1.5 2 3 4}"

# Per-class calibrated baseline P95 (1× P95). E_M1 SLO uses 2× these.
SHORT_P95_TTFT_MS="${SHORT_P95_TTFT_MS:-456}"
SHORT_P95_TPOT_MS="${SHORT_P95_TPOT_MS:-22}"
LONG_P95_TTFT_MS="${LONG_P95_TTFT_MS:-1951}"
LONG_P95_TPOT_MS="${LONG_P95_TPOT_MS:-26}"

BASELINES_M1="vllm_fcfs reroute_no_ckpt ours_no_picker ours"
BASELINES_M3="vllm_fcfs reroute_no_ckpt ours"

RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
mkdir -p "${RESULTS_DIR}"
export EVAL_RESULTS_DIR="${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/master_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[master] log: ${LOG_FILE}"
echo "[master] HARDWARE_TAG: ${HARDWARE_TAG}"
echo "[master] mixed ratio: ${MIXED_SHORT_RATIO} short + $(echo 1.0 - ${MIXED_SHORT_RATIO} | bc) long"
echo "[master] short P95: TTFT=${SHORT_P95_TTFT_MS} TPOT=${SHORT_P95_TPOT_MS}"
echo "[master] long  P95: TTFT=${LONG_P95_TTFT_MS} TPOT=${LONG_P95_TPOT_MS}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup_shm() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
}

# Guard: verify GPU is mostly free before starting. We expect <2GB used
# on each visible GPU (Xorg etc reserve a few MB).
check_gpu_free() {
  local max_used_mib
  max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "${max_used_mib}" -gt 2000 ]; then
    echo "[master] FAIL: GPU has ${max_used_mib} MiB used at startup; expected <2GB. Aborting to avoid silent OOM. Run: pkill -9 -f 'api_server|VLLM::EngineCore' and retry."
    exit 1
  fi
  echo "[master] GPU pre-check OK: max used ${max_used_mib} MiB"
}

check_gpu_free

# Compute 2x default SLOs.
SHORT_TTFT_SLO_MS=$((SHORT_P95_TTFT_MS * 2))
SHORT_TPOT_SLO_MS=$((SHORT_P95_TPOT_MS * 2))
LONG_TTFT_SLO_MS=$((LONG_P95_TTFT_MS * 2))
LONG_TPOT_SLO_MS=$((LONG_P95_TPOT_MS * 2))

echo ""
echo "================================================================"
echo "[master] PHASE 1: E_M1 mixed_short_long — per-class SLO"
echo "                  short: TTFT<${SHORT_TTFT_SLO_MS}ms / TPOT<${SHORT_TPOT_SLO_MS}ms"
echo "                  long:  TTFT<${LONG_TTFT_SLO_MS}ms / TPOT<${LONG_TPOT_SLO_MS}ms"
echo "================================================================"
for seed in ${SEEDS}; do
  for qps in ${E_M1_QPS}; do
    for baseline in ${BASELINES_M1}; do
      echo "[master] E_M1 mixed baseline=${baseline} qps=${qps} seed=${seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset mixed_short_long \
        --mixed-short-ratio "${MIXED_SHORT_RATIO}" \
        --arrival-rate-qps "${qps}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${seed}" \
        --slo-mode mixed \
        --short-ttft-slo-ms ${SHORT_TTFT_SLO_MS} \
        --short-tpot-slo-ms ${SHORT_TPOT_SLO_MS} \
        --long-ttft-slo-ms ${LONG_TTFT_SLO_MS} \
        --long-tpot-slo-ms ${LONG_TPOT_SLO_MS} \
        2>&1 || echo "[master] WARN: E_M1 mixed baseline=${baseline} qps=${qps} seed=${seed} FAILED"
    done
  done
done

echo ""
echo "================================================================"
echo "[master] PHASE 2: E_M2 mixed tightness @ QPS=${E_M2_FIXED_QPS}"
echo "================================================================"
for seed in ${E_M2_SEEDS}; do
  for factor in ${E_M2_TIGHTNESS_FACTORS}; do
    short_ttft=$(echo "scale=0; ${SHORT_P95_TTFT_MS} * ${factor} / 1" | bc)
    short_tpot=$(echo "scale=0; ${SHORT_P95_TPOT_MS} * ${factor} / 1" | bc)
    long_ttft=$(echo "scale=0; ${LONG_P95_TTFT_MS} * ${factor} / 1" | bc)
    long_tpot=$(echo "scale=0; ${LONG_P95_TPOT_MS} * ${factor} / 1" | bc)
    for baseline in ${BASELINES_M1}; do
      tag_seed=$((seed + 100))
      echo "[master] E_M2 mixed baseline=${baseline} factor=${factor}x seed=${seed}->${tag_seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset mixed_short_long \
        --mixed-short-ratio "${MIXED_SHORT_RATIO}" \
        --arrival-rate-qps "${E_M2_FIXED_QPS}" \
        --num-requests "${NUM_REQUESTS}" \
        --seed "${tag_seed}" \
        --slo-mode mixed \
        --short-ttft-slo-ms ${short_ttft} \
        --short-tpot-slo-ms ${short_tpot} \
        --long-ttft-slo-ms ${long_ttft} \
        --long-tpot-slo-ms ${long_tpot} \
        2>&1 || echo "[master] WARN: E_M2 mixed baseline=${baseline} factor=${factor} seed=${seed} FAILED"
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
      2>&1 || echo "[master] WARN: E_M3 baseline=${baseline} seed=${seed} FAILED"
  done
done

echo ""
echo "================================================================"
echo "[master] ALL PHASES COMPLETE. Results in ${RESULTS_DIR}"
echo "================================================================"
