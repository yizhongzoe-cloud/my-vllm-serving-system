#!/bin/bash
# Master driver: ShareGPT paper sweep for E_M1 / E_M2 / E_M3 / E_M4.
#
# Hardware-tagged outputs: pass HARDWARE_TAG env var (defaults to
# "default") to write into experiments_v2/eval/results/<HARDWARE_TAG>/.
# Run on both A6000 and L40S to compare which platform's data we
# ship in the paper.
#
# Usage on A6000:
#   HARDWARE_TAG=a6000 bash experiments_v2/eval/scripts/run_paper_sweep_sharegpt.sh
#
# Usage on L40S:
#   HARDWARE_TAG=l40s bash experiments_v2/eval/scripts/run_paper_sweep_sharegpt.sh
#
# Optional env vars:
#   SEEDS="0 1 2"            (default: "0 1 2" — 3-seed rigor)
#   E_M1_QPS_SWEEP="1.0 2.0 4.0 6.0 8.0"
#   E_M2_FIXED_QPS="4.0"     (single QPS used for SLO-tightness sweep)
#   E_M4_QPS_SWEEP="1.0 2.0 4.0 6.0 8.0"
#   NUM_REQUESTS="60"
#   BASELINES_M1="vllm_fcfs reroute_no_ckpt ours"
#   BASELINES_M3="vllm_fcfs reroute_no_ckpt ours"
#
# Run order (shortest → longest):
#   E_M3 → E_M4 → E_M2 → E_M1
# so that quick sanity checks (E_M3) surface infra issues before the
# longest sweep.
#
# Total time on A6000: ~13-15 GPU-hours for 3 seeds.

set -u  # do NOT set -e — we want fail-tolerant across runs

REPO_ROOT="/home/yzhong76/code/my-vllm-serving-system"
cd "$REPO_ROOT"

HARDWARE_TAG="${HARDWARE_TAG:-default}"
SEEDS="${SEEDS:-0 1 2}"
E_M1_QPS_SWEEP="${E_M1_QPS_SWEEP:-1.0 2.0 4.0 6.0 8.0}"
E_M2_FIXED_QPS="${E_M2_FIXED_QPS:-4.0}"
E_M4_QPS_SWEEP="${E_M4_QPS_SWEEP:-1.0 2.0 4.0 6.0 8.0}"
NUM_REQUESTS="${NUM_REQUESTS:-60}"
BASELINES_M1="${BASELINES_M1:-vllm_fcfs reroute_no_ckpt ours}"
BASELINES_M3="${BASELINES_M3:-vllm_fcfs reroute_no_ckpt ours}"

# Tier SLO numbers from ShareGPT calibration (baseline P95 × 1.5/3/6).
#
# !!! INCONSISTENT WITH DOCS / RULER_16K SWEEP !!!
# Paper methodology was changed on 2026-05-13 from {1.5, 3, 6} →
# {2, 3, 6} (tight bumped because 1.5× was inside batch-size jitter
# on ShareGPT). All docs and the RULER_16K sweep already use 2/3/6.
# This ShareGPT sweep is intentionally left at 1.5× because the
# A6000 ShareGPT data already in results/a6000/ was collected with
# these values and we don't want to invalidate it yet.
#
# Before re-running this sweep — decide: keep 1.5× to match old
# data, or bump to {912, 1368, 2736} TTFT / {44, 66, 132} TPOT to
# match the paper methodology.
E_M1_TTFT_TIGHT_MS="684"
E_M1_TTFT_NORMAL_MS="1368"
E_M1_TTFT_LOOSE_MS="2736"
E_M1_TPOT_TIGHT_MS="33"
E_M1_TPOT_NORMAL_MS="66"
E_M1_TPOT_LOOSE_MS="132"

RESULTS_DIR="experiments_v2/eval/results/${HARDWARE_TAG}"
mkdir -p "${RESULTS_DIR}"

export EVAL_RESULTS_DIR="${RESULTS_DIR}"
export PYTHONPATH="${REPO_ROOT}"

LOG_FILE="${RESULTS_DIR}/paper_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "[master] writing all output to: ${LOG_FILE}"
echo "[master] hardware tag: ${HARDWARE_TAG}"
echo "[master] results dir:  ${RESULTS_DIR}"
echo "[master] seeds:        ${SEEDS}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cleanup_shm() {
  rm -rf /dev/shm/vllm_ft_preempt_queue \
         /dev/shm/vllm_ft_engine_status \
         /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_checkpoints 2>/dev/null
}

# ============================================================
# E_M3: System overhead (healthy, low load) — 3 baselines × 3 seeds
# ============================================================
echo ""
echo "================================================"
echo "[master] E_M3 — system overhead under healthy load"
echo "================================================"
for seed in ${SEEDS}; do
  for baseline in ${BASELINES_M3}; do
    echo "[master] E_M3 baseline=${baseline} seed=${seed}"
    cleanup_shm
    python -m experiments_v2.eval.scripts.e_m3_overhead \
      --baseline "${baseline}" \
      --seed "${seed}" \
      --num-requests "${NUM_REQUESTS}" \
      2>&1 || echo "[master] WARNING: E_M3 run failed"
  done
done

# ============================================================
# E_M4: Picker ablation — 5 qps × 2 baselines × 3 seeds
# ============================================================
echo ""
echo "================================================"
echo "[master] E_M4 — picker ablation (ours w/ vs w/o picker)"
echo "================================================"
for seed in ${SEEDS}; do
  echo "[master] E_M4 seed=${seed}"
  cleanup_shm
  python -m experiments_v2.eval.scripts.e_m4_picker_ablation \
    --dataset sharegpt \
    --num-requests "${NUM_REQUESTS}" \
    --seed "${seed}" \
    --qps-sweep ${E_M4_QPS_SWEEP} \
    --ttft-slo-tight-ms ${E_M1_TTFT_TIGHT_MS} \
    --ttft-slo-normal-ms ${E_M1_TTFT_NORMAL_MS} \
    --ttft-slo-loose-ms ${E_M1_TTFT_LOOSE_MS} \
    --tpot-slo-tight-ms ${E_M1_TPOT_TIGHT_MS} \
    --tpot-slo-normal-ms ${E_M1_TPOT_NORMAL_MS} \
    --tpot-slo-loose-ms ${E_M1_TPOT_LOOSE_MS} \
    2>&1 || echo "[master] WARNING: E_M4 seed=${seed} failed"
done

# ============================================================
# E_M2: SLO tightness sweep — 3 tightness × 3 baselines × 3 seeds
# ============================================================
echo ""
echo "================================================"
echo "[master] E_M2 — SLO tightness sweep at QPS=${E_M2_FIXED_QPS}"
echo "================================================"
for seed in ${SEEDS}; do
  echo "[master] E_M2 seed=${seed}"
  cleanup_shm
  python -m experiments_v2.eval.scripts.e_m2_slo_tightness \
    --dataset sharegpt \
    --arrival-rate-qps "${E_M2_FIXED_QPS}" \
    --num-requests "${NUM_REQUESTS}" \
    --seed "${seed}" \
    --baselines ${BASELINES_M1} \
    2>&1 || echo "[master] WARNING: E_M2 seed=${seed} failed"
done

# ============================================================
# E_M1: SLO attainment vs load — 5 qps × 3 baselines × 3 seeds
# ============================================================
echo ""
echo "================================================"
echo "[master] E_M1 — SLO attainment vs load (main figure)"
echo "================================================"
for seed in ${SEEDS}; do
  for baseline in ${BASELINES_M1}; do
    for qps in ${E_M1_QPS_SWEEP}; do
      echo "[master] E_M1 baseline=${baseline} qps=${qps} seed=${seed}"
      cleanup_shm
      python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
        --baseline "${baseline}" \
        --dataset sharegpt \
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
        2>&1 || echo "[master] WARNING: E_M1 run failed"
    done
  done
done

echo ""
echo "[master] ALL DONE"
echo "[master] outputs in: ${RESULTS_DIR}"
