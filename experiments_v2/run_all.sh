#!/bin/bash
# Full experiment execution pipeline (v2).
# Usage: bash experiments_v2/run_all.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
PORT="${PORT:-8300}"
V2="experiments_v2"

echo "============================================================"
echo " FT Serving Experiments v2"
echo " Python: $PYTHON"
echo " Port:   $PORT"
echo "============================================================"

# ======= Phase 0: Datasets =======
echo ""
echo "=== Phase 0: Downloading datasets ==="
$PYTHON $V2/datasets/download.py

# ======= Phase 0: Profiling (1B) =======
echo ""
echo "=== Phase 0: Profiling 1B model ==="
$PYTHON $V2/profile_checkpoint_costs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output $V2/checkpoint_cost_profile_1b.json \
    --port $PORT

# ======= Phase 0: Calibration (1B) =======
echo ""
echo "=== Phase 0: Calibrating 1B ==="
$PYTHON $V2/calibrate.py \
    --config $V2/config_1b.yaml \
    --port $PORT \
    --skip-recovery \
    --output $V2/config_1b_calibrated.yaml

# ======= Phase 1: Smoke Test (1B) =======
echo ""
echo "=== Phase 1: Smoke Test (1B) ==="
$PYTHON $V2/suite.py \
    --config $V2/config_1b_calibrated.yaml \
    --experiment E0_Smoke \
    --port $PORT

echo ""
echo "=== Smoke Test Analysis ==="
$PYTHON $V2/analyze.py \
    results_v2/1B/E0_Smoke \
    --output figures_v2/1B/E0_Smoke

echo ""
echo "============================================================"
echo " Smoke test complete."
echo " Check figures_v2/1B/E0_Smoke/ before continuing."
echo " Press Enter to continue to 8B experiments, or Ctrl+C to abort."
echo "============================================================"
read -r

# ======= Phase 2: Profiling + Calibration (8B) =======
echo ""
echo "=== Phase 2: Profiling 8B model ==="
$PYTHON $V2/profile_checkpoint_costs.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output $V2/checkpoint_cost_profile_8b.json \
    --port $PORT

echo ""
echo "=== Phase 2: Calibrating 8B ==="
$PYTHON $V2/calibrate.py \
    --config $V2/config_8b.yaml \
    --port $PORT \
    --output $V2/config_8b_calibrated.yaml

# ======= Phase 2: Core Experiments (8B) =======
for EXP in E0_Smoke E1a_Main E3_Ablation E2_Recovery E4_Checkpoint_Tradeoff E5_Controller E6_SLO_Sensitivity; do
    echo ""
    echo "=== Running $EXP (8B) ==="
    $PYTHON $V2/suite.py \
        --config $V2/config_8b_calibrated.yaml \
        --experiment $EXP \
        --resume \
        --port $PORT

    echo "=== Analyzing $EXP (8B) ==="
    $PYTHON $V2/analyze.py \
        results_v2/8B/$EXP \
        --output figures_v2/8B/$EXP
done

echo ""
echo "============================================================"
echo " 8B experiments complete."
echo " Press Enter to continue to 70B, or Ctrl+C to stop here."
echo "============================================================"
read -r

# ======= Phase 3: 70B Validation =======
echo ""
echo "=== Phase 3: Profiling 70B model ==="
$PYTHON $V2/profile_checkpoint_costs.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output $V2/checkpoint_cost_profile_70b.json \
    --port $PORT

echo ""
echo "=== Phase 3: Calibrating 70B ==="
$PYTHON $V2/calibrate.py \
    --config $V2/config_70b.yaml \
    --port $PORT \
    --output $V2/config_70b_calibrated.yaml

for EXP in E0_Smoke E1b_Main E2_Recovery E4_Checkpoint_Tradeoff E5_Controller; do
    echo ""
    echo "=== Running $EXP (70B) ==="
    $PYTHON $V2/suite.py \
        --config $V2/config_70b_calibrated.yaml \
        --experiment $EXP \
        --resume \
        --port $PORT

    echo "=== Analyzing $EXP (70B) ==="
    $PYTHON $V2/analyze.py \
        results_v2/70B/$EXP \
        --output figures_v2/70B/$EXP
done

# ======= Cross-model analysis (E2) =======
echo ""
echo "=== Cross-model E2 analysis ==="
$PYTHON $V2/analyze.py \
    results_v2/8B/E2_Recovery results_v2/70B/E2_Recovery \
    --output figures_v2/cross_model/E2_Recovery

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
