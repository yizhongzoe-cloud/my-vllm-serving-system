#!/usr/bin/env bash
set -euo pipefail

# Write the experiment ids you want to run here.
# Examples:
#   RUN_IDS=(1 2 3 4 5)
#   RUN_IDS=(1 4)
#   RUN_IDS=()
RUN_IDS=(4 5)

# 1 = run suite.py, 0 = skip experiment execution
RUN_EXPERIMENTS=1
# 1 = run analyze.py, 0 = skip figure/table generation
RUN_ANALYSIS=0

PYTHON_BIN="${PYTHON_BIN:-/home/yzhong76/envs/sd_env/bin/python}"
PORT="${PORT:-8300}"

run_experiment() {
  local exp_name="$1"

  echo
  echo "========================================================================"
  echo "Experiment: ${exp_name}"
  echo "========================================================================"

  if [[ "${RUN_EXPERIMENTS}" == "1" ]]; then
    local suite_cmd=(
      "${PYTHON_BIN}" experiments/suite.py
      --experiment "${exp_name}"
      --port "${PORT}"
    )
    echo "\$ ${suite_cmd[*]}"
    "${suite_cmd[@]}"
  fi

  if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    local analyze_cmd=(
      "${PYTHON_BIN}" experiments/analyze.py
      "results/${exp_name}"
      --output "figures/${exp_name}"
    )
    echo "\$ ${analyze_cmd[*]}"
    "${analyze_cmd[@]}"
  fi
}

map_id_to_experiment() {
  case "$1" in
    1) echo "E1_Main" ;;
    2) echo "E2_Recovery" ;;
    3) echo "E3_Ablation" ;;
    4) echo "E4_Checkpoint_Tradeoff" ;;
    5) echo "E5_Controller" ;;
    *)
      echo "ERROR: unknown experiment id '$1' (expected 1..5)" >&2
      exit 1
      ;;
  esac
}

if [[ "${#RUN_IDS[@]}" -eq 0 ]]; then
  echo "No experiments selected. Edit RUN_IDS at the top of this script."
  exit 0
fi

for id in "${RUN_IDS[@]}"; do
  run_experiment "$(map_id_to_experiment "${id}")"
done

echo
echo "All selected experiments completed."
