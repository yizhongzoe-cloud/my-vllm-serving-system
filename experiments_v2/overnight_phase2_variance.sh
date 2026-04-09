#!/usr/bin/env bash
# ============================================================================
# overnight_phase2_variance.sh — Multi-seed variance baseline.
#
# We need to know the run-to-run noise floor BEFORE we can claim that any
# optimization is real. Today's session showed reload goodput swinging
# from 116 to 142 across runs depending on stochastic active@fault.
# This script runs 3 seeds × 2 baselines = 6 runs to bound the variance.
#
# Sequential (see overnight_2026-04-09.sh for why).
# Expected runtime: ~45 minutes (6 × 7 min each + 1 min restart overhead).
# ============================================================================

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_BASE="results_v2/8B_overnight_2026-04-09/phase2_variance"
mkdir -p "$OUT_BASE"

WORKLOAD="W1_Chat"
LOAD="Heavy"
FAULT="F2_Mid"
CONFIG="experiments_v2/config_8b.yaml"
PORT=8400

run_seed() {
    local baseline="$1"
    local seed="$2"
    local out_dir="${OUT_BASE}/${baseline}/seed${seed}"
    mkdir -p "$out_dir"

    echo "================================================================="
    echo "[$(date +%T)] START: phase2 ${baseline} seed=${seed}"
    echo "================================================================="

    rm -rf /dev/shm/vllm_ft_checkpoints

    python experiments_v2/run.py \
        --config "$CONFIG" \
        --baseline "$baseline" \
        --workload "$WORKLOAD" \
        --load "$LOAD" \
        --fault "$FAULT" \
        --seed "$seed" \
        --port "$PORT" \
        --output-dir "$out_dir" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    echo "[$(date +%T)] END:   ${baseline} seed=${seed} rc=$rc"
    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  goodput={m.get(\"goodput\", -1):.1f}  '
      f'ttft_p50={m.get(\"ttft_p50_ms\", -1):.0f}ms  '
      f'tpot_p50={m.get(\"tpot_p50_ms\", -1):.1f}ms  '
      f'slo_viol={m.get(\"slo_violation_rate\", -1)*100:.1f}%  '
      f'completion={m.get(\"completion_rate\", -1)*100:.1f}%  '
      f'active@fault={m.get(\"active_requests_at_fault\", \"?\")}')
"
    fi
    echo ""
}

echo "########################################################################"
echo "# Phase 2: Multi-seed variance baseline — $(date)"
echo "########################################################################"

# 3 seeds × Our-System (reload default) and Our-System-NoCkpt
for seed in 42 123 456; do
    run_seed "Our-System"        "$seed"
    run_seed "Our-System-NoCkpt" "$seed"
done

echo "########################################################################"
echo "# Phase 2 finished at $(date)"
echo "########################################################################"

# Summary table
python3 << 'PYEOF'
import json
from pathlib import Path
import statistics

base = Path("results_v2/8B_overnight_2026-04-09/phase2_variance")
data = {}
for baseline_dir in sorted(base.iterdir()):
    if not baseline_dir.is_dir():
        continue
    runs = []
    for seed_dir in sorted(baseline_dir.iterdir()):
        mf = seed_dir / "metrics.json"
        if mf.exists():
            try:
                m = json.load(open(mf))
                runs.append({
                    "seed": seed_dir.name,
                    "goodput": m.get("goodput", -1),
                    "ttft_p50": m.get("ttft_p50_ms", -1),
                    "tpot_p50": m.get("tpot_p50_ms", -1),
                    "slo_viol": m.get("slo_violation_rate", -1) * 100,
                    "active": m.get("active_requests_at_fault", "?"),
                })
            except Exception as e:
                print(f"  parse {mf}: {e}")
    if runs:
        data[baseline_dir.name] = runs

print("\n=== Phase 2 Variance Summary ===\n")
print(f"{'baseline':<25} {'seed':>10} {'goodput':>10} {'ttft_p50':>10} {'tpot_p50':>10} {'slo%':>7} {'act@flt':>8}")
print("-" * 90)
for baseline, runs in data.items():
    for r in runs:
        print(f"{baseline:<25} {r['seed']:>10} {r['goodput']:>10.1f} {r['ttft_p50']:>10.0f} "
              f"{r['tpot_p50']:>10.1f} {r['slo_viol']:>7.1f} {str(r['active']):>8}")
    if len(runs) > 1:
        gs = [r['goodput'] for r in runs]
        print(f"{baseline+' MEAN±STD':<25} {'':>10} {statistics.mean(gs):>10.1f} ± {statistics.stdev(gs):.1f} tok/s")
        print()
PYEOF
