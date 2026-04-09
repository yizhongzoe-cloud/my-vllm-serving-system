#!/usr/bin/env bash
# ============================================================================
# overnight_2026-04-09.sh — Sequential test runner for framework-overhead
# investigation. Generated 2026-04-09 night session.
#
# Why sequential (not parallel): /dev/shm/vllm_ft_checkpoints is hardcoded
# in vllm/v1/engine/ft_client.py:53, so two parallel ft_benders_centralized
# runs collide on os.replace() of checkpoint chunks. We saw this break
# runs on 2026-04-09 (active_requests_at_fault=0 but goodput≈0).
# ----------------------------------------------------------------------------
# How to run:
#   cd /home/jlpang/my-vllm-serving-system
#   nohup bash experiments_v2/overnight_2026-04-09.sh > /tmp/overnight.log 2>&1 &
#   disown
#
# To monitor:
#   tail -f /tmp/overnight.log
#   ls results_v2/8B_overnight_2026-04-09/
# ----------------------------------------------------------------------------
# Expected total runtime: ~50-60 minutes (7 experiments × ~7 min each).
# ============================================================================

set -u  # NOTE: no -e — we want to keep going past failures
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

# Hardware / env
export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Output base
OUT_BASE="results_v2/8B_overnight_2026-04-09"
mkdir -p "$OUT_BASE"

# Experiment cell (held constant across all runs for apples-to-apples)
WORKLOAD="W1_Chat"
LOAD="Heavy"
SEED=42
CONFIG="experiments_v2/config_8b.yaml"
PORT=8400

run_cell() {
    local tag="$1"
    local baseline="$2"
    local fault="$3"
    local extra_env="${4:-}"

    local out_dir="${OUT_BASE}/${tag}/${baseline}/${WORKLOAD}/${LOAD}/${fault}/${SEED}"
    mkdir -p "$out_dir"

    echo "================================================================="
    echo "[$(date +%T)] START: ${tag} (${baseline} / ${fault})"
    [ -n "$extra_env" ] && echo "  env: $extra_env"
    echo "================================================================="

    # Always wipe shared ckpt dir before each run to avoid stale state
    rm -rf /dev/shm/vllm_ft_checkpoints

    eval "$extra_env python experiments_v2/run.py \
        --config $CONFIG \
        --baseline $baseline \
        --workload $WORKLOAD \
        --load $LOAD \
        --fault $fault \
        --seed $SEED \
        --port $PORT \
        --output-dir $out_dir" \
        > "${out_dir}/stdout.log" 2>&1

    local rc=$?
    echo "[$(date +%T)] END:   ${tag} rc=$rc"
    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  goodput={m.get(\"goodput\", -1):.1f} t/s  '
      f'ttft_p50={m.get(\"ttft_p50_ms\", -1):.0f}ms  '
      f'tpot_p50={m.get(\"tpot_p50_ms\", -1):.1f}ms  '
      f'slo_viol={m.get(\"slo_violation_rate\", -1)*100:.1f}%  '
      f'completion={m.get(\"completion_rate\", -1)*100:.1f}%  '
      f'active@fault={m.get(\"active_requests_at_fault\", \"?\")}')
" 2>/dev/null || echo "  (metrics.json present but parse failed)"
    else
        echo "  (NO metrics.json — run failed; check stdout.log)"
    fi
    echo ""
}

echo "########################################################################"
echo "# Overnight framework-overhead investigation — 2026-04-09"
echo "# Started at $(date)"
echo "########################################################################"
echo ""

# ===========================================================================
# Experiment 1: Baseline reload (default Our-System) — establishes a fresh
# data point. Tells us the variance bounds for subsequent comparisons.
# Expected: goodput ~115-130, completion 100%, SLO violation ~65%.
# ===========================================================================
run_cell "01_reload_baseline" "Our-System" "F2_Mid"

# ===========================================================================
# Experiment 2: No-FT baseline — upper-bound goal.
# Expected: goodput ~310-330, completion 98-99%, SLO violation ~20%.
# ===========================================================================
run_cell "02_noft_baseline" "No-FT" "F2_Mid"

# ===========================================================================
# Experiment 3: Our-System-NoCkpt — ft_benders_centralized policy WITH
# checkpoint controller OFF. KEY EXPERIMENT: isolates the cost of
# CheckpointController + KV checkpoint pool from the rest of the framework.
#
# If goodput jumps significantly toward No-FT (e.g. 200+ tok/s) →
# CheckpointController is the dominant overhead → next attack target.
# If goodput barely changes (still ~120 tok/s) → overhead is somewhere
# else (ft_client output processing / scheduler wrapper / snapshot
# collection). Need to dig further.
# ===========================================================================
run_cell "03_nockpt" "Our-System-NoCkpt" "F2_Mid"

# ===========================================================================
# Experiment 4: cProfile dump on Our-System (with fault).
# FT_PROFILE_MAX_CALLS=300 captures the first 300 schedule() calls and
# dumps cumulative stats to FT_PROFILE_OUTPUT. Use the dump to find the
# actual hot functions in the FT scheduler path.
#
# After it finishes, look at:
#   results_v2/8B_overnight_2026-04-09/04_cprofile/.../ft_schedule_profile.txt
# Top 20 cumulative-time entries will tell us where the GIL time goes.
# ===========================================================================
run_cell "04_cprofile" "Our-System" "F2_Mid" \
    "FT_PROFILE_MAX_CALLS=300 FT_PROFILE_OUTPUT=${OUT_BASE}/04_cprofile/ft_schedule_profile.txt"

# ===========================================================================
# Experiment 5: Our-System-NoCkpt + cProfile.
# Compare the schedule() profile WITH and WITHOUT checkpoint controller.
# The diff between this and exp 4 isolates the checkpoint controller
# functions in the cProfile output.
# ===========================================================================
run_cell "05_nockpt_cprofile" "Our-System-NoCkpt" "F2_Mid" \
    "FT_PROFILE_MAX_CALLS=300 FT_PROFILE_OUTPUT=${OUT_BASE}/05_nockpt_cprofile/ft_schedule_profile.txt"

# ===========================================================================
# Experiment 6: Our-System with new A5000-rough-estimate decode capacity
# profile. Replaces the A6000 default (decode_capacity=10, planning=0.5s)
# with rough A5000 dp=2 estimates from today's measurements
# (decode_capacity=26, planning=1.0s). See:
#   experiments_v2/decode_capacity_profile_8b_a5000.json
#
# Expected: Benders convergence rate jumps from ~2% to ~60-80%, but
# goodput likely DROPS (we saw this in v4 today: 124→43 because
# Benders rejects ~70% of requests when its capacity model is honest).
# This run documents the trade-off curve. NOT a fix — just a data point.
# ===========================================================================
echo "[$(date +%T)] Swapping in A5000 profile for experiment 6..."
cp experiments_v2/decode_capacity_profile_8b.json /tmp/decode_capacity_profile_8b.json.overnight_backup
cp experiments_v2/decode_capacity_profile_8b_a5000.json experiments_v2/decode_capacity_profile_8b.json
run_cell "06_reload_a5000_profile" "Our-System" "F2_Mid"
echo "[$(date +%T)] Restoring original profile..."
cp /tmp/decode_capacity_profile_8b.json.overnight_backup experiments_v2/decode_capacity_profile_8b.json

# ===========================================================================
# Experiment 7: cProfile WITHOUT fault (fault=none). Isolates steady-state
# baseline overhead from the recovery path. The fault path adds noise to
# the cProfile aggregates because failover-related code runs once per fault.
# A fault=none run gives a cleaner profile of the per-step hot path.
# ===========================================================================
run_cell "07_cprofile_nofault" "Our-System" "none" \
    "FT_PROFILE_MAX_CALLS=300 FT_PROFILE_OUTPUT=${OUT_BASE}/07_cprofile_nofault/ft_schedule_profile.txt"

echo "########################################################################"
echo "# Overnight investigation finished at $(date)"
echo "# Results in: $OUT_BASE"
echo "########################################################################"

# Quick summary table
echo ""
echo "=== Summary table ==="
python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results_v2/8B_overnight_2026-04-09")
rows = []
for tag_dir in sorted(base.iterdir()):
    if not tag_dir.is_dir():
        continue
    metrics_files = list(tag_dir.glob("**/metrics.json"))
    for mf in metrics_files:
        try:
            m = json.load(open(mf))
            rows.append({
                "tag": tag_dir.name,
                "goodput": m.get("goodput", -1),
                "ttft_p50": m.get("ttft_p50_ms", -1),
                "tpot_p50": m.get("tpot_p50_ms", -1),
                "slo_viol": m.get("slo_violation_rate", -1) * 100,
                "completion": m.get("completion_rate", -1) * 100,
                "active_at_fault": m.get("active_requests_at_fault", "?"),
            })
        except Exception as e:
            print(f"  {tag_dir.name}: parse failed: {e}")

if rows:
    print(f"{'tag':<32} {'goodput':>10} {'ttft_p50':>10} {'tpot_p50':>10} {'slo%':>7} {'comp%':>7} {'act@flt':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['tag']:<32} {r['goodput']:>10.1f} {r['ttft_p50']:>10.0f} "
              f"{r['tpot_p50']:>10.1f} {r['slo_viol']:>7.1f} {r['completion']:>7.1f} "
              f"{str(r['active_at_fault']):>8}")
PYEOF
