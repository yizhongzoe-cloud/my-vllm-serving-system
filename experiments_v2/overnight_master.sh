#!/usr/bin/env bash
# ============================================================================
# overnight_master.sh — End-to-end overnight orchestrator (2026-04-09).
#
# Runs Phases 2, 3, 4 sequentially after Phase 1 completes. Phase 1 (the
# 7-experiment investigation suite at experiments_v2/overnight_2026-04-09.sh)
# is launched separately and this script waits for its summary table to
# appear before proceeding.
#
# Total expected runtime: ~6-7 hours (well within the 8h budget).
#
# DESIGN CONSTRAINTS:
#   - Sequential only (/dev/shm/vllm_ft_checkpoints collision risk)
#   - GPU 4-5 only (the rest are in use by other workloads)
#   - All optimizations are env-var-gated default-off, no behavior change
#     unless explicitly toggled
#   - Every code change is committed BEFORE running validation
#   - Every result is dumped to results_v2/8B_overnight_2026-04-09/
#
# How to monitor in the morning:
#   tail -50 /tmp/overnight.log
#   tail -50 /tmp/overnight_master.log
#   cat results_v2/8B_overnight_2026-04-09/SUMMARY.md
# ============================================================================

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_BASE="results_v2/8B_overnight_2026-04-09"
mkdir -p "$OUT_BASE"

WORKLOAD="W1_Chat"
LOAD="Heavy"
FAULT="F2_Mid"
CONFIG="experiments_v2/config_8b.yaml"
PORT=8400

run_cell() {
    local tag="$1"
    local baseline="$2"
    local fault="$3"
    local seed="$4"
    local extra_env="${5:-}"

    local out_dir="${OUT_BASE}/${tag}/${baseline}/${WORKLOAD}/${LOAD}/${fault}/${seed}"
    mkdir -p "$out_dir"

    echo "=================================================================" >&2
    echo "[$(date +%T)] START: ${tag} (${baseline}/${fault} seed=${seed})" >&2
    [ -n "$extra_env" ] && echo "  env: $extra_env" >&2
    echo "=================================================================" >&2

    rm -rf /dev/shm/vllm_ft_checkpoints

    eval "$extra_env python experiments_v2/run.py \
        --config $CONFIG \
        --baseline $baseline \
        --workload $WORKLOAD \
        --load $LOAD \
        --fault $fault \
        --seed $seed \
        --port $PORT \
        --output-dir $out_dir" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END:   ${tag} rc=$rc" >&2

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
" 2>&1 || echo "  (metrics parse failed)"
    else
        echo "  (NO metrics.json — run failed)"
    fi
    echo "" >&2
}

# ===========================================================================
# Wait for Phase 1 to complete
# ===========================================================================
echo "########################################################################" >&2
echo "# Master orchestrator started at $(date)" >&2
echo "# Waiting for Phase 1 (overnight_2026-04-09.sh) to finish..." >&2
echo "########################################################################" >&2

# Phase 1 finishes when 07_cprofile_nofault has metrics.json
PHASE1_DONE_FILE="${OUT_BASE}/07_cprofile_nofault/Our-System/W1_Chat/Heavy/none/42/metrics.json"
WAIT_START=$(date +%s)
while [ ! -f "$PHASE1_DONE_FILE" ]; do
    sleep 30
    elapsed=$(( $(date +%s) - WAIT_START ))
    if [ $elapsed -gt 5400 ]; then  # 90 min hard cap
        echo "[$(date +%T)] Phase 1 not done after 90 min — proceeding anyway" >&2
        break
    fi
done
echo "[$(date +%T)] Phase 1 finished (or timeout). Sleeping 10s for safety..." >&2
sleep 10

# Make sure Phase 1's child processes are gone
while pgrep -f "overnight_2026-04-09.sh" > /dev/null; do
    sleep 5
done
while pgrep -f "experiments_v2/run.py" > /dev/null; do
    sleep 5
done
sleep 5

echo "########################################################################" >&2
echo "# Phase 2: Multi-seed variance baselines (3 seeds × 2 baselines)" >&2
echo "########################################################################" >&2

# ===========================================================================
# Phase 2: Multi-seed variance — establishes the noise floor.
# 3 seeds × {Our-System reload, Our-System-NoCkpt} = 6 runs ~ 45 min
# ===========================================================================
for seed in 42 123 456; do
    run_cell "phase2_variance/reload_s${seed}" "Our-System"        "$FAULT" "$seed"
    run_cell "phase2_variance/nockpt_s${seed}" "Our-System-NoCkpt" "$FAULT" "$seed"
done

echo "########################################################################" >&2
echo "# Phase 3: Multi-seed cProfile + steady-state (no fault)" >&2
echo "########################################################################" >&2

# ===========================================================================
# Phase 3: Steady-state cProfile dumps for diff analysis.
# 3 seeds × {Our-System no-fault, Our-System-NoCkpt no-fault} cProfile dumps
# ===========================================================================
for seed in 42 123 456; do
    run_cell "phase3_steady/reload_s${seed}" "Our-System" "none" "$seed" \
        "FT_PROFILE_MAX_CALLS=300 FT_PROFILE_OUTPUT=${OUT_BASE}/phase3_steady/reload_s${seed}/ft_profile.txt"
    run_cell "phase3_steady/nockpt_s${seed}" "Our-System-NoCkpt" "none" "$seed" \
        "FT_PROFILE_MAX_CALLS=300 FT_PROFILE_OUTPUT=${OUT_BASE}/phase3_steady/nockpt_s${seed}/ft_profile.txt"
done

echo "########################################################################" >&2
echo "# Phase 4: Snapshot-bypass ablation (FT_DISABLE_SNAPSHOTS=1)" >&2
echo "########################################################################" >&2

# ===========================================================================
# Phase 4: Validate the FT_DISABLE_SNAPSHOTS optimization (committed in
# 8a7a98c59 before this script was launched).
# 3 seeds × Our-System with bypass enabled. Compare to phase2 reload runs.
# ===========================================================================
for seed in 42 123 456; do
    run_cell "phase4_no_snapshots/reload_s${seed}" "Our-System" "$FAULT" "$seed" \
        "FT_DISABLE_SNAPSHOTS=1"
done

# Bonus: 3 seeds with FT_DISABLE_SNAPSHOTS=1 + Our-System-NoCkpt
# (combined: skip ckpt + skip snapshots)
for seed in 42 123 456; do
    run_cell "phase4_combined/nockpt_nosnap_s${seed}" "Our-System-NoCkpt" "$FAULT" "$seed" \
        "FT_DISABLE_SNAPSHOTS=1"
done

echo "########################################################################" >&2
echo "# Phase 5: Final summary + doc update" >&2
echo "########################################################################" >&2

# ===========================================================================
# Phase 5: Generate SUMMARY.md with all results in a single table.
# ===========================================================================
python3 << 'PYEOF' > "${OUT_BASE}/SUMMARY.md"
import json
import statistics
from pathlib import Path
from datetime import datetime

base = Path("results_v2/8B_overnight_2026-04-09")

def collect(pattern):
    rows = []
    for mf in sorted(base.glob(pattern)):
        try:
            m = json.load(open(mf))
            rows.append({
                "path": str(mf.relative_to(base)),
                "goodput": m.get("goodput", -1),
                "ttft_p50": m.get("ttft_p50_ms", -1),
                "ttft_p95": m.get("ttft_p95_ms", -1),
                "tpot_p50": m.get("tpot_p50_ms", -1),
                "tpot_p95": m.get("tpot_p95_ms", -1),
                "slo_viol": m.get("slo_violation_rate", -1) * 100,
                "completion": m.get("completion_rate", -1) * 100,
                "active_at_fault": m.get("active_requests_at_fault", "?"),
                "failover_p50": m.get("failover_gap_p50_ms", 0),
            })
        except Exception as e:
            rows.append({"path": str(mf), "error": str(e)})
    return rows

def md_table(rows, title):
    print(f"## {title}\n")
    print(f"_{len(rows)} rows_\n")
    if not rows or "error" in rows[0]:
        print("(no data)\n")
        return
    print("| run | goodput | ttft_p50 | ttft_p95 | tpot_p50 | tpot_p95 | slo% | comp% | act@flt | failover_p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['path']} | ERROR: {r['error']} |")
            continue
        print(f"| `{r['path']}` | "
              f"{r['goodput']:.1f} | "
              f"{r['ttft_p50']:.0f} | "
              f"{r['ttft_p95']:.0f} | "
              f"{r['tpot_p50']:.1f} | "
              f"{r['tpot_p95']:.1f} | "
              f"{r['slo_viol']:.1f} | "
              f"{r['completion']:.1f} | "
              f"{r['active_at_fault']} | "
              f"{r['failover_p50']:.0f} |")
    if len(rows) > 1:
        good = [r['goodput'] for r in rows if 'goodput' in r and r['goodput'] >= 0]
        if good:
            print(f"\n**goodput mean ± stdev**: {statistics.mean(good):.1f} ± "
                  f"{statistics.stdev(good) if len(good) > 1 else 0:.1f} tok/s "
                  f"(min {min(good):.1f}, max {max(good):.1f})")
    print()

print(f"# Overnight 2026-04-09 — Final Summary\n")
print(f"_Generated at {datetime.now().isoformat(timespec='seconds')}_\n")
print("Investigation goal: localize Our-System framework baseline overhead "
      "(the ~200 tok/s gap to No-FT on W1_Chat/Heavy that today's drop-mode "
      "test showed is NOT from recovery).\n")
print("---\n")

md_table(collect("01_reload_baseline/**/metrics.json"), "Phase 1 — Baseline reload (1 run)")
md_table(collect("02_noft_baseline/**/metrics.json"), "Phase 1 — Baseline No-FT (1 run, upper bound)")
md_table(collect("03_nockpt/**/metrics.json"), "Phase 1 — Our-System-NoCkpt (1 run, ckpt-off ablation)")
md_table(collect("04_cprofile/**/metrics.json"), "Phase 1 — cProfile WITH ckpt + fault")
md_table(collect("05_nockpt_cprofile/**/metrics.json"), "Phase 1 — cProfile WITHOUT ckpt + fault")
md_table(collect("06_reload_a5000_profile/**/metrics.json"), "Phase 1 — A5000 profile swap")
md_table(collect("07_cprofile_nofault/**/metrics.json"), "Phase 1 — cProfile no fault (steady state)")

md_table(collect("phase2_variance/reload_s*/**/metrics.json"), "Phase 2 — Variance: Our-System reload (3 seeds)")
md_table(collect("phase2_variance/nockpt_s*/**/metrics.json"), "Phase 2 — Variance: Our-System-NoCkpt (3 seeds)")

md_table(collect("phase3_steady/reload_s*/**/metrics.json"), "Phase 3 — Steady state: Our-System (no fault, 3 seeds)")
md_table(collect("phase3_steady/nockpt_s*/**/metrics.json"), "Phase 3 — Steady state: Our-System-NoCkpt (no fault, 3 seeds)")

md_table(collect("phase4_no_snapshots/reload_s*/**/metrics.json"), "Phase 4 — Our-System + FT_DISABLE_SNAPSHOTS=1 (3 seeds)")
md_table(collect("phase4_combined/nockpt_nosnap_s*/**/metrics.json"), "Phase 4 — Our-System-NoCkpt + FT_DISABLE_SNAPSHOTS=1 (3 seeds)")

print("\n## cProfile dump locations\n")
for prof in sorted(base.glob("**/ft_profile*.txt")) + sorted(base.glob("**/ft_schedule_profile.txt")):
    print(f"- `{prof.relative_to(base)}`")
PYEOF

echo "[$(date +%T)] SUMMARY.md generated at ${OUT_BASE}/SUMMARY.md" >&2
cat "${OUT_BASE}/SUMMARY.md"

echo "########################################################################" >&2
echo "# Master orchestrator finished at $(date)" >&2
echo "########################################################################" >&2
