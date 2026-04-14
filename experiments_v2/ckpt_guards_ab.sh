#!/usr/bin/env bash
# A/B test for 3 adaptive checkpoint guards (fix the per-block save degeneration).
#
# 5 variants × 3 seeds × 1 cell = 15 runs sequential, ~105 min.
# Cell: W1_Chat/Moderate/none — no-fault cell where checkpoint overhead is the
#   dominant cost (~1-3% goodput), most likely to show guard benefit.
#
# All variants also enable: FT_ASYNC_RESTORE=1 + FT_GATED_SOLVER=1 (best Phase 1).
#
# Variants:
#   base  : Phase 1 best (C1 + A3), no checkpoint guard
#   guard_c : + FT_CKPT_MIN_INTERVAL_BLOCKS=4 (save every 64 tokens, not 16)
#   guard_b : + FT_CKPT_LOAD_GUARD=0.7 (skip save when batch > 70% of cap=26)
#   guard_a : + FT_CKPT_SLO_GUARD=0.2 (skip save when step_time > 80% of tpot_slo)
#   all     : all 3 guards active
#
# Also run on Heavy/F2_Mid to measure recovery quality impact (bigger uncovered suffix).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1
export FT_GATED_SOLVER=1

OUT_BASE="results_v2/8B/ckpt_guards_ab"
CONFIG="experiments_v2/config_8b.yaml"
SEEDS=(42 123 456)

run_one() {
    local variant="$1"
    local load="$2"
    local fault="$3"
    local seed="$4"
    local out_dir="${OUT_BASE}/${variant}/${load}_${fault}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${variant}/${load}/${fault}/s${seed}" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local extra_env=""
    case "$variant" in
        base)    extra_env="" ;;
        guard_c) extra_env="FT_CKPT_MIN_INTERVAL_BLOCKS=4" ;;
        guard_b) extra_env="FT_CKPT_LOAD_GUARD=0.7" ;;
        guard_a) extra_env="FT_CKPT_SLO_GUARD=0.2" ;;
        all)     extra_env="FT_CKPT_MIN_INTERVAL_BLOCKS=4 FT_CKPT_LOAD_GUARD=0.7 FT_CKPT_SLO_GUARD=0.2" ;;
    esac

    echo "[$(date +%H:%M:%S)] START ${variant}/${load}/${fault}/s${seed} (${extra_env:-no guard})" >&2

    eval "CUDA_VISIBLE_DEVICES=6,7 ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline Our-System \
        --workload W1_Chat \
        --load ${load} \
        --fault ${fault} \
        --seed ${seed} \
        --port 8500 \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${variant}/${load}/${fault}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  FAILED" >&2
    fi
}

echo "########################################################################"
echo "# Checkpoint guards A/B test — $(date)"
echo "# 5 variants: base | guard_c(min_blocks=4) | guard_b(load>0.7) | guard_a(headroom<0.2) | all"
echo "# Cells: Moderate/none (no-fault overhead) + Heavy/F2_Mid (recovery impact)"
echo "########################################################################"

# Moderate/none — 5 variants × 3 seeds = 15 runs
for seed in "${SEEDS[@]}"; do
    for v in base guard_c guard_b guard_a all; do
        run_one "$v" "Moderate" "none" "$seed"
    done
done

# Heavy/F2_Mid — 5 variants × 3 seeds = 15 runs
for seed in "${SEEDS[@]}"; do
    for v in base guard_c guard_b guard_a all; do
        run_one "$v" "Heavy" "F2_Mid" "$seed"
    done
done

echo ""
echo "########################################################################"
echo "# Finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics

base_dir = "results_v2/8B/ckpt_guards_ab"
variants = ["base", "guard_c", "guard_b", "guard_a", "all"]

print("\n=== Checkpoint guards A/B summary ===\n")
for cell in ["Moderate_none", "Heavy_F2_Mid"]:
    print(f"--- W1_Chat/{cell.replace('_','/')} ---")
    print(f"{'variant':<12} {'mean±std':>16} {'slo%':>8}")
    print("-" * 40)
    cell_stats = {}
    for v in variants:
        vals = []
        slos = []
        for f in sorted(glob.glob(f"{base_dir}/{v}/{cell}/*/metrics.json")):
            try:
                m = json.load(open(f))
                vals.append(m.get("goodput", -1))
                slos.append(m.get("slo_violation_rate", -1) * 100)
            except: pass
        if len(vals) >= 2:
            m_ = statistics.mean(vals); s_ = statistics.stdev(vals)
            sl_ = statistics.mean(slos)
            cell_stats[v] = m_
            print(f"{v:<12} {m_:>7.1f}±{s_:>4.0f}    {sl_:>6.1f}%")
    if "base" in cell_stats:
        print()
        for v in variants:
            if v != "base" and v in cell_stats:
                d = cell_stats[v] - cell_stats["base"]
                print(f"  {v:<12} Δ = {d:+.1f} tok/s ({d/cell_stats['base']*100:+.1f}%)")
    print()
PYEOF
