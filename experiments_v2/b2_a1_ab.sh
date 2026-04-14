#!/usr/bin/env bash
# Phase 2 B2 + A1 A/B test.
#
# 4 variants × 3 seeds × 1 cell = 12 runs sequential.
# Cell: W1_Chat/Heavy/F2_Mid (Our-System's hardest, where improvements matter most).
# Baseline: Our-System on A5000 profile (config_8b.yaml), FT_ASYNC_RESTORE=1.
#
# Variants (all layered on top of C1 FT_ASYNC_RESTORE):
#   base   : C1 only (matches current committed state)
#   b2     : C1 + FT_BATCH_CKPT_EVAL=1 (pre-filter checkpoint eval loop)
#   a1     : C1 + FT_SLO_AWARE_OBJECTIVE=1 (SLO-aware solver objective)
#   both   : C1 + B2 + A1
#
# Important: NO FT_GATED_SOLVER here — A1 requires the solver to actually run.
# Benders solver will converge 100% on A5000 profile (default=26).
#
# Total: 12 × ~7 min ≈ 85 min sequential on GPU 4-5 port 8400.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1

OUT_BASE="results_v2/8B/b2_a1_ab"
CONFIG="experiments_v2/config_8b.yaml"
SEEDS=(42 123 456)
CELL_LOAD="Heavy"
CELL_FAULT="F2_Mid"

run_one() {
    local variant="$1"   # base | b2 | a1 | both
    local seed="$2"
    local out_dir="${OUT_BASE}/${variant}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${variant}/s${seed}" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local extra_env=""
    case "$variant" in
        base) extra_env="" ;;
        b2)   extra_env="FT_BATCH_CKPT_EVAL=1" ;;
        a1)   extra_env="FT_SLO_AWARE_OBJECTIVE=1" ;;
        both) extra_env="FT_BATCH_CKPT_EVAL=1 FT_SLO_AWARE_OBJECTIVE=1" ;;
    esac

    echo "[$(date +%H:%M:%S)] START ${variant}/s${seed} (${extra_env:-C1 only})" >&2

    eval "CUDA_VISIBLE_DEVICES=4,5 ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline Our-System \
        --workload W1_Chat \
        --load ${CELL_LOAD} \
        --fault ${CELL_FAULT} \
        --seed ${seed} \
        --port 8400 \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${variant}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  ttft_p50={m.get(\"ttft_p50_ms\",-1):.0f}  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  ${variant}/s${seed}: FAILED (rc=$rc)" >&2
    fi
}

echo "########################################################################"
echo "# B2 + A1 A/B test (on top of C1 FT_ASYNC_RESTORE) — $(date)"
echo "# Cell: W1_Chat/${CELL_LOAD}/${CELL_FAULT}, baseline: Our-System"
echo "# 4 variants: base | b2 | a1 | both"
echo "########################################################################"

# Interleave: each seed runs all 4 variants before moving to next seed
# (reduces cross-seed warmup bias; within a seed, try all variants).
for seed in "${SEEDS[@]}"; do
    run_one "base" "$seed"
    run_one "b2"   "$seed"
    run_one "a1"   "$seed"
    run_one "both" "$seed"
done

echo ""
echo "########################################################################"
echo "# Finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics

base = "results_v2/8B/b2_a1_ab"
print("\n=== B2+A1 A/B summary (3 seeds each) ===\n")

stats = {}
for variant in ["base", "b2", "a1", "both"]:
    vals = []
    ttfts = []
    slos = []
    for f in sorted(glob.glob(f"{base}/{variant}/*/metrics.json")):
        try:
            m = json.load(open(f))
            seed = f.split("/")[-2]
            g = m.get("goodput", -1)
            c = m.get("completion_rate", -1) * 100
            t = m.get("ttft_p50_ms", -1)
            s = m.get("slo_violation_rate", -1) * 100
            print(f"  {variant}/s{seed}: goodput={g:.1f}  comp={c:.1f}%  ttft_p50={t:.0f}  slo={s:.1f}%")
            vals.append(g); ttfts.append(t); slos.append(s)
        except: pass
    if len(vals) >= 2:
        m_ = statistics.mean(vals); s_ = statistics.stdev(vals)
        t_ = statistics.mean(ttfts); sl_ = statistics.mean(slos)
        stats[variant] = (m_, s_, t_, sl_)
        print(f"  MEAN ± STD: {m_:.1f} ± {s_:.1f}  ttft_p50 {t_:.0f}  slo {sl_:.1f}%  (n={len(vals)})")
    print()

print("=" * 60)
print("Delta vs base (goodput)")
print("=" * 60)
if "base" in stats:
    b = stats["base"][0]
    for v in ["b2", "a1", "both"]:
        if v in stats:
            d = stats[v][0] - b
            print(f"  {v:<8} {d:+.1f} tok/s ({d/b*100:+.1f}%)")
PYEOF
