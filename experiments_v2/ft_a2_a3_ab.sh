#!/usr/bin/env bash
# A/B for Phase 1 improvements A2 + A3, layered on top of C1 (FT_ASYNC_RESTORE).
#
# Variant "base":  FT_ASYNC_RESTORE=1  (already committed, +22% validated)
# Variant "both":  FT_ASYNC_RESTORE=1 + FT_PLANNING_HORIZON_SEC=0.3 + FT_GATED_SOLVER=1
#
# Cell: W1_Chat/Heavy/F2_Mid (Our-System's hardest cell)
# Baseline: Our-System
# Profile: A5000 default=26 (config_8b.yaml)
# Seeds: 42 / 123 / 456 (same as FT_ASYNC_RESTORE A/B for direct comparison)
#
# Total: 2 variants × 3 seeds = 6 runs sequential on GPU 4-5, ~45 min
#
# Reference for base variant: results_v2/8B/ft_async_restore_ab/on/{42,123,456}
# (already have FT_ASYNC_RESTORE=1 only data; "base" rerun here for clean
# pair comparison in case of env drift).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1

OUT_BASE="results_v2/8B/ft_a2_a3_ab"
CONFIG="experiments_v2/config_8b.yaml"
SEEDS=(42 123 456)

run_one() {
    local variant="$1"   # base | both
    local seed="$2"
    local out_dir="${OUT_BASE}/${variant}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${variant}/s${seed}" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local extra_env=""
    if [ "$variant" = "both" ]; then
        extra_env="FT_PLANNING_HORIZON_SEC=0.3 FT_GATED_SOLVER=1"
    fi

    echo "[$(date +%H:%M:%S)] START ${variant}/s${seed} (${extra_env:-C1 only})" >&2

    eval "CUDA_VISIBLE_DEVICES=4,5 ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline Our-System \
        --workload W1_Chat \
        --load Heavy \
        --fault F2_Mid \
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
echo "# A2 + A3 A/B test (on top of C1 FT_ASYNC_RESTORE) — $(date)"
echo "# Cell: W1_Chat/Heavy/F2_Mid, baseline: Our-System, A5000 profile"
echo "# base = FT_ASYNC_RESTORE=1 only"
echo "# both = base + FT_PLANNING_HORIZON_SEC=0.3 + FT_GATED_SOLVER=1"
echo "########################################################################"

# Interleave base/both per seed to reduce warm-up bias
for seed in "${SEEDS[@]}"; do
    run_one "base" "$seed"
    run_one "both" "$seed"
done

echo ""
echo "########################################################################"
echo "# Finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics
base = "results_v2/8B/ft_a2_a3_ab"
print("\n=== A/B summary (3 seeds each) ===\n")
stats = {}
for variant in ["base", "both"]:
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
        t_ = statistics.mean(ttfts)
        sl_ = statistics.mean(slos)
        stats[variant] = (m_, s_, t_, sl_)
        print(f"  MEAN ± STD: {m_:.1f} ± {s_:.1f}  ttft_p50 {t_:.0f}  slo {sl_:.1f}%  (n={len(vals)})")
    print()

if "base" in stats and "both" in stats:
    delta = stats["both"][0] - stats["base"][0]
    delta_pct = delta / stats["base"][0] * 100
    print(f"=== A2+A3 delta on top of C1: {delta:+.1f} tok/s ({delta_pct:+.1f}%) ===")
PYEOF
