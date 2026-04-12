#!/usr/bin/env bash
# A/B test for FT_ASYNC_RESTORE=1 (pipelined KV restore) vs OFF (default).
# Same cell, same baseline, 3 seeds each side, sequential on GPU 4-5.
#
# Cell: W1_Chat/Heavy/F2_Mid (the cell where Our-System hurts the most)
# Baseline: Our-System (the one that touches the restore path)
# Profile: A5000 default=26 (standard config_8b.yaml, not phase8)
#
# Total: 2 variants × 3 seeds = 6 runs sequential ≈ 45 min
#
# Output: results_v2/8B/ft_async_restore_ab/{off,on}/<seed>/

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1

OUT_BASE="results_v2/8B/ft_async_restore_ab"
CONFIG="experiments_v2/config_8b.yaml"
SEEDS=(42 123 456)

run_one() {
    local variant="$1"   # off | on
    local seed="$2"
    local out_dir="${OUT_BASE}/${variant}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${variant}/s${seed}" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local extra_env=""
    if [ "$variant" = "on" ]; then
        extra_env="FT_ASYNC_RESTORE=1"
    fi

    echo "[$(date +%H:%M:%S)] START ${variant}/s${seed} (${extra_env:-no extra env})" >&2

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
echo "# FT_ASYNC_RESTORE A/B test — $(date)"
echo "# Cell: W1_Chat/Heavy/F2_Mid, baseline: Our-System, A5000 profile"
echo "########################################################################"

# Interleave off/on by seed to reduce systematic GPU warm-up bias
for seed in "${SEEDS[@]}"; do
    run_one "off" "$seed"
    run_one "on"  "$seed"
done

echo ""
echo "########################################################################"
echo "# A/B test finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics
base = "results_v2/8B/ft_async_restore_ab"
print("\n=== A/B summary (3 seeds each) ===\n")
for variant in ["off", "on"]:
    vals = []
    for f in sorted(glob.glob(f"{base}/{variant}/*/metrics.json")):
        try:
            m = json.load(open(f))
            seed = f.split("/")[-2]
            g = m.get("goodput", -1)
            c = m.get("completion_rate", -1) * 100
            t = m.get("ttft_p50_ms", -1)
            s = m.get("slo_violation_rate", -1) * 100
            print(f"  {variant}/s{seed}: goodput={g:.1f}  comp={c:.1f}%  ttft_p50={t:.0f}  slo={s:.1f}%")
            vals.append(g)
        except: pass
    if len(vals) >= 2:
        m_ = statistics.mean(vals)
        s_ = statistics.stdev(vals)
        print(f"  MEAN ± STD: {m_:.1f} ± {s_:.1f}  (n={len(vals)})")
    print()
PYEOF
