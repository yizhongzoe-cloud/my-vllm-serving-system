#!/usr/bin/env bash
# B1 A/B test: original cost model parameters vs A5000-calibrated.
#
# Variant "orig":  config_8b.yaml (prefill=4000, decode=2000, load_bw=10e9)
# Variant "cal":   config_8b_a5000_calibrated.yaml (prefill=2000, decode=400, load_bw=0.5e9)
#
# Calibration source:
#   - prefill_throughput 2000 from E1a_3seed Our-System Heavy p95=2268 per engine
#   - decode_throughput 400 from E1a_3seed Our-System Heavy p95=479 per engine
#   - load_bandwidth 0.5e9 from kv_restore_done event intervals (median ~0.4 GB/s)
#     — 25× less than the original config value of 10e9
#
# Cell: W1_Chat/Heavy/F2_Mid (hard cell where cost model matters)
# Baseline: Our-System (the one that uses the cost model)
# Also add a no-fault cell (W1_Chat/Moderate/none) to measure no-fault overhead impact
#
# GPU 6-7 (port 8500), sequential to avoid race, complements GPU 4-5 running A2+A3 AB.
#
# Also enables FT_ASYNC_RESTORE=1 (the committed improvement) so we compare apples
# to apples with the current best baseline.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1

OUT_BASE="results_v2/8B/b1_cost_ab"
SEEDS=(42 123 456)

run_one() {
    local variant="$1"    # orig | cal
    local cell="$2"       # e.g. "Heavy/F2_Mid"
    local seed="$3"

    local load="${cell%/*}"
    local fault="${cell#*/}"
    local out_dir="${OUT_BASE}/${variant}/${load}_${fault}/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${variant}/${cell}/s${seed}" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    local config
    if [ "$variant" = "orig" ]; then
        config="experiments_v2/config_8b.yaml"
    else
        config="experiments_v2/config_8b_a5000_calibrated.yaml"
    fi

    echo "[$(date +%H:%M:%S)] START ${variant}/${cell}/s${seed} (config=${config##*/})" >&2

    CUDA_VISIBLE_DEVICES=6,7 python experiments_v2/run.py \
        --config "$config" \
        --baseline Our-System \
        --workload W1_Chat \
        --load "$load" \
        --fault "$fault" \
        --seed "$seed" \
        --port 8500 \
        --output-dir "$out_dir" \
        > "${out_dir}/stdout.log" 2>&1
    local rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${variant}/${cell}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  ttft_p50={m.get(\"ttft_p50_ms\",-1):.0f}  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  ${variant}/${cell}/s${seed}: FAILED (rc=$rc)" >&2
    fi
}

echo "########################################################################"
echo "# B1 cost-model recalibration A/B — $(date)"
echo "# orig: config_8b.yaml               (A6000 dp=1 values: prefill=4000, decode=2000, load_bw=10e9)"
echo "# cal:  config_8b_a5000_calibrated.yaml (A5000 dp=2 calibrated: prefill=2000, decode=400, load_bw=0.5e9)"
echo "# Both variants enable FT_ASYNC_RESTORE=1"
echo "########################################################################"

# Two cells × 2 variants × 3 seeds = 12 runs
# - Heavy/F2_Mid: hardest, biggest cost model impact
# - Moderate/none: measures no-fault framework overhead
# Interleave to reduce warmup bias
for seed in "${SEEDS[@]}"; do
    for cell in "Heavy/F2_Mid" "Moderate/none"; do
        run_one "orig" "$cell" "$seed"
        run_one "cal"  "$cell" "$seed"
    done
done

echo ""
echo "########################################################################"
echo "# Finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics
base = "results_v2/8B/b1_cost_ab"
print("\n=== B1 A/B summary (3 seeds each cell) ===\n")

def summarize(variant, cell):
    load, fault = cell.split("/")
    key = f"{load}_{fault}"
    vals = []
    for f in sorted(glob.glob(f"{base}/{variant}/{key}/*/metrics.json")):
        try:
            m = json.load(open(f))
            vals.append((
                int(f.split("/")[-2]),
                m.get("goodput", -1),
                m.get("completion_rate", -1) * 100,
                m.get("ttft_p50_ms", -1),
                m.get("slo_violation_rate", -1) * 100,
            ))
        except: pass
    return vals

for cell in ["Heavy/F2_Mid", "Moderate/none"]:
    print(f"--- {cell} ---")
    for variant in ["orig", "cal"]:
        vals = summarize(variant, cell)
        for seed, g, c, t, s in vals:
            print(f"  {variant}/s{seed}: goodput={g:.1f}  comp={c:.1f}%  ttft_p50={t:.0f}  slo={s:.1f}%")
        if len(vals) >= 2:
            gs = [v[1] for v in vals]
            m_ = statistics.mean(gs); s_ = statistics.stdev(gs)
            print(f"  MEAN ± STD: {m_:.1f} ± {s_:.1f}")
        print()
PYEOF
