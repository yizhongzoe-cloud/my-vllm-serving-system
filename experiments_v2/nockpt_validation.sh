#!/usr/bin/env bash
# V2-NoCkpt validation: extend 3-seed finding to 6 seeds + cross-workload.
# Hypothesis: solver admission control alone beats NR. Checkpoint is dead weight.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/nockpt_2026-04-21"
mkdir -p "$ROOT"

pkill -9 -f "api_server|EngineCore" 2>/dev/null
sleep 3
rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null

V2_SOLVER='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_SOLVER_TIME_CAP_MS=100 FT_SOLVER_GREEDY_SEED=1 FT_RECOVERY_MODE=reprefill'
NR_ENV="FT_RECOVERY_MODE=reprefill"

run_cell() {
    local tag="$1" baseline="$2" wl="$3" load="$4" seed="$5" env_str="$6"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    pkill -9 -f "api_server|EngineCore" 2>/dev/null
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    sleep 3
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} ${baseline} ${wl} ${load}"
    eval "CUDA_VISIBLE_DEVICES=0,1 ${env_str} timeout 600 python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault F2_Mid \
        --seed ${seed} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
"
    fi
}

echo "######## V2-NoCkpt validation $(date) ########"

# Phase A: W7 Heavy, 3 new seeds (we already have 42/123/456 from overnight_0421/P4/V2_NoCkpt)
for s in 789 1234 22222; do
    run_cell "W7_NoCkpt_Heavy"    "Our-System-NoCkpt" "W7_Saturated" "Heavy" "$s" "$V2_SOLVER" || true
done

# Phase B: W5 Moderate, 6 seeds (no prior data)
for s in 42 123 456 789 1234 22222; do
    run_cell "W5_NoCkpt_Moderate" "Our-System-NoCkpt" "W5_LongDoc" "Moderate" "$s" "$V2_SOLVER" || true
done

# Phase C: W7 Saturated (RPS=2.5), 3 seeds — same as P3 W7 but with NoCkpt baseline
for s in 42 123 456; do
    run_cell "W7_NoCkpt_Saturated" "Our-System-NoCkpt" "W7_Saturated" "Saturated" "$s" "$V2_SOLVER" || true
done

echo ""
echo "######## FINAL NoCkpt PAIRED ANALYSIS $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st

def read_dir(path):
    out = {}
    if not os.path.isdir(path): return out
    for seed in sorted(os.listdir(path)):
        p = os.path.join(path, seed, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                out[seed] = (m['goodput'], m.get('failover_gap_p95_ms', 0))
    return out

def paired(v2_dirs, nr_dirs, label):
    v2 = {}
    for d in v2_dirs: v2.update(read_dir(d))
    nr = {}
    for d in nr_dirs: nr.update(read_dir(d))
    common = sorted(set(v2.keys()) & set(nr.keys()))
    if not common:
        print(f"\n=== {label} === no paired data")
        return 0, 0
    wins = 0
    print(f"\n=== {label} (n={len(common)}) ===")
    for s in common:
        vg, vf = v2[s]; ng, nf = nr[s]
        gp_w = '✓' if vg >= ng else '✗'
        fg_w = '✓' if vf <= nf else '✗'
        strict = ' 🎯' if (vg >= ng and vf <= nf) else ''
        print(f"  s{s}: V2 {vg:.1f}/{vf:.0f} vs NR {ng:.1f}/{nf:.0f} [{gp_w}][{fg_w}]{strict}")
        if gp_w == '✓' and fg_w == '✓': wins += 1
    v2_gps = [v2[s][0] for s in common]
    v2_fgs = [v2[s][1] for s in common]
    nr_gps = [nr[s][0] for s in common]
    nr_fgs = [nr[s][1] for s in common]
    print(f"  strict wins: {wins}/{len(common)}")
    print(f"  V2 mean: gp={st.mean(v2_gps):.1f}  fg_p95={st.mean(v2_fgs):.0f}")
    print(f"  NR mean: gp={st.mean(nr_gps):.1f}  fg_p95={st.mean(nr_fgs):.0f}")
    return wins, len(common)

OV = "results_v2/8B/overnight_2026-04-21"
NC = "results_v2/8B/nockpt_2026-04-21"

# W7 Heavy: merge P4/V2_NoCkpt (42/123/456) + new runs (789/1234/22222) vs P1/W7_NR (6 seeds)
paired(
    [f"{OV}/P4/V2_NoCkpt", f"{NC}/W7_NoCkpt_Heavy"],
    [f"{OV}/P1/W7_NR", f"{OV}/P5/W7_NR_extra"],
    "W7 Heavy V2-NoCkpt (up to 6 seeds)"
)

# W5 Moderate
paired(
    [f"{NC}/W5_NoCkpt_Moderate"],
    [f"{OV}/P1/W5_NR"],
    "W5 Moderate V2-NoCkpt"
)

# W7 Saturated
paired(
    [f"{NC}/W7_NoCkpt_Saturated"],
    [f"{OV}/P3/W7_NR_Sat"],
    "W7 Saturated V2-NoCkpt"
)
PYEOF

echo ""
echo "######## NoCkpt validation DONE $(date) ########"
