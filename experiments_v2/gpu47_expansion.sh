#!/usr/bin/env bash
# GPU 4-7 parallel expansion: firm up the W5 protection claim + W7 Saturated.
# Track A (GPU 4-5, port 8505): W5 Moderate 6 new seeds V2-NoCkpt + 6 NR
# Track B (GPU 6-7, port 8507): W7 Saturated 6 seeds V2-NoCkpt + 6 NR
#                             + W4_Mixed 3 seeds V2-NoCkpt + 3 NR
# NoCkpt + NR don't use /dev/shm checkpoints so parallel is race-free.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/gpu47_2026-04-21"
mkdir -p "$ROOT"

V2_SOLVER='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_SOLVER_TIME_CAP_MS=100 FT_SOLVER_GREEDY_SEED=1 FT_RECOVERY_MODE=reprefill'
NR_ENV="FT_RECOVERY_MODE=reprefill"

run_cell() {
    local track="$1" tag="$2" gpus="$3" port="$4" baseline="$5" wl="$6" load="$7" fault="$8" seed="$9" env_str="${10}"
    local out="${ROOT}/${track}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    sleep 2
    echo "[$(date +%H:%M:%S)] [${track}] ${tag}/s${seed} ${baseline} ${wl} ${load} ${fault}"
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_str} timeout 600 python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  [${track}] ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
"
    fi
}

track_A() {
    # W5 Moderate: 6 new seeds for V2-NoCkpt + 6 NR paired
    local SEEDS_NEW="5678 9999 11111 98765 54321 77777"
    for s in $SEEDS_NEW; do
        run_cell "A" "W5_V2NC" "4,5" "8505" "Our-System-NoCkpt" "W5_LongDoc" "Moderate" "F2_Mid" "$s" "$V2_SOLVER" || true
    done
    for s in $SEEDS_NEW; do
        run_cell "A" "W5_NR" "4,5" "8505" "NoFT-Reprefill" "W5_LongDoc" "Moderate" "F2_Mid" "$s" "$NR_ENV" || true
    done
    echo "[$(date +%H:%M:%S)] Track A done"
}

track_B() {
    # W7 Saturated: 6 new seeds (+ existing 42/123/456 from overnight = 9 total)
    local SEEDS_SAT="789 1234 22222 5678 9999 11111"
    for s in $SEEDS_SAT; do
        run_cell "B" "W7_Sat_V2NC" "6,7" "8507" "Our-System-NoCkpt" "W7_Saturated" "Saturated" "F2_Mid" "$s" "$V2_SOLVER" || true
    done
    for s in $SEEDS_SAT; do
        run_cell "B" "W7_Sat_NR"    "6,7" "8507" "NoFT-Reprefill"    "W7_Saturated" "Saturated" "F2_Mid" "$s" "$NR_ENV" || true
    done
    # W4_Mixed: production mix, per-request SLO differentiation — solver's natural strength
    local SEEDS_W4="42 123 456 789 1234 22222"
    for s in $SEEDS_W4; do
        run_cell "B" "W4_V2NC" "6,7" "8507" "Our-System-NoCkpt" "W4_Mixed" "Heavy" "F2_Mid" "$s" "$V2_SOLVER" || true
    done
    for s in $SEEDS_W4; do
        run_cell "B" "W4_NR"   "6,7" "8507" "NoFT-Reprefill"    "W4_Mixed" "Heavy" "F2_Mid" "$s" "$NR_ENV" || true
    done
    echo "[$(date +%H:%M:%S)] Track B done"
}

# Launch both tracks in parallel
echo "######## GPU 4-7 expansion start $(date) ########"
track_A > "${ROOT}/track_A.log" 2>&1 &
PID_A=$!
track_B > "${ROOT}/track_B.log" 2>&1 &
PID_B=$!
echo "Track A PID=$PID_A  Track B PID=$PID_B"

# Wait for both
wait $PID_A $PID_B
echo "Both tracks finished $(date)"

##################################################
# Final paired analysis                          #
##################################################
echo ""
echo "######## FINAL ANALYSIS $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st

def read_dir(path, comp_filter=0.95):
    out = {}
    if not os.path.isdir(path): return out
    for s in sorted(os.listdir(path)):
        p = os.path.join(path, s, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= comp_filter:
                out[s] = (m['goodput'], m.get('failover_gap_p95_ms',0), m['completion_rate']*100)
    return out

def read_dir_all(path):
    return read_dir(path, comp_filter=0.0)

def paired_strict(v2_dirs, nr_dirs, label):
    v2 = {}; [v2.update(read_dir(d)) for d in v2_dirs]
    nr = {}; [nr.update(read_dir(d)) for d in nr_dirs]
    common = sorted(set(v2) & set(nr))
    if not common:
        print(f"\n=== {label} === no paired stable data")
        return
    wins = 0
    print(f"\n=== {label} strict (n={len(common)}, comp>=95%) ===")
    for s in common:
        vg, vf, _ = v2[s]; ng, nf, _ = nr[s]
        strict = ' 🎯' if (vg >= ng and vf <= nf) else ''
        gp_w = '✓' if vg >= ng else '✗'
        fg_w = '✓' if vf <= nf else '✗'
        print(f"  s{s}: V2 {vg:.1f}/{vf:.0f} vs NR {ng:.1f}/{nf:.0f} [{gp_w}][{fg_w}]{strict}")
        if vg >= ng and vf <= nf: wins += 1
    v_gps = [v2[s][0] for s in common]; v_fgs = [v2[s][1] for s in common]
    n_gps = [nr[s][0] for s in common]; n_fgs = [nr[s][1] for s in common]
    print(f"  strict wins: {wins}/{len(common)}")
    print(f"  V2 mean: gp={st.mean(v_gps):.1f}  fg_p95={st.mean(v_fgs):.0f}")
    print(f"  NR mean: gp={st.mean(n_gps):.1f}  fg_p95={st.mean(n_fgs):.0f}")
    try:
        from scipy import stats as ss
        t_g, p_g = ss.ttest_rel(v_gps, n_gps)
        t_f, p_f = ss.ttest_rel(v_fgs, n_fgs)
        print(f"  paired t-test gp: t={t_g:.2f} p={p_g:.3f}  fg_p95: t={t_f:.2f} p={p_f:.3f}")
    except ImportError: pass

def comp_protection(v2_dirs, nr_dirs, label):
    v2 = {}; [v2.update(read_dir_all(d)) for d in v2_dirs]
    nr = {}; [nr.update(read_dir_all(d)) for d in nr_dirs]
    common = sorted(set(v2) & set(nr))
    if not common:
        print(f"\n=== {label} comp-protection === no paired")
        return
    print(f"\n=== {label} comp-protection (all seeds, n={len(common)}) ===")
    v_comps = []; n_comps = []
    for s in common:
        vg,vf,vc = v2[s]; ng,nf,nc = nr[s]
        v_comps.append(vc); n_comps.append(nc)
        print(f"  s{s}: V2 {vg:.1f}/{vf:.0f}/{vc:.0f}% vs NR {ng:.1f}/{nf:.0f}/{nc:.0f}%  Δcomp={vc-nc:+.0f}")
    print(f"  V2 comp mean: {st.mean(v_comps):.1f}%")
    print(f"  NR comp mean: {st.mean(n_comps):.1f}%")
    print(f"  Δ comp:      {st.mean(v_comps)-st.mean(n_comps):+.1f}%")

OV = "results_v2/8B/overnight_2026-04-21"
NC = "results_v2/8B/nockpt_2026-04-21"
G47 = "results_v2/8B/gpu47_2026-04-21"

# W5 Moderate 12-seed (6 existing NC + 6 new) vs NR paired
print("\n#### W5 Moderate expanded ####")
paired_strict(
    [f"{NC}/W5_NoCkpt_Moderate", f"{G47}/A/W5_V2NC"],
    [f"{OV}/P1/W5_NR", f"{G47}/A/W5_NR"],
    "W5 Moderate V2-NoCkpt"
)
comp_protection(
    [f"{NC}/W5_NoCkpt_Moderate", f"{G47}/A/W5_V2NC"],
    [f"{OV}/P1/W5_NR", f"{G47}/A/W5_NR"],
    "W5 Moderate"
)

# W7 Saturated 9-seed
print("\n#### W7 Saturated expanded ####")
paired_strict(
    [f"{OV}/P4/V2_NoCkpt".replace('V2_NoCkpt','_dummy'), f"{NC}/W7_NoCkpt_Saturated", f"{G47}/B/W7_Sat_V2NC"],
    [f"{OV}/P3/W7_NR_Sat", f"{G47}/B/W7_Sat_NR"],
    "W7 Saturated V2-NoCkpt"
)
comp_protection(
    [f"{NC}/W7_NoCkpt_Saturated", f"{G47}/B/W7_Sat_V2NC"],
    [f"{OV}/P3/W7_NR_Sat", f"{G47}/B/W7_Sat_NR"],
    "W7 Saturated"
)

# W4_Mixed 6-seed
print("\n#### W4_Mixed Heavy (new angle: production mix, per-req SLO) ####")
paired_strict(
    [f"{G47}/B/W4_V2NC"],
    [f"{G47}/B/W4_NR"],
    "W4_Mixed Heavy V2-NoCkpt"
)
comp_protection(
    [f"{G47}/B/W4_V2NC"],
    [f"{G47}/B/W4_NR"],
    "W4_Mixed Heavy"
)
PYEOF

echo ""
echo "######## GPU 4-7 expansion DONE $(date) ########"
