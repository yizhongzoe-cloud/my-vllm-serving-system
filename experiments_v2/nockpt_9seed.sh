#!/usr/bin/env bash
# Extend V2-NoCkpt W7 Heavy from 6 to 9 seeds (+5678/9999/11111).
# Uses existing NR data from overnight_0421 P5 extras.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/nockpt_2026-04-21/W7_NoCkpt_Heavy"
mkdir -p "$ROOT"

pkill -9 -f "api_server|EngineCore" 2>/dev/null
sleep 3
rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null

V2_SOLVER='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_SOLVER_TIME_CAP_MS=100 FT_SOLVER_GREEDY_SEED=1 FT_RECOVERY_MODE=reprefill'

for s in 5678 9999 11111; do
    out="${ROOT}/${s}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && { echo "s${s} skipped"; continue; }
    pkill -9 -f "api_server|EngineCore" 2>/dev/null
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    sleep 3
    echo "[$(date +%H:%M:%S)] s${s}"
    eval "CUDA_VISIBLE_DEVICES=0,1 ${V2_SOLVER} timeout 600 python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System-NoCkpt \
        --workload W7_Saturated --load Heavy --fault F2_Mid \
        --seed ${s} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  s${s}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
"
    fi
done

echo ""
echo "######## 9-seed W7 Heavy paired analysis $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
OV = "results_v2/8B/overnight_2026-04-21"
NC = "results_v2/8B/nockpt_2026-04-21"

def read_all(path):
    out = {}
    if not os.path.isdir(path): return out
    for s in sorted(os.listdir(path)):
        p = os.path.join(path, s, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                out[s] = (m['goodput'], m.get('failover_gap_p95_ms',0))
    return out

# V2-NoCkpt sources: overnight P4 (42/123/456) + nockpt (789/1234/22222/5678/9999/11111)
v2 = {}
v2.update(read_all(f"{OV}/P4/V2_NoCkpt"))
v2.update(read_all(f"{NC}/W7_NoCkpt_Heavy"))

# NR sources: overnight P1 (6 seeds) + P5 extras (5678/9999/11111)
nr = {}
nr.update(read_all(f"{OV}/P1/W7_NR"))
nr.update(read_all(f"{OV}/P5/W7_NR_extra"))

common = sorted(set(v2.keys()) & set(nr.keys()))
print(f"=== W7 Heavy V2-NoCkpt paired (n={len(common)}) ===")
wins = 0; partial_gp = 0; partial_fg = 0
for s in common:
    vg, vf = v2[s]; ng, nf = nr[s]
    gp_w = '✓' if vg >= ng else '✗'
    fg_w = '✓' if vf <= nf else '✗'
    strict = ' 🎯' if (vg >= ng and vf <= nf) else ''
    print(f"  s{s}: V2 {vg:.1f}/{vf:.0f} vs NR {ng:.1f}/{nf:.0f} [{gp_w}][{fg_w}]{strict}")
    if gp_w == '✓' and fg_w == '✓': wins += 1
    elif gp_w == '✓': partial_gp += 1
    elif fg_w == '✓': partial_fg += 1

v_gps = [v2[s][0] for s in common]
v_fgs = [v2[s][1] for s in common]
n_gps = [nr[s][0] for s in common]
n_fgs = [nr[s][1] for s in common]
print(f"\n  strict wins: {wins}/{len(common)}")
print(f"  gp-only wins: {partial_gp}  fg-only wins: {partial_fg}")
print(f"  V2 mean: gp={st.mean(v_gps):.1f}  fg_p95={st.mean(v_fgs):.0f}")
print(f"  NR mean: gp={st.mean(n_gps):.1f}  fg_p95={st.mean(n_fgs):.0f}")
print(f"  Δ V2-NR: gp={st.mean(v_gps)-st.mean(n_gps):+.1f} ({100*(st.mean(v_gps)-st.mean(n_gps))/st.mean(n_gps):+.1f}%)  fg_p95={st.mean(v_fgs)-st.mean(n_fgs):+.0f} ({100*(st.mean(v_fgs)-st.mean(n_fgs))/st.mean(n_fgs):+.1f}%)")

# Paired t-test for significance
try:
    from scipy import stats as ss
    t_gp, p_gp = ss.ttest_rel(v_gps, n_gps)
    t_fg, p_fg = ss.ttest_rel(v_fgs, n_fgs)
    print(f"\n  paired t-test gp: t={t_gp:.2f} p={p_gp:.3f}")
    print(f"  paired t-test fg_p95: t={t_fg:.2f} p={p_fg:.3f}")
except ImportError:
    pass
PYEOF

echo ""
echo "######## 9-seed extension DONE $(date) ########"
