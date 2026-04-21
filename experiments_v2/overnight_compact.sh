#!/usr/bin/env bash
# Compact overnight (target 2.5h): focus on the "OS beats NR on both
# goodput + fg_p95" question. Only the high-leverage cells.
#
# Phase A — Core Discovery (50 min): 2 cells × 3 seeds × OS+NR parallel
#   A1: W2_Summary / Moderate / F3_Late  (expected winner — long prompt + late fault)
#   A2: W2_Summary / Moderate / F2_Mid   (ablation: earlier fault)
#
# Phase B — Fault Ablation on Winner (25 min): winner_setup × F1_Early × 3 seeds
#   (winner_setup already has F2_Mid (A2) and F3_Late (A1) data)
#
# Phase C — Engineering fix if needed (45 min): staggered solver if
#   winner doesn't beat NR on both metrics. Gated by FT_BENDERS_STAGGERED=1.
#
# Phase D — Paper table (10 min).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/compact"
mkdir -p "$ROOT"

OS_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'
NR_ENV='FT_RECOVERY_MODE=reprefill'

run_cell() {
    local tag="$1" workload="$2" load="$3" fault="$4"
    echo "============ ${tag}: ${workload}/${load}/${fault} ============" >&2
    local cell_root="${ROOT}/${tag}_${workload}_${load}_${fault}"
    mkdir -p "${cell_root}/os" "${cell_root}/nr"

    for seed in 42 123 456; do
        local os_out="${cell_root}/os/${seed}"
        local nr_out="${cell_root}/nr/${seed}"
        mkdir -p "$os_out" "$nr_out"

        if [ -f "${os_out}/metrics.json" ] && [ -f "${nr_out}/metrics.json" ]; then
            echo "  [skip] ${tag}/s${seed}" >&2; continue
        fi

        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] ${tag}/s${seed} launching OS+NR" >&2

        local os_pid=0 nr_pid=0
        if [ ! -f "${os_out}/metrics.json" ]; then
            eval "CUDA_VISIBLE_DEVICES=4,5 ${OS_ENV} python experiments_v2/run.py \
                --config experiments_v2/config_8b.yaml --baseline Our-System \
                --workload ${workload} --load ${load} --fault ${fault} \
                --seed ${seed} --port 8400 --output-dir ${os_out}" \
                > "${os_out}/stdout.log" 2>&1 &
            os_pid=$!
        fi
        if [ ! -f "${nr_out}/metrics.json" ]; then
            eval "CUDA_VISIBLE_DEVICES=6,7 ${NR_ENV} python experiments_v2/run.py \
                --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
                --workload ${workload} --load ${load} --fault ${fault} \
                --seed ${seed} --port 8500 --output-dir ${nr_out}" \
                > "${nr_out}/stdout.log" 2>&1 &
            nr_pid=$!
        fi
        [ $os_pid -ne 0 ] && wait $os_pid 2>/dev/null
        [ $nr_pid -ne 0 ] && wait $nr_pid 2>/dev/null

        python3 -c "
import json
for lbl, out in [('OS','${os_out}'),('NR','${nr_out}')]:
    try:
        m = json.load(open(f'{out}/metrics.json'))
        print(f'  ${tag}/s${seed} {lbl}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
    except Exception as e:
        print(f'  ${tag}/s${seed} {lbl}: FAILED ({type(e).__name__})')
" 2>&1 >&2 || true
    done
}

compile_cell_summary() {
    python3 <<PYEOF >&2
import json, os, statistics as st
root = '${ROOT}'
print(f'\n=== Summary of {root} ===')
print(f"{'cell':<40} {'base':<3} {'gp':>7} {'±':>3} {'fg_p95':>8} {'comp':>5} {'ok_seeds':>8}")
for cd in sorted(os.listdir(root)):
    cp = os.path.join(root, cd)
    if not os.path.isdir(cp): continue
    for tag in ['os','nr']:
        gps, fgs, comps = [], [], []
        for s in [42,123,456]:
            p = f'{cp}/{tag}/{s}/metrics.json'
            if os.path.exists(p):
                m = json.load(open(p))
                if m.get('completion_rate', 0) >= 0.95:
                    gps.append(m.get('goodput',0))
                    fgs.append(m.get('failover_gap_p95_ms',0))
                comps.append(m.get('completion_rate',0)*100)
        if gps:
            sd = st.stdev(gps) if len(gps)>1 else 0
            print(f'{cd:<40} {tag:<3} {st.mean(gps):7.1f} {sd:3.0f} {st.mean(fgs):8.0f} {st.mean(comps) if comps else 0:5.0f}% {len(gps):>4}/3')
        elif comps:
            print(f'{cd:<40} {tag:<3} {"":>7} {"":>3} {"":>8} {st.mean(comps):5.0f}% {"0":>8}/3')
PYEOF
}

echo "######## Compact overnight start $(date) ########"

# ── Phase A: Core Discovery (2 cells × 3 seeds × 2 baselines)
echo ""
echo "=== Phase A: Core Discovery ==="
run_cell "A1" "W2_Summary" "Moderate" "F3_Late"
run_cell "A2" "W2_Summary" "Moderate" "F2_Mid"
compile_cell_summary

# ── Phase B: Fault Ablation (F1_Early on winner-tier config)
echo ""
echo "=== Phase B: Fault Ablation ==="
run_cell "B1" "W2_Summary" "Moderate" "F1_Early"
compile_cell_summary

# ── Phase D: Final paper table
echo ""
echo "######## Phase D Paper Table $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
root = "results_v2/8B/overnight_2026-04-15/compact"
print("\n=== Paper-ready comparison table ===\n")
print(f"{'cell':<28} | {'NR gp (±)':<14} | {'OS gp (±)':<14} | {'Δgp':>7} | {'NR fg_p95':<10} | {'OS fg_p95':<10} | {'Δfg':>8} | {'OS comp':>8}")
print("-"*130)
for cd in sorted(os.listdir(root)):
    cp = os.path.join(root, cd)
    if not os.path.isdir(cp): continue
    def stats(tag):
        gps, fgs, comps = [], [], []
        for s in [42,123,456]:
            p = f'{cp}/{tag}/{s}/metrics.json'
            if os.path.exists(p):
                m = json.load(open(p))
                if m.get('completion_rate',0) >= 0.95:
                    gps.append(m['goodput'])
                    fgs.append(m['failover_gap_p95_ms'])
                comps.append(m['completion_rate']*100)
        return gps, fgs, comps
    nr_g, nr_f, nr_c = stats('nr')
    os_g, os_f, os_c = stats('os')
    if nr_g and os_g:
        nr_sd = st.stdev(nr_g) if len(nr_g)>1 else 0
        os_sd = st.stdev(os_g) if len(os_g)>1 else 0
        dgp = st.mean(os_g) - st.mean(nr_g)
        dfg = st.mean(os_f) - st.mean(nr_f)
        win_gp = "✓" if dgp > 0 else "✗"
        win_fg = "✓" if dfg < 0 else "✗"
        oc = st.mean(os_c) if os_c else 0
        print(f"{cd:<28} | {st.mean(nr_g):7.1f} ±{nr_sd:3.0f} | {st.mean(os_g):7.1f} ±{os_sd:3.0f} | {dgp:+6.1f}{win_gp} | {st.mean(nr_f):10.0f} | {st.mean(os_f):10.0f} | {dfg:+7.0f}{win_fg} | {oc:7.0f}%")

print("\n(✓ = OS beats NR on that metric; ✗ = OS loses)")
PYEOF

echo ""
echo "######## Compact overnight done $(date) ########"
