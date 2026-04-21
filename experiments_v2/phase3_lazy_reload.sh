#!/usr/bin/env bash
# Phase 3 — Staggered Admission (engine-layer via FT_LAZY_RELOAD).
# Head-to-head: v4 vs v4+LAZY_RELOAD (MAX_CONCURRENT=3) on cells where
# OS lost most in compact run: A1, A4, A5.
# NR data reused from compact run (gpu_util=0.9, FT_RECOVERY_MODE=reprefill).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/phase3"
mkdir -p "$ROOT"

OS_ENV_LAZY='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1 FT_LAZY_RELOAD=1 FT_LAZY_MAX_CONCURRENT=3'

run_cell_lazy() {
    local cell_id="$1" workload="$2" load="$3" fault="$4"
    echo "============ ${cell_id}+LAZY: ${workload}/${load}/${fault} ============" >&2
    local cell_root="${ROOT}/${cell_id}_LAZY_${workload}_${load}_${fault}"
    mkdir -p "${cell_root}"

    for seed in 42 123 456; do
        local out="${cell_root}/${seed}"
        mkdir -p "$out"
        [ -f "${out}/metrics.json" ] && continue

        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] ${cell_id}+LAZY/s${seed}" >&2
        eval "CUDA_VISIBLE_DEVICES=4,5 ${OS_ENV_LAZY} python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline Our-System \
            --workload ${workload} --load ${load} --fault ${fault} \
            --seed ${seed} --port 8400 --output-dir ${out}" \
            > "${out}/stdout.log" 2>&1

        python3 -c "
import json
try:
    m = json.load(open('${out}/metrics.json'))
    print(f'  ${cell_id}+LAZY/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
except Exception:
    print(f'  ${cell_id}+LAZY/s${seed}: FAILED')
" 2>&1
    done
}

echo "######## Phase 3 (LAZY_RELOAD head-to-head) start $(date) ########"
run_cell_lazy "A1" "W2_Summary" "Moderate" "F3_Late"
run_cell_lazy "A4" "W2_Summary" "Heavy" "F2_Mid"
run_cell_lazy "A5" "W4_Mixed" "Heavy" "F2_Mid"
echo "######## Phase 3 done $(date) ########"

# Head-to-head comparison table
python3 << 'PYEOF'
import json, os, statistics as st
compact = "results_v2/8B/overnight_2026-04-15/compact"
phase3 = "results_v2/8B/overnight_2026-04-15/phase3"
print("\n=== Head-to-head: v4 (compact) vs v4+LAZY_RELOAD (phase3) ===")
print(f"{'cell':<30} {'variant':<22} {'gp±std':>12} {'fg_p95±std':>15} {'comp':>6}")

def mean_stdev(xs):
    if not xs: return 0, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0)

cells = [
    ("A1_W2_Summary_Moderate_F3_Late",),
    ("A4_W2_Summary_Heavy_F2_Mid",),
    ("A5_W4_Mixed_Heavy_F2_Mid",),
]
for (cell,) in cells:
    for variant, path_base in [
        ("v4 (no LAZY_RELOAD)", f"{compact}/{cell}/os"),
        ("v4 + LAZY_RELOAD=3",  f"{phase3}/{cell.replace('A1','A1_LAZY').replace('A4','A4_LAZY').replace('A5','A5_LAZY')}"),
    ]:
        gps, fgs, comps = [], [], []
        for s in [42,123,456]:
            p = f"{path_base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                if m.get('completion_rate',0) >= 0.95:
                    gps.append(m['goodput']); fgs.append(m.get('failover_gap_p95_ms',0))
                comps.append(m.get('completion_rate',0)*100)
        gp_m, gp_s = mean_stdev(gps)
        fg_m, fg_s = mean_stdev(fgs)
        cm = st.mean(comps) if comps else 0
        print(f"{cell[:28]:<30} {variant:<22} {gp_m:6.1f}±{gp_s:3.0f}   {fg_m:7.0f}±{fg_s:4.0f}   {cm:4.0f}%")
    # NR reference
    p_nr_base = f"{compact}/{cell}/nr"
    gps, fgs, comps = [], [], []
    for s in [42,123,456]:
        p = f"{p_nr_base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m.get('completion_rate',0) >= 0.95:
                gps.append(m['goodput']); fgs.append(m.get('failover_gap_p95_ms',0))
            comps.append(m.get('completion_rate',0)*100)
    gp_m, gp_s = mean_stdev(gps)
    fg_m, fg_s = mean_stdev(fgs)
    cm = st.mean(comps) if comps else 0
    print(f"{cell[:28]:<30} {'NR baseline':<22} {gp_m:6.1f}±{gp_s:3.0f}   {fg_m:7.0f}±{fg_s:4.0f}   {cm:4.0f}%")
    print()
PYEOF
