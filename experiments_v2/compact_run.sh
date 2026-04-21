#!/usr/bin/env bash
# Compact version (~2h): 3 key cells × 3 seeds × OS+NR parallel.
#   A1: W2_Summary/Moderate/F3_Late  (main hypothesis - OS wins both metrics)
#   A2: W2_Summary/Moderate/F2_Mid   (alternative fault timing)
#   A3: W1_Chat/Moderate/F3_Late     (control - short prompt)
# OS on GPU 4-5, NR on GPU 6-7 parallel. ~20 min per cell.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/compact"
mkdir -p "$ROOT"

OS_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

NR_ENV='FT_RECOVERY_MODE=reprefill'

run_cell() {
    local cell_id="$1" workload="$2" load="$3" fault="$4"
    echo "============ ${cell_id}: ${workload}/${load}/${fault} ============" >&2
    local cell_root="${ROOT}/${cell_id}_${workload}_${load}_${fault}"
    mkdir -p "${cell_root}/os" "${cell_root}/nr"

    for seed in 42 123 456; do
        local os_out="${cell_root}/os/${seed}"
        local nr_out="${cell_root}/nr/${seed}"
        mkdir -p "$os_out" "$nr_out"
        [ -f "${os_out}/metrics.json" ] && [ -f "${nr_out}/metrics.json" ] && continue

        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] ${cell_id}/s${seed} launching" >&2

        eval "CUDA_VISIBLE_DEVICES=4,5 ${OS_ENV} python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline Our-System \
            --workload ${workload} --load ${load} --fault ${fault} \
            --seed ${seed} --port 8400 --output-dir ${os_out}" \
            > "${os_out}/stdout.log" 2>&1 &
        local os_pid=$!

        eval "CUDA_VISIBLE_DEVICES=6,7 ${NR_ENV} python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
            --workload ${workload} --load ${load} --fault ${fault} \
            --seed ${seed} --port 8500 --output-dir ${nr_out}" \
            > "${nr_out}/stdout.log" 2>&1 &
        local nr_pid=$!

        wait $os_pid 2>/dev/null
        wait $nr_pid 2>/dev/null

        python3 -c "
import json, os
for tag, out in [('OS', '${os_out}'), ('NR', '${nr_out}')]:
    try:
        m = json.load(open(f'{out}/metrics.json'))
        print(f'  ${cell_id}/s${seed} {tag}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
    except Exception as e:
        print(f'  ${cell_id}/s${seed} {tag}: FAILED ({type(e).__name__})')
" 2>&1
    done
}

echo "######## Compact run start $(date) ########"
run_cell "A1" "W2_Summary" "Moderate" "F3_Late"
run_cell "A2" "W2_Summary" "Moderate" "F2_Mid"
run_cell "A3" "W1_Chat" "Moderate" "F3_Late"
echo "######## Compact run done $(date) ########"

# Final comparison
python3 << 'PYEOF'
import json, os, statistics as st
root = "results_v2/8B/overnight_2026-04-15/compact"
print("\n=== Compact Run — Paper-Ready Table ===")
print(f"{'cell':<40} {'tag':<3} {'gp_mean±std':>15} {'fg_p95_mean±std':>20} {'comp_avg':>9}  {'Δ vs NR':>10}")
for cell_dir in sorted(os.listdir(root)):
    cp = os.path.join(root, cell_dir)
    if not os.path.isdir(cp): continue
    stats = {}
    for tag in ['os','nr']:
        gps, fgs, comps = [], [], []
        for s in [42,123,456]:
            p = f'{cp}/{tag}/{s}/metrics.json'
            if os.path.exists(p):
                m = json.load(open(p))
                if m.get('completion_rate', 0) >= 0.95:
                    gps.append(m['goodput'])
                    fgs.append(m.get('failover_gap_p95_ms', 0))
                comps.append(m.get('completion_rate', 0)*100)
        stats[tag] = (gps, fgs, comps)
    for tag in ['nr','os']:
        gps, fgs, comps = stats[tag]
        if len(gps) >= 1:
            gp_s = st.stdev(gps) if len(gps)>1 else 0
            fg_s = st.stdev(fgs) if len(fgs)>1 else 0
            cm = st.mean(comps) if comps else 0
            if tag == 'os' and stats['nr'][0]:
                d = st.mean(gps) - st.mean(stats['nr'][0])
                d_str = f"{d:+.1f}"
            else:
                d_str = "—"
            print(f"{cell_dir:<40} {tag.upper():<3} {st.mean(gps):7.1f}±{gp_s:3.0f}     {st.mean(fgs):10.0f}±{fg_s:4.0f}      {cm:5.0f}%  {d_str:>10}")
        else:
            print(f"{cell_dir:<40} {tag.upper():<3} {'no valid':>15} {'—':>20} {'—':>9}  {'—':>10}")
PYEOF
