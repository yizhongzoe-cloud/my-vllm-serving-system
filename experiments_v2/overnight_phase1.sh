#!/usr/bin/env bash
# Phase 1 Discovery — 6 cells × 3 seeds × (OS v4 + NR).
# OS on GPU 4-5, NR on GPU 6-7, parallel per cell. Cells sequential.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/phase1"
mkdir -p "$ROOT"

OS_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

NR_ENV='FT_RECOVERY_MODE=reprefill'

run_cell() {
    local cell_id="$1" workload="$2" load="$3" fault="$4"
    echo "============ Cell ${cell_id}: ${workload}/${load}/${fault} ============" >&2
    local cell_root="${ROOT}/${cell_id}_${workload}_${load}_${fault}"
    mkdir -p "${cell_root}/os" "${cell_root}/nr"

    for seed in 42 123 456; do
        local os_out="${cell_root}/os/${seed}"
        local nr_out="${cell_root}/nr/${seed}"
        mkdir -p "$os_out" "$nr_out"

        if [ -f "${os_out}/metrics.json" ] && [ -f "${nr_out}/metrics.json" ]; then
            echo "  [skip] ${cell_id}/s${seed}" >&2; continue
        fi

        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] ${cell_id}/s${seed} launching OS+NR" >&2

        # OS on GPU 4-5 port 8400
        if [ ! -f "${os_out}/metrics.json" ]; then
            eval "CUDA_VISIBLE_DEVICES=4,5 ${OS_ENV} python experiments_v2/run.py \
                --config experiments_v2/config_8b.yaml --baseline Our-System \
                --workload ${workload} --load ${load} --fault ${fault} \
                --seed ${seed} --port 8400 --output-dir ${os_out}" \
                > "${os_out}/stdout.log" 2>&1 &
            local os_pid=$!
        else
            local os_pid=0
        fi

        # NR on GPU 6-7 port 8500
        if [ ! -f "${nr_out}/metrics.json" ]; then
            eval "CUDA_VISIBLE_DEVICES=6,7 ${NR_ENV} python experiments_v2/run.py \
                --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
                --workload ${workload} --load ${load} --fault ${fault} \
                --seed ${seed} --port 8500 --output-dir ${nr_out}" \
                > "${nr_out}/stdout.log" 2>&1 &
            local nr_pid=$!
        else
            local nr_pid=0
        fi

        [ $os_pid -ne 0 ] && wait $os_pid 2>/dev/null
        [ $nr_pid -ne 0 ] && wait $nr_pid 2>/dev/null

        python3 -c "
import json
for tag, out in [('OS', '${os_out}'), ('NR', '${nr_out}')]:
    try:
        m = json.load(open(f'{out}/metrics.json'))
        print(f'  ${cell_id}/s${seed} {tag}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
    except Exception as e:
        print(f'  ${cell_id}/s${seed} {tag}: FAILED ({type(e).__name__})')
" >&2 2>&1 || true
    done

    # Cell summary
    python3 <<PYEOF >&2
import json, os, statistics as st
cell_root = '${cell_root}'
for tag in ['os', 'nr']:
    gps, fgs = [], []
    for s in [42, 123, 456]:
        p = f'{cell_root}/{tag}/{s}/metrics.json'
        if os.path.exists(p):
            m = json.load(open(p))
            if m.get('completion_rate', 0) >= 0.95:
                gps.append(m['goodput'])
                fgs.append(m.get('failover_gap_p95_ms', 0))
    if len(gps) == 3:
        print(f'  [${cell_id}] {tag.upper():2s} 3-seed: gp={st.mean(gps):.1f}±{st.stdev(gps):.0f} fg_p95={st.mean(fgs):.0f}±{st.stdev(fgs):.0f}')
    else:
        print(f'  [${cell_id}] {tag.upper():2s} only {len(gps)}/3 comp>=95%')
PYEOF
}

echo "######## Phase 1 start $(date) ########"

# Cell 1: W2 Moderate F3_Late (expected winner)
run_cell "c1" "W2_Summary" "Moderate" "F3_Late"

# Cell 2: W2 Moderate F2_Mid
run_cell "c2" "W2_Summary" "Moderate" "F2_Mid"

# Cell 3: W2 Heavy F3_Late (boundary)
run_cell "c3" "W2_Summary" "Heavy" "F3_Late"

# Cell 4: W4_Mixed Moderate F3_Late
run_cell "c4" "W4_Mixed" "Moderate" "F3_Late"

# Cell 5: W1_Chat Moderate F3_Late (control)
run_cell "c5" "W1_Chat" "Moderate" "F3_Late"

# Cell 6: W3_Instruct Moderate F2_Mid (short prompt control)
run_cell "c6" "W3_Instruct" "Moderate" "F2_Mid"

echo ""
echo "######## Phase 1 complete $(date) ########"

# Final summary
python3 << 'PYEOF'
import json, os, statistics as st
root = "results_v2/8B/overnight_2026-04-15/phase1"
print("\n=== Phase 1 Summary Table ===")
print(f"{'cell':<28} {'baseline':<3} {'gp_mean':>8} {'fg_p95':>8} {'comp':>5}")
for cell_dir in sorted(os.listdir(root)):
    cp = os.path.join(root, cell_dir)
    if not os.path.isdir(cp): continue
    for tag in ['os','nr']:
        gps, fgs, comps = [], [], []
        for s in [42,123,456]:
            p = f'{cp}/{tag}/{s}/metrics.json'
            if os.path.exists(p):
                m = json.load(open(p))
                gps.append(m.get('goodput',0))
                fgs.append(m.get('failover_gap_p95_ms',0))
                comps.append(m.get('completion_rate',0)*100)
        if gps:
            gp_m = st.mean(gps) if len(gps)>0 else 0
            fg_m = st.mean(fgs) if len(fgs)>0 else 0
            cm_m = st.mean(comps) if len(comps)>0 else 0
            print(f"{cell_dir:<28} {tag:<3} {gp_m:8.1f} {fg_m:8.0f} {cm_m:5.0f}%")
PYEOF
