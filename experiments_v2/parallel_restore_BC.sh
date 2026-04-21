#!/usr/bin/env bash
# Compare Variant B vs C to eliminate per-req sync's fg_p95 cost while
# still avoiding OOM.
#
#   B: max_workers=2, sync=off, gpu_util=0.9
#      temp peak = 2 × 224 MB = 448 MB → fits in 2.5 GB headroom
#
#   C: max_workers=8, sync=off, gpu_util=0.85
#      temp peak = 8 × 224 MB = 1.8 GB → fits in ~3.7 GB headroom
#
# Sequential to avoid /dev/shm wipe race between variants.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

COMMON_ENV_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RECOVERY_PREBUDGET=1'

run_one() {
    local tag="$1" seed="$2" config="$3" extra="$4"
    local out="results_v2/8B/parallel_restore_${tag}/${seed}"
    [ -f "${out}/metrics.json" ] && return 0
    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} starting" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV_BASE} ${extra} python experiments_v2/run.py \
        --config ${config} --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  ${tag}/s${seed}: FAILED" >&2
}

echo "######## B + C comparison $(date) ########"

echo ""
echo "=== Variant B: max_workers=2, sync=off, gpu_util=0.9 ==="
for s in 42 123 456; do
    run_one "v5B" "$s" "experiments_v2/config_8b.yaml" "FT_RESTORE_MAX_WORKERS=2 FT_RESTORE_PER_REQ_SYNC=0"
done

echo ""
echo "=== Variant C: max_workers=8, sync=off, gpu_util=0.85 ==="
for s in 42 123 456; do
    run_one "v5C" "$s" "experiments_v2/config_8b_gpu085.yaml" "FT_RESTORE_MAX_WORKERS=8 FT_RESTORE_PER_REQ_SYNC=0"
done

echo ""
echo "######## done $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
def load(p):
    try: return json.load(open(p))
    except: return None

nr = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42,123,456]]
nr_gp = [m['goodput'] for m in nr if m]
nr_fg = [m['failover_gap_p95_ms'] for m in nr if m]
print(f"\nNR Heavy: gp={st.mean(nr_gp):.1f}±{st.stdev(nr_gp):.0f}  fg_p95={st.mean(nr_fg):.0f}")

for tag in ["v5B","v5C"]:
    print(f"\n--- {tag} ---")
    rows = []
    for s in [42,123,456]:
        m = load(f"results_v2/8B/parallel_restore_{tag}/{s}/metrics.json")
        if m:
            gp, cr, fg = m['goodput'], m['completion_rate']*100, m['failover_gap_p95_ms']
            mark = "OK" if cr>=95 else "CRASH"
            print(f"  [{mark}] s{s}: gp={gp:.1f} comp={cr:.0f}% fg_p95={fg:.0f}")
            if cr>=95: rows.append((gp,fg))
    if len(rows)==3:
        gp_m = st.mean([r[0] for r in rows])
        fg_m = st.mean([r[1] for r in rows])
        d = gp_m - st.mean(nr_gp)
        print(f"  3-seed mean: gp={gp_m:.1f} fg_p95={fg_m:.0f}  Δvs.NR={d:+.1f}  {'WIN' if d>0 else 'LOSE'}")
PYEOF
