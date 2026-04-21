#!/usr/bin/env bash
# Parallel-restore validation v2 — sequential 3-seed on GPU 4-5 only.
# Avoids the /dev/shm wipe race that corrupted checkpoints in v1.
# Validates os_parallel_pb (batch_rpc + parallel load + prebudget) —
# same config whose s42 previously hit gp=295.3 comp=100% fg_p95=3227ms.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/parallel_restore_v2"
mkdir -p "$ROOT"

COMMON_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RECOVERY_PREBUDGET=1'

run_one() {
    local seed="$1"
    local out="${ROOT}/${seed}"
    [ -f "${out}/metrics.json" ] && return 0
    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] s${seed} starting" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV} python experiments_v2/run.py \
        --config ${CONFIG} --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  s${seed}: FAILED" >&2
}

echo "######## parallel_restore_v2 $(date) ########"
for s in 42 123 456; do
    run_one "$s"
done
echo "######## done $(date) ########"

python3 << 'PYEOF'
import json, os, statistics as st
def load(p):
    try: return json.load(open(p))
    except: return None

nr = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42,123,456]]
nr_gp = [m['goodput'] for m in nr if m]
nr_fg = [m['failover_gap_p95_ms'] for m in nr if m]
print(f"\nNR Heavy: gp={st.mean(nr_gp):.1f}±{st.stdev(nr_gp):.0f}  fg_p95={st.mean(nr_fg):.0f}±{st.stdev(nr_fg):.0f}")

rows = []
for s in [42,123,456]:
    m = load(f"results_v2/8B/parallel_restore_v2/{s}/metrics.json")
    if m:
        gp = m['goodput']
        cr = m['completion_rate']*100
        fg = m['failover_gap_p95_ms']
        mark = "OK" if cr >= 95 else "CRASH"
        print(f"  [{mark}] s{s}: gp={gp:.1f} comp={cr:.0f}% fg_p95={fg:.0f}")
        if cr>=95: rows.append((gp,fg))

if len(rows)==3:
    gp_m = st.mean([r[0] for r in rows])
    fg_m = st.mean([r[1] for r in rows])
    gp_s = st.stdev([r[0] for r in rows])
    d = gp_m - st.mean(nr_gp)
    print(f"\nos_parallel_pb 3-seed: gp={gp_m:.1f}±{gp_s:.0f}  fg={fg_m:.0f}  Δvs.NR={d:+.1f}  {'WIN' if d>0 else 'LOSE'}")
PYEOF
