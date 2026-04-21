#!/usr/bin/env bash
# V7: gpu_util=0.80 (more headroom) + sequential restore (stable path).
# Goal: 3/3 comp=100% on Heavy/F2_Mid.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/v7_gpu080"
mkdir -p "$ROOT"

COMMON_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=0 FT_RESTORE_PARALLEL_LOAD=0 FT_RECOVERY_PREBUDGET=1'

run_one() {
    local seed="$1"
    local out="${ROOT}/${seed}"
    [ -f "${out}/metrics.json" ] && return 0
    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] s${seed} starting" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV} python experiments_v2/run.py \
        --config experiments_v2/config_8b_gpu080.yaml --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f}ms fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  s${seed}: FAILED" >&2
}

echo "######## V7 gpu_util=0.80 $(date) ########"
for s in 42 123 456; do
    run_one "$s"
done

python3 << 'PYEOF'
import json, os, statistics as st
def load(p):
    try: return json.load(open(p))
    except: return None

nr = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42,123,456]]
nr_gp = [m['goodput'] for m in nr]
nr_fg = [m['failover_gap_p95_ms'] for m in nr]
print(f"\nNR: gp={st.mean(nr_gp):.1f}±{st.stdev(nr_gp):.0f} fg_p95={st.mean(nr_fg):.0f}")
print(f"\n{'seed':<6}{'NR_gp':>8}{'NR_fg':>8}{'OS_gp':>9}{'OS_comp':>9}{'OS_fg50':>10}{'OS_fg95':>10}{'Δgp':>8}")
vgps, vfgs = [], []
for i, s in enumerate([42,123,456]):
    m = load(f"results_v2/8B/v7_gpu080/{s}/metrics.json")
    if m:
        cr=m['completion_rate']*100; mark="" if cr>=95 else "!"
        d=m['goodput']-nr[i]['goodput']
        print(f"s{s:<5}{nr[i]['goodput']:8.1f}{nr[i]['failover_gap_p95_ms']:8.0f}{m['goodput']:8.1f}{mark:>1}{cr:8.0f}%{m.get('failover_gap_p50_ms',0):10.0f}{m.get('failover_gap_p95_ms',0):10.0f}{d:+8.1f}")
        if cr>=95: vgps.append(m['goodput']); vfgs.append(m['failover_gap_p95_ms'])
if len(vgps)==3:
    d=st.mean(vgps)-st.mean(nr_gp)
    print(f"\n3-seed mean: gp={st.mean(vgps):.1f} fg_p95={st.mean(vfgs):.0f} Δvs.NR={d:+.1f} {'WIN' if d>0 else 'LOSE'}")
PYEOF
