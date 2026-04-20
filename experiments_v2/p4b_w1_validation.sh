#!/usr/bin/env bash
# Parallel P4b on GPU 2-3: validate best variants on W1_Chat/Heavy
# (cross-workload check for paper breadth).
# Tests: V1 no_prebudget, V2 interval=2, baseline cap=3+rep.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p4b_w1"
mkdir -p "$ROOT"

BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3'

run_v() {
    local tag="$1" extra="$2" seed="$3"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints_w1 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=2,3 ${BASE} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8401 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P4b W1_Chat/Heavy validation $(date) ########"
for s in 42 123 456; do
    run_v "W1_baseline"         "FT_RECOVERY_PREBUDGET=1"                        "$s"
    run_v "W1_V1_noprebudget"   "FT_RECOVERY_PREBUDGET=0"                        "$s"
    run_v "W1_V2_interval2"     "FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2" "$s"
done

# Also NR baseline on W1/Heavy for fair comparison
echo ""
echo "=== W1_Chat/Heavy NR baseline ==="
for s in 42 123 456; do
    out="${ROOT}/W1_NR/${s}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && continue
    rm -rf /dev/shm/vllm_ft_checkpoints_w1 2>/dev/null
    echo "[$(date +%H:%M:%S)] W1_NR/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=2,3 FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${s} --port 8401 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  W1_NR/s${s}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
done

echo ""
echo "=== P4b summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
print(f"{'variant':<22} {'gp':>8} {'fg_p50':>12} {'fg_p95':>12} {'comp':>5}")
cfgs = [
    ("W1_NR",               "results_v2/8B/overnight_2026-04-16/p4b_w1/W1_NR"),
    ("W1_baseline cap3+rep","results_v2/8B/overnight_2026-04-16/p4b_w1/W1_baseline"),
    ("W1_V1 no_prebudget",  "results_v2/8B/overnight_2026-04-16/p4b_w1/W1_V1_noprebudget"),
    ("W1_V2 ckpt_interval2","results_v2/8B/overnight_2026-04-16/p4b_w1/W1_V2_interval2"),
]
for name, base in cfgs:
    gps, fg50s, fg95s, comps = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            comps.append(m['completion_rate']*100)
            if m.get('completion_rate',0) >= 0.95:
                gps.append(m['goodput'])
                fg50s.append(m.get('failover_gap_p50_ms',0))
                fg95s.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gp_s = st.stdev(gps) if len(gps)>1 else 0
        fg50_s = st.stdev(fg50s) if len(fg50s)>1 else 0
        fg95_s = st.stdev(fg95s) if len(fg95s)>1 else 0
        cm = st.mean(comps) if comps else 0
        print(f"{name:<22} {st.mean(gps):6.1f}±{gp_s:3.0f} {st.mean(fg50s):7.0f}±{fg50_s:4.0f} {st.mean(fg95s):7.0f}±{fg95_s:4.0f} {cm:3.0f}%")
PYEOF
