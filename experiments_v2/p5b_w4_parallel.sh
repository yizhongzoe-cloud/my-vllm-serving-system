#!/usr/bin/env bash
# P5b: parallel W4_Mixed/Heavy baseline on GPU 2-3 while P5 ablation runs on GPU 0-1.
# Saves Phase 2 time — pre-computes NR + V2 reference on W4 production mix.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p5b_w4"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_os() {
    local tag="$1" seed="$2"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints_w4 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=2,3 ${V2_BASE} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W4_Mixed --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8402 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

run_nr() {
    local seed="$1"
    local out="${ROOT}/W4_NR/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints_w4 2>/dev/null
    echo "[$(date +%H:%M:%S)] W4_NR/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=2,3 FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
        --workload W4_Mixed --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8402 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  W4_NR/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P5b W4_Mixed/Heavy parallel baseline $(date) ########"
for s in 42 123 456; do
    run_nr "$s"
    run_os "W4_V2" "$s"
done

echo ""
echo "=== P5b W4 summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
cfgs = [
    ("W4 NR", "results_v2/8B/overnight_2026-04-16/p5b_w4/W4_NR"),
    ("W4 V2 ckpt_interval=2", "results_v2/8B/overnight_2026-04-16/p5b_w4/W4_V2"),
]
print(f"{'variant':<28} {'gp':>8} {'fg_p50':>12} {'fg_p95':>12}")
for name, base in cfgs:
    gps, fg50s, fg95s = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m.get('completion_rate',0) >= 0.95:
                gps.append(m['goodput'])
                fg50s.append(m.get('failover_gap_p50_ms',0))
                fg95s.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gp_s = st.stdev(gps) if len(gps)>1 else 0
        fg50_s = st.stdev(fg50s) if len(fg50s)>1 else 0
        fg95_s = st.stdev(fg95s) if len(fg95s)>1 else 0
        print(f"{name:<28} {st.mean(gps):6.1f}±{gp_s:3.0f} {st.mean(fg50s):7.0f}±{fg50_s:4.0f} {st.mean(fg95s):7.0f}±{fg95_s:4.0f}")
PYEOF
