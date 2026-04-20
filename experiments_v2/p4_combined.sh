#!/usr/bin/env bash
# P4: combine both fixes + more aggressive throttle.
#   V3: no_prebudget + interval=2 (combined)
#   V4: no_prebudget + interval=10 (near-off ckpt ctrl)
#   V5: no_prebudget + interval=10 + no FT_RECOVERY_PREBUDGET also in restore path
# On A4 3-seed. GPU 0-1.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p4_combined"
mkdir -p "$ROOT"

BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=0'

run_v() {
    local tag="$1" extra="$2" seed="$3"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=0,1 ${BASE} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P4 combined $(date) ########"
for s in 42 123 456; do
    run_v "V3_noprebudget_i2"   "FT_CHECKPOINT_STEP_INTERVAL=2"  "$s"
    run_v "V4_noprebudget_i10"  "FT_CHECKPOINT_STEP_INTERVAL=10" "$s"
done

echo ""
echo "=== P4 summary + all vs NR ==="
python3 << 'PYEOF'
import json, os, statistics as st
cfgs = [
    ("NR",                 "results_v2/8B/overnight_2026-04-16/p0_nr_fresh"),
    ("Baseline cap3+rep",  "results_v2/8B/overnight_2026-04-16/p1_os_cap3_reprefill"),
    ("V1 no_prebudget",    "results_v2/8B/overnight_2026-04-16/p3_prefault_pause/V1_no_prebudget"),
    ("V2 ckpt_interval2",  "results_v2/8B/overnight_2026-04-16/p3_prefault_pause/V2_interval2"),
    ("V3 V1+V2",           "results_v2/8B/overnight_2026-04-16/p4_combined/V3_noprebudget_i2"),
    ("V4 V1+interval10",   "results_v2/8B/overnight_2026-04-16/p4_combined/V4_noprebudget_i10"),
]
print(f"{'variant':<22} {'gp':>8} {'fg_p50':>12} {'fg_p95':>12}")
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
        print(f"{name:<22} {st.mean(gps):6.1f}±{gp_s:3.0f} {st.mean(fg50s):7.0f}±{fg50_s:4.0f} {st.mean(fg95s):7.0f}±{fg95_s:4.0f}")

# Per-seed breakdown
print("\n=== per-seed fg_p95 ===")
for s in [42, 123, 456]:
    print(f"s{s}:")
    for name, base in cfgs:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            print(f"  {name:<22}: fg_p95={m.get('failover_gap_p95_ms',0):.0f}")
PYEOF
