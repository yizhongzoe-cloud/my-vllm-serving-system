#!/usr/bin/env bash
# P6 (Phase 4): V2 × {F1_Early, F3_Late} × W2_Summary/Heavy × 3 seeds.
# F2_Mid already done. Gives fault-timing robustness story for paper.
# GPU 0-1.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p6_fault_timing"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_os() {
    local fault="$1" seed="$2"
    local out="${ROOT}/V2_${fault}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] V2_${fault}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=0,1 ${V2_BASE} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault ${fault} \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  V2_${fault}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

run_nr() {
    local fault="$1" seed="$2"
    local out="${ROOT}/NR_${fault}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] NR_${fault}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=0,1 FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
        --workload W2_Summary --load Heavy --fault ${fault} \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  NR_${fault}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P6 fault timing (V2 + NR × F1/F3 × W2/Heavy) $(date) ########"
for s in 42 123 456; do
    run_os "F1_Early" "$s"
    run_nr "F1_Early" "$s"
    run_os "F3_Late" "$s"
    run_nr "F3_Late" "$s"
done

echo ""
echo "=== P6 summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
faults = ["F1_Early", "F3_Late"]
print(f"{'variant':<22} {'fault':<10} {'gp':>8} {'fg_p50':>12} {'fg_p95':>12}")
for f in faults:
    for name, tag in [("NR", f"NR_{f}"), ("V2", f"V2_{f}")]:
        base = f"results_v2/8B/overnight_2026-04-16/p6_fault_timing/{tag}"
        gps, fg50s, fg95s = [], [], []
        for s in [42,123,456]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                if m.get('completion_rate',0) >= 0.95:
                    gps.append(m['goodput']); fg50s.append(m.get('failover_gap_p50_ms',0))
                    fg95s.append(m.get('failover_gap_p95_ms',0))
        if gps:
            gp_s = st.stdev(gps) if len(gps)>1 else 0
            fg50_s = st.stdev(fg50s) if len(fg50s)>1 else 0
            fg95_s = st.stdev(fg95s) if len(fg95s)>1 else 0
            print(f"{name:<22} {f:<10} {st.mean(gps):6.1f}±{gp_s:3.0f} {st.mean(fg50s):7.0f}±{fg50_s:4.0f} {st.mean(fg95s):7.0f}±{fg95_s:4.0f}")
PYEOF
