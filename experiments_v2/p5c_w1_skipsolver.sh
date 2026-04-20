#!/usr/bin/env bash
# P5c: parallel cross-workload validation of B_skip_solver on W1_Chat/Heavy.
# If B (FT_SKIP_SOLVER=1) is the W2 winner (fg_p95 ~81ms on s42), verify on W1.
# GPU 2-3. No dependency on P5 progress.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p5c_w1_skipsolver"
mkdir -p "$ROOT"

BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_v() {
    local tag="$1" extra="$2" seed="$3"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints_w1c 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=2,3 ${BASE} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8403 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P5c W1 skip_solver cross-workload $(date) ########"
for s in 42 123 456; do
    run_v "W1_B_skip_solver" "FT_SKIP_SOLVER=1" "$s"
done

echo ""
echo "=== P5c W1 summary (vs known W1 V2=3267, NR=3220) ==="
python3 << 'PYEOF'
import json, os, statistics as st
base = "results_v2/8B/overnight_2026-04-16/p5c_w1_skipsolver/W1_B_skip_solver"
gps, fg50s, fg95s = [], [], []
for s in [42,123,456]:
    p = f"{base}/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        if m.get('completion_rate',0) >= 0.95:
            gps.append(m['goodput'])
            fg50s.append(m.get('failover_gap_p50_ms',0))
            fg95s.append(m.get('failover_gap_p95_ms',0))
            print(f"  W1_B_skip_solver/s{s}: gp={m['goodput']:.1f} fg_p95={m.get('failover_gap_p95_ms',0):.0f}")
if gps:
    print(f"\nW1_B_skip_solver: gp={st.mean(gps):.1f} fg_p95={st.mean(fg95s):.0f}±{st.stdev(fg95s) if len(fg95s)>1 else 0:.0f}")
    print(f"(reference: W1 V2=3267±938, W1 NR=3220±1270)")
PYEOF
