#!/usr/bin/env bash
# Diag13: CRITICAL TEST — V2 with FT_RECOVERY_MODE=reload (not reprefill).
# All prior 128 experiments disabled V2's main contribution by using reprefill.
# This test enables KV checkpoint reload — V2's actual paper idea.
#
# Prediction: on W5 long-prompt workload, reload (50-100ms KV load) should
# beat NR reprefill (~1.3s prefill for 5k prompts) despite V2's 5s orchestration.
#
# Launch: nohup bash experiments_v2/diag13_reload_mode.sh > /tmp/diag13.log 2>&1 &
# Requires: GPU 0-3 free with ≥22GB each (currently blocked by yxwang).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag13"
mkdir -p "$ROOT"

# CRITICAL: FT_RECOVERY_MODE=reload (the real V2 path, not reprefill)
V2_BASE_RELOAD='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

# Wait for GPU 0-1 and 2-3 to be free (≥22GB each)
echo "[$(date +%H:%M:%S)] diag13: waiting for GPU availability..."
while true; do
    free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    free2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
    free3=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)
    if [ "$free0" -gt 22000 ] && [ "$free1" -gt 22000 ] && [ "$free2" -gt 22000 ] && [ "$free3" -gt 22000 ]; then
        break
    fi
    echo "[$(date +%H:%M:%S)] GPU free: 0=${free0} 1=${free1} 2=${free2} 3=${free3} MB — waiting..."
    sleep 120
done
echo "[$(date +%H:%M:%S)] GPUs free! Starting..."

run_cell() {
    local tag="$1" gpus="$2" port="$3" baseline="$4" wl="$5" load="$6" seed="$7"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" = "NoFT-Reprefill" ]; then
        env_base="FT_RECOVERY_MODE=reprefill"
    else
        env_base="$V2_BASE_RELOAD"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} ${baseline}/${wl}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag13: RELOAD mode (V2 real path) vs NR $(date) ########"

# GPU 0-1: W5 V2 reload + NR
(
    for s in 42 123 456; do
        run_cell "W5_V2_reload"  "0,1" "8500" "Our-System"     "W5_LongDoc" "Moderate" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag13_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: W5 NR baseline fresh + W4 V2 reload
(
    for s in 42 123 456; do
        run_cell "W5_NR_fresh"   "2,3" "8501" "NoFT-Reprefill" "W5_LongDoc" "Moderate" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W4_V2_reload"  "2,3" "8501" "Our-System"     "W4_Mixed"    "Heavy"    "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag13_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag13 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20/diag13"
CFGS = [
    ("W5 NR fresh",         f"{ROOT}/W5_NR_fresh"),
    ("W5 V2_FIXED RELOAD",  f"{ROOT}/W5_V2_reload"),
    ("W4 V2_FIXED RELOAD",  f"{ROOT}/W4_V2_reload"),
]
print(f"{'variant':<30} {'gp':>12} {'comp':>5} {'fg_p95':>9}")
for name, base in CFGS:
    gps, c, f95 = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        print(f"{name:<30} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")

# Key: does V2 reload finally beat NR on W5 F2?
print("\n### Beat-NR check ###")
nr_gp, nr_fg = 30.4, 6115  # W5 historical NR baseline
gps, f95 = [], []
for s in [42,123,456]:
    p = f"{ROOT}/W5_V2_reload/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
if gps:
    mg = st.mean(gps); mf = st.mean(f95)
    gp_w = "✓" if mg >= nr_gp else "✗"
    fg_w = "✓" if mf <= nr_fg else "✗"
    win = "🎯 STRICT WIN" if (mg >= nr_gp and mf <= nr_fg) else ""
    print(f"W5 V2 reload: gp={mg:.1f}[{gp_w}] fg={mf:.0f}[{fg_w}] {win}")
PYEOF
echo "[$(date +%H:%M:%S)] diag13 complete"
