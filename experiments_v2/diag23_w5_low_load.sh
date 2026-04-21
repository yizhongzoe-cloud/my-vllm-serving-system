#!/usr/bin/env bash
# Diag23: W5 V2 reload with LOWER load (Light=0.5 RPS vs Moderate=1.0 RPS).
# Hypothesis: reduced concurrency prevents KV pressure → no preemption → no
# attention-kernel cuda_assert after reload.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag23"
mkdir -p "$ROOT"

# Wait for diag22
while pgrep -f "diag22_tuning" > /dev/null 2>&1; do sleep 30; done
sleep 10
pkill -9 -f "api_server" 2>/dev/null
sleep 5
rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" baseline="$2" load="$3" seed="$4"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    sleep 2
    local env_base=""
    if [ "$baseline" = "NoFT-Reprefill" ]; then
        env_base="FT_RECOVERY_MODE=reprefill"
    else
        env_base="$V2_BASE"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} ${baseline} ${load}"
    eval "CUDA_VISIBLE_DEVICES=0,1 ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload W5_LongDoc --load ${load} --fault F2_Mid \
        --seed ${seed} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag23: W5 low-load reload $(date) ########"

# Light RPS=0.5 — half the concurrency
for s in 42 123 456; do
    run_cell "W5_V2_Light"  "Our-System"     "Light" "$s" || true
done

# NR Light for paired comparison
for s in 42 123 456; do
    run_cell "W5_NR_Light"  "NoFT-Reprefill" "Light" "$s" || true
done

echo ""
echo "######## Diag23 summary ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20/diag23"
def get(tag, s):
    p = f"{ROOT}/{tag}/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        return (m['goodput'], m.get('failover_gap_p95_ms',0), m['completion_rate']*100)
    return None

print("=== W5 Light load (V2 reload vs NR) ===")
wins = 0
for s in [42, 123, 456]:
    v = get("W5_V2_Light", s)
    n = get("W5_NR_Light", s)
    if v and n:
        print(f"  s{s}: V2 gp={v[0]:.1f}/fg={v[1]:.0f}/comp={v[2]:.0f}%  vs  NR gp={n[0]:.1f}/fg={n[1]:.0f}/comp={n[2]:.0f}%")
        if v[2] >= 95 and n[2] >= 95:
            gp_w = v[0] >= n[0]
            fg_w = v[1] <= n[1]
            if gp_w and fg_w:
                print(f"    🎯 STRICT WIN on s{s}")
                wins += 1
            elif gp_w or fg_w:
                print(f"    partial win (gp={int(gp_w)} fg={int(fg_w)})")
print(f"\ntotal strict wins: {wins}/3")
PYEOF
