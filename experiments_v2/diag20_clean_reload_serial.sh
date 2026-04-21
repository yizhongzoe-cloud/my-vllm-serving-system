#!/usr/bin/env bash
# Diag20: CLEAN reload serial on one GPU pair at a time.
# Avoids parallel cleanup race on /dev/shm/vllm_ft_checkpoints/.
# This is the REAL test of V2 reload with working checkpoints.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag20"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" gpus="$2" port="$3" baseline="$4" wl="$5" load="$6" seed="$7"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    # Safe cleanup: only done when NO other experiment is writing
    rm -rf /dev/shm/vllm_ft_checkpoints/ 2>/dev/null
    sleep 2
    local env_base=""
    if [ "$baseline" = "NoFT-Reprefill" ]; then
        env_base="FT_RECOVERY_MODE=reprefill"
    else
        env_base="$V2_BASE"
    fi
    local shm_free_mb=$(df -m /dev/shm | awk 'NR==2 {print $4}')
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (shm_free=${shm_free_mb}MB)" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
gp = m['goodput']
comp = m['completion_rate'] * 100
fg = m.get('failover_gap_p95_ms', 0)
enospc = 0
try:
    with open('${out}/server.log') as f:
        for line in f:
            if 'No space left' in line:
                enospc += 1
except:
    pass
print(f'  s${seed}: gp={gp:.1f} comp={comp:.0f}% fg_p95={fg:.0f} enospc={enospc}')
" 2>&1
    fi
}

echo "######## Diag20: SERIAL clean reload $(date) ########"
df -h /dev/shm | tail -1

# Run W7 first on GPU 0-1, sequentially
for s in 42 123 456 789 1234 5678; do
    run_cell "W7_clean" "0,1" "8500" "Our-System" "W7_Saturated" "Heavy" "$s" || true
done

# Then W5 on GPU 0-1 (still sequential to avoid race)
for s in 42 123 456 789 1234 5678; do
    run_cell "W5_clean" "0,1" "8500" "Our-System" "W5_LongDoc" "Moderate" "$s" || true
done

echo ""
echo "######## Diag20 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"

# W7 paired
print("=== W7 V2 reload CLEAN vs NR (paired per-seed) ===")
v2_base = f"{ROOT}/diag20/W7_clean"
nr_seeds = {}
for base, seeds in [(f"{ROOT}/w7/W7_NR_Heavy", [42,123,456]), (f"{ROOT}/diag18/W7_NR_extra", [789,1234,5678])]:
    for s in seeds:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                nr_seeds[s] = (m['goodput'], m.get('failover_gap_p95_ms',0))

wins_both = 0; total = 0
for s in [42,123,456,789,1234,5678]:
    p = f"{v2_base}/{s}/metrics.json"
    if os.path.exists(p) and s in nr_seeds:
        m = json.load(open(p))
        if m['completion_rate'] < 0.95: continue
        v_gp, v_fg = m['goodput'], m.get('failover_gap_p95_ms',0)
        n_gp, n_fg = nr_seeds[s]
        gp_w = "✓" if v_gp >= n_gp else "✗"
        fg_w = "✓" if v_fg <= n_fg else "✗"
        strict = " 🎯" if (v_gp >= n_gp and v_fg <= n_fg) else ""
        print(f"  s{s}: V2 {v_gp:.1f}/{v_fg:.0f} vs NR {n_gp:.1f}/{n_fg:.0f} [{gp_w}][{fg_w}]{strict}")
        total += 1
        if gp_w=='✓' and fg_w=='✓': wins_both += 1
print(f"\nPaired wins: {wins_both}/{total}")

# W5
print("\n=== W5 V2 reload CLEAN ===")
for s in [42,123,456,789,1234,5678]:
    p = f"{ROOT}/diag20/W5_clean/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"  s{s}: gp={m['goodput']:.1f} comp={m['completion_rate']*100:.0f}% fg_p95={m.get('failover_gap_p95_ms',0):.0f}")
PYEOF
