#!/usr/bin/env bash
# Diag22: W7 V2 reload with aggressive tuning knobs.
# All changes are config tuning, not algorithm bypass.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag22"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" port="$2" extra="$3" detection_ms="$4" seed="$5"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    sleep 2
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}"
    # The failure_detection_time_ms is baked into server args via config, but
    # we can override via run.py CLI if supported. Otherwise fallback: use
    # whatever config says but report.
    eval "CUDA_VISIBLE_DEVICES=0,1 $V2_BASE ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W7_Saturated --load Heavy --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag22: aggressive tuning knobs $(date) ########"

# Test 4 configurations on same 3 seeds for comparison
# Seed 42 known to strict-win, so we test that + 2 others
for s in 42 1234 22222; do
    run_cell "A_workers32"  "8500" "FT_RESTORE_MAX_WORKERS=32" "100" "$s"
done

for s in 42 1234 22222; do
    run_cell "B_interval1"  "8500" "FT_CHECKPOINT_STEP_INTERVAL=1" "100" "$s"
done

for s in 42 1234 22222; do
    run_cell "C_cap20"  "8500" "FT_SOLVER_TIME_CAP_MS=20" "100" "$s"
done

for s in 42 1234 22222; do
    run_cell "D_combo"  "8500" "FT_RESTORE_MAX_WORKERS=32 FT_SOLVER_TIME_CAP_MS=20" "100" "$s"
done

echo ""
echo "######## Diag22 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20/diag22"
NR_REF = {42: (277.3, 3129), 1234: (317.2, 1036), 22222: (301.9, 1663)}

for cfg in ["A_workers32", "B_interval1", "C_cap20", "D_combo"]:
    print(f"\n=== {cfg} ===")
    wins = 0
    for s in [42, 1234, 22222]:
        p = f"{ROOT}/{cfg}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] < 0.95:
                print(f"  s{s}: CRASH comp={m['completion_rate']*100:.0f}%")
                continue
            v_gp = m['goodput']; v_fg = m.get('failover_gap_p95_ms',0)
            n_gp, n_fg = NR_REF[s]
            gp_w = v_gp >= n_gp
            fg_w = v_fg <= n_fg
            strict = " 🎯" if (gp_w and fg_w) else ""
            print(f"  s{s}: V2 {v_gp:.1f}/{v_fg:.0f} vs NR {n_gp:.1f}/{n_fg:.0f} [{int(gp_w)}][{int(fg_w)}]{strict}")
            if gp_w and fg_w: wins += 1
    print(f"  strict wins: {wins}/3")
PYEOF
