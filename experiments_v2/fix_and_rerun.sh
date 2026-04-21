#!/usr/bin/env bash
# ============================================================================
# fix_and_rerun.sh — comprehensive rerun post bug fixes
#
# Changes from diagnostic_2026-04-14:
#   + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (OOM mitigation)
#   + Added os_prebudget and os_w50_prebudget configs to validate recovery speedup
#   + Cleaner test matrix (3-seed for all key configs)
#
# Targets: find config that BEATS NR on Heavy + Moderate (goodput + fg_p95).
# ============================================================================

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

echo "[$(date +%H:%M:%S)] Waiting for prior experiments..."
while pgrep -f "python experiments_v2/run.py" > /dev/null 2>&1; do
    sleep 20
done
sleep 5
echo "[$(date +%H:%M:%S)] Starting fix+rerun"

CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/fix_rerun_2026-04-14"
mkdir -p "$ROOT"

# Env with OOM mitigation; single-line critical for passthrough
COMMON_ENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1'

wait_for_gpu() {
    local gpus="$1"
    local required_mib=20000
    while true; do
        local min_free=1000000
        for g in ${gpus//,/ }; do
            local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
            if [ -n "$free" ] && [ "$free" -lt "$min_free" ]; then
                min_free=$free
            fi
        done
        if [ "$min_free" -ge "$required_mib" ]; then
            return 0
        fi
        sleep 30
    done
}

run_one() {
    local tag="$1" cell="$2" seed="$3" baseline="$4" extra_env="${5:-}" gpu="$6" port="$7"
    local out="${ROOT}/${tag}/${cell}_F2_Mid/${seed}"
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  [skip] ${tag}/${cell}/s${seed}: gp={m[\"goodput\"]:.1f}')" 2>/dev/null || true
        return 0
    fi
    wait_for_gpu "${gpu}"
    mkdir -p "$out" && rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] ${tag}/${cell}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpu} ${COMMON_ENV} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} --baseline ${baseline} \
        --workload W1_Chat --load ${cell} --fault F2_Mid \
        --seed ${seed} --port ${port} \
        --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/${cell}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  ${tag}/${cell}/s${seed}: FAILED" >&2
}

echo "########################################################################"
echo "# Fix + Rerun — $(date)"
echo "########################################################################"

# ── Phase A: 3-seed targeted configs ────────────────────────────
# Only what matters: os_clean, os_warmup50, os_prebudget, os_prebudget_warmup50
echo ""
echo "=== Phase A: 3-seed configs (with OOM mitigation) ==="
(
    for seed in 42 123 456; do
        run_one "os_clean"             "Heavy" "$seed" "Our-System" "" "4,5" "8400"
        run_one "os_warmup50"          "Heavy" "$seed" "Our-System" "FT_CKPT_WARMUP_TOKENS=50" "4,5" "8400"
        run_one "os_prebudget"         "Heavy" "$seed" "Our-System" "FT_RECOVERY_PREBUDGET=1" "4,5" "8400"
        run_one "os_prebudget_w50"     "Heavy" "$seed" "Our-System" "FT_RECOVERY_PREBUDGET=1 FT_CKPT_WARMUP_TOKENS=50" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Heavy Phase A done" >&2
) &
HP=$!
(
    for seed in 42 123 456; do
        run_one "os_clean"             "Moderate" "$seed" "Our-System" "" "6,7" "8500"
        run_one "os_warmup50"          "Moderate" "$seed" "Our-System" "FT_CKPT_WARMUP_TOKENS=50" "6,7" "8500"
        run_one "os_prebudget"         "Moderate" "$seed" "Our-System" "FT_RECOVERY_PREBUDGET=1" "6,7" "8500"
        run_one "os_prebudget_w50"     "Moderate" "$seed" "Our-System" "FT_RECOVERY_PREBUDGET=1 FT_CKPT_WARMUP_TOKENS=50" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Moderate Phase A done" >&2
) &
MP=$!
wait $HP $MP

echo ""
echo "########################################################################"
echo "# Fix + rerun complete — $(date)"
echo "########################################################################"

# Summary
python3 << 'PYEOF'
import json, glob, os, statistics
def load(f): return json.load(open(f)) if os.path.exists(f) else None
def stats(v): return (statistics.mean(v), statistics.stdev(v) if len(v)>1 else 0, len(v)) if v else (None,None,0)

print("\n=== Fix+Rerun Summary ===\n")
nr_heavy = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42,123,456]]
nr_mod = [load(f"results_v2/8B/overnight_opt/noft_reprefill/W1_Chat/Moderate/F2_Mid/{s}/metrics.json") for s in [42,123,456]]

for cell, nrs in [("Heavy", nr_heavy), ("Moderate", nr_mod)]:
    print(f"\n--- {cell}/F2_Mid ---")
    nr_gps = [m['goodput'] for m in nrs if m]
    nr_fgs = [m.get('failover_gap_p95_ms', 0) for m in nrs if m]
    nr_mean, nr_std, _ = stats(nr_gps)
    nr_fg_m, nr_fg_s, _ = stats(nr_fgs)
    print(f"  {'NR':<28} gp={nr_mean:6.1f}±{nr_std:3.0f}  fg={nr_fg_m:5.0f}±{nr_fg_s:4.0f}  Δ=    0")
    for tag in ["os_clean", "os_warmup50", "os_prebudget", "os_prebudget_w50"]:
        rows = []
        for s in [42, 123, 456]:
            m = load(f"results_v2/8B/fix_rerun_2026-04-14/{tag}/{cell}_F2_Mid/{s}/metrics.json")
            if m and m.get('completion_rate', 0) >= 0.95:
                rows.append((m['goodput'], m.get('failover_gap_p95_ms', 0), m['completion_rate']*100))
        if rows:
            gp_m, gp_s, n = stats([r[0] for r in rows])
            fg_m, fg_s, _ = stats([r[1] for r in rows])
            delta = gp_m - nr_mean
            print(f"  {tag:<28} gp={gp_m:6.1f}±{gp_s:3.0f}  fg={fg_m:5.0f}±{fg_s:4.0f}  Δ={delta:+6.1f}  (n={n})")
        else:
            print(f"  {tag:<28} (no clean data)")
PYEOF
