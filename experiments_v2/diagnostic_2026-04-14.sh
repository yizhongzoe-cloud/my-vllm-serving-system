#!/usr/bin/env bash
# ============================================================================
# diagnostic_2026-04-14.sh — comprehensive A/B post bug fix
#
# Tests:
#   Phase A (3-seed, the heavyweight baselines):
#     1. OS clean (no prebudget, no warmup)        → true OS-with-restore baseline
#     2. OS-NoCkpt                                  → framework cost without checkpoint
#     3. OS + warmup=50 only                        → warmup contribution
#
#   Phase B (1-seed s42 variants, optimization scan):
#     4. OS + fast_detect alone
#     5. OS + batch_rpc alone
#     6. OS + warmup=50 + fast_detect
#     7. OS + warmup=50 + batch_rpc
#     8. OS + warmup=50 + fast + batch (kitchen sink)
#
# Auto-waits for any prior python experiments. ~50 min wall clock.
# ============================================================================

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

# Wait for prior experiments
echo "[$(date +%H:%M:%S)] Waiting for prior experiments to finish..."
while pgrep -f "python experiments_v2/run.py" > /dev/null 2>&1; do
    sleep 20
done
sleep 5
echo "[$(date +%H:%M:%S)] Starting diagnostic suite"

CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/diagnostic_2026-04-14"
mkdir -p "$ROOT"

# Common env (fixed restore path now actually works).
# IMPORTANT: single-line so inline env vars all apply to the command
# that follows. Multi-line broke CUDA_VISIBLE_DEVICES passthrough.
# Added expandable_segments:True to mitigate OOM from fragmentation
# (batch_rpc + warmup=50 crashed with OOM under aggressive alloc).
COMMON_ENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1'

wait_for_gpu() {
    local gpus="$1"  # e.g. "4,5"
    local required_mib=20000  # need ~20GB free for 8B model + KV
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
        echo "[$(date +%H:%M:%S)] GPU ${gpus} busy (min free ${min_free} MiB < ${required_mib}); waiting 30s..." >&2
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
echo "# Diagnostic — $(date)"
echo "########################################################################"

# ── Phase A: 3-seed baselines ───────────────────────────────────
echo ""
echo "=== Phase A: 3-seed core baselines ==="
(
    # Heavy 3-seed baselines on GPU 4-5
    for seed in 123 456; do
        run_one "os_clean" "Heavy" "$seed" "Our-System" "" "4,5" "8400"
    done
    for seed in 42 123 456; do
        run_one "os_no_ckpt" "Heavy" "$seed" "Our-System-NoCkpt" "" "4,5" "8400"
    done
    for seed in 42 123 456; do
        run_one "os_warmup50" "Heavy" "$seed" "Our-System" "FT_CKPT_WARMUP_TOKENS=50" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Heavy Phase A done" >&2
) &
HP=$!
(
    # Moderate 3-seed baselines on GPU 6-7
    for seed in 123 456; do
        run_one "os_clean" "Moderate" "$seed" "Our-System" "" "6,7" "8500"
    done
    for seed in 42 123 456; do
        run_one "os_no_ckpt" "Moderate" "$seed" "Our-System-NoCkpt" "" "6,7" "8500"
    done
    for seed in 42 123 456; do
        run_one "os_warmup50" "Moderate" "$seed" "Our-System" "FT_CKPT_WARMUP_TOKENS=50" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Moderate Phase A done" >&2
) &
MP=$!
wait $HP $MP

# ── Phase B: 1-seed s42 optimization scan ───────────────────────
echo ""
echo "=== Phase B: 1-seed s42 optimization variants ==="
for cell_pair in "Heavy:4,5:8400" "Moderate:6,7:8500"; do
    cell=${cell_pair%%:*}
    rest=${cell_pair#*:}
    gpu=${rest%:*}
    port=${rest#*:}
    (
        run_one "os_fast_detect_only"      "$cell" "42" "Our-System" "FT_FAST_FAULT_DETECT=1" "$gpu" "$port"
        run_one "os_batch_rpc_only"        "$cell" "42" "Our-System" "FT_RESTORE_BATCH_RPC=1" "$gpu" "$port"
        run_one "os_w50_fast"              "$cell" "42" "Our-System" "FT_CKPT_WARMUP_TOKENS=50 FT_FAST_FAULT_DETECT=1" "$gpu" "$port"
        run_one "os_w50_batch"             "$cell" "42" "Our-System" "FT_CKPT_WARMUP_TOKENS=50 FT_RESTORE_BATCH_RPC=1" "$gpu" "$port"
        run_one "os_w50_fast_batch"        "$cell" "42" "Our-System" "FT_CKPT_WARMUP_TOKENS=50 FT_FAST_FAULT_DETECT=1 FT_RESTORE_BATCH_RPC=1" "$gpu" "$port"
        echo "[$(date +%H:%M:%S)] ${cell} Phase B done" >&2
    ) &
done
wait

echo ""
echo "########################################################################"
echo "# Diagnostic complete — $(date)"
echo "########################################################################"

# Final summary
python3 << 'PYEOF'
import json, glob, os, statistics

def load(f):
    return json.load(open(f)) if os.path.exists(f) else None

def mean_std(values):
    if not values: return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0)

print("\n=== Diagnostic Summary ===\n")

# NR references (3-seed)
nr_h = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42, 123, 456]]
nr_m = [load(f"results_v2/8B/overnight_opt/noft_reprefill/W1_Chat/Moderate/F2_Mid/{s}/metrics.json") for s in [42, 123, 456]]

for cell, nr_seeds in [("Heavy", nr_h), ("Moderate", nr_m)]:
    print(f"\n--- {cell}/F2_Mid ---")
    print(f"{'config':<25} {'gp mean ±std':>16} {'fg_p95 mean ±std':>20} {'n':>3}")

    # NR
    gps = [m['goodput'] for m in nr_seeds if m]
    fgs = [m.get('failover_gap_p95_ms', 0) for m in nr_seeds if m]
    if gps:
        print(f"  {'NR':<23} {statistics.mean(gps):8.1f} ±{statistics.stdev(gps):5.0f}    {statistics.mean(fgs):10.0f} ±{statistics.stdev(fgs):5.0f}    {len(gps)}")

    # 3-seed configs (look in clean_test for s42 os_clean, diagnostic for the rest)
    for tag in ["os_clean", "os_no_ckpt", "os_warmup50"]:
        runs = []
        for s in [42, 123, 456]:
            if tag == "os_clean" and s == 42:
                m = load(f"results_v2/8B/clean_test/{cell}_F2_Mid/os_clean/metrics.json")
            else:
                m = load(f"results_v2/8B/diagnostic_2026-04-14/{tag}/{cell}_F2_Mid/{s}/metrics.json")
            if m and m.get('completion_rate', 0) >= 0.95:
                runs.append((m['goodput'], m.get('failover_gap_p95_ms', 0)))
        if runs:
            gp_m, gp_s = mean_std([r[0] for r in runs])
            fg_m, fg_s = mean_std([r[1] for r in runs])
            print(f"  {tag:<23} {gp_m:8.1f} ±{gp_s:5.0f}    {fg_m:10.0f} ±{fg_s:5.0f}    {len(runs)}")

    # 1-seed s42 variants
    for tag in ["os_fast_detect_only", "os_batch_rpc_only", "os_w50_fast", "os_w50_batch", "os_w50_fast_batch"]:
        m = load(f"results_v2/8B/diagnostic_2026-04-14/{tag}/{cell}_F2_Mid/42/metrics.json")
        if m and m.get('completion_rate', 0) >= 0.95:
            print(f"  {tag:<23} {m['goodput']:8.1f}            {m.get('failover_gap_p95_ms',0):10.0f}              1")
PYEOF
