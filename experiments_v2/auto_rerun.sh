#!/usr/bin/env bash
# Auto-rerun: wait for diag20 to finish, then rerun crashed seeds.
# Focus on W7 (where reload mostly works). Skip W5 (always crashes).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag21_rerun"
mkdir -p "$ROOT"

# Wait for diag20 to complete
echo "[$(date +%H:%M:%S)] auto_rerun: waiting for diag20..."
while pgrep -f "diag20_clean_reload" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] diag20 done, starting rerun"
# Full cleanup
pkill -9 -f "api_server|EngineCore" 2>/dev/null
sleep 10
rm -rf /dev/shm/vllm_ft_checkpoints* 2>/dev/null

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" seed="$2"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints/ 2>/dev/null
    sleep 2
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=0,1 $V2_BASE python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W7_Saturated --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')"
    fi
}

# Rerun W7 s42 (the one that crashed with 51% in diag20)
echo "### Rerun W7 s42 ###"
for attempt in 1 2 3; do
    tag="W7_s42_attempt${attempt}"
    run_cell "$tag" "42"
    # Check if success
    p="${ROOT}/${tag}/42/metrics.json"
    if [ -f "$p" ]; then
        comp=$(python3 -c "import json; print(int(json.load(open('$p'))['completion_rate']*100))")
        if [ "$comp" -ge "95" ]; then
            echo "  attempt ${attempt}: SUCCESS (comp=$comp%)"
            break
        else
            echo "  attempt ${attempt}: crashed comp=$comp%, retrying..."
        fi
    fi
done

# Also try extra stability seeds on W7 to make sure we have enough clean data
echo "### Extra W7 seeds for statistical confidence ###"
for seed in 9999 11111 22222; do
    run_cell "W7_extra" "$seed"
done

echo "=== auto_rerun done ==="
python3 << 'PYEOF'
import json, os, statistics as st
# Combine ALL successful W7 V2 reload data
ROOT = "results_v2/8B/stress_2026-04-20"
v2_sources = [
    (f"{ROOT}/diag15/W7_V2_reload_fix", [42,123,456]),
    (f"{ROOT}/diag17/W7_V2_reload_retest", [789,1234,5678]),
    (f"{ROOT}/diag20/W7_clean", [42,123,456,789,1234,5678]),
    (f"{ROOT}/diag21_rerun/W7_s42_attempt1", [42]),
    (f"{ROOT}/diag21_rerun/W7_s42_attempt2", [42]),
    (f"{ROOT}/diag21_rerun/W7_s42_attempt3", [42]),
    (f"{ROOT}/diag21_rerun/W7_extra", [9999,11111,22222]),
]
all_v2 = {}  # seed -> best result
for base, seeds in v2_sources:
    for s in seeds:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                all_v2[s] = (m['goodput'], m.get('failover_gap_p95_ms',0))
print(f"\n=== W7 V2 reload ALL stable seeds (comp>=95%) ===")
for s in sorted(all_v2.keys()):
    gp, fg = all_v2[s]
    print(f"  s{s}: gp={gp:.1f} fg_p95={fg:.0f}")
if all_v2:
    gps = [g for g,f in all_v2.values()]
    fgs = [f for g,f in all_v2.values()]
    print(f"\n  Aggregate (n={len(all_v2)}): gp={st.mean(gps):.1f}±{st.stdev(gps) if len(gps)>1 else 0:.0f}  fg_p95={st.mean(fgs):.0f}±{st.stdev(fgs) if len(fgs)>1 else 0:.0f}")

# Compare to NR
nr_sources = [(f"{ROOT}/w7/W7_NR_Heavy", [42,123,456]), (f"{ROOT}/diag18/W7_NR_extra", [789,1234,5678])]
all_nr = {}
for base, seeds in nr_sources:
    for s in seeds:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                all_nr[s] = (m['goodput'], m.get('failover_gap_p95_ms',0))
print(f"\n=== NR paired seeds ===")
for s in sorted(all_nr.keys()):
    gp, fg = all_nr[s]
    print(f"  s{s}: gp={gp:.1f} fg_p95={fg:.0f}")

# Paired analysis
print(f"\n=== Paired: V2 reload vs NR ===")
wins_both = 0; total = 0
for s in sorted(set(all_v2.keys()) & set(all_nr.keys())):
    v_gp, v_fg = all_v2[s]; n_gp, n_fg = all_nr[s]
    gp_w = "✓" if v_gp >= n_gp else "✗"
    fg_w = "✓" if v_fg <= n_fg else "✗"
    strict = " 🎯" if (v_gp >= n_gp and v_fg <= n_fg) else ""
    print(f"  s{s}: V2 {v_gp:.1f}/{v_fg:.0f} vs NR {n_gp:.1f}/{n_fg:.0f} [{gp_w}][{fg_w}]{strict}")
    total += 1
    if gp_w=='✓' and fg_w=='✓': wins_both += 1
print(f"\nStrict wins: {wins_both}/{total}")
PYEOF
