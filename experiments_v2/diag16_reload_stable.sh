#!/usr/bin/env bash
# Diag16: reload mode with lower memory pressure (gpu_memory_utilization=0.85).
# Hypothesis: OOM + preemption crashes at high mem pressure prevent reload
# mode from demonstrating its advantage. Lower mem pressure should stabilize
# more seeds, letting us measure true reload vs NR performance.
#
# Focus: W7 (already shows 100% comp w/ OOB fix) and W1 (strict win on 1 seed)
# with 3 seeds each. If stable, add W4/W5 variants.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag16"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

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
        env_base="$V2_BASE"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    # Use safe_mem config (gpu_memory_utilization=0.85)
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b_safe_mem.yaml --baseline ${baseline} \
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

echo "######## Diag16: reload @ gpu_mem=0.85 $(date) ########"

# GPU 0-1: W7 + W1 × V2 reload × 3 seeds
(
    for s in 42 123 456; do
        run_cell "W7_reload_safe" "0,1" "8500" "Our-System" "W7_Saturated" "Heavy" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W1_reload_safe" "0,1" "8500" "Our-System" "W1_Chat" "Heavy" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag16_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: W4 + W5 × V2 reload × 3 seeds + NR baselines
(
    for s in 42 123 456; do
        run_cell "W4_reload_safe" "2,3" "8501" "Our-System" "W4_Mixed" "Heavy" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W5_reload_safe" "2,3" "8501" "Our-System" "W5_LongDoc" "Moderate" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag16_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag16 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20/diag16"
ORIG = "results_v2/8B/stress_2026-04-20"

print(f"{'variant':<34} {'gp':>12} {'comp':>5} {'fg_p95':>9}")
for name, base, seeds in [
    ("W7 NR hist (target 263.6/3236)", f"{ORIG}/w7/W7_NR_Heavy", [42,123,456]),
    ("W7 V2 reload SAFE",               f"{ROOT}/W7_reload_safe", [42,123,456]),
    ("W1 V2 reload SAFE",               f"{ROOT}/W1_reload_safe", [42,123,456]),
    ("W4 V2 reload SAFE",               f"{ROOT}/W4_reload_safe", [42,123,456]),
    ("W5 V2 reload SAFE",               f"{ROOT}/W5_reload_safe", [42,123,456]),
]:
    gps, c, f95 = [], [], []
    for s in seeds:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        print(f"{name:<34} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")

# Scorecard comp>=80% filter
print("\n=== Beat-NR scorecard (comp>=80%) ===")
for tgt, base, nr_gp, nr_fg in [
    ("W7", f"{ROOT}/W7_reload_safe", 263.6, 3236),
    ("W1", f"{ROOT}/W1_reload_safe", 263.6, 3220),
    ("W4", f"{ROOT}/W4_reload_safe", 253.6, 1935),
    ("W5", f"{ROOT}/W5_reload_safe", 30.4, 6115),
]:
    gps, f95 = [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.80:
                gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        mg = st.mean(gps); mf = st.mean(f95)
        gp_w = "✓" if mg >= nr_gp else "✗"
        fg_w = "✓" if mf <= nr_fg else "✗"
        strict = " 🎯 STRICT WIN" if (mg >= nr_gp and mf <= nr_fg) else ""
        print(f"  {tgt}: n={len(gps)} gp={mg:.1f}[{gp_w}] fg={mf:.0f}[{fg_w}]{strict}")
PYEOF
