#!/usr/bin/env bash
# Diag9: validate real optimizations (no bypass).
# - baseline_clean: V2 with NO env overrides (trivial_skip default OFF)
# - opt_pool: baseline_clean + 8GB ckpt pool (from config)
# - opt_epoch: + epoch=100ms
# - opt_timecap: + FT_SOLVER_TIME_CAP_MS=100
# - opt_full: all optimizations stacked (pool + epoch + timecap + warm-start + presolve)
#
# Key: ALL variants run Benders MIP every epoch (no trivial-skip, no solver-skip).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag9"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" wl="$5" fault="$6" seed="$7"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} $V2_BASE ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload ${wl} --load Moderate --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% ttft_p50={m.get(\"ttft_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

echo "######## Diag9: real optimizations (solver+ckpt always on) $(date) ########"

# GPU 0-1: no-fault baseline vs opt_full
# GPU 2-3: W5 F2 variants
(
    # W5 no-fault: measures pure steady-state overhead
    for s in 42 123 456; do
        run_cell "W5none_baseline_8GBpool" "0,1" "8500" "" "W5_LongDoc" "none" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W5none_opt_full" "0,1" "8500" "FT_EPOCH_INTERVAL_MS=100 FT_SOLVER_TIME_CAP_MS=100" "W5_LongDoc" "none" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag9_gpu01.log 2>&1 &
P1=$!

(
    # W5 F2: measures fault-recovery scenario
    for s in 42 123 456; do
        run_cell "W5F2_baseline_8GBpool" "2,3" "8501" "" "W5_LongDoc" "F2_Mid" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W5F2_opt_full" "2,3" "8501" "FT_EPOCH_INTERVAL_MS=100 FT_SOLVER_TIME_CAP_MS=100" "W5_LongDoc" "F2_Mid" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag9_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag9 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("W5 NR none (target)",                 f"{ROOT}/w5_ext/W5_NR_none"),
    ("W5 V2 none — OLD (32GB pool)",        f"{ROOT}/w5_ext/W5_V2_none"),
    ("W5 V2 none — 8GB pool (NEW)",         f"{ROOT}/diag9/W5none_baseline_8GBpool"),
    ("W5 V2 none — opt_full (NEW)",         f"{ROOT}/diag9/W5none_opt_full"),
    ("W5 NR F2 (target)",                   f"{ROOT}/w5/W5_NR"),
    ("W5 V2 F2 — OLD (32GB pool)",          f"{ROOT}/w5/W5_V2"),
    ("W5 V2 F2 — 8GB pool (NEW)",           f"{ROOT}/diag9/W5F2_baseline_8GBpool"),
    ("W5 V2 F2 — opt_full (NEW)",           f"{ROOT}/diag9/W5F2_opt_full"),
]
print(f"{'variant':<42} {'gp':>12} {'comp':>5} {'ttft_p50':>9} {'fg_p95':>9}")
for name, base in CFGS:
    gps, c, t50, f95 = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            t50.append(m.get('ttft_p50_ms',0)); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        print(f"{name:<42} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(t50):5.0f} {st.mean(f95):5.0f}")
    else: print(f"{name:<42}   NO DATA")

# Beat-NR scorecard for W5 F2
print("\n### W5 F2 beat-NR scorecard (target: gp >= 30.4 AND fg_p95 <= 6115) ###")
for name, base in CFGS:
    if "F2" not in name or "target" in name: continue
    gps, f95 = [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        mg = st.mean(gps); mf = st.mean(f95)
        gp_w = "✓" if mg >= 30.4 else "✗"
        fg_w = "✓" if mf <= 6115 else "✗"
        strict = " 🎯" if (mg >= 30.4 and mf <= 6115) else ""
        print(f"  {name:<42} gp={mg:5.1f}[{gp_w}] fg={mf:5.0f}[{fg_w}]{strict}")
PYEOF
echo "[$(date +%H:%M:%S)] diag9 complete"
