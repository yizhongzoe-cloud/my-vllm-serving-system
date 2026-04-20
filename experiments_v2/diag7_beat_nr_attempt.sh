#!/usr/bin/env bash
# Diag7: attempt to beat NR on W5 F2 by bypassing solver during fault window.
# 4 variants × 3 seeds on W5_LongDoc Moderate F2_Mid.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag7"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

CAP_BASE='FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=50'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" seed="$6"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" != "NoFT-Reprefill" ]; then
        env_base="$V2_BASE"
    else
        env_base="FT_RECOVERY_MODE=reprefill"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline})" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload W5_LongDoc --load Moderate --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

echo "######## Diag7: beat NR W5 F2 attempts $(date) ########"

# GPU 0-1: X (NoCkpt + FAST_FAILOVER + cap50) + W (FIXED + FAST_FAILOVER)
(
    for s in 42 123 456; do
        run_cell "X_NoCkpt_fastfail" "0,1" "8500" "$CAP_BASE FT_FAST_FAILOVER=1" "Our-System-NoCkpt" "$s" || true
        run_cell "W_FIXED_fastfail"   "0,1" "8500" "$CAP_BASE FT_FAST_FAILOVER=1" "Our-System"        "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 track done" >&2
) > /tmp/diag7_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: Y (NoCkpt + SKIP_SOLVER) + Z (NoCkpt + GATED_SOLVER)
(
    for s in 42 123 456; do
        run_cell "Y_NoCkpt_skipsolver" "2,3" "8501" "$CAP_BASE FT_SKIP_SOLVER=1"    "Our-System-NoCkpt" "$s" || true
        run_cell "Z_NoCkpt_gatedsolver" "2,3" "8501" "$CAP_BASE FT_GATED_SOLVER=1"   "Our-System-NoCkpt" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 track done" >&2
) > /tmp/diag7_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag7 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    ("W5 NR F2 (TARGET to beat)",           f"{ROOT}/w5/W5_NR"),
    ("W5 V2_orig F2",                        f"{ROOT}/w5/W5_V2"),
    ("W5 OS_NoCkpt F2 (diag3)",              f"{ROOT}/diag3/OS_NoCkpt_F2"),
    ("W5 FIXED+NoCkpt F2 (diag6 D2)",        f"{ROOT}/diag6/D_fixed_noCkpt"),
    # NEW in diag7
    ("W5 X: NoCkpt+FastFail+cap50",          f"{ROOT}/diag7/X_NoCkpt_fastfail"),
    ("W5 W: FIXED+FastFail",                 f"{ROOT}/diag7/W_FIXED_fastfail"),
    ("W5 Y: NoCkpt+SkipSolver+cap50",        f"{ROOT}/diag7/Y_NoCkpt_skipsolver"),
    ("W5 Z: NoCkpt+GatedSolver+cap50",       f"{ROOT}/diag7/Z_NoCkpt_gatedsolver"),
]
print(f"{'variant':<42} {'gp':>14} {'comp':>6} {'fg_p95':>10}")
print("-"*85)
for name, base in CFGS:
    gps, c, f95 = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        print(f"{name:<42} {st.mean(gps):6.1f}±{st.stdev(gps) if len(gps)>1 else 0:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")
    else: print(f"{name:<42}   NO DATA")

# Beat-NR scorecard
print("\n### Beat-NR scorecard (target: gp >= 30.4 AND fg_p95 <= 6115) ###")
nr_gp = 30.4
nr_fg = 6115
for name, base in CFGS:
    gps, f95 = [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        mg = st.mean(gps); mf = st.mean(f95)
        gp_win = "✓" if mg >= nr_gp else "✗"
        fg_win = "✓" if mf <= nr_fg else "✗"
        strict = "🎯 WIN" if (mg >= nr_gp and mf <= nr_fg) else ""
        print(f"  {name:<42} gp={mg:5.1f} [{gp_win}] fg={mf:5.0f} [{fg_win}] {strict}")
PYEOF
echo "[$(date +%H:%M:%S)] diag7 complete"
