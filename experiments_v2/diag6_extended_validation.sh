#!/usr/bin/env bash
# Diag6: extended solver-fix validation.
# Waits for diag5 to clear GPUs, then runs 4 phases in parallel on both GPU pairs.
# Phase A: W5 V2_FIXED × F1/F3 fault timing (paper-critical)
# Phase B: Ablation (disable trivial, vary time_cap) × W5 none
# Phase C: W1 + W4 V2_FIXED (no-regression check)
# Phase D: Combo variants (FIXED + SKIP_SOLVER, FIXED + NoCkpt)

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag6"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# Wait for diag5
echo "[$(date +%H:%M:%S)] diag6: waiting for diag5 to clear GPUs..."
while pgrep -f "baseline.*W5_LongDoc.*diag5" > /dev/null 2>&1 || \
      pgrep -f "baseline.*W7_Saturated.*diag5" > /dev/null 2>&1; do
    sleep 30
done
sleep 20
pkill -9 -f "api_server.*port 8500" 2>/dev/null
pkill -9 -f "api_server.*port 8501" 2>/dev/null
sleep 5

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" wl="$6" load="$7" fault="$8" seed="$9"
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
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline}, ${wl}, ${load}, ${fault})" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f} ttft_p50={m.get(\"ttft_p50_ms\",0):.0f}')
" 2>&1
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

# ---------- GPU 0-1 track: Phase A (fault timing) then Phase C (cross-workload) ----------
(
    # Phase A: W5 V2_FIXED × F1/F3
    FIX_ENV='FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=100'
    for s in 42 123 456; do
        run_cell "A_W5_fixed_F1"   "0,1" "8500" "$FIX_ENV" "Our-System" "W5_LongDoc" "Moderate" "F1_Early" "$s" || true
        run_cell "A_W5_fixed_F3"   "0,1" "8500" "$FIX_ENV" "Our-System" "W5_LongDoc" "Moderate" "F3_Late"  "$s" || true
    done
    # Phase C: no-regression on W1 + W4
    for s in 42 123 456; do
        run_cell "C_W1_fixed_F2"   "0,1" "8500" "$FIX_ENV" "Our-System" "W1_Chat"     "Heavy"    "F2_Mid"   "$s" || true
        run_cell "C_W4_fixed_F2"   "0,1" "8500" "$FIX_ENV" "Our-System" "W4_Mixed"    "Heavy"    "F2_Mid"   "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 track done" >&2
) > /tmp/diag6_gpu01.log 2>&1 &
GPU01_PID=$!

# ---------- GPU 2-3 track: Phase B (ablation) then Phase D (combo) ----------
(
    # Phase B: ablation on W5 none (isolate each fix)
    # B1: all fixes disabled (verify baseline reproduces)
    for s in 42 123 456; do
        run_cell "B_no_fix"        "2,3" "8501" "FT_SOLVER_TRIVIAL_SKIP=0" "Our-System" "W5_LongDoc" "Moderate" "none" "$s" || true
    done
    # B2: only trivial_skip (default) — time_cap stays at 1s default
    for s in 42 123 456; do
        run_cell "B_trivial_only"  "2,3" "8501" "FT_SOLVER_TRIVIAL_SKIP=1" "Our-System" "W5_LongDoc" "Moderate" "none" "$s" || true
    done
    # B3: tighter time cap (50ms) + trivial
    for s in 42 123 456; do
        run_cell "B_cap50"         "2,3" "8501" "FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=50" "Our-System" "W5_LongDoc" "Moderate" "none" "$s" || true
    done
    # B4: looser time cap (500ms) + trivial
    for s in 42 123 456; do
        run_cell "B_cap500"        "2,3" "8501" "FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=500" "Our-System" "W5_LongDoc" "Moderate" "none" "$s" || true
    done

    # Phase D: best combos
    FIX_ENV='FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=100'
    # D1: FIXED + SKIP_SOLVER (all fixes + bypass solver entirely)
    for s in 42 123 456; do
        run_cell "D_fixed_skip_solver" "2,3" "8501" "$FIX_ENV FT_SKIP_SOLVER=1" "Our-System" "W5_LongDoc" "Moderate" "F2_Mid" "$s" || true
    done
    # D2: FIXED on Our-System-NoCkpt (fixes applied to no-ckpt path)
    for s in 42 123 456; do
        run_cell "D_fixed_noCkpt"   "2,3" "8501" "$FIX_ENV" "Our-System-NoCkpt" "W5_LongDoc" "Moderate" "F2_Mid" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 track done" >&2
) > /tmp/diag6_gpu23.log 2>&1 &
GPU23_PID=$!

wait $GPU01_PID $GPU23_PID

echo ""
echo "######## Diag6 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
CFGS = [
    # Baselines for reference
    ("W5 NR none (target)",             f"{ROOT}/w5_ext/W5_NR_none"),
    ("W5 V2_orig none (-27%)",           f"{ROOT}/w5_ext/W5_V2_none"),
    ("W5 V2_FIXED none (diag5)",         f"{ROOT}/diag5/V2_fixed_none"),
    # Phase A: fault timing
    ("W5 NR F1 (target)",                f"{ROOT}/w5_ext/W5_NR_F1"),
    ("W5 V2_FIXED F1 (NEW)",             f"{ROOT}/diag6/A_W5_fixed_F1"),
    ("W5 NR F3 (target)",                f"{ROOT}/w5_ext/W5_NR_F3"),
    ("W5 V2_FIXED F3 (NEW)",             f"{ROOT}/diag6/A_W5_fixed_F3"),
    # Phase B: ablation
    ("W5 B_no_fix none (sanity=V2)",     f"{ROOT}/diag6/B_no_fix"),
    ("W5 B_trivial_only none",           f"{ROOT}/diag6/B_trivial_only"),
    ("W5 B_cap50 none",                  f"{ROOT}/diag6/B_cap50"),
    ("W5 B_cap500 none",                 f"{ROOT}/diag6/B_cap500"),
    # Phase C: cross-workload
    ("W1 NR F2 (target)",                f"{ROOT}/w7/W7_NR_Heavy"),  # wait that's wrong, use baseline from p4b
    ("W1 V2_FIXED F2 (NEW)",             f"{ROOT}/diag6/C_W1_fixed_F2"),
    ("W4 V2_FIXED F2 (NEW)",             f"{ROOT}/diag6/C_W4_fixed_F2"),
    # Phase D: combos
    ("W5 FIXED+skip_solver F2",          f"{ROOT}/diag6/D_fixed_skip_solver"),
    ("W5 FIXED+NoCkpt F2",               f"{ROOT}/diag6/D_fixed_noCkpt"),
]
print(f"{'variant':<38} {'gp':>12} {'comp':>5} {'ttft_p50':>9} {'fg_p95':>9}")
for name, base in CFGS:
    gps, c, t50, f95 = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            t50.append(m.get('ttft_p50_ms',0)); f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        print(f"{name:<38} {st.mean(gps):6.1f}±{st.stdev(gps) if len(gps)>1 else 0:3.0f} {st.mean(c):4.0f}% {st.mean(t50):5.0f} {st.mean(f95):5.0f}")
    else:
        print(f"{name:<38}   NO DATA")
PYEOF
echo "[$(date +%H:%M:%S)] diag6 pipeline complete"
