#!/usr/bin/env bash
# Diag8: vary FT_SOLVER_TRIVIAL_MAX threshold, combined with best X config.
# Tests hypothesis: higher trivial threshold helps W5 long-prompt fault window.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag8"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# X-base: NoCkpt + FastFail + cap50 + trivial (best variant from diag7)
X_BASE='FT_SOLVER_TRIVIAL_SKIP=1 FT_SOLVER_TIME_CAP_MS=50 FT_FAST_FAILOVER=1'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" seed="$5"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} $V2_BASE ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System-NoCkpt \
        --workload W5_LongDoc --load Moderate --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag8: trivial threshold sweep $(date) ########"

# GPU 0-1: X_thr20 + X_thr40
(
    for s in 42 123 456; do
        run_cell "X_trivial20" "0,1" "8500" "$X_BASE FT_SOLVER_TRIVIAL_MAX=20" "$s" || true
        run_cell "X_trivial40" "0,1" "8500" "$X_BASE FT_SOLVER_TRIVIAL_MAX=40" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag8_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: extra NR seeds for better stats + more seeds on X_best
(
    # Extra NR seeds to check if NR s123 "lucky" holds with more trials
    for s in 789 1234 5678; do
        out="$ROOT/NR_extra_seed/${s}"
        mkdir -p "$out"
        [ -f "${out}/metrics.json" ] && continue
        rm -rf /dev/shm/vllm_ft_checkpoints_8501 2>/dev/null
        echo "[$(date +%H:%M:%S)] NR_extra/s${s}" >&2
        CUDA_VISIBLE_DEVICES=2,3 FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline NoFT-Reprefill \
            --workload W5_LongDoc --load Moderate --fault F2_Mid \
            --seed ${s} --port 8501 --output-dir "${out}" > "${out}/stdout.log" 2>&1
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  NR/s${s}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1 || true
    done
    # Extra X seeds with best diag7 config
    for s in 789 1234 5678; do
        run_cell "X_extra_seed" "2,3" "8501" "$X_BASE" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag8_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag8 final $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"
print("=== W5 F2 full beat-NR scorecard (6 seeds where available) ===\n")
nr_gp, nr_fg = 30.4, 6115

# Merge 3-seed and 6-seed data
def agg(base_list):
    gps, f95, comps = [], [], []
    for base in base_list:
        for s in [42,123,456,789,1234,5678]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                gps.append(m['goodput']); f95.append(m.get('failover_gap_p95_ms',0))
                comps.append(m['completion_rate']*100)
    return gps, f95, comps

rows = [
    ("NR F2 (3+3 seeds)",             [f"{ROOT}/w5/W5_NR", f"{ROOT}/diag8/NR_extra_seed"]),
    ("X orig (diag7, 3 seeds)",       [f"{ROOT}/diag7/X_NoCkpt_fastfail"]),
    ("X + 3 extra seeds (6 total)",    [f"{ROOT}/diag7/X_NoCkpt_fastfail", f"{ROOT}/diag8/X_extra_seed"]),
    ("X_trivial20 (3 seeds)",          [f"{ROOT}/diag8/X_trivial20"]),
    ("X_trivial40 (3 seeds)",          [f"{ROOT}/diag8/X_trivial40"]),
]
print(f"{'variant':<38} {'n':>3} {'gp':>14} {'comp':>6} {'fg_p95':>14}")
print("-"*85)
for name, bases in rows:
    gps, f95, comps = agg(bases)
    n = len(gps)
    if not gps:
        print(f"{name:<38} NO DATA")
        continue
    mg = st.mean(gps); mf = st.mean(f95); mc = st.mean(comps)
    gs = st.stdev(gps) if n>1 else 0
    fs = st.stdev(f95) if n>1 else 0
    gp_w = "✓" if mg >= nr_gp else "✗"
    fg_w = "✓" if mf <= nr_fg else "✗"
    strict = " 🎯" if (mg >= nr_gp and mf <= nr_fg) else ""
    print(f"{name:<38} {n:3d} {mg:6.1f}±{gs:3.0f}[{gp_w}] {mc:4.0f}% {mf:5.0f}±{fs:5.0f}[{fg_w}]{strict}")
PYEOF
echo "[$(date +%H:%M:%S)] diag8 complete"
