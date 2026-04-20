#!/usr/bin/env bash
# Stress test pipeline 2026-04-20.
# - Gates on W5 smoke (NR @ RPS=1.0). If smoke produces valid metrics.json, proceeds.
# - Runs W5 × {NR, V2, V2+skip_solver} × {42,123,456} on GPU 0-1
# - Parallel W7 × {NR, V2} × {42,123,456} on GPU 2-3
# - Auto-debug: failed cells retry at lower RPS; continues past individual failures.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/stress_2026-04-20"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

# ------------------------ helpers ------------------------

run_cell() {
    # $1=tag  $2=gpus  $3=port  $4=extra_env  $5=baseline  $6=workload  $7=load  $8=fault  $9=seed  ${10}=root_tag
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" wl="$6" load="$7" fault="$8" seed="$9" root_tag="${10}"
    local out="${ROOT}/${root_tag}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    local env_base=""
    if [ "$baseline" != "NoFT-Reprefill" ]; then
        env_base="$V2_BASE"
    else
        env_base="FT_RECOVERY_MODE=reprefill"
    fi
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (${baseline}, ${wl}, ${load}, ${fault}) GPU=${gpus}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault ${fault} \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
        return 0
    else
        echo "  ${tag}/s${seed}: FAILED — no metrics.json" >&2
        if grep -q "AssertionError\|CUDA out of memory" "${out}/server.log" 2>/dev/null; then
            echo "    cause: preemption/OOM assertion" >&2
        fi
        return 1
    fi
}

# ------------------------ Phase A: wait for existing smoke then decide ------------------------

SMOKE_OUT="${ROOT}/smoke/W5_NR_RPS1/42"
echo "######## Stress pipeline 2026-04-20 $(date) ########" >&2

# Wait up to 15min for the already-running smoke to produce metrics.json
echo "Phase A: waiting for smoke at ${SMOKE_OUT}" >&2
for _ in $(seq 1 180); do
    [ -f "${SMOKE_OUT}/metrics.json" ] && break
    if grep -qE "AssertionError|CUDA out of memory|ENGINE_CORE_DEAD" "${SMOKE_OUT}/server.log" 2>/dev/null; then
        echo "  smoke server errored, aborting wait" >&2
        break
    fi
    sleep 5
done

SMOKE_OK=0
if [ -f "${SMOKE_OUT}/metrics.json" ]; then
    SMOKE_OK=$(python3 -c "
import json
try:
    m = json.load(open('${SMOKE_OUT}/metrics.json'))
    print(1 if m.get('completion_rate', 0) >= 0.8 and m.get('goodput', 0) > 0 else 0)
except Exception:
    print(0)
")
fi

if [ "$SMOKE_OK" != "1" ]; then
    echo "Phase A: smoke at RPS=1.0 FAILED. Retrying at Light (RPS=0.5) ..." >&2
    # Wait for any lingering server process to die
    pkill -9 -f "port 8500" 2>/dev/null || true
    sleep 5
    run_cell "W5_NR_RPS0p5" "0,1" "8500" "" "NoFT-Reprefill" "W5_LongDoc" "Light" "none" "42" "smoke" || true
    if [ -f "${ROOT}/smoke/W5_NR_RPS0p5/42/metrics.json" ]; then
        SMOKE_OK=$(python3 -c "
import json
m = json.load(open('${ROOT}/smoke/W5_NR_RPS0p5/42/metrics.json'))
print(1 if m.get('completion_rate', 0) >= 0.8 and m.get('goodput', 0) > 0 else 0)
")
        [ "$SMOKE_OK" = "1" ] && { SMOKE_LOAD="Light"; SMOKE_PORT="8500"; echo "  smoke@Light OK, using Light for main run" >&2; }
    fi
fi

if [ "$SMOKE_OK" != "1" ]; then
    echo "Phase A: all smokes FAILED — giving up on W5. Running W7 only." >&2
    SKIP_W5=1
else
    SKIP_W5=0
    # Decide load level based on what passed
    if [ -f "${ROOT}/smoke/W5_NR_RPS0p5/42/metrics.json" ]; then
        MAIN_LOAD="Light"
    else
        MAIN_LOAD="Moderate"
    fi
    echo "Phase A: smoke OK. Using MAIN_LOAD=${MAIN_LOAD} for W5 full runs" >&2
fi

# ------------------------ Phase B: W7 on GPU 2-3 (parallel, independent) ------------------------

echo "Phase B: kicking off W7 on GPU 2-3 in background" >&2
(
    for s in 42 123 456; do
        run_cell "W7_NR_Heavy"       "2,3" "8501" "" "NoFT-Reprefill" "W7_Saturated" "Heavy" "F2_Mid" "$s" "w7" || true
        run_cell "W7_V2_Heavy"       "2,3" "8501" "" "Our-System"     "W7_Saturated" "Heavy" "F2_Mid" "$s" "w7" || true
    done
    echo "[$(date +%H:%M:%S)] W7 done" >&2
) > /tmp/stress_w7.log 2>&1 &
W7_PID=$!

# ------------------------ Phase C: W5 on GPU 0-1 (serial) ------------------------

if [ "$SKIP_W5" != "1" ]; then
    echo "Phase C: W5 full matrix on GPU 0-1, load=${MAIN_LOAD}" >&2
    for s in 42 123 456; do
        run_cell "W5_NR"              "0,1" "8500" "" "NoFT-Reprefill" "W5_LongDoc" "${MAIN_LOAD}" "F2_Mid" "$s" "w5" || true
        run_cell "W5_V2"              "0,1" "8500" "" "Our-System"     "W5_LongDoc" "${MAIN_LOAD}" "F2_Mid" "$s" "w5" || true
        run_cell "W5_V2_skip_solver"  "0,1" "8500" "FT_SKIP_SOLVER=1" "Our-System" "W5_LongDoc" "${MAIN_LOAD}" "F2_Mid" "$s" "w5" || true
    done
fi

# Wait for W7 background to finish
wait $W7_PID 2>/dev/null || true

# ------------------------ Phase D: summary ------------------------

echo "" >&2
echo "######## Summary ########" >&2
python3 << 'PYEOF'
import json, os, statistics as st
CFGS = [
    ("W5 NR (filtered)",      "results_v2/8B/stress_2026-04-20/w5/W5_NR"),
    ("W5 V2",                 "results_v2/8B/stress_2026-04-20/w5/W5_V2"),
    ("W5 V2+skip_solver",     "results_v2/8B/stress_2026-04-20/w5/W5_V2_skip_solver"),
    ("W7 NR",                 "results_v2/8B/stress_2026-04-20/w7/W7_NR_Heavy"),
    ("W7 V2",                 "results_v2/8B/stress_2026-04-20/w7/W7_V2_Heavy"),
]
print(f"{'variant':<28} {'gp':>12} {'fg_p50':>12} {'fg_p95':>12} {'comp':>6}")
for name, base in CFGS:
    gps, f50, f95, comps = [], [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            comps.append(m.get('completion_rate',0)*100)
            if m.get('completion_rate',0) >= 0.9:
                gps.append(m['goodput']); f50.append(m.get('failover_gap_p50_ms',0))
                f95.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gs = st.stdev(gps) if len(gps)>1 else 0
        f50s = st.stdev(f50) if len(f50)>1 else 0
        f95s = st.stdev(f95) if len(f95)>1 else 0
        cm = st.mean(comps) if comps else 0
        print(f"{name:<28} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(f50):7.0f}±{f50s:4.0f} {st.mean(f95):7.0f}±{f95s:4.0f} {cm:4.0f}%")
    else:
        print(f"{name:<28} NO VALID RUNS (comp<90% or all failed)")
PYEOF
echo "[$(date +%H:%M:%S)] Pipeline complete" >&2
