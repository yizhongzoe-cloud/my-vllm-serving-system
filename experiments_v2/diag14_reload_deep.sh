#!/usr/bin/env bash
# Diag14: deeper reload mode validation.
# 1. More seeds on W5 reload to reduce variance (6 total: 42/123/456/789/1234/5678)
# 2. Test reload on W7 (does reload help short-prompt too?)
# 3. Test reload + FT_RESTORE_MAX_WORKERS=16 (more parallelism)

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag14"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" gpus="$2" port="$3" extra="$4" baseline="$5" wl="$6" load="$7" seed="$8"
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
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
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

echo "######## Diag14: reload deep validation $(date) ########"

# GPU 0-1: W5 V2 reload 更多 seeds + W7 reload validation
(
    for s in 789 1234 5678; do
        run_cell "W5_V2_reload_extra" "0,1" "8500" "" "Our-System" "W5_LongDoc" "Moderate" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W7_V2_reload" "0,1" "8500" "" "Our-System" "W7_Saturated" "Heavy" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag14_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: W5 reload + 更多 workers + W4/W1 no-regression with reload
(
    for s in 42 123 456; do
        run_cell "W5_reload_w16" "2,3" "8501" "FT_RESTORE_MAX_WORKERS=16" "Our-System" "W5_LongDoc" "Moderate" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W1_V2_reload" "2,3" "8501" "" "Our-System" "W1_Chat" "Heavy" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag14_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag14 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT_D = "results_v2/8B/stress_2026-04-20/diag14"
ROOT_ORIG = "results_v2/8B/stress_2026-04-20"
print(f"{'variant':<34} {'gp':>12} {'comp':>6} {'fg_p95':>10}")
for name, base, seeds in [
    ("NR W5 (historical)",     f"{ROOT_ORIG}/w5/W5_NR", [42,123,456]),
    ("V2 reload W5 (diag13)",  f"{ROOT_ORIG}/diag13/W5_V2_reload", [42,123,456]),
    ("V2 reload W5 (extra 3)", f"{ROOT_D}/W5_V2_reload_extra", [789,1234,5678]),
    ("V2 reload W5 w16",       f"{ROOT_D}/W5_reload_w16", [42,123,456]),
    ("NR W7 (historical)",     f"{ROOT_ORIG}/w7/W7_NR_Heavy", [42,123,456]),
    ("V2 reload W7",           f"{ROOT_D}/W7_V2_reload", [42,123,456]),
    ("V2 reload W1",           f"{ROOT_D}/W1_V2_reload", [42,123,456]),
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

# 6-seed combined W5 reload
print("\n=== W5 reload 6 seeds combined ===")
gps, f95, c = [], [], []
for base in [f"{ROOT_ORIG}/diag13/W5_V2_reload", f"{ROOT_D}/W5_V2_reload_extra"]:
    for s in [42,123,456,789,1234,5678]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
if gps:
    gs = st.stdev(gps) if len(gps)>1 else 0
    print(f"W5 V2 reload 6-seed: gp={st.mean(gps):.1f}±{gs:.0f} comp={st.mean(c):.0f}% fg_p95={st.mean(f95):.0f}")
    # Also filter comp>=75% subset
    gps2, f952 = [], []
    for base in [f"{ROOT_ORIG}/diag13/W5_V2_reload", f"{ROOT_D}/W5_V2_reload_extra"]:
        for s in [42,123,456,789,1234,5678]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                if m['completion_rate'] >= 0.75:
                    gps2.append(m['goodput']); f952.append(m.get('failover_gap_p95_ms',0))
    if gps2:
        print(f"W5 V2 reload 6-seed (comp>=75%): n={len(gps2)} gp={st.mean(gps2):.1f} fg_p95={st.mean(f952):.0f}")
PYEOF
