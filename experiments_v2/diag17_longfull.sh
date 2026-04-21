#!/usr/bin/env bash
# Diag17: test V2 reload on unfiltered ArXiv (p95=7108, max=7383).
# Longer prompts amplify reprefill cost → reload advantage should grow.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag17"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" gpus="$2" port="$3" baseline="$4" wl="$5" seed="$6"
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
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load Moderate --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
    fi
}

echo "######## Diag17: unfiltered long-context $(date) ########"

# GPU 0-1: W5b (unfiltered) NR baseline + V2 reload
(
    for s in 42 123 456; do
        run_cell "W5b_NR"         "0,1" "8500" "NoFT-Reprefill" "W5b_LongFull" "$s" || true
    done
    for s in 42 123 456; do
        run_cell "W5b_V2_reload"  "0,1" "8500" "Our-System"     "W5b_LongFull" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag17_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: more W5b V2 reload seeds for stats + W7 reload stable retest
(
    for s in 789 1234 5678; do
        run_cell "W5b_V2_reload_extra" "2,3" "8501" "Our-System" "W5b_LongFull" "$s" || true
    done
    # Also retest W7 reload to confirm OOB fix stability with 3 new seeds
    for s in 789 1234 5678; do
        run_cell "W7_V2_reload_retest" "2,3" "8501" "Our-System" "W7_Saturated" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag17_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag17 summary $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20/diag17"

print(f"{'variant':<36} {'n':>3} {'gp':>12} {'comp':>5} {'fg_p95':>9}")
for name, base, seeds in [
    ("W5b unfiltered NR",        f"{ROOT}/W5b_NR", [42,123,456]),
    ("W5b unfiltered V2 reload", f"{ROOT}/W5b_V2_reload", [42,123,456]),
    ("W5b V2 reload extra",      f"{ROOT}/W5b_V2_reload_extra", [789,1234,5678]),
    ("W7 V2 reload retest",      f"{ROOT}/W7_V2_reload_retest", [789,1234,5678]),
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
        print(f"{name:<36} {len(gps):3d} {st.mean(gps):6.1f}±{gs:3.0f} {st.mean(c):4.0f}% {st.mean(f95):5.0f}")

# W5b 6-seed combined
print("\n=== W5b V2 reload 6 seeds combined ===")
gps, c, f95 = [], [], []
for base in [f"{ROOT}/W5b_V2_reload", f"{ROOT}/W5b_V2_reload_extra"]:
    for s in [42,123,456,789,1234,5678]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            gps.append(m['goodput']); c.append(m['completion_rate']*100)
            f95.append(m.get('failover_gap_p95_ms',0))
if gps:
    print(f"all: n={len(gps)} gp={st.mean(gps):.1f}±{st.stdev(gps) if len(gps)>1 else 0:.0f} comp={st.mean(c):.0f}% fg_p95={st.mean(f95):.0f}")
    # Filter comp>=80%
    gps2, f952 = [], []
    for base in [f"{ROOT}/W5b_V2_reload", f"{ROOT}/W5b_V2_reload_extra"]:
        for s in [42,123,456,789,1234,5678]:
            p = f"{base}/{s}/metrics.json"
            if os.path.exists(p):
                m = json.load(open(p))
                if m['completion_rate'] >= 0.80:
                    gps2.append(m['goodput']); f952.append(m.get('failover_gap_p95_ms',0))
    if gps2:
        print(f"comp>=80%: n={len(gps2)} gp={st.mean(gps2):.1f} fg_p95={st.mean(f952):.0f}")

# Paired per-seed W5b vs NR
print("\n=== W5b paired per-seed comparison ===")
for s in [42, 123, 456]:
    nr_p = f"{ROOT}/W5b_NR/{s}/metrics.json"
    v2_p = f"{ROOT}/W5b_V2_reload/{s}/metrics.json"
    if os.path.exists(nr_p) and os.path.exists(v2_p):
        n = json.load(open(nr_p))
        v = json.load(open(v2_p))
        gp_w = "✓" if v['goodput'] >= n['goodput'] else "✗"
        fg_w = "✓" if v.get('failover_gap_p95_ms',0) <= n.get('failover_gap_p95_ms',0) else "✗"
        comp_ok = v['completion_rate'] >= 0.80 and n['completion_rate'] >= 0.80
        strict = " 🎯" if (gp_w=='✓' and fg_w=='✓' and comp_ok) else ""
        print(f"  s{s}: V2 {v['goodput']:.1f}/{v.get('failover_gap_p95_ms',0):.0f}/{v['completion_rate']*100:.0f}% "
              f"vs NR {n['goodput']:.1f}/{n.get('failover_gap_p95_ms',0):.0f}/{n['completion_rate']*100:.0f}% "
              f"[{gp_w}][{fg_w}]{strict}")
PYEOF
