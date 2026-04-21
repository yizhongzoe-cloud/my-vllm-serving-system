#!/usr/bin/env bash
# Extend solver improvement to 3-seed: s123 + s456 for all 3 variants.
# Runs after smoke (s42) completes. Total ~36 min (6 runs × ~6 min).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/solver_improve"
mkdir -p "$ROOT"

BASE_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

run_variant_seed() {
    local tag="$1" seed="$2" extra_env="$3"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && { echo "  [skip] ${tag}/s${seed}"; return 0; }

    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${BASE_ENV} ${extra_env} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1

    python3 -c "
import json
try:
    m = json.load(open('${out}/metrics.json'))
    print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
except Exception:
    print(f'  ${tag}/s${seed}: FAILED')
" 2>&1
}

echo "######## 3-seed extension start $(date) ########"

for seed in 123 456; do
    run_variant_seed "V_A_penalty0.01" "$seed" "FT_SOLVER_RECOVERY_PENALTY=0.01"
    run_variant_seed "V_C_cap3" "$seed" "FT_SOLVER_RUNNING_CAP=3"
    run_variant_seed "V_AC_both" "$seed" "FT_SOLVER_RECOVERY_PENALTY=0.01 FT_SOLVER_RUNNING_CAP=3"
done

echo ""
echo "######## 3-seed extension done $(date) ########"

# Full comparison
python3 << 'PYEOF'
import json, os, statistics as st
cells = [
    ("v4-greedy", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/os"),
    ("NR", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/nr"),
    ("real-solver (baseline)", "results_v2/8B/overnight_2026-04-15/solver_real/A4_W2_Heavy_F2"),
    ("V_A (penalty=0.01)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_A_penalty0.01"),
    ("V_C (cap=3)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_C_cap3"),
    ("V_AC (both)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_AC_both"),
]

print(f"\n=== Full 3-seed A4 W2/Heavy/F2_Mid ===")
print(f"{'variant':<30} {'gp±std':>12} {'fg_p50±std':>14} {'fg_p95±std':>14} {'comp':>5}")
print("-" * 90)

for name, base in cells:
    gps, fg50s, fg95s, comps = [], [], [], []
    for s in [42, 123, 456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            comps.append(m['completion_rate']*100)
            if m['completion_rate'] >= 0.95:
                gps.append(m['goodput'])
                fg50s.append(m.get('failover_gap_p50_ms', 0))
                fg95s.append(m.get('failover_gap_p95_ms', 0))
    if gps:
        gp_s = st.stdev(gps) if len(gps) > 1 else 0
        fg50_s = st.stdev(fg50s) if len(fg50s) > 1 else 0
        fg95_s = st.stdev(fg95s) if len(fg95s) > 1 else 0
        cm = st.mean(comps) if comps else 0
        print(f"{name:<30} {st.mean(gps):6.1f}±{gp_s:3.0f}    {st.mean(fg50s):7.0f}±{fg50_s:3.0f}    {st.mean(fg95s):7.0f}±{fg95_s:3.0f}    {cm:4.0f}%")
PYEOF
