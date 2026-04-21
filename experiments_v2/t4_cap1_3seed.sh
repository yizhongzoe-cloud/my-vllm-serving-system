#!/usr/bin/env bash
# T4 winner validation: FT_SOLVER_RUNNING_CAP=1 on A4 3-seed.
# Compare with NR + V_C_N3 baseline.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/t4_cap1_3seed"
mkdir -p "$ROOT"

BASE_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1 FT_SOLVER_RUNNING_CAP=1'

echo "######## T4 cap=1 3-seed validation $(date) ########"
for seed in 42 123 456; do
    out="${ROOT}/${seed}"
    mkdir -p "$out"
    # Reuse s42 smoke
    if [ "$seed" = "42" ] && [ -f "results_v2/8B/overnight_2026-04-15/solver_tune/T4_C_N1/42/metrics.json" ]; then
        cp results_v2/8B/overnight_2026-04-15/solver_tune/T4_C_N1/42/*.json "$out/" 2>/dev/null
        cp results_v2/8B/overnight_2026-04-15/solver_tune/T4_C_N1/42/*.csv "$out/" 2>/dev/null
        cp results_v2/8B/overnight_2026-04-15/solver_tune/T4_C_N1/42/*.log "$out/" 2>/dev/null
        echo "  [reuse] s42 from smoke"
        python3 -c "import json; m=json.load(open('$out/metrics.json')); print(f'  s42: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')"
        continue
    fi
    [ -f "${out}/metrics.json" ] && continue

    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] T4/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${BASE_ENV} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1

    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
done

echo ""
echo "=== T4 (cap=1) vs NR vs V_C(cap=3) 3-seed ==="
python3 << 'PYEOF'
import json, os, statistics as st
for name, base in [
    ("T4 cap=1", "results_v2/8B/overnight_2026-04-15/t4_cap1_3seed"),
    ("V_C cap=3 (prev)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_C_cap3"),
    ("NR", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/nr"),
    ("v4-greedy", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/os"),
]:
    gps, fg50s, fg95s, comps = [], [], [], []
    for s in [42, 123, 456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            comps.append(m['completion_rate']*100)
            if m['completion_rate'] >= 0.95:
                gps.append(m['goodput'])
                fg50s.append(m.get('failover_gap_p50_ms',0))
                fg95s.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gp_s = st.stdev(gps) if len(gps)>1 else 0
        fg50_s = st.stdev(fg50s) if len(fg50s)>1 else 0
        fg95_s = st.stdev(fg95s) if len(fg95s)>1 else 0
        print(f"{name:<20} gp={st.mean(gps):6.1f}±{gp_s:3.0f} fg_p50={st.mean(fg50s):7.0f}±{fg50_s:4.0f} fg_p95={st.mean(fg95s):7.0f}±{fg95_s:4.0f} comp={st.mean(comps):3.0f}%")
PYEOF
