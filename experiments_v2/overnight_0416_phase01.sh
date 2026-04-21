#!/usr/bin/env bash
# Overnight 2026-04-16 Phase 0 + Phase 1
# Phase 0: fresh NR baseline (with random fault code) 3-seed
# Phase 1: 3 OS ablations × 3 seeds
#   P1-01: OS reprefill only (no cap, no penalty)
#   P1-02: OS cap=1 + reprefill
#   P1-03: OS cap=3 + reprefill
# All on A4 W2_Summary/Heavy/F2_Mid, sequential on GPU 4-5.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16"
mkdir -p "$ROOT"

OS_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

run_cfg() {
    local cfg_name="$1" baseline="$2" extra_env="$3"
    echo "============ ${cfg_name}: baseline=${baseline} extras=${extra_env} ============" >&2
    local cfg_root="${ROOT}/${cfg_name}"
    mkdir -p "${cfg_root}"

    for seed in 42 123 456; do
        local out="${cfg_root}/${seed}"
        mkdir -p "$out"
        [ -f "${out}/metrics.json" ] && { echo "  [skip] ${cfg_name}/s${seed}"; continue; }

        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] ${cfg_name}/s${seed}" >&2
        eval "CUDA_VISIBLE_DEVICES=4,5 ${extra_env} python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline ${baseline} \
            --workload W2_Summary --load Heavy --fault F2_Mid \
            --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1

        python3 -c "
import json
try:
    m = json.load(open('${out}/metrics.json'))
    print(f'  ${cfg_name}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
except Exception:
    print(f'  ${cfg_name}/s${seed}: FAILED')
" 2>&1
    done
}

echo "######## Overnight 2026-04-16 Phase 0+1 start $(date) ########"

# Phase 0: fresh NR baseline (random fault code)
run_cfg "p0_nr_fresh" "NoFT-Reprefill" "FT_RECOVERY_MODE=reprefill"

# Phase 1: OS ablations
run_cfg "p1_os_reprefill_only" "Our-System" "${OS_BASE} FT_RECOVERY_MODE=reprefill"
run_cfg "p1_os_cap1_reprefill"  "Our-System" "${OS_BASE} FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=1"
run_cfg "p1_os_cap3_reprefill"  "Our-System" "${OS_BASE} FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3"

echo ""
echo "######## P0+P1 done $(date) ########"

# Final comparison
python3 << 'PYEOF'
import json, os, statistics as st
root = "results_v2/8B/overnight_2026-04-16"
print(f"\n=== Overnight 2026-04-16 P0+P1 Summary (A4 W2/Heavy/F2_Mid) ===")
print(f"{'config':<28} {'gp±std':>12} {'fg_p50±std':>14} {'fg_p95±std':>14} {'comp':>5}")
print("-"*80)

configs = [
    ("NR (fresh, random fault)", f"{root}/p0_nr_fresh"),
    ("OS reprefill only", f"{root}/p1_os_reprefill_only"),
    ("OS cap=1 + reprefill", f"{root}/p1_os_cap1_reprefill"),
    ("OS cap=3 + reprefill", f"{root}/p1_os_cap3_reprefill"),
]
for name, base in configs:
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
        cm = st.mean(comps) if comps else 0
        print(f"{name:<28} {st.mean(gps):6.1f}±{gp_s:3.0f}  {st.mean(fg50s):7.0f}±{fg50_s:4.0f}  {st.mean(fg95s):7.0f}±{fg95_s:4.0f}  {cm:3.0f}%")
    else:
        print(f"{name:<28} (no valid seeds)")

# Per-seed breakdown
print("\n=== Per-seed details ===")
for s in [42, 123, 456]:
    print(f"\n-- s{s} --")
    for name, base in configs:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            print(f"  {name:<28}: gp={m['goodput']:6.1f} comp={m['completion_rate']*100:3.0f}% fg_p50={m.get('failover_gap_p50_ms',0):7.0f} fg_p95={m.get('failover_gap_p95_ms',0):7.0f}")
PYEOF
