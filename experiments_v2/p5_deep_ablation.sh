#!/usr/bin/env bash
# P5: deep ablation to locate the remaining 633ms pre-fault pause on W2.
# Layer on V2 baseline, turn off one component at a time.
# Target: find the component whose removal gets OS W2 fg_p95 close to NR 400ms.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-16/p5_deep"
mkdir -p "$ROOT"

# V2 base (current best universal): ckpt_interval=2 + cap=3 + reprefill + prebudget
V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reprefill FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2'

run_v() {
    local tag="$1" extra="$2" seed="$3"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=0,1 ${V2_BASE} ${extra} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1
}

echo "######## P5 deep ablation (W2/Heavy) $(date) ########"

# V2 baseline (control)
for s in 42 123 456; do
    run_v "V2_base"              ""  "$s"
    # A: disable snapshots (engine-side overhead)
    run_v "A_no_snapshots"       "FT_DISABLE_SNAPSHOTS=1"  "$s"
    # B: skip solver (ft_client side overhead)
    run_v "B_skip_solver"        "FT_SKIP_SOLVER=1"  "$s"
    # C: both A+B
    run_v "C_no_snap_skip_solver" "FT_DISABLE_SNAPSHOTS=1 FT_SKIP_SOLVER=1"  "$s"
    # D: very aggressive ckpt throttle (interval=20, near-off)
    run_v "D_interval20"         "FT_CHECKPOINT_STEP_INTERVAL=20"  "$s"
done

echo ""
echo "=== P5 summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
configs = [
    ("NR W2 ref", "results_v2/8B/overnight_2026-04-16/p0_nr_fresh"),
    ("V2 base (control)", "results_v2/8B/overnight_2026-04-16/p5_deep/V2_base"),
    ("A no_snapshots", "results_v2/8B/overnight_2026-04-16/p5_deep/A_no_snapshots"),
    ("B skip_solver", "results_v2/8B/overnight_2026-04-16/p5_deep/B_skip_solver"),
    ("C both A+B", "results_v2/8B/overnight_2026-04-16/p5_deep/C_no_snap_skip_solver"),
    ("D interval=20", "results_v2/8B/overnight_2026-04-16/p5_deep/D_interval20"),
]
print(f"{'variant':<28} {'gp':>8} {'fg_p50':>12} {'fg_p95':>12}")
for name, base in configs:
    gps, fg50s, fg95s = [], [], []
    for s in [42,123,456]:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            if m.get('completion_rate', 0) >= 0.95:
                gps.append(m['goodput']); fg50s.append(m.get('failover_gap_p50_ms',0))
                fg95s.append(m.get('failover_gap_p95_ms',0))
    if gps:
        gp_s = st.stdev(gps) if len(gps)>1 else 0
        fg50_s = st.stdev(fg50s) if len(fg50s)>1 else 0
        fg95_s = st.stdev(fg95s) if len(fg95s)>1 else 0
        print(f"{name:<28} {st.mean(gps):6.1f}±{gp_s:3.0f} {st.mean(fg50s):7.0f}±{fg50_s:4.0f} {st.mean(fg95s):7.0f}±{fg95_s:4.0f}")

# Per-seed
print("\n=== Per-seed fg_p95 ===")
for s in [42,123,456]:
    print(f"s{s}:")
    for name, base in configs:
        p = f"{base}/{s}/metrics.json"
        if os.path.exists(p):
            m = json.load(open(p))
            print(f"  {name:<28}: fg_p95={m.get('failover_gap_p95_ms',0):.0f}")
PYEOF
