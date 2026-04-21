#!/usr/bin/env bash
# Parameter tuning smoke — 6 variants on A4 W2_Summary/Heavy/F2_Mid/s42.
# ~36 min total (6 × ~6 min).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/solver_tune"
mkdir -p "$ROOT"

BASE_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

run_variant() {
    local tag="$1" extra_env="$2"
    local out="${ROOT}/${tag}/42"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && { echo "  [skip] ${tag}"; return 0; }

    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] ${tag}: ${extra_env}" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${BASE_ENV} ${extra_env} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed 42 --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1

    python3 -c "
import json, os, re
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
with open('${out}/server.log') as f:
    content = re.sub(r'\x1b\[[0-9;]*m', '', f.read())
rs = re.findall(r'Running: (\d+) reqs', content)
if rs:
    vals = [int(v) for v in rs[-40:-20]]
    if vals:
        import statistics as st
        print(f'    pre-fault running: mean={st.mean(vals):.1f} max={max(vals)}')
print(f'    reroutes: {content.count(\"re-routed\")}')
" 2>&1
}

echo "######## Parameter tuning smoke $(date) ########"

# V_A tuning (alpha sweep)
run_variant "T1_A_a0.05"    "FT_SOLVER_RECOVERY_PENALTY=0.05"
run_variant "T2_A_a0.1"     "FT_SOLVER_RECOVERY_PENALTY=0.1"
run_variant "T3_A_a0.5"     "FT_SOLVER_RECOVERY_PENALTY=0.5"

# V_C tuning (N sweep)
run_variant "T4_C_N1"       "FT_SOLVER_RUNNING_CAP=1"
run_variant "T5_C_N2"       "FT_SOLVER_RUNNING_CAP=2"

# Best combo
run_variant "T6_AC_a0.1_N2" "FT_SOLVER_RECOVERY_PENALTY=0.1 FT_SOLVER_RUNNING_CAP=2"

echo ""
echo "=== Tuning summary ==="
python3 << 'PYEOF'
import json, os, statistics as st
print(f"{'variant':<24} {'gp':>7} {'comp':>5} {'fg_p50':>8} {'fg_p95':>8}")
# Baselines
for name, path in [
    ("v4-greedy (ref)", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/os/42/metrics.json"),
    ("NR (ref)", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/nr/42/metrics.json"),
    ("V_AC α=0.01 N=3 (prev)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_AC_both/42/metrics.json"),
    ("V_C N=3 (prev)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_C_cap3/42/metrics.json"),
]:
    if os.path.exists(path):
        m = json.load(open(path))
        print(f"{name:<24} {m['goodput']:7.1f} {m['completion_rate']*100:4.0f}% {m.get('failover_gap_p50_ms',0):8.0f} {m.get('failover_gap_p95_ms',0):8.0f}")
print()
# New tuning results
for tag in ["T1_A_a0.05", "T2_A_a0.1", "T3_A_a0.5", "T4_C_N1", "T5_C_N2", "T6_AC_a0.1_N2"]:
    p = f"results_v2/8B/overnight_2026-04-15/solver_tune/{tag}/42/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{tag:<24} {m['goodput']:7.1f} {m['completion_rate']*100:4.0f}% {m.get('failover_gap_p50_ms',0):8.0f} {m.get('failover_gap_p95_ms',0):8.0f}")
PYEOF
