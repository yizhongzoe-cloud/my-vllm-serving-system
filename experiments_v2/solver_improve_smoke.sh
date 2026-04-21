#!/usr/bin/env bash
# Solver improvement smoke test — 3 variants on A4/s42 (single seed, ~6 min each):
#   V-A: FT_SOLVER_RECOVERY_PENALTY=0.01
#   V-C: FT_SOLVER_RUNNING_CAP=3
#   V-AC: both
# Baseline: real-solver (no penalty, no cap) from previous run.
# NR and v4-greedy already in compact run.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/solver_improve"
mkdir -p "$ROOT"

BASE_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

run_variant() {
    local tag="$1" extra_env="$2"
    local out="${ROOT}/${tag}/42"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && { echo "  [skip] ${tag}/s42"; return 0; }

    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    echo "[$(date +%H:%M:%S)] variant ${tag}: ${extra_env}" >&2
    eval "CUDA_VISIBLE_DEVICES=4,5 ${BASE_ENV} ${extra_env} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline Our-System \
        --workload W2_Summary --load Heavy --fault F2_Mid \
        --seed 42 --port 8400 --output-dir ${out}" > "${out}/stdout.log" 2>&1

    python3 -c "
import json, os, re
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s42: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
# Running batch stats pre-fault
with open('${out}/server.log') as f:
    content = re.sub(r'\x1b\[[0-9;]*m', '', f.read())
running_stats = re.findall(r'Running: (\d+) reqs', content)
if running_stats:
    vals = [int(v) for v in running_stats[-40:-20]]
    if vals:
        import statistics as st
        print(f'    pre-fault running: mean={st.mean(vals):.1f} max={max(vals)}')
routes = re.findall(r're-routed', content)
print(f'    reroutes: {len(routes)}')
" 2>&1
}

echo "######## Solver improvement smoke $(date) ########"

run_variant "V_A_penalty0.01" "FT_SOLVER_RECOVERY_PENALTY=0.01"
run_variant "V_C_cap3" "FT_SOLVER_RUNNING_CAP=3"
run_variant "V_AC_both" "FT_SOLVER_RECOVERY_PENALTY=0.01 FT_SOLVER_RUNNING_CAP=3"

echo ""
echo "=== Summary comparison (all A4/s42) ==="
python3 << 'PYEOF'
import json, os, re, statistics as st

baselines = [
    ("v4-greedy", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/os/42/metrics.json"),
    ("NR", "results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/nr/42/metrics.json"),
    ("real-solver (baseline)", "results_v2/8B/overnight_2026-04-15/solver_real/A4_W2_Heavy_F2/42/metrics.json"),
]
variants = [
    ("V_A (penalty=0.01)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_A_penalty0.01/42/metrics.json"),
    ("V_C (cap=3)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_C_cap3/42/metrics.json"),
    ("V_AC (both)", "results_v2/8B/overnight_2026-04-15/solver_improve/V_AC_both/42/metrics.json"),
]

print(f"{'variant':<32} {'gp':>7} {'comp':>5} {'fg_p50':>8} {'fg_p95':>8} {'Δfg vs greedy':>14}")
print("-" * 90)
greedy_fg = None
for name, path in baselines + variants:
    if os.path.exists(path):
        m = json.load(open(path))
        fg = m.get('failover_gap_p95_ms', 0)
        if 'greedy' in name: greedy_fg = fg
        d = f"{fg - greedy_fg:+.0f}ms" if greedy_fg and 'greedy' not in name else "baseline"
        print(f"{name:<32} {m['goodput']:7.1f} {m['completion_rate']*100:4.0f}% {m.get('failover_gap_p50_ms',0):8.0f} {fg:8.0f} {d:>14}")
    else:
        print(f"{name:<32} {'n/a':>7}")
PYEOF
