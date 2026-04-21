#!/usr/bin/env bash
# Real Benders solver 3-seed: NO FT_GATED_SOLVER, NO FT_SKIP_SOLVER.
# A4 W2/Heavy/F2_Mid × 3 seeds. Also runs NR baseline for comparison.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/overnight_2026-04-15/solver_real/A4_W2_Heavy_F2"
mkdir -p "$ROOT"

OS_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'
NR_ENV='FT_RECOVERY_MODE=reprefill'

echo "######## Real Solver 3-seed $(date) ########"
for seed in 42 123 456; do
    os_out="${ROOT}/${seed}"
    nr_out="results_v2/8B/overnight_2026-04-15/compact/A4_W2_Summary_Heavy_F2_Mid/nr/${seed}"

    # Skip if already done (reuse s42 from smoke)
    if [ -f "${os_out}/metrics.json" ]; then
        echo "  [skip] s${seed} OS already done" >&2
    else
        mkdir -p "$os_out"
        rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
        echo "[$(date +%H:%M:%S)] solver-real/s${seed}" >&2
        eval "CUDA_VISIBLE_DEVICES=4,5 ${OS_ENV} python experiments_v2/run.py \
            --config experiments_v2/config_8b.yaml --baseline Our-System \
            --workload W2_Summary --load Heavy --fault F2_Mid \
            --seed ${seed} --port 8400 --output-dir ${os_out}" \
            > "${os_out}/stdout.log" 2>&1
    fi

    python3 -c "
import json
for tag, out in [('OS-solver', '${os_out}'), ('NR', '${nr_out}')]:
    try:
        m = json.load(open(f'{out}/metrics.json'))
        print(f'  s${seed} {tag}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
    except Exception:
        print(f'  s${seed} {tag}: N/A')
" 2>&1
done

echo ""
echo "=== Solver invocation analysis ==="
for seed in 42 123 456; do
    slog="${ROOT}/${seed}/server.log"
    [ -f "$slog" ] || continue
    echo "--- s${seed} ---"
    grep -c "Greedy fallback\|greedy dispatch\|greedy bootstrap" "$slog"
    echo "  greedy calls"
    grep -c "Solver\|solver\|BENDERS\|Benders\|admitted" "$slog"
    echo "  solver/admitted calls"
    # Show displaced req routing decisions
    grep "reroute_plan\|recovery_target\|displaced\|re-routed" "$slog" | head -10 | sed 's/\x1b\[[0-9;]*m//g'
done
echo "######## done $(date) ########"
