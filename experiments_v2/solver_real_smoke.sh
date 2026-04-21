#!/usr/bin/env bash
# Real Benders solver test — NO FT_GATED_SOLVER, NO FT_SKIP_SOLVER.
# Solver actually runs on every admission decision.
# Test A4 W2/Heavy/F2_Mid s42 (single seed smoke).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
OUT="results_v2/8B/overnight_2026-04-15/solver_real/A4_W2_Heavy_F2/42"
mkdir -p "$OUT"

# Key diff: removed FT_GATED_SOLVER=1 and FT_SKIP_SOLVER
COMMON_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_PREBUDGET=1'

echo "[$(date +%H:%M:%S)] Real solver smoke: A4 W2/Heavy/F2 s42"
eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV} python experiments_v2/run.py \
    --config experiments_v2/config_8b.yaml --baseline Our-System \
    --workload W2_Summary --load Heavy --fault F2_Mid \
    --seed 42 --port 8400 --output-dir ${OUT}" > "${OUT}/stdout.log" 2>&1

python3 -c "
import json
m = json.load(open('${OUT}/metrics.json'))
print(f'A4 solver-real s42: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "FAILED"

# Check solver invocation count
echo "--- solver vs greedy count ---"
grep -c "Greedy fallback\|greedy dispatch\|greedy bootstrap" "${OUT}/server.log" 2>/dev/null
echo "greedy"
grep -c "Solver.*solution\|solver.*admit\|BENDERS\|Benders" "${OUT}/server.log" 2>/dev/null
echo "solver"
