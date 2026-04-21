#!/usr/bin/env bash
# V8: FT_LAZY_RELOAD + FT_LAZY_MAX_CONCURRENT=3 @ gpu_util=0.9.
# Rerouted reqs held in engine-side deque, drained 3 at a time so
# scheduler never admits more than 3 displaced reqs to running
# simultaneously — eliminating the fault-time batch-size spike
# that caused OOM/CUDA-assert on s456 (19 displaced).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
OUT="results_v2/8B/v8_lazy/456"
mkdir -p "$OUT"

COMMON_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=0 FT_RESTORE_PARALLEL_LOAD=0 FT_RECOVERY_PREBUDGET=1 FT_LAZY_RELOAD=1 FT_LAZY_MAX_CONCURRENT=1'

echo "[$(date +%H:%M:%S)] V8 LAZY_RELOAD smoke test s456 (max_concurrent=3)"
eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV} python experiments_v2/run.py \
    --config experiments_v2/config_8b.yaml --baseline Our-System \
    --workload W1_Chat --load Heavy --fault F2_Mid \
    --seed 456 --port 8400 --output-dir ${OUT}" > "${OUT}/stdout.log" 2>&1

python3 -c "
import json
m = json.load(open('${OUT}/metrics.json'))
print(f's456 V8: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f}ms fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "FAILED"
