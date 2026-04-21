#!/usr/bin/env bash
# Quick single-seed test: gpu_util=0.75, sync=off, max_workers=8.
# Goal: verify if dropping gpu_util from 0.85 to 0.75 (extra ~1.2 GB
# headroom, total ~4.9 GB for temp+activation) is enough to absorb
# the worst-case s456 accumulation (19 displaced × 224 MB ≈ 4.3 GB).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
OUT="results_v2/8B/parallel_restore_v5D/456"
mkdir -p "$OUT"

COMMON_ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RECOVERY_PREBUDGET=1 FT_RESTORE_MAX_WORKERS=8 FT_RESTORE_PER_REQ_SYNC=0'

echo "[$(date +%H:%M:%S)] gpu_util=0.75 test, s456"
eval "CUDA_VISIBLE_DEVICES=4,5 ${COMMON_ENV} python experiments_v2/run.py \
    --config experiments_v2/config_8b_gpu075.yaml --baseline Our-System \
    --workload W1_Chat --load Heavy --fault F2_Mid \
    --seed 456 --port 8400 --output-dir ${OUT}" > "${OUT}/stdout.log" 2>&1

python3 -c "
import json
m = json.load(open('${OUT}/metrics.json'))
print(f's456: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "FAILED"
