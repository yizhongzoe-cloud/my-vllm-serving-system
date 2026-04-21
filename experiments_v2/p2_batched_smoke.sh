#!/usr/bin/env bash
# P2 smoke: FT_BATCHED_RELOAD=1 on A4 s42 single seed.
# Expect: metrics.json written, kv_restore_done fires, no new Traceback in server.log.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

OUT="results_v2/8B/overnight_2026-04-16/p2_batched_smoke/42"
mkdir -p "$OUT"
rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null

# Explicitly enable reload mode (not reprefill) so we exercise the batched path.
ENV='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RECOVERY_PREBUDGET=1 FT_BATCHED_RELOAD=1'

echo "[$(date +%H:%M:%S)] P2 batched smoke s42"
eval "CUDA_VISIBLE_DEVICES=4,5 ${ENV} python experiments_v2/run.py \
    --config experiments_v2/config_8b.yaml --baseline Our-System \
    --workload W2_Summary --load Heavy --fault F2_Mid \
    --seed 42 --port 8400 --output-dir ${OUT}" > "${OUT}/stdout.log" 2>&1

echo ""
echo "=== smoke result ==="
python3 -c "
import json
m = json.load(open('${OUT}/metrics.json'))
print(f'gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p50={m.get(\"failover_gap_p50_ms\",0):.0f} fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
" 2>&1 || echo "METRICS READ FAILED"

echo ""
echo "=== error check ==="
grep -cE "EngineCore encountered a fatal|OutOfMemory|device-side assert|AssertionError|CUBLAS" "${OUT}/server.log" | head -3
echo "error count ^^^"

echo ""
echo "=== batched_v2 path fired? ==="
grep -c "path=batched_v2" "${OUT}/server.log"
echo "batched_v2 fires ^^^"
grep -c "FAULT_EVENT kv_restore_done" "${OUT}/server.log"
echo "total kv_restore_done ^^^"
