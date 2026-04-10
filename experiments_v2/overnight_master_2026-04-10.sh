#!/usr/bin/env bash
# Wait for NoFT-Reprefill to finish, then launch 3-seed overnight.
set -u

cd /home/jlpang/my-vllm-serving-system

echo "[$(date +%H:%M:%S)] Waiting for NoFT-Reprefill suite to finish..."
while pgrep -f "suite.py.*E1a_NoFT_Reprefill" > /dev/null 2>&1; do
    sleep 30
done

# Also wait for any leftover run.py / api_server
sleep 10
while pgrep -f "run.py.*8400" > /dev/null 2>&1; do
    sleep 10
done

echo "[$(date +%H:%M:%S)] NoFT-Reprefill done. Launching 3-seed overnight..."
exec bash experiments_v2/overnight_3seed_2026-04-10.sh
