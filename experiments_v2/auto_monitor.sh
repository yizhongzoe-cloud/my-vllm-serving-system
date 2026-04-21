#!/usr/bin/env bash
# Auto-monitor: every 20s, check experiment state.
# Log summary to /tmp/auto_monitor.log.
# Detect crashes (comp<75%) and flag for rerun.

set -u
cd /home/jlpang/my-vllm-serving-system
MONITOR_LOG="/tmp/auto_monitor.log"
CRASH_LOG="/tmp/auto_crash.log"
> "$MONITOR_LOG"
> "$CRASH_LOG"

while true; do
    now=$(date +%H:%M:%S)
    # Count metrics
    n_metrics=$(find results_v2/8B/stress_2026-04-20/diag20/ -name "metrics.json" 2>/dev/null | wc -l)
    # Check running procs
    n_procs=$(pgrep -af "run.py" | grep -v claude | wc -l)
    # Check diag20 shell alive
    diag_alive=$(pgrep -af "diag20_clean_reload" | grep -v claude | wc -l)
    # GPU usage
    gpu0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    # Latest log line
    latest=$(tail -1 /tmp/diag20_v3.log 2>/dev/null | head -c 120)

    echo "[$now] metrics=$n_metrics procs=$n_procs diag=$diag_alive gpu0=${gpu0}MiB | $latest" >> "$MONITOR_LOG"

    # Detect recent crashes (last 5 completed runs)
    for mfile in $(find results_v2/8B/stress_2026-04-20/diag20/ -name "metrics.json" -newer /tmp/last_crash_check 2>/dev/null); do
        comp=$(python3 -c "import json; m=json.load(open('$mfile')); print(int(m['completion_rate']*100))" 2>/dev/null)
        if [ -n "$comp" ] && [ "$comp" -lt 75 ]; then
            seed=$(basename $(dirname "$mfile"))
            wl=$(basename $(dirname $(dirname "$mfile")))
            echo "[$now] CRASH: $wl/$seed comp=$comp%" >> "$CRASH_LOG"
        fi
    done
    touch /tmp/last_crash_check 2>/dev/null

    # Detect if diag20 shell died but should still be running
    if [ "$diag_alive" = "0" ] && [ "$n_metrics" -lt "12" ]; then
        echo "[$now] WARNING: diag20 shell died with only $n_metrics/12 done" >> "$MONITOR_LOG"
    fi

    sleep 20
done
