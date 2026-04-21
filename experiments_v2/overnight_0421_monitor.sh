#!/usr/bin/env bash
# Monitor overnight_0421 every 30s. Log to /tmp/overnight_0421_monitor.log.
# Auto-kill hung runs (>600s wall, GPU mem <500MiB).

set -u
cd /home/jlpang/my-vllm-serving-system
MON="/tmp/overnight_0421_monitor.log"
> "$MON"
ROOT="results_v2/8B/overnight_2026-04-21"

while true; do
    now=$(date +%H:%M:%S)
    n_metrics=$(find "$ROOT" -name metrics.json 2>/dev/null | wc -l)
    n_stable=$(find "$ROOT" -name metrics.json 2>/dev/null | xargs -I {} python3 -c "import json,sys; m=json.load(open('{}'))\nprint(1 if m['completion_rate']>=0.95 else 0)" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    run_alive=$(pgrep -f "experiments_v2/run.py" 2>/dev/null | wc -l)
    script_alive=$(pgrep -f "overnight_0421.sh" | grep -v claude | grep -v monitor | wc -l)
    gpu0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    gpu1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
    shm=$(df -m /dev/shm | awk 'NR==2 {print $3}')

    # Detect hung run: a run.py PID that's been running >600s with low GPU usage
    hung=""
    for pid in $(pgrep -f "experiments_v2/run.py" 2>/dev/null); do
        etime=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -z "$etime" ] && continue
        if [ "$etime" -gt 700 ] && [ "$gpu0" -lt 500 ]; then
            hung="$pid"
            echo "[$now] KILL-HUNG pid=$pid etime=${etime}s gpu0=${gpu0}MiB" >> "$MON"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    echo "[$now] metrics=$n_metrics stable=$n_stable runpy=$run_alive script=$script_alive gpu0=${gpu0} gpu1=${gpu1} shm=${shm}MB" >> "$MON"

    # Exit if script died AND no runs alive
    if [ "$script_alive" = "0" ] && [ "$run_alive" = "0" ]; then
        echo "[$now] script finished, monitor exits" >> "$MON"
        break
    fi
    sleep 30
done
