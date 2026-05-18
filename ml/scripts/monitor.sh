#!/usr/bin/env bash
# Live training monitor — runs in terminal, refreshes every 30s
# Usage: bash ml/scripts/monitor.sh

LOG=ml/logs/v7.0.log
INTERVAL=30

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SENTINEL v7.0 Training Monitor  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Process status
    PID=$(pgrep -f "train.py.*v7.0" | head -1)
    if [ -n "$PID" ]; then
        RSS=$(ps -p $PID -o rss= 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        CPU=$(ps -p $PID -o %cpu= 2>/dev/null | xargs)
        echo "  Process: PID $PID | RAM: $RSS | CPU: ${CPU}%"
    else
        echo "  Process: *** NOT RUNNING ***"
    fi

    # GPU
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{printf "  GPU:     %s%% util | %s/%s MiB VRAM | %s°C\n", $1,$2,$3,$4}'

    echo ""

    # Latest epoch summary
    echo "── Latest epoch ─────────────────────────────────────────────────"
    grep "Epoch.*Loss=\|F1-macro=" $LOG 2>/dev/null | tail -3

    echo ""

    # Latest step (within current epoch)
    echo "── Current step ─────────────────────────────────────────────────"
    grep "Step " $LOG 2>/dev/null | tail -1

    echo ""

    # JK weights (last seen)
    echo "── JK attention weights ─────────────────────────────────────────"
    grep "JK attention" $LOG 2>/dev/null | tail -1

    echo ""

    # Per-class F1 (last epoch)
    echo "── Per-class F1 (last epoch) ────────────────────────────────────"
    grep -E "f1_|Top3|Bottom3" $LOG 2>/dev/null | grep -E "Top3|Bottom3" | tail -2

    echo ""

    # Warnings / alerts
    echo "── Alerts ───────────────────────────────────────────────────────"
    grep -E "WARNING|CRITICAL|collapse|DEATH|NaN|Inf" $LOG 2>/dev/null | tail -5

    echo ""
    echo "  MLflow UI → http://localhost:5000    Refresh: ${INTERVAL}s    Ctrl+C to quit"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sleep $INTERVAL
done
