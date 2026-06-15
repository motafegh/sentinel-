#!/bin/bash
# Watchdog/monitor for the SmartBugs Wild full eval.
# Run with: bash /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/watch_smartbugs_eval.sh
# 
# Modes:
#   bash .../watch_smartbugs_eval.sh status    # one-shot status check
#   bash .../watch_smartbugs_eval.sh tail      # tail the live log (Ctrl-C to exit)
#   bash .../watch_smartbugs_eval.sh watch      # loop: status every 60s
#   bash .../watch_smartbugs_eval.sh stop       # send SIGTERM to eval
#   bash .../watch_smartbugs_eval.sh force-kill # send SIGKILL (last resort)
#   bash .../watch_smartbugs_eval.sh resume     # resume the eval (--resume)

set -e
STATE=/home/motafeq/projects/sentinel/ml/data/smartbugs_wild_eval_state.json
LOG=$(ls -t /home/motafeq/projects/sentinel/ml/logs/smartbugs_wild_eval_*.log 2>/dev/null | head -1)
PID_FILE=/tmp/sb_wild_full.pid

if [ "$1" == "status" ]; then
    echo "=== STATUS ==="
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "RUNNING: PID $PID"
            ps -o pid,pcpu,pmem,etime,cmd -p "$PID" | head -2
        else
            echo "NOT RUNNING (PID $PID from pidfile is dead)"
        fi
    else
        echo "NO PID FILE"
    fi
    if [ -f "$STATE" ]; then
        echo
        echo "=== STATE ==="
        ml/.venv/bin/python -c "
import json, sys
try:
    d = json.load(open('$STATE'))
    n = len(d['processed_set'])
    errs = sum(1 for r in d['processed_set'].values() if r.get('error'))
    cfg = d.get('config', {})
    print(f'Processed: {n}')
    print(f'Errors: {errs}')
    if 'baseline_time' in d.get('stats', {}):
        print(f'Baseline time (from resume): {d[\"stats\"][\"baseline_time\"]:.1f}s')
    last_upd = d.get('last_updated', 'never')
    print(f'Last state update: {last_upd}')
    if cfg:
        total = cfg.get('max_contracts') or cfg.get('n_total_contracts')
        print(f'Total target: {total}')
        if n and total:
            print(f'Progress: {100*n/total:.2f}%')
except Exception as e:
    print(f'ERROR: {e}')
"
    else
        echo
        echo "=== NO STATE FILE ==="
    fi
    if [ -n "$LOG" ]; then
        echo
        echo "=== LAST 5 LOG LINES ==="
        tail -5 "$LOG"
    fi
elif [ "$1" == "tail" ]; then
    if [ -z "$LOG" ]; then echo "No log file"; exit 1; fi
    tail -f "$LOG"
elif [ "$1" == "watch" ]; then
    while true; do
        clear
        bash "$0" status
        sleep 60
    done
elif [ "$1" == "stop" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        echo "Sending SIGTERM to PID $PID..."
        kill -TERM "$PID"
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Still running after SIGTERM; sending SIGKILL"
            kill -9 "$PID"
        else
            echo "Stopped cleanly"
        fi
    else
        echo "No running eval found (PID file: $PID_FILE)"
    fi
elif [ "$1" == "force-kill" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "Sending SIGKILL to PID $PID"
        kill -9 "$PID"
    fi
elif [ "$1" == "resume" ]; then
    cd /home/motafeq/projects/sentinel
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH=.
    export TRITON_CACHE_DIR=/tmp/triton_cache
    if [ -f "$PID_FILE" ] && ps -p "$(cat $PID_FILE)" > /dev/null 2>&1; then
        echo "ALREADY RUNNING (PID $(cat $PID_FILE)). Stop first with: bash $0 stop"
        exit 1
    fi
    nohup ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py --checkpoint 500 --log-every 30 --report-every 2000 --resume > /tmp/sb_wild_full.log 2>&1 &
    echo $! > "$PID_FILE"
    disown
    sleep 2
    echo "Resumed: PID $(cat $PID_FILE)"
else
    echo "Usage: $0 {status|tail|watch|stop|force-kill|resume}"
fi
