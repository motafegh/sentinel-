#!/bin/bash
# Wrapper to launch the full eval in background, capture PID and log
cd /home/motafeq/projects/sentinel
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.
export TRITON_CACHE_DIR=/tmp/triton_cache
nohup ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py --checkpoint 500 --log-every 30 --report-every 2000 > /tmp/sb_wild_full.log 2>&1 &
echo $! > /tmp/sb_wild_full.pid
disown
sleep 1
echo "Launched PID: $(cat /tmp/sb_wild_full.pid)"
