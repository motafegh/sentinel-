#!/usr/bin/env bash
# Run 8 epoch monitor — writes summary to /tmp/run8_monitor_log.txt
# Called by OS crontab every ~35 min (epoch cadence)

LOG=/tmp/run8_v10.log
OUT=/tmp/run8_monitor_log.txt
TS=$(date '+%Y-%m-%d %H:%M:%S')

if [[ ! -f "$LOG" ]]; then
    echo "[$TS] LOG NOT FOUND: $LOG" >> "$OUT"
    exit 0
fi

# Extract most recent epoch summary line
EPOCH_LINE=$(grep "^2026.*Epoch [0-9]*/100 |" "$LOG" | tail -1)
BEST_LINE=$(grep "★ New best F1-macro" "$LOG" | tail -1)
JK_LINE=$(grep "JK attention weights" "$LOG" | tail -1)
STEP_LINE=$(grep "Step [0-9]*/455" "$LOG" | tail -1)
WARN_LINES=$(grep -v "prefix_attention_mean diagnostic" "$LOG" | grep "WARNING\|ERROR" | tail -5)

if [[ -z "$EPOCH_LINE" ]]; then
    echo "[$TS] No epoch data yet in log." >> "$OUT"
    exit 0
fi

# Parse epoch number and F1
EP=$(echo "$EPOCH_LINE" | grep -oP 'Epoch \K[0-9]+')
F1=$(echo "$EPOCH_LINE" | grep -oP 'F1-macro=\K[0-9.]+')
LOSS=$(echo "$EPOCH_LINE" | grep -oP 'Loss=\K[0-9.]+')
SPEED=$(echo "$EPOCH_LINE" | grep -oP '[0-9.]+ min/ep')
VRAM=$(echo "$EPOCH_LINE" | grep -oP 'VRAM: [0-9.]+/[0-9.]+ GiB \([0-9.]+%\)')
TOP3=$(grep -A2 "Epoch $EP/100 |" "$LOG" | grep "Top3:" | tail -1 | sed 's/.*Top3://')
BOT3=$(grep -A3 "Epoch $EP/100 |" "$LOG" | grep "Bottom3:" | tail -1 | sed 's/.*Bottom3://')

# Parse best F1
BEST_F1=$(echo "$BEST_LINE" | grep -oP 'New best F1-macro: \K[0-9.]+')
BEST_EP=$(grep "★ New best" "$LOG" | tail -1 | grep -oP 'Epoch \K[0-9]+' || \
          grep -B1 "★ New best" "$LOG" | grep "Epoch.*F1-macro" | tail -1 | grep -oP 'Epoch \K[0-9]+')

# Parse JK Phase3
PH3=$(echo "$JK_LINE" | grep -oP 'Phase3=\K[0-9.]+')
PH3_FLAG=""
if (( $(echo "$PH3 > 0.40" | bc -l 2>/dev/null) )); then PH3_FLAG=" ⚠ RED";
elif (( $(echo "$PH3 > 0.38" | bc -l 2>/dev/null) )); then PH3_FLAG=" ⚠ WARN"; fi

# Parse current step (if epoch in progress)
CUR_STEP=$(echo "$STEP_LINE" | grep -oP 'Step \K[0-9]+/455')

# Check if training finished
FINISHED=$(grep -E "Training complete|Early stopping triggered|TrainingAbortError" "$LOG" | tail -1)

{
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[$TS]  RUN 8 MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$FINISHED" ]]; then
    echo "  STATUS: $FINISHED"
else
    echo "  Completed: ep${EP}  |  In progress: step $CUR_STEP"
fi
echo "  F1 (last ep):  $F1"
echo "  Best F1:       $BEST_F1  (best_epoch=$BEST_EP)"
echo "  Loss:          $LOSS  |  Speed: $SPEED  |  VRAM: $VRAM"
echo "  JK Phase3:     $PH3${PH3_FLAG}"
echo "  Top3: $TOP3"
echo "  Bot3: $BOT3"
if [[ -n "$WARN_LINES" ]]; then
    echo "  WARNINGS:"
    echo "$WARN_LINES" | sed 's/^/    /'
fi
echo ""
} >> "$OUT"

# Notify via notify-send if available
if command -v notify-send &>/dev/null; then
    MSG="Run8 ep${EP} F1=${F1} (best=${BEST_F1})"
    notify-send -t 8000 "SENTINEL Run 8" "$MSG" 2>/dev/null || true
fi
