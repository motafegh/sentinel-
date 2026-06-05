#!/usr/bin/env bash
# Run 8 log watcher — event-driven, not time-based.
# Monitors /tmp/run8_v10.log with tail -F.
# Fires when trigger words appear, writes to /tmp/run8_events.txt.
# Usage: bash run8_watcher.sh [start|stop|status]

LOG=/tmp/run8_v10.log
EVENTS=/tmp/run8_events.txt
PID_FILE=/tmp/run8_watcher.pid
WATCHER_LOG=/tmp/run8_watcher_err.log

# ── Trigger patterns ────────────────────────────────────────────────────────
# Each pattern maps to an event tag shown in the events file.
# Ordered from most frequent to least (grep short-circuits on first match).
TRIGGER_NEW_BEST="★ New best F1-macro"
TRIGGER_EPOCH_END="Epoch [0-9]*/100 | Loss="
TRIGGER_JK="JK attention weights"
TRIGGER_DONE="Training complete\|Early stopping triggered\|TrainingAbortError"

stop_watcher() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null
            echo "[$(date '+%H:%M:%S')] Watcher PID $PID stopped."
        fi
        rm -f "$PID_FILE"
    else
        echo "No watcher running."
    fi
}

status_watcher() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Watcher running (PID $PID)."
            echo "Events file: $EVENTS ($(wc -l < "$EVENTS" 2>/dev/null || echo 0) lines)"
        else
            echo "PID file exists but process $PID is dead."
        fi
    else
        echo "Watcher not running."
    fi
}

start_watcher() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Watcher already running (PID $PID). Use 'stop' first."
            exit 0
        fi
    fi

    touch "$EVENTS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started — watching $LOG" >> "$EVENTS"

    # Inner loop: process log lines as they appear
    (
        # Wait up to 60s for the log to appear
        for i in $(seq 1 12); do
            [[ -f "$LOG" ]] && break
            sleep 5
        done
        [[ ! -f "$LOG" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] LOG NEVER APPEARED: $LOG" >> "$EVENTS" && exit 1

        tail -F "$LOG" 2>/dev/null | while IFS= read -r line; do

            # ── NEW BEST ──────────────────────────────────────────────────
            if echo "$line" | grep -q "$TRIGGER_NEW_BEST"; then
                F1=$(echo "$line" | grep -oP 'New best F1-macro: \K[0-9.]+')
                EP=$(echo "$line" | grep -oP 'Epoch \K[0-9]+' 2>/dev/null || echo "?")
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] 🏆 NEW_BEST  ep=${EP}  F1=${F1}" >> "$EVENTS"
                command -v notify-send &>/dev/null && \
                    notify-send -t 6000 "SENTINEL Run8 ★" "New best F1=${F1} at ep${EP}" 2>/dev/null || true

            # ── EPOCH COMPLETE ────────────────────────────────────────────
            elif echo "$line" | grep -qP "Epoch [0-9]+/100 \| Loss="; then
                EP=$(echo "$line"  | grep -oP 'Epoch \K[0-9]+')
                F1=$(echo "$line"  | grep -oP 'F1-macro=\K[0-9.]+')
                LOSS=$(echo "$line" | grep -oP 'Loss=\K[0-9.]+')
                SPEED=$(echo "$line" | grep -oP '[0-9.]+ min/ep')
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] ✓ EPOCH_END ep=${EP}  F1=${F1}  loss=${LOSS}  ${SPEED}" >> "$EVENTS"

            # ── JK PHASE3 WARNING ─────────────────────────────────────────
            elif echo "$line" | grep -q "$TRIGGER_JK"; then
                PH3=$(echo "$line" | grep -oP 'Phase3=\K[0-9.]+')
                # Only log if Phase3 >= 0.38 (warn threshold)
                if (( $(echo "$PH3 >= 0.38" | bc -l 2>/dev/null) )); then
                    EP=$(grep -oP 'Epoch \K[0-9]+' "$LOG" 2>/dev/null | tail -1)
                    TS=$(date '+%Y-%m-%d %H:%M:%S')
                    echo "[$TS] ⚠ JK_DRIFT   ep=${EP}  Phase3=${PH3}  ← approaching 0.40 red line" >> "$EVENTS"
                    command -v notify-send &>/dev/null && \
                        notify-send -u critical -t 10000 "SENTINEL Run8 ⚠" "JK Phase3=${PH3} (red line=0.40)" 2>/dev/null || true
                fi

            # ── TRAINING DONE ─────────────────────────────────────────────
            elif echo "$line" | grep -qE "Training complete|Early stopping triggered|TrainingAbortError"; then
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] 🏁 DONE  $line" >> "$EVENTS"
                command -v notify-send &>/dev/null && \
                    notify-send -u critical -t 0 "SENTINEL Run8 DONE" "$line" 2>/dev/null || true
            fi

        done
    ) >> "$WATCHER_LOG" 2>&1 &

    echo $! > "$PID_FILE"
    echo "[$(date '+%H:%M:%S')] Watcher started (PID $(cat $PID_FILE)) — events → $EVENTS"
}

case "${1:-start}" in
    start)  start_watcher ;;
    stop)   stop_watcher ;;
    status) status_watcher ;;
    *)      echo "Usage: $0 [start|stop|status]" ;;
esac
