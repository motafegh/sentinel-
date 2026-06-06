#!/usr/bin/env bash
# Run 9 event-driven log watcher.
# Fires on specific trigger words in /tmp/run9_v11.log.
# Sends Windows toast notifications (WSL2) + appends to /tmp/run9_events.txt.
# Usage: bash run9_watcher.sh [start|stop|status]

LOG=/tmp/run9_v11.log
EVENTS=/tmp/run9_events.txt
PID_FILE=/tmp/run9_watcher.pid
WATCHER_LOG=/tmp/run9_watcher_err.log

# ── Windows toast notification via PowerShell ────────────────────────────────
toast() {
    local title="$1"
    local body="$2"
    powershell.exe -Command "
\$xml = '<toast><visual><binding template=\"ToastText02\"><text id=\"1\">$title</text><text id=\"2\">$body</text></binding></visual></toast>'
\$toastXml = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime]::New()
\$toastXml.LoadXml(\$xml)
\$toast = [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime]::New(\$toastXml)
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]::CreateToastNotifier('SENTINEL Run9').Show(\$toast)
" 2>/dev/null &
}

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
            echo "Watcher RUNNING (PID $PID)"
            echo "Events: $EVENTS ($(wc -l < "$EVENTS" 2>/dev/null || echo 0) lines)"
            echo "Last event: $(tail -1 "$EVENTS" 2>/dev/null)"
        else
            echo "PID file exists but process $PID is dead — run 'start' to restart."
        fi
    else
        echo "Watcher NOT running."
    fi
}

start_watcher() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Watcher already running (PID $PID)."
            exit 0
        fi
    fi

    touch "$EVENTS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started — watching $LOG" >> "$EVENTS"

    (
        # Wait up to 60s for the log to appear
        for i in $(seq 1 12); do [[ -f "$LOG" ]] && break; sleep 5; done
        [[ ! -f "$LOG" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: log never appeared" >> "$EVENTS" && exit 1

        tail -F "$LOG" 2>/dev/null | while IFS= read -r line; do

            # ── NEW BEST ──────────────────────────────────────────────────
            if echo "$line" | grep -q "★ New best F1-macro"; then
                F1=$(echo "$line" | grep -oP 'New best F1-macro: \K[0-9.]+')
                EP=$(grep -oP 'Epoch \K[0-9]+' "$LOG" 2>/dev/null | tail -1)
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] 🏆 NEW_BEST  ep=${EP}  F1=${F1}" >> "$EVENTS"
                toast "★ Run9 New Best!" "F1=${F1} at ep${EP}"

            # ── EPOCH COMPLETE ────────────────────────────────────────────
            elif echo "$line" | grep -qP "Epoch [0-9]+/100 \| Loss="; then
                EP=$(echo "$line"   | grep -oP 'Epoch \K[0-9]+')
                F1=$(echo "$line"   | grep -oP 'F1-macro=\K[0-9.]+')
                LOSS=$(echo "$line" | grep -oP 'Loss=\K[0-9.]+')
                SPEED=$(echo "$line" | grep -oP '[0-9.]+ min/ep')
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] ✓ EPOCH_END ep=${EP}  F1=${F1}  loss=${LOSS}  ${SPEED}" >> "$EVENTS"
                # Toast only on round epochs (every 5) to avoid spam
                if (( EP % 5 == 0 )); then
                    toast "Run9 ep${EP} done" "F1=${F1}  loss=${LOSS}"
                fi

            # ── JK PHASE3 WARNING ─────────────────────────────────────────
            elif echo "$line" | grep -q "JK attention weights"; then
                PH3=$(echo "$line" | grep -oP 'Phase3=\K[0-9.]+')
                if (( $(echo "$PH3 >= 0.38" | bc -l 2>/dev/null) )); then
                    EP=$(grep -oP 'Epoch \K[0-9]+' "$LOG" 2>/dev/null | tail -1)
                    TS=$(date '+%Y-%m-%d %H:%M:%S')
                    echo "[$TS] ⚠ JK_DRIFT  ep=${EP}  Phase3=${PH3}  (red=0.40)" >> "$EVENTS"
                    toast "⚠ Run9 JK Drift!" "Phase3=${PH3} at ep${EP} — red line is 0.40"
                fi

            # ── TRAINING DONE ─────────────────────────────────────────────
            elif echo "$line" | grep -qE "Training complete|Early stopping triggered|TrainingAbortError"; then
                TS=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$TS] 🏁 DONE  $line" >> "$EVENTS"
                BEST_F1=$(grep "★ New best F1-macro" "$LOG" | tail -1 | grep -oP 'New best F1-macro: \K[0-9.]+')
                toast "🏁 Run9 FINISHED" "Best F1=${BEST_F1:-?} — training complete"
            fi

        done
    ) >> "$WATCHER_LOG" 2>&1 &

    echo $! > "$PID_FILE"
    echo "[$(date '+%H:%M:%S')] Watcher started (PID $(cat $PID_FILE))"
    echo "  Events  → $EVENTS"
    echo "  Errors  → $WATCHER_LOG"
    echo "  Toasts  → Windows notifications on NEW_BEST + every 5th epoch + DONE"
}

case "${1:-start}" in
    start)  start_watcher ;;
    stop)   stop_watcher ;;
    status) status_watcher ;;
    *)      echo "Usage: $0 [start|stop|status]" ;;
esac
