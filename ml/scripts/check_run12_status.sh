#!/bin/bash
# check_run12_status.sh — Run 12 training status checker
#
# Runs every 5 min via cron. Two outputs:
#   1. ALWAYS log to ml/logs/run12_status_checks.log (silent, the user can tail)
#   2. FIRE notify-send ONLY on important events (new best, epoch complete, NaN, died)
#
# State file ml/logs/.run12_check_state tracks last-known best F1 + last epoch,
# so "new best" notifications fire only when something ACTUALLY changes (not on
# every check). The state file is in ml/logs/ which is gitignored.
#
# Reusable for future runs — just change RUN_NAME and the corresponding
# LOG_DIR / LAUNCH_LOG paths at the top.
#
# Cron entry (in user's crontab, NOT root):
#   */5 * * * * /home/motafeq/projects/sentinel/ml/scripts/check_run12_status.sh

set -uo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
RUN_NAME="GCB-P1-Run12-v3dospatched-20260613"
PROJECT_ROOT="$HOME/projects/sentinel"
LOG_DIR="$PROJECT_ROOT/ml/logs/$RUN_NAME"
LAUNCH_LOG="$PROJECT_ROOT/ml/logs/run12_launch_2026-06-13.log"
STATE_FILE="$PROJECT_ROOT/ml/logs/.run12_check_state"
CHECK_LOG="$PROJECT_ROOT/ml/logs/run12_status_checks.log"
NOTIFY_LOG="$PROJECT_ROOT/ml/logs/run12_notifications.log"
PYTHON="$PROJECT_ROOT/ml/.venv/bin/python"

# ─── Helper functions ─────────────────────────────────────────────────────────
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2" >> "$CHECK_LOG"
}

notify() {
  # Usage: notify <priority> <title> <body>
  # priority: low | normal | critical
  #
  # Tries 3 notification paths in order (first success wins):
  #   1. notify-send        (Linux desktop, may fail in WSL2)
  #   2. PowerShell toast    (Windows desktop via WSL → powershell.exe)
  #   3. Always log to NOTIFY_LOG (silent, the user can tail)
  local priority="$1"
  local title="$2"
  local body="$3"
  local icon="dialog-information"
  case "$priority" in
    critical) icon="dialog-error";;
  esac
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$timestamp] [NOTIFY/${priority^^}] $title — $body" >> "$NOTIFY_LOG"

  # Attempt 1: notify-send (Linux)
  if command -v notify-send >/dev/null 2>&1; then
    if notify-send -u "$priority" -i "$icon" "$title" "$body" 2>/dev/null; then
      return 0
    fi
  fi

  # Attempt 2: PowerShell toast notification (Windows desktop via WSL)
  # Uses System.Windows.Forms.MessageBox for guaranteed availability.
  # Quiet MessageBox (no sound, single OK button that auto-closes) for low-priority.
  # For critical: regular MessageBox so user has to dismiss.
  if command -v powershell.exe >/dev/null 2>&1; then
    # Escape single quotes for PowerShell
    local ps_title="${title//\'/\'\'}"
    local ps_body="${body//\'/\'\'}"
    local ps_msg="Run 12 Monitor\n[$priority] $ps_title\n\n$ps_body"
    if [ "$priority" = "critical" ]; then
      # Regular MessageBox, user must click OK
      powershell.exe -NoProfile -Command "
        Add-Type -AssemblyName System.Windows.Forms;
        [System.Windows.Forms.MessageBox]::Show('$ps_msg', 'SENTINEL Run 12 CRITICAL', 'OK', 'Warning') | Out-Null
      " 2>/dev/null &
    else
      # Auto-closing messagebox for low/normal (no user interaction)
      powershell.exe -NoProfile -Command "
        Add-Type -AssemblyName System.Windows.Forms;
        \$msg = New-Object System.Windows.Forms.Form;
        \$msg.Text = 'SENTINEL Run 12 - $ps_title';
        \$msg.Size = New-Object System.Drawing.Size(500, 200);
        \$label = New-Object System.Windows.Forms.Label;
        \$label.Text = '$ps_msg';
        \$label.AutoSize = \$true;
        \$label.Location = New-Object System.Drawing.Point(10, 10);
        \$msg.Controls.Add(\$label);
        \$msg.Topmost = \$true;
        \$msg.Show();
        Start-Sleep -Seconds 5;
        \$msg.Close()
      " 2>/dev/null &
    fi
    return 0
  fi

  # Attempt 3: just log (silent) — user can tail NOTIFY_LOG
  return 0
}

# ─── 1. Process check ──────────────────────────────────────────────────────────
PID=$(ps aux | grep "train.py" | grep "$RUN_NAME" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$PID" ]; then
  log "CRITICAL" "Run 12 process NOT FOUND"
  notify "critical" "Run 12 DEAD" "Process $RUN_NAME not running. Last 5 launch-log lines: $(tail -5 "$LAUNCH_LOG" 2>/dev/null | tr '\n' ' ' | head -c 300)"
  exit 0
fi

ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')

# ─── 2. Check if we have any epoch results yet ──────────────────────────────
if [ ! -f "$LOG_DIR/epoch_summary.jsonl" ]; then
  log "OK" "Run 12 alive ($ELAPSED elapsed), epoch 1 in progress (no summary yet)"
  exit 0
fi

# ─── 3. Read latest epoch summary ────────────────────────────────────────────
EPOCH_SUMMARY=$(tail -1 "$LOG_DIR/epoch_summary.jsonl")
if [ -z "$EPOCH_SUMMARY" ]; then
  log "WARN" "Run 12 alive ($ELAPSED), epoch_summary.jsonl empty"
  exit 0
fi

# Parse with python (safer than jq on edge cases; handles None for f1_macro_tuned
# which is None between threshold_tune_interval runs)
read EPOCH_NUM F1_FIXED F1_TUNED DOS_F1 DOS_AUCPR <<<"$(echo "$EPOCH_SUMMARY" | "$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
e = d['epoch']
f1f = d.get('per_class_f1', {}).get('ExternalBug') or 0.0
f1t = d.get('f1_macro_tuned') or 'nan'  # None between tune intervals
dos_f1 = d.get('per_class_f1', {}).get('DenialOfService') or 0.0
dos_pr = d.get('auc_pr_per_label', {}).get('DenialOfService') or 0.0
print(f'{e} {f1f:.4f} {f1t if isinstance(f1t, str) else f\"{f1t:.4f}\"} {dos_f1:.4f} {dos_pr:.4f}')
")"
# Handle NaN safely
[ -z "$F1_TUNED" ] && F1_TUNED="nan"

# ─── 4. State diff: detect new epoch or new best F1 ─────────────────────────
LAST_STATE=""
if [ -f "$STATE_FILE" ]; then
  LAST_STATE=$(cat "$STATE_FILE" 2>/dev/null)
fi
LAST_BEST=$(echo "$LAST_STATE" | "$PYTHON" -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('best_f1', 0))" 2>/dev/null || echo "0")
LAST_EPOCH_NUM=$(echo "$LAST_STATE" | "$PYTHON" -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('last_epoch', 0))" 2>/dev/null || echo "0")

NEW_BEST=0
if [ -n "$LAST_BEST" ] && [ "$F1_TUNED" != "nan" ] && [ "$(echo "$F1_TUNED > $LAST_BEST" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  NEW_BEST=1
fi

NEW_EPOCH=0
if [ "$EPOCH_NUM" != "$LAST_EPOCH_NUM" ]; then
  NEW_EPOCH=1
fi

# ─── 5. NaN / Inf / loss-spike detection ─────────────────────────────────────
NAN_ALERT=""
if echo "$EPOCH_SUMMARY" | grep -qiE '"(nan|inf)"' ; then
  NAN_MSG=$(echo "$EPOCH_SUMMARY" | "$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
fields = {k: v for k, v in d.items() if isinstance(v, float) and (v != v or v == float('inf'))}
print(','.join(fields.keys()) if fields else 'loss_spike')
" 2>/dev/null || echo "unknown")
  if [ -n "$NAN_MSG" ]; then
    NAN_ALERT="NaN/Inf in: $NAN_MSG"
  fi
fi
if [ -n "$NAN_ALERT" ]; then
  log "CRITICAL" "NaN/Inf detected in ep$EPOCH_NUM metrics: $NAN_ALERT"
  notify "critical" "Run 12 NaN/Inf" "ep$EPOCH_NUM has $NAN_ALERT"
fi

# ─── 6. Alerts.jsonl check ──────────────────────────────────────────────────
NEW_ALERT_MSG=""
if [ -s "$LOG_DIR/alerts.jsonl" ]; then
  LAST_ALERT_COUNT=$(echo "$LAST_STATE" | "$PYTHON" -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('alert_count', 0))" 2>/dev/null || echo "0")
  CUR_ALERT_COUNT=$(wc -l < "$LOG_DIR/alerts.jsonl")
  if [ "$CUR_ALERT_COUNT" -gt "$LAST_ALERT_COUNT" ]; then
    NEW_ALERT_LINE=$(tail -1 "$LOG_DIR/alerts.jsonl")
    NEW_ALERT_MSG=$(echo "$NEW_ALERT_LINE" | "$PYTHON" -c "
import json, sys
d = json.loads(sys.stdin.read())
msg = d.get('message', '')
print(msg[:300])
" 2>/dev/null)
    if [ -z "$NEW_ALERT_MSG" ]; then
      NEW_ALERT_MSG="(unparseable alert — see $LOG_DIR/alerts.jsonl)"
    fi
  fi
fi

# ─── 7. Stalled training detection ──────────────────────────────────────────
STALL_ALERT=""
if [ -f "$LOG_DIR/step_metrics.jsonl" ]; then
  STEP_MTIME=$(stat -c %Y "$LOG_DIR/step_metrics.jsonl" 2>/dev/null)
  NOW=$(date +%s)
  if [ -n "$STEP_MTIME" ]; then
    AGE=$(( NOW - STEP_MTIME ))
    if [ "$AGE" -gt 1800 ]; then  # 30 min no new step
      STALL_ALERT="step_metrics.jsonl not updated in ${AGE}s"
    fi
  fi
fi

# ─── 8. Decide whether to fire notification ──────────────────────────────────
if [ "$NEW_EPOCH" = "1" ]; then
  log "OK" "ep$EPOCH_NUM DONE — f1_tuned=$F1_TUNED, DoS_F1=$DOS_F1, DoS_AUCPR=$DOS_AUCPR, ExternalBug_F1=$F1_FIXED (elapsed $ELAPSED)"
  TITLE="Run 12 ep$EPOCH_NUM complete"
  BODY="f1_tuned=$F1_TUNED (was $LAST_BEST)
DoS F1=$DOS_F1, AUC-PR=$DOS_AUCPR
ExternalBug F1=$F1_FIXED
Elapsed: $ELAPSED"
  if [ "$NEW_BEST" = "1" ]; then
    BODY="$BODY
NEW BEST F1 ↑"
  fi
  if [ -n "$NAN_ALERT" ]; then
    notify "critical" "$TITLE (NaN/Inf!)" "$BODY"
  elif [ -n "$STALL_ALERT" ]; then
    notify "normal" "$TITLE (stalled?)" "$BODY
$STALL_ALERT"
  elif [ "$NEW_BEST" = "1" ] && [ -n "$NEW_ALERT_MSG" ]; then
    notify "normal" "$TITLE (new best, new alert)" "$BODY
Latest alert: $NEW_ALERT_MSG"
  elif [ "$NEW_BEST" = "1" ]; then
    notify "normal" "$TITLE (new best)" "$BODY"
  else
    notify "low" "$TITLE" "$BODY"
  fi
elif [ -n "$NAN_ALERT" ]; then
  # Already notified above
  true
elif [ -n "$STALL_ALERT" ]; then
  log "WARN" "$STALL_ALERT"
  if ! grep -q "$STALL_ALERT" "$NOTIFY_LOG" 2>/dev/null; then
    notify "normal" "Run 12 STALLED?" "ep$EPOCH_NUM: $STALL_ALERT. Process $PID still alive."
  fi
elif [ -n "$NEW_ALERT_MSG" ]; then
  log "WARN" "New alert in ep$EPOCH_NUM: $NEW_ALERT_MSG"
  if ! grep -qF "$NEW_ALERT_MSG" "$NOTIFY_LOG" 2>/dev/null; then
    notify "normal" "Run 12 NEW ALERT" "$NEW_ALERT_MSG"
  fi
else
  log "OK" "Run 12 ep$EPOCH_NUM (no new epoch, f1_tuned=$F1_TUNED, elapsed $ELAPSED)"
fi

# ─── 9. Save state ──────────────────────────────────────────────────────────
echo "{\"best_f1\": $F1_TUNED, \"last_epoch\": $EPOCH_NUM, \"alert_count\": $CUR_ALERT_COUNT, \"last_check\": $(date +%s)}" > "$STATE_FILE"

exit 0
