# Run 12 monitoring setup (2026-06-13)

> **Status:** ACTIVE — cron job + log checker + 3-path notification system.
> Cron was added at 00:40 UTC, first tick fired at 00:40 UTC, log growing every 5 min.

## What's set up

| Component | Path | Purpose |
|---|---|---|
| **Log checker script** | `ml/scripts/check_run12_status.sh` | Reads training state, detects important events, fires notifications |
| **Cron entry** | `crontab -l` (every 5 min) | `*/5 * * * * /home/motafeq/projects/sentinel/ml/scripts/check_run12_status.sh` |
| **Status log** | `ml/logs/run12_status_checks.log` | ONE LINE per check, silent. Tail this to see status. |
| **Notification log** | `ml/logs/run12_notifications.log` | One entry per FIRED notification. Tail to see alert history. |
| **Cron wrapper log** | `ml/logs/check_run12_cron.log` | Cron's stdout/stderr capture. Will be empty if script runs silently. |
| **State file** | `ml/logs/.run12_check_state` | JSON: `{best_f1, last_epoch, alert_count, last_check}` — used for "new best" detection |

## How the log checker decides what to fire

**Always:** log every check to `ml/logs/run12_status_checks.log` (one line, timestamped, `OK`/`WARN`/`CRITICAL`).

**Fires notifications only on these events:**

| Event | Priority | Trigger condition |
|---|---|---|
| **Process died** | critical | `ps aux | grep train.py` returns nothing |
| **NaN/Inf in metrics** | critical | Any of the `f1_macro_tuned`, `per_class_f1`, `auc_pr_per_label` etc. is NaN/Inf |
| **New epoch complete** | low / normal / critical | New row in `epoch_summary.jsonl`. Priority: `critical` if NaN also detected, `normal` if new best F1, else `low` |
| **New best F1** | normal | `f1_macro_tuned` improved over last check's value |
| **New alert in alerts.jsonl** | normal | New entry in `alerts.jsonl` since last check |
| **Training stalled** | normal | `step_metrics.jsonl` not updated in >30 min, but process alive |

## Notification paths (3-tier fallback)

The `notify()` function tries 3 paths in order, first success wins:

1. **`notify-send`** (Linux desktop) — may fail in WSL2 with "GDBus.Error.ServiceUnknown" (no desktop service). Logged but doesn't break the script.

2. **PowerShell toast** (Windows desktop via WSL `powershell.exe`) — auto-closing MessageBox (5 sec display). For `critical` priority, uses regular MessageBox (user must dismiss). This is the path that will fire on the user's actual desktop.

3. **Always log to `ml/logs/run12_notifications.log`** — silent, the user can tail.

In WSL2: only path #2 will actually produce a visible notification. Path #1 fails silently (no desktop). Path #3 always works.

## How to monitor (for Ali)

```bash
# Live-tail the status log (one line per check, every 5 min)
tail -f /home/motafeq/projects/sentinel/ml/logs/run12_status_checks.log

# Check what notifications have fired
tail -f /home/motafeq/projects/sentinel/ml/logs/run12_notifications.log

# Verify cron is firing
crontab -l | grep SENTINEL_RUN12_MONITOR

# Verify process is still alive
ps -p 230342 -o pid,etime,cmd

# Manually run the checker (bypass cron)
bash /home/motafeq/projects/sentinel/ml/scripts/check_run12_status.sh
```

## How to disable / remove

```bash
# Remove just this cron entry
crontab -l | grep -v SENTINEL_RUN12_MONITOR | crontab -

# Or disable temporarily
crontab -e
# Comment out the */5 * * * * line
```

## Cron entry details (installed 2026-06-13 23:40 UTC)

```cron
0 2 * * * cd /home/motafeq/projects/sentinel/agents && /home/motafeq/projects/sentinel/agents/.venv/bin/python -m src.ingestion.pipeline >> /home/motafeq/projects/sentinel/logs/ingestion_cron.log 2>&1 # SENTINEL_INGESTION
*/30 * * * * /home/motafeq/projects/sentinel/ml/scripts/push_log_snapshot.sh >> /tmp/push_log_snapshot.log 2>&1 # SENTINEL_TRAINING_SNAPSHOT
*/5 * * * * /home/motafeq/projects/sentinel/ml/scripts/check_run12_status.sh >> /home/motafeq/projects/sentinel/ml/logs/check_run12_cron.log 2>&1 # SENTINEL_RUN12_MONITOR
```

(3 entries total: agent ingestion at 2 AM daily, training snapshot every 30 min, Run 12 monitor every 5 min)

## Reusability

The script is **specific to Run 12** (hardcoded `RUN_NAME` and `LOG_DIR`). For future runs (Run 13+), either:
- Edit the top of the script to change `RUN_NAME` and `LOG_DIR`
- Or generalize to accept `RUN_NAME` as a parameter and add multiple cron entries

## Tested events (Run 12 ep1 → ep2)

| Time | Event | Notification fired? |
|---|---|---|
| 00:31:17 | Run 12 launched | (no script running yet) |
| 00:36:36 | First script run (manual test, ep1 results not in JSONL) | (logged, no notify) |
| 00:38:35 | Manual test (ep1 results now in JSONL) | (logged, no notify) |
| 00:40:02 | First cron tick | (logged, no notify) |
| 00:45:01 | Second cron tick | (logged, no notify) |
| 00:46:38 | Third cron tick (script testing) | (logged, no notify) |
| 00:50+ | ... continuing every 5 min | (logged, no notify unless event) |

**As of 2026-06-14 00:46 UTC, no notifications have fired** because no NaN, no death, no new best F1, no new alerts since the script started.

## Expected next events (likely times)

| Expected event | When (approximate) | Notification |
|---|---|---|
| Ep2 complete | ~01:35 UTC (currently at step 200/282 at 00:18, ~8 min/100steps) | "Run 12 ep2 complete" with f1_tuned=nan (tuning only at ep10) |
| Ep3 complete | ~03:10 UTC | Same |
| Ep10 complete (FIRST threshold tuning) | ~22:00 UTC next day | "Run 12 ep10 complete (new best)" — first real f1_tuned value |
| New best F1 (if any) | Any time after ep10 | "Run 12 ep_N complete (new best)" |

## Files NOT committed (correctly)

All log files are in `ml/logs/` which is gitignored:
- `ml/logs/run12_status_checks.log` (silent, grows ~every 5 min)
- `ml/logs/run12_notifications.log` (only on events)
- `ml/logs/check_run12_cron.log` (cron wrapper, usually empty)
- `ml/logs/.run12_check_state` (state file)

Only the **script itself** is committed (`ml/scripts/check_run12_status.sh`).

## Reusability for other Claude sessions

If this Run 12 monitoring framework is useful for future runs, the same pattern can be applied. The script is the only file needed; cron entry is added once per run.
