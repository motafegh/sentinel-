#!/usr/bin/env bash
# Snapshot active training run JSONL logs → ml/training_snapshots/ → git push
# Run via local cron every 30 min while a training job is active.
set -euo pipefail

REPO_DIR="/home/motafeq/projects/sentinel"
LOG_DIR="$REPO_DIR/ml/logs"
SNAP_BASE="$REPO_DIR/ml/training_snapshots"

# Find the most recently modified epoch_summary.jsonl (= active run)
LATEST_EPOCH_FILE=$(find "$LOG_DIR" -name "epoch_summary.jsonl" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}')

if [ -z "$LATEST_EPOCH_FILE" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No epoch_summary.jsonl found — no active run." >&2
    exit 0
fi

RUN_DIR=$(dirname "$LATEST_EPOCH_FILE")
RUN_NAME=$(basename "$RUN_DIR")
DEST="$SNAP_BASE/$RUN_NAME"
mkdir -p "$DEST"

# Copy all three JSONL streams
cp "$RUN_DIR/epoch_summary.jsonl" "$DEST/"
cp "$RUN_DIR/alerts.jsonl"        "$DEST/" 2>/dev/null || true
cp "$RUN_DIR/step_metrics.jsonl"  "$DEST/" 2>/dev/null || true

# Write snapshot metadata
EPOCH_COUNT=$(wc -l < "$DEST/epoch_summary.jsonl" | tr -d ' ')
cat > "$DEST/snapshot_meta.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "snapshot_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "epochs_completed": $EPOCH_COUNT
}
EOF

cd "$REPO_DIR"

# Stage ONLY snapshot files — do not touch anything else that may be staged
git add ml/training_snapshots/

# Check if specifically the snapshot files differ from HEAD (not the full index)
if git diff --cached --quiet -- ml/training_snapshots/; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No changes — snapshot unchanged."
    exit 0
fi

# Commit ONLY the snapshot paths, leaving anything else staged untouched
git commit -- ml/training_snapshots/ -m "chore(snapshot): $RUN_NAME — epoch $EPOCH_COUNT @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin main
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushed snapshot for $RUN_NAME (ep $EPOCH_COUNT)."
