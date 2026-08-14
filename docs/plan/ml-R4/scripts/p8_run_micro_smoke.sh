#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
RUNNER="$SCRIPT_DIR/p8_run_micro_smoke.py"

if [[ -n "${SENTINEL_PYTHON:-}" ]]; then
  PYTHON="$SENTINEL_PYTHON"
elif [[ -x "$REPO_ROOT/ml/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/ml/.venv/bin/python"
elif [[ -x "$HOME/projects/sentinel/ml/.venv/bin/python" ]]; then
  PYTHON="$HOME/projects/sentinel/ml/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

exec "$PYTHON" "$RUNNER" "$@"
