"""
scheduler_cron.py

Manages cron-based scheduling for SENTINEL's ingestion pipeline.

RECALL — Cron is Linux's built-in job scheduler.
This script installs/removes/checks the crontab entry programmatically
so we never have to manually edit crontab files.

RECALL — WSL2 limitation:
Cron daemon must be started manually in WSL2 and stops when the
session closes. For always-on scheduling, use GitHub Actions.
Cron is useful for local development and testing.

CHANGES (2026-04-11):
  FIX-27: _check_cron_daemon() now checks exit code instead of
          string "running" in stdout. The old check:
            "running" in result.stdout.lower()
          is broken because "not running" contains the substring "running"
          → a stopped cron daemon was silently reported as RUNNING.
  FIX-28: Removed duplicate `import subprocess` inside run_now().
          subprocess is already imported at module level (line 5).

Run from agents/ directory:
  poetry run python -m src.ingestion.scheduler_cron install
  poetry run python -m src.ingestion.scheduler_cron remove
  poetry run python -m src.ingestion.scheduler_cron status
  poetry run python -m src.ingestion.scheduler_cron run-now
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────────
# Absolute paths required — cron runs without your shell's PATH or CWD set.
AGENTS_DIR   = Path(__file__).parent.parent.parent.resolve()
PYTHON_PATH  = AGENTS_DIR / ".venv" / "bin" / "python"
LOG_PATH     = AGENTS_DIR.parent / "logs" / "ingestion_cron.log"

CRON_SCHEDULE = "0 2 * * *"          # daily at 2 AM UTC
CRON_MARKER   = "# SENTINEL_INGESTION"   # unique marker for find/update

# cd to agents/ first so relative paths inside pipeline.py resolve correctly.
# >> log_path 2>&1 captures both stdout and stderr.
CRON_COMMAND = (
    f"cd {AGENTS_DIR} && "
    f"{PYTHON_PATH} -m src.ingestion.pipeline "
    f">> {LOG_PATH} 2>&1"
)

CRON_LINE = f"{CRON_SCHEDULE} {CRON_COMMAND} {CRON_MARKER}"


def get_current_crontab() -> str:
    """Read current crontab content. Returns empty string if none exists."""
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    # crontab -l returns exit code 1 if no crontab exists — not an error for us
    if result.returncode != 0:
        return ""
    return result.stdout


def set_crontab(content: str) -> bool:
    """Write new crontab content."""
    result = subprocess.run(["crontab", "-"], input=content, capture_output=True, text=True)
    return result.returncode == 0


def install() -> None:
    """
    Install SENTINEL ingestion job into user crontab (idempotent).

    Checks for existing CRON_MARKER entry — updates it if found,
    appends a new entry if not. Safe to run multiple times.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    current = get_current_crontab()

    # Remove any existing SENTINEL entry (handles update case)
    lines = [line for line in current.splitlines() if CRON_MARKER not in line]
    lines.append(CRON_LINE)
    new_crontab = "\n".join(lines) + "\n"

    if set_crontab(new_crontab):
        logger.info("Cron job installed successfully")
        logger.info(f"  Schedule: {CRON_SCHEDULE} (daily at 2 AM)")
        logger.info(f"  Command:  {CRON_COMMAND[:60]}...")
        logger.info(f"  Logs:     {LOG_PATH}")
        logger.info(f"  Verify:   crontab -l")
    else:
        logger.error("Failed to install cron job")

    _check_cron_daemon()


def remove() -> None:
    """Remove SENTINEL ingestion job from crontab."""
    current     = get_current_crontab()
    lines       = [line for line in current.splitlines() if CRON_MARKER not in line]
    new_crontab = "\n".join(lines) + "\n"

    if set_crontab(new_crontab):
        logger.info("Cron job removed")
    else:
        logger.error("Failed to remove cron job")


def status() -> None:
    """Show current crontab status for SENTINEL jobs."""
    current         = get_current_crontab()
    sentinel_lines  = [line for line in current.splitlines() if CRON_MARKER in line]

    if sentinel_lines:
        logger.info("SENTINEL cron job: INSTALLED")
        for line in sentinel_lines:
            logger.info(f"  {line}")
    else:
        logger.info("SENTINEL cron job: NOT INSTALLED")
        logger.info("  Run: poetry run python -m src.ingestion.scheduler_cron install")

    _check_cron_daemon()

    if LOG_PATH.exists():
        logger.info(f"\nRecent log ({LOG_PATH}):")
        lines = LOG_PATH.read_text().splitlines()
        for line in lines[-10:]:
            logger.info(f"  {line}")
    else:
        logger.info(f"No log file yet at {LOG_PATH}")


def run_now() -> None:
    """
    Run the ingestion pipeline immediately (for testing).

    Bypasses cron entirely — useful to verify the pipeline works
    before installing the schedule.

    FIX-28: Removed duplicate `import subprocess` that was here.
            subprocess is imported at module level.
    """
    logger.info("Running ingestion pipeline manually...")
    result = subprocess.run(
        [str(PYTHON_PATH), "-m", "src.ingestion.pipeline"],
        cwd=str(AGENTS_DIR),
        capture_output=False,   # show output directly in terminal
    )
    sys.exit(result.returncode)


def _check_cron_daemon() -> None:
    """
    Check if cron daemon is running.

    FIX-27: Now checks exit code instead of string matching.
            Old: "running" in result.stdout.lower()
            Bug: "cron is not running" contains the substring "running"
                 → a stopped daemon was reported as RUNNING (false positive).
            New: result.returncode == 0 is the reliable OS-level signal.
                 `service cron status` exits 0 when running, non-zero when not.

    RECALL — In WSL2, cron must be started manually:
      sudo service cron start
    For auto-start, add to /etc/wsl.conf:
      [boot]
      command = service cron start
    """
    result = subprocess.run(
        ["service", "cron", "status"],
        capture_output=True,
        text=True,
    )

    # FIX-27: Use exit code — reliable across all distros and cron variants.
    if result.returncode == 0:
        logger.info("Cron daemon: RUNNING")
    else:
        logger.warning("Cron daemon: NOT RUNNING")
        logger.warning("  Start with: sudo service cron start")
        logger.warning("  For WSL2 auto-start, add to /etc/wsl.conf:")
        logger.warning("    [boot]")
        logger.warning("    command = service cron start")


if __name__ == "__main__":
    commands = {
        "install":  install,
        "remove":   remove,
        "status":   status,
        "run-now":  run_now,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python -m src.ingestion.scheduler_cron [{' | '.join(commands)}]")
        sys.exit(1)

    commands[sys.argv[1]]()
