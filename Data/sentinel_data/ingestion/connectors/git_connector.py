"""Git connector — clone a repo at a pinned commit and collect .sol files.

Clone strategy:
- If `pin` is empty  → shallow clone (--depth 1) of the default branch at HEAD.
- If `pin` is set    → clone with full history then checkout the pinned commit,
                       so the SHA is verifiable and the audit trail is complete.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from sentinel_data.ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    PullResult,
    SourceConfig,
)


class GitConnector(BaseConnector):

    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        if not cfg.url:
            raise ConnectorError(f"[{cfg.name}] git connector requires a url")

        repo_dir = dest / "repo"

        if repo_dir.exists():
            resolved = self._current_commit(repo_dir)
        else:
            resolved = self._clone(cfg, repo_dir)

        # run any post-clone command (e.g. `python checkout_sources.py` for scabench)
        post_cmd = cfg.extra.get("post_clone_cmd")
        if post_cmd and not (dest / ".post_clone_done").exists():
            _run(post_cmd.split(), cwd=repo_dir)
            (dest / ".post_clone_done").touch()

        sol_files = self.find_sol_files(repo_dir)
        return PullResult(
            source=cfg.name,
            local_dir=repo_dir,
            resolved_pin=resolved,
            sol_files=sol_files,
            fetched_at="",     # filled by BaseConnector.pull()
            duration_s=0.0,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clone(self, cfg: SourceConfig, repo_dir: Path) -> str:
        if cfg.pin:
            # Full clone so we can checkout an arbitrary commit
            _run(["git", "clone", "--quiet", cfg.url, str(repo_dir)])
            _run(["git", "checkout", cfg.pin], cwd=repo_dir)
            return cfg.pin
        else:
            # Shallow clone for speed when no pin is specified
            _run(["git", "clone", "--depth", "1", "--quiet", cfg.url, str(repo_dir)])
            return self._current_commit(repo_dir)

    @staticmethod
    def _current_commit(repo_dir: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConnectorError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
