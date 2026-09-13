"""Best-effort local provenance capture for persisted experimental records."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def capture_git_commit() -> str | None:
    """Return Jaull's checked-out commit without making provenance a hard dependency.

    Installed distributions and source archives legitimately have no ``.git``
    directory. Experimental execution must still work there, so failures are
    represented as unavailable provenance rather than propagated.
    """

    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


__all__ = ["capture_git_commit"]
