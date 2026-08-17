"""Per-user data locations that follow each OS's convention.

One tiny helper so every subsystem that persists something to disk (model
artifacts, exported reports, future run logs, …) agrees on a single root
under the user profile — no scattered files in the current working
directory, no dependency on ``platformdirs``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def user_data_dir(*subdirs: str) -> Path:
    """Return ``<per-user jaull data root>/<subdirs...>``.

    - Linux / other: ``$XDG_DATA_HOME/jaull`` or ``~/.local/share/jaull``.
    - macOS: ``~/Library/Application Support/jaull``.
    - Windows: ``%LOCALAPPDATA%\\jaull`` (falls back to
      ``~/AppData/Local/jaull`` if the env var is unset).

    The path is **not** created; call ``.mkdir(parents=True, exist_ok=True)``
    at the site that actually writes into it.
    """
    return _root().joinpath(*subdirs)


def user_cache_dir(*subdirs: str) -> Path:
    """Return ``<per-user jaull cache root>/<subdirs...>``.

    - Linux / other: ``$XDG_CACHE_HOME/jaull`` or ``~/.cache/jaull``.
    - macOS: ``~/Library/Caches/jaull``.
    - Windows: ``%LOCALAPPDATA%\\jaull\\cache`` (falls back to
      ``~/AppData/Local/jaull/cache`` if the env var is unset).

    Cache files are disposable and may be deleted without losing user data.
    The path is **not** created; call ``.mkdir(parents=True, exist_ok=True)``
    at the write site.
    """
    return _cache_root().joinpath(*subdirs)


def _root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / "jaull"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "jaull"
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "jaull"


def _cache_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / "jaull" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "jaull"
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "jaull"


__all__ = ["user_cache_dir", "user_data_dir"]
