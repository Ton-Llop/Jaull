"""Add the server to the opencode the person already uses, keeping everything else theirs.

Someone with opencode usually has providers and a default model already. Only the ``jaull``
provider is written; their other providers and settings are untouched, their default model is
kept, and the file is copied aside before the first change. A file that cannot be read safely
(comments, odd shapes) is left alone.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROVIDER_ID = "jaull"
_BACKUP_SUFFIX = ".jaull-backup"


@dataclass(frozen=True)
class OpencodeMerge:
    #: The merged configuration; None when the existing file cannot be merged safely.
    config: dict[str, Any] | None
    #: created, added, updated, unchanged or unreadable.
    outcome: str
    #: True when the default model is now the server's.
    selected: bool = False


@dataclass(frozen=True)
class OpencodeInstall:
    path: Path
    outcome: str
    backup: Path | None = None
    selected: bool = False


def merge_opencode_config(existing: str | None, ours: dict[str, Any]) -> OpencodeMerge:
    """``ours`` added to the configuration in ``existing`` (the file's text, if any)."""
    if existing is None or not existing.strip():
        return OpencodeMerge(ours, "created", selected=True)
    try:
        current = json.loads(existing)
    except ValueError:  # includes comments: opencode accepts them, json does not
        return OpencodeMerge(None, "unreadable")
    if not isinstance(current, dict):
        return OpencodeMerge(None, "unreadable")
    providers = current.get("provider", {})
    if not isinstance(providers, dict):
        return OpencodeMerge(None, "unreadable")

    merged = dict(current)
    merged.setdefault("$schema", ours["$schema"])
    merged["provider"] = {**providers, _PROVIDER_ID: ours["provider"][_PROVIDER_ID]}
    merged.setdefault("model", ours["model"])
    selected = merged["model"] == ours["model"]
    if merged == current:
        return OpencodeMerge(merged, "unchanged", selected)
    outcome = "updated" if _PROVIDER_ID in providers else "added"
    return OpencodeMerge(merged, outcome, selected)


def opencode_directory() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "opencode"


def install_opencode_config(ours: dict[str, Any]) -> OpencodeInstall | None:
    """Merge ``ours`` into this user's opencode configuration; None when they have no opencode."""
    directory = opencode_directory()
    if shutil.which("opencode") is None and not directory.is_dir():
        return None
    target = directory / "opencode.json"
    jsonc = directory / "opencode.jsonc"
    if jsonc.exists():  # opencode reads this one too; do not guess which wins
        return OpencodeInstall(jsonc, "unreadable")

    try:
        existing = target.read_text(encoding="utf-8") if target.is_file() else None
    except (OSError, UnicodeDecodeError):
        return OpencodeInstall(target, "unreadable")
    merge = merge_opencode_config(existing, ours)
    if merge.config is None:
        return OpencodeInstall(target, "unreadable")
    if merge.outcome == "unchanged":
        return OpencodeInstall(target, "unchanged", selected=merge.selected)

    backup: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if existing is not None:
            backup = target.with_name(target.name + _BACKUP_SUFFIX)
            if not backup.exists():  # the first one is the file as the person had it
                shutil.copy2(target, backup)
        staged = target.with_name(target.name + ".tmp")
        staged.write_text(json.dumps(merge.config, indent=2) + "\n", encoding="utf-8")
        staged.chmod(0o600)  # the file now holds the API key
        os.replace(staged, target)
    except OSError:
        return OpencodeInstall(target, "unreadable")
    return OpencodeInstall(target, merge.outcome, backup, merge.selected)


__all__ = [
    "OpencodeInstall",
    "OpencodeMerge",
    "install_opencode_config",
    "merge_opencode_config",
    "opencode_directory",
]
