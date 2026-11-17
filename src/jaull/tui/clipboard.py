"""Copying the server's API key and opencode config, shared by Setup.

The operating system's clipboard is tried first: it works whatever sits between Jaull and the
screen. The terminal's own copy sequence (OSC 52) is the fallback, and it is the one tmux
blocks unless ``set-clipboard on`` is set.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

    from jaull.advisor.service import AdvisorService

# Windows and WSL have clip.exe; the rest are the usual Linux and macOS tools.
_SYSTEM_TOOLS = (
    ("clip.exe",),
    ("clip",),
    ("wl-copy",),
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
    ("pbcopy",),
)
_TOOL_TIMEOUT_SECONDS = 5
# Inside WSL the Windows folders are often not on PATH, but clip.exe is always here.
_WSL_CLIP = "/mnt/c/Windows/System32/clip.exe"
# A Windows program started from a Linux folder fails ("Invalid argument"); /mnt/c is a
# folder it understands.
_WINDOWS_CWD = "/mnt/c"

_NO_CONFIG = "There is no opencode config stored. Restart the server to create one."
_NO_KEY = "There is no API key stored. Restart the server to create one."
_TERMINAL_ONLY = (" If paste gives nothing, your terminal blocks copying: the text is in the "
                  "file named in the server output.")
_TMUX_HINT = " Inside tmux, add `set -g set-clipboard on` to ~/.tmux.conf and restart tmux."


def _system_copy(text: str) -> bool:
    """Copy with the system's clipboard tool. False when there is none or it failed."""
    for command in _SYSTEM_TOOLS:
        tool = shutil.which(command[0])
        if tool is None and command[0] == "clip.exe" and Path(_WSL_CLIP).is_file():
            tool = _WSL_CLIP
        if tool is None:
            continue
        windows_program = tool.lower().endswith(".exe") and sys.platform != "win32"
        try:
            # The text goes through stdin, never the command line. Output is discarded, not
            # captured: xclip and wl-copy stay alive in the background holding the clipboard.
            subprocess.run(
                [tool, *command[1:]], input=text.encode("utf-8"), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=_TOOL_TIMEOUT_SECONDS, cwd=_WINDOWS_CWD if windows_program else None,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def _copy(app: App[object], text: str, done: str) -> str:
    """Put ``text`` on the clipboard and say so, without ever showing it."""
    if _system_copy(text):
        return done
    app.copy_to_clipboard(text)  # the terminal's copy sequence
    return done + _TERMINAL_ONLY + (_TMUX_HINT if os.environ.get("TMUX") else "")


def copy_text(app: App[object], text: str, what: str) -> str:
    """Copy any text; ``what`` names it in the message ("Address" -> "Address copied.")."""
    return _copy(app, text, f"{what} copied.")


def copy_api_key(app: App[object], advisor: AdvisorService) -> str:
    key = advisor.server_api_key()
    return _NO_KEY if key is None else _copy(app, key, "API key copied.")


def copy_opencode_config(app: App[object], advisor: AdvisorService) -> str:
    text = advisor.server_opencode_config()
    if text is None:
        return _NO_CONFIG
    return _copy(
        app, text,
        "opencode config copied. Paste it into ~/.config/opencode/opencode.json on the PC "
        "that runs opencode (it contains the API key).",
    )


__all__ = ["copy_api_key", "copy_opencode_config", "copy_text"]
