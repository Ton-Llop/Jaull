"""Copying to the clipboard: the system tool first, the terminal's sequence as the fallback."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from jaull.tui import clipboard


def test_the_system_clipboard_tool_gets_the_text_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        seen.update(command=command, **kwargs)

    monkeypatch.setattr(
        clipboard.shutil, "which", lambda name: "/bin/clip" if name == "clip.exe" else None)
    monkeypatch.setattr(clipboard.subprocess, "run", run)

    assert clipboard._system_copy("secret")
    assert seen["command"] == ["/bin/clip"] and seen["input"] == b"secret"  # never in argv


def test_no_tool_or_a_failing_one_means_no_system_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    monkeypatch.setattr(clipboard, "_WSL_CLIP", "/nowhere/clip.exe")  # this PC may be WSL
    assert not clipboard._system_copy("x")

    def fails(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(clipboard.subprocess, "run", fails)
    assert not clipboard._system_copy("x")


def test_the_terminal_is_the_fallback_and_tmux_gets_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clipboard, "_system_copy", lambda text: False)
    copied: list[str] = []
    app = SimpleNamespace(copy_to_clipboard=copied.append)
    advisor = SimpleNamespace(server_opencode_config=lambda: None, server_api_key=lambda: None)

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    message = clipboard.copy_text(app, "text", "Address")  # type: ignore[arg-type]
    assert copied == ["text"] and "set-clipboard on" in message
    monkeypatch.delenv("TMUX")
    assert "tmux" not in clipboard.copy_text(app, "text", "Address")  # type: ignore[arg-type]

    # Nothing stored: say so instead of copying an empty string.
    assert "no opencode config" in clipboard.copy_opencode_config(app, advisor).lower()  # type: ignore[arg-type]
    assert "no API key" in clipboard.copy_api_key(app, advisor)  # type: ignore[arg-type]
    assert copied == ["text", "text"]


def test_in_wsl_clip_is_found_without_windows_on_the_path_and_run_from_a_windows_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: None)
    # The module's own names are replaced, not pathlib's or sys's: the test must not depend
    # on the OS it runs on (a Windows path prints with backslashes).
    monkeypatch.setattr(
        clipboard, "Path",
        lambda text: SimpleNamespace(is_file=lambda: text == clipboard._WSL_CLIP),
    )
    monkeypatch.setattr(clipboard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(clipboard.subprocess, "run", lambda command, **kw: seen.update(
        command=command, **kw))

    assert clipboard._system_copy("x")
    assert seen["command"] == [clipboard._WSL_CLIP] and seen["cwd"] == "/mnt/c"
