"""The no-argument invocation and the unchanged subcommand surface."""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from local_ai_check.cli import app as cli_app
from local_ai_check.cli.app import app


def test_no_args_prints_help_when_output_is_not_a_terminal(monkeypatch: Any) -> None:
    """A piped or redirected invocation must never launch a full-screen UI."""
    launched: list[bool] = []
    monkeypatch.setattr(cli_app, "_is_interactive_terminal", lambda: False)
    monkeypatch.setattr(cli_app, "_launch_ui", lambda: launched.append(True))

    result = CliRunner().invoke(app, [], catch_exceptions=False)

    assert result.exit_code == 0
    assert launched == []
    assert "Usage" in result.stdout
    for command in ("scan", "inspect", "estimate", "doctor", "ui"):
        assert command in result.stdout


def test_no_args_launches_the_ui_on_a_real_terminal(monkeypatch: Any) -> None:
    launched: list[bool] = []
    monkeypatch.setattr(cli_app, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli_app, "_launch_ui", lambda: launched.append(True))

    result = CliRunner().invoke(app, [], catch_exceptions=False)

    assert result.exit_code == 0
    assert launched == [True]


def test_interactive_detection_requires_both_streams(monkeypatch: Any) -> None:
    class _Stream:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(cli_app.sys, "stdin", _Stream(True))
    monkeypatch.setattr(cli_app.sys, "stdout", _Stream(False))
    assert cli_app._is_interactive_terminal() is False

    monkeypatch.setattr(cli_app.sys, "stdout", _Stream(True))
    assert cli_app._is_interactive_terminal() is True


def test_interactive_detection_survives_replaced_streams(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli_app.sys, "stdin", object())
    assert cli_app._is_interactive_terminal() is False


def test_ui_subcommand_still_launches_the_interface(monkeypatch: Any) -> None:
    launched: list[bool] = []
    monkeypatch.setattr(cli_app, "_launch_ui", lambda: launched.append(True))

    result = CliRunner().invoke(app, ["ui"], catch_exceptions=False)

    assert result.exit_code == 0
    assert launched == [True]


def test_help_flag_is_unchanged() -> None:
    result = CliRunner().invoke(app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Analyze local hardware" in result.stdout
