from __future__ import annotations

import subprocess

from jaull.observability.provenance import capture_git_commit


def test_capture_git_commit_returns_head_from_the_repository(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr("jaull.observability.provenance.subprocess.run", fake_run)

    assert capture_git_commit() == "abc123"
    assert seen["args"] == (("git", "rev-parse", "HEAD"),)
    assert "cwd" in seen["kwargs"]


def test_capture_git_commit_is_optional_when_git_is_unavailable(monkeypatch) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr("jaull.observability.provenance.subprocess.run", missing)

    assert capture_git_commit() is None
