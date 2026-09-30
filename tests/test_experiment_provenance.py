from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jaull.observability.provenance import capture_git_commit


@pytest.mark.parametrize("worktree", [False, True])
def test_capture_git_commit_returns_head_from_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worktree: bool,
) -> None:
    marker = tmp_path / ".git"
    if worktree:
        marker.write_text("gitdir: /elsewhere/worktrees/jaull\n", encoding="utf-8")
    else:
        marker.mkdir()
    monkeypatch.setattr("jaull.observability.provenance._REPOSITORY_ROOT", tmp_path)
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr("jaull.observability.provenance.subprocess.run", fake_run)

    assert capture_git_commit() == "abc123"
    assert seen["args"] == (("git", "rev-parse", "HEAD"),)
    assert "cwd" in seen["kwargs"]


def test_capture_git_commit_does_not_attribute_a_parent_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    archive = tmp_path / "source-archive"
    archive.mkdir()
    calls: list[object] = []

    def parent_head(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="parent-head\n", stderr="")

    monkeypatch.setattr("jaull.observability.provenance._REPOSITORY_ROOT", archive)
    monkeypatch.setattr("jaull.observability.provenance.subprocess.run", parent_head)

    assert capture_git_commit() is None
    assert calls == []


def test_capture_git_commit_is_optional_when_git_is_unavailable(monkeypatch) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr("jaull.observability.provenance.subprocess.run", missing)

    assert capture_git_commit() is None
