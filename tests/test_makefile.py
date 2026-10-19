"""Exercise GNU Make with fake tools; never install dependencies or build images."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

MAKE = shutil.which("make")
pytestmark = pytest.mark.skipif(MAKE is None or os.name != "posix", reason="GNU Make/POSIX test")
ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, target, *, uv=True, docker=True, daemon=True):
    log = tmp_path / "calls"
    for name, installed in (("uv", uv), ("docker", docker)):
        if installed:
            tool = tmp_path / name
            tool.write_text(
                "#!/bin/sh\nprintf '%s\\n' \""
                + name
                + ' $*" >> "$TEST_LOG"\n'
                + ('if [ "$1" = info ]; then exit 1; fi\n' if not daemon else "")
            )
            tool.chmod(0o700)
    result = subprocess.run(
        [MAKE, "-f", str(ROOT / "Makefile"), target],
        cwd=ROOT,
        env={**os.environ, "PATH": str(tmp_path), "TEST_LOG": str(log)},
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result, log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    "target,expected",
    [
        ("sync", ["uv sync --locked"]),
        (
            "docker-build",
            [
                "docker info",
                "docker build --platform linux/amd64 --tag "
                "jaull-quality-eval:gguf-v1 pilot/quality_eval",
            ],
        ),
        (
            "setup",
            [
                "uv sync --locked",
                "docker info",
                "docker build --platform linux/amd64 --tag "
                "jaull-quality-eval:gguf-v1 pilot/quality_eval",
            ],
        ),
    ],
)
def test_make_targets_only_run_the_requested_preparation(tmp_path, target, expected):
    result, calls = _run(tmp_path, target)
    assert result.returncode == 0, result.stderr
    assert calls == expected


@pytest.mark.parametrize(
    "failure,target,message",
    [
        ("uv", "sync", "uv is required"),
        ("docker", "docker-build", "Docker CLI is required"),
        ("daemon", "docker-build", "Docker daemon is unavailable"),
    ],
)
def test_make_missing_tools_and_daemon_are_actionable(tmp_path, failure, target, message):
    result, calls = _run(
        tmp_path, target, uv=failure != "uv", docker=failure != "docker", daemon=failure != "daemon"
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not any("docker build" in call for call in calls)


def test_make_help_needs_no_dependencies(tmp_path):
    result, calls = _run(tmp_path, "help", uv=False, docker=False)
    assert result.returncode == 0
    assert "make setup" in result.stdout and "GNU Make" in result.stdout
    assert not calls
