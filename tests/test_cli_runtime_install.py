from __future__ import annotations

import pytest
from typer.testing import CliRunner

from jaull.cli.app import app
from jaull.cli.runtime import run_runtime_install
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppInstallation, RuntimeSource
from jaull.runtime.llama_cpp_release import UnsupportedPlatformError


class _Advisor:
    def __init__(self) -> None:
        self.backends: list[ComputeBackend] = []

    def install_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation:
        self.backends.append(backend)
        return LlamaCppInstallation(
            root="/x", llama_cli="/x/llama-cli", source=RuntimeSource.MANAGED, version_text="b1",
        )


class _Refusing(_Advisor):
    def install_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation:
        raise UnsupportedPlatformError("No pinned llama.cpp build for linux/x86_64/hip.")


def test_runtime_install_calls_the_facade(capsys: pytest.CaptureFixture[str]) -> None:
    advisor = _Advisor()
    assert run_runtime_install(ComputeBackend.VULKAN, advisor=advisor) == 0  # type: ignore[arg-type]
    assert advisor.backends == [ComputeBackend.VULKAN]
    assert "/x/llama-cli" in capsys.readouterr().out


def test_runtime_install_reports_unsupported_combinations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_runtime_install(ComputeBackend.HIP, advisor=_Refusing()) == 2  # type: ignore[arg-type]
    assert "No pinned llama.cpp build" in capsys.readouterr().err


def test_the_command_is_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ComputeBackend] = []

    def fake(backend: ComputeBackend) -> int:
        seen.append(backend)
        return 0

    monkeypatch.setattr("jaull.cli.app.run_runtime_install", fake)
    result = CliRunner().invoke(app, ["runtime", "install", "--backend", "vulkan"])
    assert result.exit_code == 0, result.output
    assert seen == [ComputeBackend.VULKAN]
