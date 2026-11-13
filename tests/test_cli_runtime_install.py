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
    seen: list[tuple[ComputeBackend, bool]] = []

    def fake(backend: ComputeBackend, *, latest: bool) -> int:
        seen.append((backend, latest))
        return 0

    monkeypatch.setattr("jaull.cli.app.run_runtime_install", fake)
    result = CliRunner().invoke(app, ["runtime", "install", "--backend", "vulkan"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(app, ["runtime", "install", "--backend", "cpu", "--latest"])
    assert result.exit_code == 0, result.output
    assert seen == [(ComputeBackend.VULKAN, False), (ComputeBackend.CPU, True)]


class _Latest(_Advisor):
    def __init__(self) -> None:
        super().__init__()
        self.latest: list[bool] = []

    def install_llama_cpp(  # type: ignore[override]
        self, backend: ComputeBackend, *, latest: bool = False,
    ) -> LlamaCppInstallation:
        self.latest.append(latest)
        return super().install_llama_cpp(backend)


def test_latest_install_is_marked_not_validated(capsys: pytest.CaptureFixture[str]) -> None:
    advisor = _Latest()
    assert run_runtime_install(
        ComputeBackend.CPU, latest=True, advisor=advisor,  # type: ignore[arg-type]
    ) == 0
    assert advisor.latest == [True]
    assert "not validated" in capsys.readouterr().out
