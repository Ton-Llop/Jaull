from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from jaull.advisor.service import AdvisorService
from jaull.artifacts.errors import ArtifactDownloadError
from jaull.cli import run as cli_run
from jaull.cli.run import RunOptions, run_model
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
    InferenceResult,
)
from jaull.domain.runtime import RuntimeFlagSource, RuntimeRecommendation
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    QuantizationNotFoundError,
)
from jaull.execution.errors import ExecutableNotFoundError, ExecutionError
from jaull.workflow.container import ServiceContainer
from tests._execution_fixtures import qwen_ctx4096_estimate, qwen_hardware


def _flag(runtime: RuntimeRecommendation | None, name: str) -> str | None:
    if runtime is None:
        return None
    return next((f.value for f in runtime.flags if f.name == name), None)


def _source(
    runtime: RuntimeRecommendation | None, name: str
) -> RuntimeFlagSource | None:
    if runtime is None:
        return None
    return next((f.source for f in runtime.flags if f.name == name), None)


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "owner/repo",
        "revision": "main",
        "filename": "model-q4_k_m.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 4,
        "local_path": Path("/tmp/model-q4_k_m.gguf"),
        "sha256": "deadbeef",
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


def _observation(
    *,
    duration_seconds: float = 0.2,
    peak_ram_bytes: int | None = 1536 * 1024**2,
    peak_vram_bytes: int | None = 1024 * 1024**2,
    exit_code: int | None = 0,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=exit_code == 0 and failure_reason is None,
        duration_seconds=duration_seconds,
        peak_ram_bytes=peak_ram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        exit_code=exit_code,
        failure_reason=failure_reason,
    )


class _FakeAdvisor:
    def __init__(
        self,
        artifact: ModelArtifact | None = None,
        *,
        observation: ExecutionObservation | None = None,
        fail_at: str | None = None,
        error: Exception | None = None,
        inspect_error: Exception | None = None,
    ) -> None:
        self.artifact = artifact or _artifact()
        self.observation = observation or _observation()
        self.fail_at = fail_at
        self.error = error
        self.inspect_error = inspect_error
        self.operations: list[str] = []
        self.resolved: list[tuple[str, str | None, str | None]] = []
        self.downloaded: list[ModelArtifact] = []
        self.verified: list[tuple[ModelArtifact, bool]] = []
        self.runs: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []
        self.planned: list[dict[str, Any]] = []
        self.last_plan: Any = None

    # -- automatic planning (AUTO path) ---------------------------------
    def scan_hardware(self, on_progress: object | None = None) -> object:
        del on_progress
        self.operations.append("scan")
        return qwen_hardware()

    def inspect_model(self, repo_id: str) -> object:
        self.operations.append("inspect")
        if self.inspect_error is not None:
            raise self.inspect_error
        return object()

    def estimate_model(
        self,
        analysis: object,
        hardware: object,
        inference_cfg: object,
        **kwargs: object,
    ) -> object:
        del analysis, hardware, inference_cfg, kwargs
        self.operations.append("estimate")
        return qwen_ctx4096_estimate(with_runtime_recommendation=True)

    def plan_launch(self, **kwargs: Any) -> RuntimeRecommendation:
        from jaull.application.execution import plan_launch

        return plan_launch(**kwargs)

    def plan_execution(self, **kwargs: Any) -> object:
        from jaull.application.execution import plan_execution

        self.planned.append(kwargs)
        self.last_plan = plan_execution(**kwargs)
        return self.last_plan

    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        self.operations.append("resolve")
        self.resolved.append((repo_id, quantization, revision))
        self._maybe_fail("resolve")
        return self.artifact

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        del on_progress
        self.operations.append("download")
        self.downloaded.append(artifact)
        self._maybe_fail("download")
        downloaded = artifact.model_copy(update={"is_downloaded": True})
        self.artifact = downloaded
        return downloaded

    def verify_artifact(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        self.operations.append("verify")
        self.verified.append((artifact, full))
        self._maybe_fail("verify")
        verified = artifact.model_copy(update={"is_verified": True})
        self.artifact = verified
        return verified

    def run_artifact(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        self.operations.append("run")
        self.runs.append((artifact, prompt, runtime))
        self._maybe_fail("run")
        return InferenceResult(
            text="generated answer",
            runtime="llama.cpp",
            model_path=artifact.local_path or Path("/tmp/model.gguf"),
            observation=self.observation,
        )

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_at == stage and self.error is not None:
            raise self.error


def test_run_model_resolves_downloads_verifies_and_runs_through_advisor(
    capsys: Any,
) -> None:
    artifact = _artifact(is_downloaded=False, is_verified=False)
    advisor = _FakeAdvisor(artifact)
    options = RunOptions(
        quantization="Q4_K_M",
        prompt="Explain local AI",
        revision="abc123",
        llama_cli_path="/opt/llama.cpp/llama-cli",
        context_size=8192,
        n_gpu_layers=12,
        timeout_seconds=42.0,
        full_verify=True,
    )

    exit_code = run_model(
        "https://huggingface.co/owner/repo/tree/main",
        options,
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Model response" in output
    assert "generated answer" in output
    assert "Execution observation" in output
    assert "Duration" in output
    assert "0.20 s" in output
    assert "Peak RAM" in output
    assert "1.5 GiB" in output
    assert "Peak VRAM" in output
    assert "1.0 GiB" in output
    assert "Exit code" in output
    assert "Reason" not in output
    assert advisor.operations == ["resolve", "download", "verify", "run"]
    assert advisor.resolved == [("owner/repo", "Q4_K_M", "abc123")]
    assert advisor.downloaded == [artifact]
    assert advisor.verified == [
        (artifact.model_copy(update={"is_downloaded": True}), True)
    ]

    ran_artifact, prompt, runtime = advisor.runs[0]
    assert ran_artifact.is_verified is True
    assert prompt == "Explain local AI"
    assert runtime is not None
    assert [(flag.name, flag.value) for flag in runtime.flags] == [
        ("--ctx-size", "8192"),
        ("--n-gpu-layers", "12"),
    ]


def test_run_model_skips_download_when_artifact_is_already_local(capsys: Any) -> None:
    artifact = _artifact(is_downloaded=True, is_verified=False)
    advisor = _FakeAdvisor(artifact)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert "generated answer" in capsys.readouterr().out
    assert advisor.operations == ["resolve", "verify", "run"]
    assert advisor.downloaded == []
    assert advisor.verified == [(artifact, False)]


def test_run_model_formats_unavailable_vram_without_zero_bytes(capsys: Any) -> None:
    advisor = _FakeAdvisor(
        observation=_observation(
            peak_ram_bytes=768 * 1024**2,
            peak_vram_bytes=None,
        )
    )

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Peak VRAM" in output
    assert "unavailable" in output
    assert "0 B" not in output


def test_run_model_renders_execution_observation_on_execution_failure(
    capsys: Any,
) -> None:
    observation = _observation(
        duration_seconds=2.4,
        peak_ram_bytes=5_700_000_000,
        peak_vram_bytes=5_900_000_000,
        exit_code=1,
        failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
    )
    advisor = _FakeAdvisor(
        fail_at="run",
        error=ExecutionError("runtime failed", observation),
    )

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    output = capsys.readouterr().out
    assert exit_code == 5
    assert "runtime failed" in output
    assert "Execution failed" in output
    assert "Duration" in output
    assert "2.40 s" in output
    assert "Peak RAM" in output
    assert "5.3 GiB" in output
    assert "Peak VRAM" in output
    assert "5.5 GiB" in output
    assert "Exit code" in output
    assert "1" in output
    assert "Reason" in output
    assert "non_zero_exit" in output


def test_run_model_configures_default_advisor_with_llama_cli_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor()

    class _AdvisorFactory:
        calls: ClassVar[list[tuple[str | Path | None, float]]] = []

        @staticmethod
        def default(
            *,
            llama_cli_path: str | Path | None = None,
            llama_cli_timeout_seconds: float = 300.0,
        ) -> _FakeAdvisor:
            _AdvisorFactory.calls.append(
                (llama_cli_path, llama_cli_timeout_seconds)
            )
            return advisor

    monkeypatch.setattr(cli_run, "AdvisorService", _AdvisorFactory)

    exit_code = run_model(
        "owner/repo",
        RunOptions(
            quantization=None,
            prompt="Hello",
            llama_cli_path="/custom/llama-cli",
            timeout_seconds=7.5,
            n_gpu_layers=0,
        ),
    )

    assert exit_code == 0
    assert "generated answer" in capsys.readouterr().out
    assert _AdvisorFactory.calls == [("/custom/llama-cli", 7.5)]
    assert advisor.operations == ["resolve", "verify", "run"]


def test_run_without_flags_plans_automatically_and_is_not_cpu_only(capsys: Any) -> None:
    advisor = _FakeAdvisor(_artifact(is_downloaded=True, is_verified=True))

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization="Q4_K_M", prompt="Hi"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert {"scan", "inspect", "estimate"} <= set(advisor.operations)
    assert advisor.planned[-1]["estimate"] is not None
    assert advisor.planned[-1]["overrides"].n_gpu_layers is None
    _, _, runtime = advisor.runs[0]
    assert _flag(runtime, "--n-gpu-layers") == "23"  # AUTO, not "0"


def test_run_with_zero_is_explicit_cpu_and_skips_estimation(capsys: Any) -> None:
    advisor = _FakeAdvisor(_artifact(is_downloaded=True, is_verified=True))

    run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hi", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    _, _, runtime = advisor.runs[0]
    assert _flag(runtime, "--n-gpu-layers") == "0"
    assert _source(runtime, "--n-gpu-layers") is RuntimeFlagSource.USER_INPUT
    assert "estimate" not in advisor.operations
    assert advisor.planned[-1]["estimate"] is None


def test_run_ctx_size_omitted_is_not_marked_user_input(capsys: Any) -> None:
    advisor = _FakeAdvisor(_artifact(is_downloaded=True, is_verified=True))

    run_model(
        "owner/repo",
        RunOptions(quantization="Q4_K_M", prompt="Hi"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    _, _, runtime = advisor.runs[0]
    assert _source(runtime, "--ctx-size") is not RuntimeFlagSource.USER_INPUT


def test_run_auto_planning_failure_does_not_fall_back_to_cpu(capsys: Any) -> None:
    advisor = _FakeAdvisor(
        _artifact(is_downloaded=True, is_verified=True),
        inspect_error=HuggingFaceUnavailableError("Hub is down"),
    )

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hi"),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 6
    output = capsys.readouterr().out
    assert "--n-gpu-layers" in output
    assert advisor.runs == []  # nothing was executed CPU-only


class _RecordingBackend:
    def __init__(self, result: ExecutionResult) -> None:
        self.requests: list[ExecutionRequest] = []
        self.result = result

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return self.result


def test_advisor_run_artifact_uses_configured_llama_cli_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_llama_cli = tmp_path / "fake-llama-cli"
    fake_llama_cli.write_bytes(b"")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    backend = _RecordingBackend(
        result=ExecutionResult(
            stdout="advisor-output",
            stderr="",
            observation=ExecutionObservation(
                success=True,
                duration_seconds=0.01,
                peak_ram_bytes=None,
                peak_vram_bytes=None,
                exit_code=0,
                failure_reason=None,
            ),
        )
    )
    monkeypatch.setattr(
        "jaull.execution.host.HostExecutionBackend", lambda *args, **kwargs: backend
    )

    advisor = AdvisorService(
        services=_unused_services(),
        llama_cli_path=fake_llama_cli,
        llama_cli_timeout_seconds=5.0,
    )

    result = advisor.run_artifact(
        artifact=_artifact(local_path=model_path, size_bytes=model_path.stat().st_size),
        prompt="Hello",
    )

    assert result.text == "advisor-output"
    assert result.model_path == model_path
    assert backend.requests[-1].command[0] == str(fake_llama_cli)


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (InvalidModelReferenceError("bad reference"), 2),
        (QuantizationNotFoundError("Q2_K", ["Q4_K_M"]), 3),
        (ArtifactDownloadError("download failed"), 4),
    ],
)
def test_run_model_maps_pre_execution_errors_to_exit_codes(
    exc: Exception,
    expected_code: int,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor(fail_at="resolve", error=exc)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == expected_code
    assert str(exc) in capsys.readouterr().out


@pytest.mark.parametrize(
    "exc",
    [
        ExecutableNotFoundError("llama-cli missing"),
        ExecutionError("runtime failed"),
    ],
)
def test_run_model_maps_advisor_execution_errors_to_exit_code_5(
    exc: ExecutionError,
    capsys: Any,
) -> None:
    advisor = _FakeAdvisor(fail_at="run", error=exc)

    exit_code = run_model(
        "owner/repo",
        RunOptions(quantization=None, prompt="Hello", n_gpu_layers=0),
        advisor=advisor,  # type: ignore[arg-type]
    )

    assert exit_code == 5
    assert str(exc) in capsys.readouterr().out


def _unused_services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
