from __future__ import annotations

from pathlib import Path

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.execution import ExecutionObservation, InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.workflow.container import ServiceContainer


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )


def _observation() -> ExecutionObservation:
    return ExecutionObservation(
        success=True,
        duration_seconds=0.1,
        peak_ram_bytes=None,
        peak_vram_bytes=None,
        exit_code=0,
    )


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "org/model",
        "revision": "main",
        "filename": "org/model",
        "format": "safetensors",
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


class _FakeRunner:
    def __init__(self, runtime: RuntimeName) -> None:
        self.runtime = runtime
        self.calls: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []

    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        self.calls.append((artifact, prompt, runtime))
        return InferenceResult(
            text=f"{self.runtime.value} result",
            runtime=self.runtime.value,
            model_path=artifact.local_path or Path(artifact.repo_id),
            observation=_observation(),
        )


def test_advisor_dispatches_transformers_runtime_to_transformers_runner() -> None:
    llama_runner = _FakeRunner(RuntimeName.LLAMA_CPP)
    transformers_runner = _FakeRunner(RuntimeName.TRANSFORMERS)
    advisor = AdvisorService(
        services=_services(),
        llama_cpp_runner=llama_runner,  # type: ignore[arg-type]
        transformers_runner=transformers_runner,  # type: ignore[arg-type]
    )
    runtime = RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        confidence=EstimationConfidence.HIGH,
    )

    result = advisor.run_artifact(
        artifact=_artifact(),
        prompt="Hello",
        runtime=runtime,
    )

    assert result.runtime == RuntimeName.TRANSFORMERS.value
    assert transformers_runner.calls == [(_artifact(), "Hello", runtime)]
    assert llama_runner.calls == []


def test_advisor_keeps_llama_cpp_as_default_runner() -> None:
    llama_runner = _FakeRunner(RuntimeName.LLAMA_CPP)
    transformers_runner = _FakeRunner(RuntimeName.TRANSFORMERS)
    advisor = AdvisorService(
        services=_services(),
        llama_cpp_runner=llama_runner,  # type: ignore[arg-type]
        transformers_runner=transformers_runner,  # type: ignore[arg-type]
    )
    runtime = RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        confidence=EstimationConfidence.HIGH,
    )

    result = advisor.run_artifact(
        artifact=_artifact(format="gguf", filename="model.gguf"),
        prompt="Hello",
        runtime=runtime,
    )

    assert result.runtime == RuntimeName.LLAMA_CPP.value
    assert len(llama_runner.calls) == 1
    assert transformers_runner.calls == []
