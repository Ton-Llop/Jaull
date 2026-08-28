"""Orchestrate one controlled inference experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.domain.experiments import (
    ExperimentBackendTrace,
    ExperimentRequest,
    ExperimentRunResult,
)
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    RuntimeCapability,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.comparison import (
    PredictionRuntimeMismatchError,
    assert_prediction_runtime_matches,
)
from jaull.evaluation.experiments import build_experiment_record
from jaull.execution.errors import ExecutionError
from jaull.execution.ports import ExecutionBackendProtocol
from jaull.experiments.errors import (
    ExperimentConfigurationError,
    ExperimentNotReadyError,
    ExperimentPersistenceError,
    ExperimentRunnerError,
    ExperimentStoreError,
)
from jaull.experiments.storage import ExperimentStore
from jaull.runtime.llama_cpp_capability import (
    evaluate_execution_readiness,
    inspect_llama_cpp_runtime,
)
from jaull.runtime.locator import RuntimeLocator, RuntimeLocatorConfig
from jaull.runtime.pytorch_capability import (
    evaluate_pytorch_execution_readiness,
    inspect_pytorch_runtime,
)


class LlamaCppExperimentRunnerProtocol(Protocol):
    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        ...


class TransformersExperimentRunnerProtocol(Protocol):
    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        ...


@dataclass(frozen=True)
class ExperimentRunner:
    """Run exactly one preconfigured experiment request."""

    execution_backend: ExecutionBackendProtocol
    llama_cpp_runner: LlamaCppExperimentRunnerProtocol
    transformers_runner: TransformersExperimentRunnerProtocol | None = None
    store: ExperimentStore | None = None
    llama_cli_path: str | Path | None = None
    python_executable: str | Path | None = None
    runtime_locator: RuntimeLocator | None = None
    capability_timeout_seconds: float = 10.0

    def run(self, request: ExperimentRequest) -> ExperimentRunResult:
        _validate_request(request)
        runtime_capability, readiness, artifact_runner = self._preflight(request)
        if readiness.status is not ExecutionReadinessStatus.READY:
            raise ExperimentNotReadyError(
                "Experiment preflight is not ready; execution was not attempted "
                f"({readiness.reason.value}).",
                readiness=readiness,
                runtime_capability=runtime_capability,
            )

        try:
            inference = artifact_runner.run(
                artifact=request.artifact,
                prompt=request.workload.prompt,
                runtime=request.runtime,
            )
            observation = inference.observation
        except ExecutionError as exc:
            if exc.observation is None:
                raise ExperimentRunnerError(str(exc)) from exc
            observation = exc.observation

        record = build_experiment_record(
            hardware=request.hardware,
            artifact=request.artifact,
            workload=request.workload,
            backend_trace=ExperimentBackendTrace(
                requested_backend=request.requested_backend,
                observed_backend=None,
                observed_source=None,
            ),
            runtime=request.runtime,
            prediction=request.prediction,
            prediction_input=request.prediction_input,
            runtime_capability=runtime_capability,
            execution_readiness=readiness,
            observation=observation,
            notes=request.notes,
        )
        persisted_path = None
        if request.persist:
            if self.store is None:
                raise ExperimentRunnerError(
                    "Experiment persistence was requested but no ExperimentStore "
                    "is configured."
                )
            try:
                persisted_path = self.store.save(record)
            except ExperimentStoreError as exc:
                raise ExperimentPersistenceError(
                    "Experiment completed, but its record could not be persisted.",
                    record=record,
                ) from exc
        return ExperimentRunResult(record=record, persisted_path=persisted_path)

    def _preflight(
        self,
        request: ExperimentRequest,
    ) -> tuple[
        RuntimeCapability,
        ExecutionReadiness,
        LlamaCppExperimentRunnerProtocol | TransformersExperimentRunnerProtocol,
    ]:
        if request.runtime.runtime is RuntimeName.TRANSFORMERS:
            if self.transformers_runner is None:
                raise ExperimentRunnerError(
                    "Transformers experiment requested but no TransformersRunner "
                    "is configured."
                )
            pytorch_capability = inspect_pytorch_runtime(
                backend=self.execution_backend,
                python_executable=self._pytorch_python(),
                timeout_seconds=self.capability_timeout_seconds,
            )
            pytorch_readiness = evaluate_pytorch_execution_readiness(
                selection=request.backend_selection,
                runtime_capability=pytorch_capability,
            )
            return pytorch_capability, pytorch_readiness, self.transformers_runner

        llama_capability = inspect_llama_cpp_runtime(
            backend=self.execution_backend,
            llama_cli_path=self._llama_cli_for_request(request),
            timeout_seconds=self.capability_timeout_seconds,
        )
        llama_readiness = evaluate_execution_readiness(
            selection=request.backend_selection,
            runtime_capability=llama_capability,
        )
        return llama_capability, llama_readiness, self.llama_cpp_runner

    def _runtime_locator(self) -> RuntimeLocator:
        if self.runtime_locator is not None:
            return self.runtime_locator
        return RuntimeLocator(
            config=RuntimeLocatorConfig(
                llama_cli_path=self.llama_cli_path,
                python_executable=self.python_executable,
            )
        )

    def _pytorch_python(self) -> str | None:
        return self._runtime_locator().resolve_pytorch().python_executable

    def _llama_cli_for_request(self, request: ExperimentRequest) -> str | None:
        readiness_by_cli: dict[str, ExecutionReadinessStatus] = {}
        fallback: str | None = None
        for installation in self._runtime_locator().discover_llama_cpp():
            if installation.llama_cli is None:
                continue
            fallback = fallback or installation.llama_cli
            capability = inspect_llama_cpp_runtime(
                backend=self.execution_backend,
                llama_cli_path=installation.llama_cli,
                timeout_seconds=self.capability_timeout_seconds,
            )
            readiness = evaluate_execution_readiness(
                selection=request.backend_selection,
                runtime_capability=capability,
            )
            readiness_by_cli[installation.llama_cli] = readiness.status
        selected = self._runtime_locator().select_llama_cpp(
            requested_backend=request.backend_selection.selected_backend,
            readiness_by_cli=readiness_by_cli,
        )
        return selected.llama_cli or fallback


def _validate_request(request: ExperimentRequest) -> None:
    try:
        assert_prediction_runtime_matches(
            estimate=request.prediction,
            runtime=request.runtime,
        )
    except PredictionRuntimeMismatchError as exc:
        raise ExperimentConfigurationError(str(exc)) from exc


__all__ = [
    "ExperimentRunner",
    "LlamaCppExperimentRunnerProtocol",
    "TransformersExperimentRunnerProtocol",
]
