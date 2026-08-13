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
from jaull.domain.runtime import ExecutionReadinessStatus, RuntimeRecommendation
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


class LlamaCppExperimentRunnerProtocol(Protocol):
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
    store: ExperimentStore | None = None
    llama_cli_path: str | Path | None = None
    capability_timeout_seconds: float = 10.0

    def run(self, request: ExperimentRequest) -> ExperimentRunResult:
        _validate_request(request)
        runtime_capability = inspect_llama_cpp_runtime(
            backend=self.execution_backend,
            llama_cli_path=self.llama_cli_path,
            timeout_seconds=self.capability_timeout_seconds,
        )
        readiness = evaluate_execution_readiness(
            selection=request.backend_selection,
            runtime_capability=runtime_capability,
        )
        if readiness.status is not ExecutionReadinessStatus.READY:
            raise ExperimentNotReadyError(
                "Experiment preflight is not ready; execution was not attempted "
                f"({readiness.reason.value}).",
                readiness=readiness,
                runtime_capability=runtime_capability,
            )

        try:
            inference = self.llama_cpp_runner.run(
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
]
