"""Build immutable experiment records from already observed Jaull runs."""

from __future__ import annotations

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.comparison import PredictionComparison
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import (
    ExperimentBackendTrace,
    ExperimentEnvironment,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentWorkload,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    ExecutionReadiness,
    LlamaCppRuntimeCapability,
    RuntimeRecommendation,
)
from jaull.evaluation.comparison import (
    assert_prediction_runtime_matches,
    compare_prediction,
)


def build_experiment_record(
    *,
    hardware: HardwareProfile,
    artifact: ModelArtifact,
    workload: ExperimentWorkload | None = None,
    backend_trace: ExperimentBackendTrace | None = None,
    runtime: RuntimeRecommendation,
    prediction: MemoryEstimate,
    runtime_capability: LlamaCppRuntimeCapability,
    execution_readiness: ExecutionReadiness,
    observation: ExecutionObservation,
    comparison: PredictionComparison | None = None,
    identity: ExperimentIdentity | None = None,
    environment: ExperimentEnvironment | None = None,
    notes: list[str] | None = None,
) -> ExperimentRecord:
    """Create an experiment snapshot without running probes or inference."""

    assert_prediction_runtime_matches(estimate=prediction, runtime=runtime)
    effective_comparison = comparison or compare_prediction(
        estimate=prediction,
        observation=observation,
        runtime=runtime,
    )
    return ExperimentRecord.create(
        hardware=hardware,
        artifact=artifact,
        workload=workload,
        backend_trace=backend_trace,
        runtime=runtime,
        prediction=prediction,
        runtime_capability=runtime_capability,
        execution_readiness=execution_readiness,
        observation=observation,
        comparison=effective_comparison,
        identity=identity,
        environment=environment,
        notes=notes,
    )


__all__ = ["build_experiment_record"]
