"""Offline re-evaluation of immutable experiment records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.comparison import PredictionComparison
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import (
    EXPERIMENT_COMPARISON_ENGINE_VERSION,
    EXPERIMENT_PREDICTION_ENGINE_VERSION,
    ExperimentEnvironment,
    ExperimentPredictionInput,
    ExperimentRecord,
    ExperimentReevaluationProvenance,
    ExperimentReevaluationResult,
    ExperimentReplayability,
    ExperimentReplayabilityStatus,
    ExperimentWorkload,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis, SafetensorsSummary
from jaull.domain.runtime import RuntimeRecommendation
from jaull.estimator.service import estimate_memory
from jaull.evaluation.comparison import compare_prediction
from jaull.exceptions import HuggingFaceUnavailableError, JaullError

PREDICTION_ENGINE_VERSION = EXPERIMENT_PREDICTION_ENGINE_VERSION
COMPARISON_ENGINE_VERSION = EXPERIMENT_COMPARISON_ENGINE_VERSION

EstimateMemoryFn = Callable[..., MemoryEstimate]
ComparePredictionFn = Callable[..., PredictionComparison]


@dataclass(frozen=True)
class ExperimentReevaluationService:
    """Re-run Jaull's current prediction logic against frozen experiment inputs.

    The service never probes the current host, never talks to Hugging Face and
    never executes a runtime. It can only work with the data already stored in
    the record.
    """

    estimate_memory_fn: EstimateMemoryFn = estimate_memory
    compare_prediction_fn: ComparePredictionFn = compare_prediction
    environment_factory: Callable[[], ExperimentEnvironment] = ExperimentEnvironment.capture

    def assess_replayability(
        self,
        record: ExperimentRecord,
    ) -> ExperimentReplayability:
        reasons: list[str] = []
        approximate_reasons: list[str] = []
        warnings: list[str] = []

        if not isinstance(getattr(record, "hardware", None), HardwareProfile):
            reasons.append("missing frozen hardware profile")
        if not isinstance(getattr(record, "artifact", None), ModelArtifact):
            reasons.append("missing artifact snapshot")
        elif record.artifact.sha256 is None:
            warnings.append(
                "artifact hash is unavailable; prediction can be recomputed, "
                "but the physical artifact identity is not cryptographically verified"
            )
        if not isinstance(getattr(record, "runtime", None), RuntimeRecommendation):
            reasons.append("missing runtime snapshot")
        if not isinstance(getattr(record, "workload", None), ExperimentWorkload):
            reasons.append("missing workload snapshot")
        if not isinstance(getattr(record, "prediction", None), MemoryEstimate):
            reasons.append("missing original prediction")
        if not isinstance(getattr(record, "observation", None), ExecutionObservation):
            reasons.append("missing execution observation")

        prediction_input = getattr(record, "prediction_input", None)
        if not isinstance(prediction_input, ExperimentPredictionInput):
            reasons.append("missing prediction input snapshot")
        else:
            if not isinstance(prediction_input.analysis, ModelAnalysis):
                reasons.append("missing model analysis snapshot")
            if not isinstance(
                prediction_input.inference_configuration,
                InferenceConfiguration,
            ):
                reasons.append("missing inference configuration snapshot")
            if prediction_input.resolve_base_model:
                approximate_reasons.append(
                    "base-model enrichment may have depended on external metadata"
                )
            approximate_reasons.extend(prediction_input.reproducibility_notes)

        if reasons:
            return ExperimentReplayability(
                status=ExperimentReplayabilityStatus.NOT_REPRODUCIBLE,
                reasons=reasons,
                warnings=warnings,
            )
        if approximate_reasons:
            return ExperimentReplayability(
                status=ExperimentReplayabilityStatus.APPROXIMATE_ONLY,
                reasons=approximate_reasons,
                warnings=warnings,
            )
        return ExperimentReplayability(
            status=ExperimentReplayabilityStatus.REPRODUCIBLE,
            reasons=[],
            warnings=warnings,
        )

    def reevaluate(self, record: ExperimentRecord) -> ExperimentReevaluationResult:
        replayability = self.assess_replayability(record)
        result = _base_result(record=record, replayability=replayability)
        if replayability.status is ExperimentReplayabilityStatus.NOT_REPRODUCIBLE:
            return result

        prediction_input = record.prediction_input
        if prediction_input is None:
            return result

        try:
            current_prediction = self.estimate_memory_fn(
                analysis=prediction_input.analysis,
                hardware=record.hardware,
                inference_cfg=prediction_input.inference_configuration,
                client=_FrozenPredictionClient(prediction_input.analysis),
                resolve_base_model=prediction_input.resolve_base_model,
                range_client=None,
                recommend_runtime=prediction_input.recommend_runtime,
            )
        except (JaullError, ValueError) as exc:
            failed = ExperimentReplayability(
                status=ExperimentReplayabilityStatus.NOT_REPRODUCIBLE,
                reasons=[*replayability.reasons, f"current prediction failed: {exc}"],
                warnings=replayability.warnings,
            )
            return _base_result(record=record, replayability=failed)

        if current_prediction.runtime_recommendation is None:
            current_prediction = current_prediction.model_copy(
                update={"runtime_recommendation": record.runtime}
            )
        current_comparison = self.compare_prediction_fn(
            estimate=current_prediction,
            observation=record.observation,
            runtime=record.runtime,
        )

        return result.model_copy(
            update={
                "current_prediction": current_prediction,
                "current_comparison": current_comparison,
                "current_provenance": ExperimentReevaluationProvenance(
                    environment=self.environment_factory(),
                    prediction_engine=PREDICTION_ENGINE_VERSION,
                    comparison_engine=COMPARISON_ENGINE_VERSION,
                ),
            }
        )


class _FrozenPredictionClient:
    """HF-client substitute backed only by the recorded model analysis."""

    def __init__(self, analysis: ModelAnalysis) -> None:
        self._analysis = analysis

    def model_info(self, repo_id: str) -> Any:
        del repo_id
        raise HuggingFaceUnavailableError(
            "Offline re-evaluation cannot query Hugging Face model info."
        )

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        del repo_id, filename
        raise HuggingFaceUnavailableError(
            "Offline re-evaluation cannot download Hugging Face files."
        )

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        if repo_id != self._analysis.repo.repo_id:
            raise HuggingFaceUnavailableError(
                "Offline re-evaluation only has safetensors metadata for "
                f"{self._analysis.repo.repo_id}."
            )
        return self._analysis.safetensors_summary


def _base_result(
    *,
    record: ExperimentRecord,
    replayability: ExperimentReplayability,
) -> ExperimentReevaluationResult:
    identity = getattr(record, "identity", None)
    return ExperimentReevaluationResult(
        experiment_id=(
            identity.experiment_id if identity is not None else "unknown-experiment"
        ),
        replayability=replayability,
        original_prediction=getattr(record, "prediction", None),
        original_comparison=getattr(record, "comparison", None),
        original_provenance=getattr(record, "environment", None),
        original_prediction_engine=getattr(record, "prediction_engine", None),
        original_comparison_engine=getattr(record, "comparison_engine", None),
        observation=getattr(record, "observation", None),
    )


def reevaluate_experiment(record: ExperimentRecord) -> ExperimentReevaluationResult:
    """Convenience wrapper for one offline re-evaluation."""

    return ExperimentReevaluationService().reevaluate(record)


__all__ = [
    "COMPARISON_ENGINE_VERSION",
    "PREDICTION_ENGINE_VERSION",
    "ExperimentReevaluationService",
    "reevaluate_experiment",
]
