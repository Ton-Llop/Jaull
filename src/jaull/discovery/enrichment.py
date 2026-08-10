"""Turn a shortlisted candidate into a fully evaluated one.

This is where the existing services do the real work: ``inspect_model``
classifies the repository and runs the right analyzer, then
``configuration.select_configuration`` drives ``estimate_memory`` over the
quantization ladder. Nothing here recomputes what those already produce.

A failure in one candidate must never end the run, so every expected error is
converted into a warning on that candidate alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.families import resolve_parameter_count
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserRequirements
from jaull.estimator.artifact_analysis import analyze_artifact
from jaull.estimator.configuration import select_configuration
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    JaullError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from jaull.runtime.executability import assess_runtime

logger = logging.getLogger(__name__)

InspectFn = Callable[[str], ModelAnalysis]
EstimateMemoryFn = Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]


def evaluate_candidate(
    candidate: ModelCandidate,
    requirements: UserRequirements,
    hardware: HardwareProfile,
    inspect_fn: InspectFn,
    estimate_fn: EstimateMemoryFn,
) -> EvaluatedCandidate:
    """Inspect and estimate one candidate, degrading instead of raising."""
    try:
        analysis = inspect_fn(candidate.repo_id)
    except ModelNotFoundError:
        return _failed(candidate, "Repository no longer exists on Hugging Face.")
    except ModelAccessDeniedError:
        return _failed(candidate, "Repository is gated or private; skipped.")
    except HuggingFaceUnavailableError as exc:
        # Propagated: a transport failure is about the run, not this candidate,
        # and the orchestrator decides whether to abort.
        raise exc
    except JaullError as exc:
        return _failed(candidate, f"Could not inspect repository: {exc}")

    try:
        choice = select_configuration(analysis, requirements, estimate_fn)
    except HuggingFaceUnavailableError as exc:
        raise exc
    except JaullError as exc:
        return _failed(
            candidate,
            f"Could not estimate memory: {exc}",
            analysis=analysis,
        )

    warnings = [*candidate.penalties, *choice.warnings]
    if choice.estimate is not None:
        warnings.extend(choice.estimate.warnings)

    updated_candidate = _with_repository_type(candidate, analysis)
    weight_estimate = (
        choice.estimate.weights if choice.estimate is not None else None
    )
    parameter_count_info = resolve_parameter_count(
        updated_candidate, analysis, weight_estimate=weight_estimate
    )
    artifact_profile = analyze_artifact(
        updated_candidate, analysis, choice.configuration
    )
    runtime_recommendation = (
        choice.estimate.runtime_recommendation if choice.estimate else None
    )
    runtime_assessment = assess_runtime(
        runtime_recommendation, hardware, artifact_profile, choice.estimate
    )

    return EvaluatedCandidate(
        candidate=updated_candidate,
        analysis=analysis,
        selected_configuration=choice.configuration,
        memory_estimate=choice.estimate,
        compatibility=choice.estimate.assessment if choice.estimate else None,
        configuration_reason=choice.reason,
        alternatives_considered=choice.considered,
        warnings=_dedupe(warnings),
        parameter_count_info=parameter_count_info,
        artifact_profile=artifact_profile,
        runtime_assessment=runtime_assessment,
    )


def _with_repository_type(
    candidate: ModelCandidate, analysis: ModelAnalysis
) -> ModelCandidate:
    """Backfill fields the search API does not expose but inspection revealed."""
    update: dict[str, object] = {
        "repository_type": analysis.classification.primary_type
    }
    if not candidate.license and analysis.repo.license:
        update["license"] = analysis.repo.license
    if not candidate.tags and analysis.repo.tags:
        update["tags"] = analysis.repo.tags
    if candidate.pipeline_tag is None and analysis.repo.pipeline_tag:
        update["pipeline_tag"] = analysis.repo.pipeline_tag
    return candidate.model_copy(update=update)


def _failed(
    candidate: ModelCandidate, reason: str, analysis: ModelAnalysis | None = None
) -> EvaluatedCandidate:
    logger.debug("Candidate %s dropped: %s", candidate.repo_id, reason)
    return EvaluatedCandidate(
        candidate=candidate,
        analysis=analysis,
        warnings=[reason],
        failed=True,
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["EstimateMemoryFn", "InspectFn", "evaluate_candidate"]
