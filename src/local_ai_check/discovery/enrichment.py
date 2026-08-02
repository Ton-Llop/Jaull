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

from local_ai_check.discovery.models import EvaluatedCandidate, ModelCandidate
from local_ai_check.domain.estimation import MemoryEstimate
from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.domain.inference import InferenceConfiguration
from local_ai_check.domain.model import ModelAnalysis
from local_ai_check.exceptions import (
    HuggingFaceUnavailableError,
    LocalAiCheckError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from local_ai_check.recommendation.configuration import select_configuration
from local_ai_check.workflow.models import UserRequirements

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
    """Inspect and estimate one candidate, degrading instead of raising.

    ``hardware`` is not read directly here — it is already bound into
    ``estimate_fn`` by the orchestrator — but it stays in the signature so the
    dependency is explicit at the call site.
    """
    del hardware

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
    except LocalAiCheckError as exc:
        return _failed(candidate, f"Could not inspect repository: {exc}")

    try:
        choice = select_configuration(analysis, requirements, estimate_fn)
    except HuggingFaceUnavailableError as exc:
        raise exc
    except LocalAiCheckError as exc:
        return _failed(
            candidate,
            f"Could not estimate memory: {exc}",
            analysis=analysis,
        )

    warnings = [*candidate.penalties, *choice.warnings]
    if choice.estimate is not None:
        warnings.extend(choice.estimate.warnings)

    return EvaluatedCandidate(
        candidate=_with_repository_type(candidate, analysis),
        analysis=analysis,
        selected_configuration=choice.configuration,
        memory_estimate=choice.estimate,
        compatibility=choice.estimate.assessment if choice.estimate else None,
        configuration_reason=choice.reason,
        alternatives_considered=choice.considered,
        warnings=_dedupe(warnings),
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
