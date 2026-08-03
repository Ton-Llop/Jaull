"""Drive the guided run end to end.

Synchronous on purpose: Textual runs this inside a thread worker (the pattern
`tui/screens/scan.py` already uses), so the event loop stays responsive, while
tests call it directly with fakes and never construct an app. Progress and
cancellation both travel through plain callables, which keeps this module free
of any UI dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from jaull.discovery import candidate_filter, enrichment, query_builder
from jaull.domain.candidates import (
    EvaluatedCandidate,
    ModelCandidate,
    SearchQuery,
)
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserAnswers, UserRequirements
from jaull.exceptions import HuggingFaceUnavailableError, JaullError
from jaull.recommendation import explanations
from jaull.recommendation import service as recommendation_service
from jaull.workflow import policies
from jaull.workflow import requirements as requirements_service
from jaull.workflow.cache import RunCache
from jaull.workflow.container import ServiceContainer
from jaull.workflow.models import WorkflowStep
from jaull.workflow.progress import (
    DISCOVERY_STEPS,
    HARDWARE_STEPS,
    ProgressCallback,
    ProgressReporter,
)
from jaull.workflow.state import RecommendationWorkflowState

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], bool]


class WorkflowCancelled(Exception):
    """Raised internally when the caller asks the run to stop."""


def scan_hardware(
    services: ServiceContainer,
    on_progress: ProgressCallback | None = None,
) -> HardwareProfile:
    """Detect hardware, reporting each real probe as it completes.

    The step callback fires from inside ``detect_hardware``; there are no
    artificial delays, so a step turns green exactly when its probe returns.
    """
    reporter = ProgressReporter(HARDWARE_STEPS, on_progress)
    remaining = [key for key, _ in HARDWARE_STEPS]

    def on_step(key: str) -> None:
        if key in remaining:
            remaining.remove(key)
        reporter.done(key)
        if remaining:
            reporter.start(remaining[0])

    reporter.start(remaining[0])
    profile = services.detect_hardware(on_step=on_step)
    for key in list(remaining):
        reporter.done(key)
    return profile


def run_workflow(
    answers: UserAnswers,
    hardware: HardwareProfile,
    services: ServiceContainer,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    state: RecommendationWorkflowState | None = None,
) -> RecommendationWorkflowState:
    """Run discovery, evaluation and ranking. Never raises for one bad model."""
    reporter = ProgressReporter(DISCOVERY_STEPS, on_progress)
    current = (state or RecommendationWorkflowState()).model_copy(
        update={
            "answers": answers,
            "hardware": hardware,
            "current_step": WorkflowStep.REQUIREMENTS,
        }
    )

    warnings: list[str] = list(hardware.warnings)
    cancelled = is_cancelled or (lambda: False)

    try:
        requirements = requirements_service.build_requirements(answers, hardware)
        current = current.model_copy(update={"requirements": requirements})

        reporter.start("queries")
        _check(cancelled)
        queries = query_builder.build_queries(
            requirements, limit=policies.SEARCH_RESULTS_PER_QUERY
        )
        reporter.done("queries", f"{len(queries)} queries")

        reporter.start("search")
        candidates, search_warnings = _search(
            services, queries, reporter, cancelled
        )
        warnings.extend(search_warnings)
        if not candidates:
            return _finish_without_results(
                current, requirements, reporter, warnings, [], queries
            )
        reporter.done("search", f"{len(candidates)} unique repositories")

        reporter.start("filter")
        outcome = candidate_filter.filter_candidates(candidates, requirements)
        shortlist = candidate_filter.shortlist(
            outcome.kept,
            requirements,
            policies.MAX_DEEP_INSPECTION,
            budget_bytes=_memory_budget(hardware),
        )
        reporter.done(
            "filter",
            f"{len(outcome.kept)} kept, {len(outcome.rejected)} filtered out",
        )
        current = current.model_copy(
            update={
                "candidates": outcome.kept,
                "search_queries": [q.label for q in queries],
                "current_step": WorkflowStep.CANDIDATE_EVALUATION,
            }
        )
        if not shortlist:
            return _finish_without_results(
                current, requirements, reporter, warnings, [], queries
            )

        evaluated = _evaluate(
            shortlist, requirements, hardware, services, reporter, cancelled
        )
        current = current.model_copy(update={"evaluated_candidates": evaluated})

        reporter.start("rank")
        usable = [item for item in evaluated if not item.failed]
        if not usable:
            return _finish_without_results(
                current, requirements, reporter, warnings, evaluated, queries
            )

        recommendations = recommendation_service.recommend(
            usable,
            requirements,
            limit=policies.MAX_RECOMMENDATIONS,
            capability_analyzer=services.capability_analyzer,
        )
        if not recommendations:
            return _finish_without_results(
                current, requirements, reporter, warnings, evaluated, queries
            )
        reporter.done("rank", f"{len(recommendations)} recommendations")

        return current.model_copy(
            update={
                "recommendations": recommendations,
                "progress": reporter.progress,
                "warnings": warnings,
                "current_step": WorkflowStep.COMPLETED,
            }
        )

    except WorkflowCancelled:
        logger.debug("Guided workflow cancelled by the caller.")
        return current.model_copy(
            update={
                "progress": reporter.progress,
                "warnings": warnings,
                "current_step": WorkflowStep.FAILED,
                "errors": ["Search cancelled."],
            }
        )
    except HuggingFaceUnavailableError as exc:
        reporter.fail(reporter.progress.current_key or "search", str(exc))
        return current.model_copy(
            update={
                "progress": reporter.progress,
                "warnings": warnings,
                "current_step": WorkflowStep.FAILED,
                "errors": [str(exc)],
            }
        )
    except JaullError as exc:
        reporter.fail(reporter.progress.current_key or "search", str(exc))
        return current.model_copy(
            update={
                "progress": reporter.progress,
                "warnings": warnings,
                "current_step": WorkflowStep.FAILED,
                "errors": [str(exc)],
            }
        )


def _search(
    services: ServiceContainer,
    queries: list[SearchQuery],
    reporter: ProgressReporter,
    cancelled: CancelCheck,
) -> tuple[list[ModelCandidate], list[str]]:
    """Run every query, tolerating individual query failures."""
    per_query: list[list[ModelCandidate]] = []
    warnings: list[str] = []
    failures = 0

    for index, query in enumerate(queries, start=1):
        _check(cancelled)
        reporter.detail("search", f"query {index} of {len(queries)}")
        try:
            per_query.append(list(services.search_client.search(query)))
        except HuggingFaceUnavailableError as exc:
            failures += 1
            per_query.append([])
            warnings.append(f"Search query {query.label!r} failed: {exc}")
            # Every query failing means the Hub is unreachable, not that this
            # particular query was unlucky.
            if failures == len(queries):
                raise

    unique = candidate_filter.deduplicate(_interleave(per_query))
    return unique[: policies.MAX_UNIQUE_CANDIDATES], warnings


def _memory_budget(hardware: HardwareProfile) -> int | None:
    """The largest pool a model could plausibly be loaded into.

    Used only to allocate the inspection budget. GPU memory when there is a GPU,
    otherwise system RAM, since a CPU-only machine runs the model in RAM.
    """
    if hardware.gpus:
        return max(gpu.vram_total_bytes for gpu in hardware.gpus)
    return hardware.memory.total_bytes or None


def _interleave(per_query: list[list[ModelCandidate]]) -> list[ModelCandidate]:
    """Round-robin the per-query results into one list.

    Concatenating instead would let the first two queries fill the whole
    candidate budget, so the format-specific and language-specific queries —
    the ones that surface GGUF builds and non-English models — would never
    reach the shortlist. Taking one result from each query in turn gives every
    angle of the search a fair share of the budget.
    """
    merged: list[ModelCandidate] = []
    depth = max((len(results) for results in per_query), default=0)
    for index in range(depth):
        for results in per_query:
            if index < len(results):
                merged.append(results[index])
    return merged


def _evaluate(
    shortlist: list[ModelCandidate],
    requirements: UserRequirements,
    hardware: HardwareProfile,
    services: ServiceContainer,
    reporter: ProgressReporter,
    cancelled: CancelCheck,
) -> list[EvaluatedCandidate]:
    """Inspect and estimate the shortlist, caching within this run."""
    analysis_cache: RunCache[str, ModelAnalysis] = RunCache()
    estimate_cache: RunCache[tuple[str, str], MemoryEstimate] = RunCache()
    range_client = _build_range_client(services)

    def inspect(repo_id: str) -> ModelAnalysis:
        return analysis_cache.get_or_compute(
            repo_id,
            lambda: services.inspect_model(repo_id, client=services.hf_client),
        )

    def make_estimate_fn(
        repo_id: str,
    ) -> Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]:
        def estimate(
            analysis: ModelAnalysis, config: InferenceConfiguration
        ) -> MemoryEstimate:
            key = (repo_id, _config_key(config))
            return estimate_cache.get_or_compute(
                key,
                lambda: services.estimate_memory(
                    analysis=analysis,
                    hardware=hardware,
                    inference_cfg=config,
                    client=services.hf_client,
                    range_client=range_client,
                ),
            )

        return estimate

    evaluated: list[EvaluatedCandidate] = []
    reporter.start("inspect")
    for index, candidate in enumerate(shortlist, start=1):
        _check(cancelled)
        reporter.detail("inspect", f"{index} of {len(shortlist)}")
        evaluated.append(
            enrichment.evaluate_candidate(
                candidate=candidate,
                requirements=requirements,
                hardware=hardware,
                inspect_fn=inspect,
                estimate_fn=make_estimate_fn(candidate.repo_id),
            )
        )
    reporter.done("inspect", f"{len(shortlist)} inspected")
    reporter.done(
        "estimate",
        f"{sum(1 for e in evaluated if e.memory_estimate is not None)} estimated",
    )
    return evaluated


def _build_range_client(services: ServiceContainer) -> object | None:
    if services.range_client_factory is None:
        return None
    try:
        return services.range_client_factory()
    except Exception:
        # GGUF header enrichment is optional; losing it must never end the run.
        logger.debug("Could not build a Range client; GGUF header reads disabled.")
        return None


def _config_key(config: InferenceConfiguration) -> str:
    return (
        f"{config.quantization or ''}|{config.precision or ''}|"
        f"{config.context_length}|{config.batch_size}|{config.target_device.value}"
    )


def _finish_without_results(
    state: RecommendationWorkflowState,
    requirements: UserRequirements,
    reporter: ProgressReporter,
    warnings: list[str],
    evaluated: list[EvaluatedCandidate],
    queries: list[SearchQuery],
) -> RecommendationWorkflowState:
    """Complete the run with an explanation instead of recommendations."""
    del requirements
    reporter.done("rank", "no compatible models")
    return state.model_copy(
        update={
            "progress": reporter.progress,
            "warnings": warnings,
            "evaluated_candidates": evaluated or state.evaluated_candidates,
            "search_queries": [q.label for q in queries] or state.search_queries,
            "no_results_reason": explanations.no_results_explanation(evaluated),
            "current_step": WorkflowStep.COMPLETED,
        }
    )


def _check(cancelled: CancelCheck) -> None:
    if cancelled():
        raise WorkflowCancelled


__all__ = ["WorkflowCancelled", "run_workflow", "scan_hardware"]
