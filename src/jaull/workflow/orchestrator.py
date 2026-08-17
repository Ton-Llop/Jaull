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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

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
from jaull.workflow import policies
from jaull.workflow import ranking as recommendation_service
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
from jaull.workflow.telemetry import PerformanceTelemetry

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
    telemetry = PerformanceTelemetry()
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
        with telemetry.timed("query_build"):
            queries = query_builder.build_queries(
                requirements, limit=policies.SEARCH_RESULTS_PER_QUERY
            )
        reporter.done("queries", f"{len(queries)} queries")

        reporter.start("search")
        with telemetry.timed("search"):
            candidates, search_warnings = _search(
                services, queries, reporter, cancelled, telemetry
            )
        warnings.extend(search_warnings)
        if not candidates:
            return _finish_without_results(
                current, requirements, reporter, warnings, [], queries, telemetry
            )
        reporter.done("search", f"{len(candidates)} unique repositories")

        reporter.start("filter")
        with telemetry.timed("filter"):
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
                current, requirements, reporter, warnings, [], queries, telemetry
            )

        evaluated = _evaluate(
            shortlist,
            requirements,
            hardware,
            services,
            reporter,
            cancelled,
            telemetry,
        )
        current = current.model_copy(update={"evaluated_candidates": evaluated})

        reporter.start("rank")
        with telemetry.timed("ranking"):
            usable = [item for item in evaluated if not item.failed]
            if not usable:
                return _finish_without_results(
                    current,
                    requirements,
                    reporter,
                    warnings,
                    evaluated,
                    queries,
                    telemetry,
                )

            recommendations = recommendation_service.recommend(
                usable,
                requirements,
                limit=policies.MAX_RECOMMENDATIONS,
                capability_analyzer=services.capability_analyzer,
            )
        if not recommendations:
            return _finish_without_results(
                current,
                requirements,
                reporter,
                warnings,
                evaluated,
                queries,
                telemetry,
            )
        reporter.done("rank", f"{len(recommendations)} recommendations")

        return current.model_copy(
            update={
                "recommendations": recommendations,
                "progress": reporter.progress,
                "warnings": warnings,
                "telemetry": telemetry.snapshot(),
                "current_step": WorkflowStep.COMPLETED,
            }
        )

    except WorkflowCancelled:
        logger.debug("Guided workflow cancelled by the caller.")
        return current.model_copy(
            update={
                "progress": reporter.progress,
                "warnings": warnings,
                "telemetry": telemetry.snapshot(),
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
                "telemetry": telemetry.snapshot(),
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
                "telemetry": telemetry.snapshot(),
                "current_step": WorkflowStep.FAILED,
                "errors": [str(exc)],
            }
        )


def _search(
    services: ServiceContainer,
    queries: list[SearchQuery],
    reporter: ProgressReporter,
    cancelled: CancelCheck,
    telemetry: PerformanceTelemetry,
) -> tuple[list[ModelCandidate], list[str]]:
    """Run every query, tolerating individual query failures."""
    per_query: list[list[ModelCandidate]] = []
    warnings: list[str] = []
    failures = 0

    for index, query in enumerate(queries, start=1):
        _check(cancelled)
        reporter.detail("search", f"query {index} of {len(queries)}")
        try:
            telemetry.increment("search_api_calls")
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
    telemetry: PerformanceTelemetry,
) -> list[EvaluatedCandidate]:
    """Inspect and estimate the shortlist, caching within this run."""
    analysis_cache: RunCache[str, ModelAnalysis] = RunCache()
    estimate_cache: RunCache[tuple[str, str], MemoryEstimate] = RunCache()

    def inspect(candidate: ModelCandidate) -> ModelAnalysis:
        return analysis_cache.get_or_compute(
            _analysis_run_key(candidate),
            lambda: _inspect_with_persistent_cache(candidate, services, telemetry),
        )

    def make_estimate_fn(
        repo_id: str,
        range_client: object | None,
    ) -> Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]:
        def estimate(
            analysis: ModelAnalysis, config: InferenceConfiguration
        ) -> MemoryEstimate:
            key = (repo_id, _config_key(config))
            return estimate_cache.get_or_compute(
                key,
                lambda: _estimate_with_timing(
                    analysis=analysis,
                    hardware=hardware,
                    config=config,
                    services=services,
                    range_client=range_client,
                    telemetry=telemetry,
                ),
            )

        return estimate

    def evaluate_one(candidate: ModelCandidate) -> EvaluatedCandidate:
        range_client = _build_range_client(services)

        def inspect_repo(repo_id: str) -> ModelAnalysis:
            if repo_id != candidate.repo_id:
                fallback = ModelCandidate(repo_id=repo_id)
                return inspect(fallback)
            return inspect(candidate)

        return enrichment.evaluate_candidate(
            candidate=candidate,
            requirements=requirements,
            hardware=hardware,
            inspect_fn=inspect_repo,
            estimate_fn=make_estimate_fn(candidate.repo_id, range_client),
        )

    evaluated: list[EvaluatedCandidate | None] = [None] * len(shortlist)
    reporter.start("inspect")
    if not shortlist:
        reporter.done("inspect", "0 inspected")
        reporter.done("estimate", "0 estimated")
        return []
    max_workers = max(1, min(policies.MAX_CONCURRENT_INSPECTIONS, len(shortlist)))
    telemetry.increment("max_concurrent_inspection_workers", max_workers)
    completed = 0
    next_index = 0
    pending: dict[Future[EvaluatedCandidate], int] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="jaull-inspect",
    ) as executor:
        try:
            while next_index < len(shortlist) and len(pending) < max_workers:
                _check(cancelled)
                future = executor.submit(evaluate_one, shortlist[next_index])
                pending[future] = next_index
                next_index += 1

            while pending:
                _check(cancelled)
                done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=0.05)
                if not done:
                    continue
                for future in done:
                    index = pending.pop(future)
                    evaluated[index] = future.result()
                    completed += 1
                    cache_stats = (
                        services.model_analysis_cache.stats
                        if services.model_analysis_cache
                        else None
                    )
                    cached = cache_stats.hits if cache_stats is not None else 0
                    reporter.detail(
                        "inspect",
                        f"{completed} of {len(shortlist)} · {cached} cached",
                    )
                    if next_index < len(shortlist):
                        _check(cancelled)
                        next_future = executor.submit(
                            evaluate_one, shortlist[next_index]
                        )
                        pending[next_future] = next_index
                        next_index += 1
        except WorkflowCancelled:
            for future in pending:
                future.cancel()
            raise
        except Exception:
            for future in pending:
                future.cancel()
            raise
    result = [item for item in evaluated if item is not None]
    telemetry.increment("run_analysis_cache_hits", analysis_cache.hits)
    telemetry.increment("run_analysis_cache_misses", analysis_cache.misses)
    telemetry.increment("run_estimate_cache_hits", estimate_cache.hits)
    telemetry.increment("run_estimate_cache_misses", estimate_cache.misses)
    reporter.done("inspect", f"{len(shortlist)} inspected")
    reporter.done(
        "estimate",
        f"{sum(1 for e in result if e.memory_estimate is not None)} estimated",
    )
    return result


def _estimate_with_timing(
    *,
    analysis: ModelAnalysis,
    hardware: HardwareProfile,
    config: InferenceConfiguration,
    services: ServiceContainer,
    range_client: object | None,
    telemetry: PerformanceTelemetry,
) -> MemoryEstimate:
    with telemetry.timed("estimation"):
        return services.estimate_memory(
            analysis=analysis,
            hardware=hardware,
            inference_cfg=config,
            client=services.hf_client,
            range_client=range_client,
        )


def _inspect_with_persistent_cache(
    candidate: ModelCandidate,
    services: ServiceContainer,
    telemetry: PerformanceTelemetry,
) -> ModelAnalysis:
    cache = services.model_analysis_cache
    if cache is not None:
        with telemetry.timed("persistent_cache_lookup"):
            cached = cache.get(candidate)
        if cached is not None:
            telemetry.increment("persistent_cache_hits")
            return cached
        telemetry.increment("persistent_cache_misses")
    telemetry.increment("deep_inspections")
    with telemetry.timed("deep_inspection"):
        analysis = services.inspect_model(
            candidate.repo_id,
            client=services.hf_client,
        )
    if cache is not None:
        with telemetry.timed("persistent_cache_write"):
            cache.put(candidate, analysis)
    return analysis


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


def _analysis_run_key(candidate: ModelCandidate) -> str:
    revision = candidate.revision_hint
    if revision is None and candidate.last_modified is not None:
        revision = candidate.last_modified.isoformat()
    return f"{candidate.repo_id}|{revision or 'unknown'}"


def _finish_without_results(
    state: RecommendationWorkflowState,
    requirements: UserRequirements,
    reporter: ProgressReporter,
    warnings: list[str],
    evaluated: list[EvaluatedCandidate],
    queries: list[SearchQuery],
    telemetry: PerformanceTelemetry,
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
            "telemetry": telemetry.snapshot(),
            "current_step": WorkflowStep.COMPLETED,
        }
    )


def _check(cancelled: CancelCheck) -> None:
    if cancelled():
        raise WorkflowCancelled


__all__ = ["WorkflowCancelled", "run_workflow", "scan_hardware"]
