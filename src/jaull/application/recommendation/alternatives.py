"""Bounded metadata-only rescue of models with no known viable execution path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from jaull.application.discovery.artifact_variants import discover_artifact_variants
from jaull.application.recommendation import policies
from jaull.discovery import candidate_filter, enrichment
from jaull.discovery.search_client import ModelSearchClient
from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.execution_plans import (
    PlanCompatibilityStatus,
    confirmed_model_repo_id,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserRequirements
from jaull.exceptions import JaullError
from jaull.execution_plans.service import resolve_model_identity
from jaull.observability.telemetry import PerformanceTelemetry
from jaull.recommendation.engine_v2 import (
    EstimatePlanFn,
    PlanRankingContext,
    generate_execution_plans,
)
from jaull.recommendation.policies import has_confirmed_memory_compatibility


def rescue_with_reupload_alternatives(
    evaluated: Sequence[EvaluatedCandidate],
    *,
    alternatives: Mapping[str, Sequence[ModelCandidate]],
    requirements: UserRequirements,
    hardware: HardwareProfile,
    inspect_model: Callable[[ModelCandidate], ModelAnalysis],
    estimate_fn: EstimatePlanFn,
    check_cancelled: Callable[[], None],
    max_inspections: int,
) -> list[EvaluatedCandidate]:
    """Inspect the merged copies of a GGUF model whose kept copy has nothing that fits.

    The search keeps the most downloaded copy of a model, but copies of one base model do not
    share quantizations: the popular one can publish only files that do not fit, or only parts,
    while a sibling has the file that does. For each kept copy without a confirmed placement the
    siblings are inspected, most downloaded first, until one has it, within ``max_inspections``.
    """
    result = list(evaluated)
    present = {item.repo_id.casefold() for item in result}
    inspections = 0
    for item in evaluated:
        siblings = alternatives.get(item.repo_id.casefold())
        if not siblings or _has_confirmed_placement(item):
            continue
        for sibling in siblings:
            if inspections >= max_inspections:
                return result
            if sibling.repo_id.casefold() in present:
                continue
            check_cancelled()
            inspections += 1
            present.add(sibling.repo_id.casefold())
            try:
                analysis = inspect_model(sibling)
            except JaullError:
                continue
            if analysis.classification.primary_type is not RepositoryType.GGUF:
                continue
            found = enrichment.evaluate_candidate(
                sibling, requirements, hardware,
                inspect_fn=_already_inspected(analysis), estimate_fn=estimate_fn,
            )
            result.append(found)
            if _has_confirmed_placement(found):
                break
    return result


def _already_inspected(analysis: ModelAnalysis) -> Callable[[str], ModelAnalysis]:
    return lambda _repo: analysis


def _has_confirmed_placement(item: EvaluatedCandidate) -> bool:
    estimate = item.memory_estimate
    return (
        not item.failed
        and estimate is not None
        and has_confirmed_memory_compatibility(estimate.assessment.status)
    )


def expand_execution_candidates(
    evaluated: Sequence[EvaluatedCandidate],
    *,
    candidates: Sequence[ModelCandidate],
    requirements: UserRequirements,
    hardware: HardwareProfile,
    search_client: ModelSearchClient,
    inspect_model: Callable[[ModelCandidate], ModelAnalysis],
    estimate_fn: EstimatePlanFn,
    check_cancelled: Callable[[], None],
    telemetry: PerformanceTelemetry,
) -> list[EvaluatedCandidate]:
    """Reuse inspected GGUFs, then inspect at most six additional repositories.

    Only models without a memory-fitting, technically supported plan trigger
    discovery. Installation readiness is deliberately not considered.
    """
    result = list(evaluated)
    inspected = {item.repo_id.casefold() for item in evaluated}
    extra_inspections = 0
    searches = 0
    context = PlanRankingContext(hardware=hardware)

    def viable(item: EvaluatedCandidate) -> bool:
        return any(
            plan.compatibility is PlanCompatibilityStatus.COMPATIBLE
            and plan.memory_prediction is not None
            and has_confirmed_memory_compatibility(plan.memory_prediction.assessment.status)
            for plan in generate_execution_plans(item, requirements, context=context)
        )

    def canonical(item: EvaluatedCandidate) -> str | None:
        identity = resolve_model_identity(candidate=item.candidate, analysis=item.analysis)
        repo = confirmed_model_repo_id(identity)
        return repo.casefold() if repo else None

    def add_confirmed(candidate: ModelCandidate, analysis: ModelAnalysis) -> None:
        if analysis.classification.primary_type is not RepositoryType.GGUF:
            return
        identity = resolve_model_identity(candidate=candidate, analysis=analysis)
        if confirmed_model_repo_id(identity) is None:
            return
        if any(item.repo_id.casefold() == candidate.repo_id.casefold() for item in result):
            return
        if not candidate_filter.filter_candidates([candidate], requirements).kept:
            return
        result.append(
            enrichment.evaluate_candidate(
                candidate,
                requirements,
                hardware,
                inspect_fn=lambda _: analysis,
                estimate_fn=estimate_fn,
            )
        )

    for item in evaluated:
        check_cancelled()
        if item.failed or item.analysis is None:
            continue
        if item.analysis.classification.primary_type is not RepositoryType.TRANSFORMERS:
            continue
        target = canonical(item)
        if target is None or any(canonical(other) == target and viable(other) for other in result):
            continue
        identity = resolve_model_identity(candidate=item.candidate, analysis=item.analysis)

        # Search hits whose structured lineage already names the model are
        # cheaper than another Hub query. Their analyses share the run cache.
        for candidate in candidates:
            if extra_inspections >= policies.MAX_VARIANT_DEEP_INSPECTION:
                break
            if candidate.repo_id.casefold() in inspected:
                continue
            if (
                not candidate.base_model_repo_id
                or candidate.base_model_repo_id.casefold() != target
            ):
                continue
            check_cancelled()
            inspected.add(candidate.repo_id.casefold())
            extra_inspections += 1
            try:
                add_confirmed(candidate, inspect_model(candidate))
            except JaullError:
                continue
        if any(canonical(other) == target and viable(other) for other in result):
            continue
        remaining = policies.MAX_VARIANT_DEEP_INSPECTION - extra_inspections
        if remaining <= 0 or searches >= policies.MAX_TARGETED_VARIANT_SEARCHES:
            continue
        searches += 1

        def inspect_discovered(repo_id: str) -> ModelAnalysis:
            nonlocal extra_inspections
            check_cancelled()
            if repo_id.casefold() not in inspected:
                inspected.add(repo_id.casefold())
                extra_inspections += 1
            return inspect_model(ModelCandidate(repo_id=repo_id))

        try:
            discover_artifact_variants(
                identity=identity,
                current=None,
                search_client=search_client,
                inspect_model=inspect_discovered,
                max_deep_inspection=remaining,
                telemetry=telemetry,
                on_confirmed=add_confirmed,
            )
        except JaullError:
            # The original diagnostic remains in the pool; no conversion is invented.
            continue
    telemetry.increment("recommendation_variant_inspections", extra_inspections)
    telemetry.increment("recommendation_variant_searches", searches)
    original_repos = {item.repo_id.casefold() for item in evaluated}
    for index, item in enumerate(result):
        if (
            item.repo_id.casefold() not in original_repos or item.failed
            or item.analysis is None
            or item.analysis.classification.primary_type is not RepositoryType.TRANSFORMERS
        ):
            continue
        target = canonical(item)
        if target is not None and not any(
            canonical(other) == target and viable(other) for other in result
        ):
            result[index] = item.model_copy(update={"warnings": [
                *item.warnings,
                "No technically viable, memory-fitting path was confirmed in the inspected "
                "pool or bounded alternative discovery. No conversion was assumed.",
            ]})
    return result
