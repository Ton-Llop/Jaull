"""Diversity-aware selection over already ranked execution plans."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

from jaull.domain.execution_plans import ExecutionPlan, ModelIdentity, confirmed_model_repo_id
from jaull.domain.recommendation import RecommendationPosition
from jaull.domain.requirements import UserRequirements
from jaull.recommendation.engine_v2 import RankedPlan, ranking_tie_key

TieKey = Callable[[RankedPlan], Hashable]


def tie_key_for(requirements: UserRequirements) -> TieKey:
    """The ranking's own definition of a tie for these requirements."""
    commercial = requirements.commercial_use_required is True
    return lambda plan: ranking_tie_key(
        plan, requirements.priority, commercial_use_required=commercial,
    )


@dataclass(frozen=True)
class DiversifiedRecommendation:
    primary: RankedPlan
    alternatives: tuple[RankedPlan, ...]
    position: RecommendationPosition


def diversify_ranked_plans(
    ranked_plans: Sequence[RankedPlan],
    *,
    limit: int,
    tie_key: TieKey | None = None,
) -> list[DiversifiedRecommendation]:
    """Select up to ``limit`` logical models without re-ranking plan internals.

    ``ranked_plans`` is already sorted by Recommendation Engine v2. The first
    plan per ``ModelIdentity`` stays the primary plan. Other plans for the same
    identity are retained as alternatives, but never consume another main slot.
    The best-ranked plan always goes first; variety only chooses between plans the
    ranking cannot tell apart. ``tie_key`` says what that means; without it, plans with
    the same assessments and downloads count as tied.
    """

    if limit <= 0:
        return []
    buckets = _best_plan_per_identity(ranked_plans)
    if not buckets:
        return []

    selected: list[_IdentityBucket] = [buckets[0]]
    remaining = buckets[1:]
    while remaining and len(selected) < limit:
        candidate = _choose_next(selected, remaining, tie_key)
        selected.append(candidate)
        remaining.remove(candidate)

    return [
        DiversifiedRecommendation(
            primary=bucket.primary,
            alternatives=tuple(bucket.alternatives),
            position=_position_for(bucket.primary, selected[0].primary, index),
        )
        for index, bucket in enumerate(selected)
    ]


@dataclass(frozen=True)
class _IdentityBucket:
    key: str
    primary: RankedPlan
    alternatives: tuple[RankedPlan, ...]
    original_index: int


def _best_plan_per_identity(
    ranked_plans: Sequence[RankedPlan],
) -> list[_IdentityBucket]:
    grouped: dict[str, list[RankedPlan]] = {}
    indexes: dict[str, int] = {}
    for index, plan in enumerate(ranked_plans):
        canonical = confirmed_model_repo_id(plan.plan.model_identity)
        key = (canonical or plan.plan.artifact.repo_id).casefold()
        grouped.setdefault(key, []).append(plan)
        if not plan.assessment.rejected:
            indexes.setdefault(key, index)
    buckets: list[_IdentityBucket] = []
    for key in sorted(indexes, key=indexes.__getitem__):
        items = grouped[key]
        primary = next(item for item in items if not item.assessment.rejected)
        buckets.append(_IdentityBucket(
            key=key, primary=primary,
            alternatives=tuple(item for item in items if item is not primary),
            original_index=indexes[key],
        ))
    return buckets


def _choose_next(
    selected: Sequence[_IdentityBucket],
    remaining: Sequence[_IdentityBucket],
    tie_key: TieKey | None = None,
) -> _IdentityBucket:
    leader = remaining[0]
    signature = tie_key or _assessment_signature
    comparable = [
        item
        for item in remaining
        if signature(item.primary) == signature(leader.primary)
    ]
    return min(
        comparable,
        key=lambda item: (_redundancy_key(item.primary.plan, selected), item.original_index),
    )


def _assessment_signature(plan: RankedPlan) -> tuple[object, ...]:
    assessment = plan.assessment
    return (
        assessment.suitability,
        assessment.capability,
        assessment.execution_fitness,
        assessment.feasibility,
        assessment.executability,
        # Not performance_evidence: measured speed orders only inside the measured
        # policy's cohorts, never by making two plans look different here.
        assessment.resource_evidence,
        assessment.license_fit,
        assessment.language_fit,
        assessment.confidence,
        # The ranking also breaks ties by downloads; plans it ranks apart on them are not tied.
        plan.evaluated.candidate.downloads,
    )


def _redundancy_key(
    plan: ExecutionPlan,
    selected: Sequence[_IdentityBucket],
) -> tuple[int, int, int, int]:
    selected_plans = [bucket.primary.plan for bucket in selected]
    family = _family_key(plan.model_identity)
    tier = _parameter_tier(plan.model_identity.parameter_count)
    runtime = plan.runtime.runtime
    profile = _execution_profile(plan)
    return (
        _count_matching(selected_plans, lambda item: _family_key(item.model_identity) == family)
        if family is not None
        else 0,
        _count_matching(
            selected_plans,
            lambda item: _parameter_tier(item.model_identity.parameter_count) == tier,
        )
        if tier is not None
        else 0,
        _count_matching(selected_plans, lambda item: item.runtime.runtime is runtime),
        _count_matching(selected_plans, lambda item: _execution_profile(item) == profile),
    )


def _family_key(identity: ModelIdentity) -> str | None:
    family = identity.family
    if family is None or family.lower() == "unknown":
        return None
    return family.lower()


def _parameter_tier(parameter_count: int | None) -> str | None:
    if parameter_count is None:
        return None
    if parameter_count < 2_000_000_000:
        return "sub_2b"
    if parameter_count < 5_000_000_000:
        return "2b_to_5b"
    if parameter_count < 10_000_000_000:
        return "5b_to_10b"
    return "10b_plus"


def _execution_profile(plan: ExecutionPlan) -> tuple[object, ...]:
    backend = (
        plan.backend_selection.selected_backend.value
        if plan.backend_selection is not None
        else None
    )
    return (
        plan.runtime.runtime,
        backend,
        plan.artifact.quantization or plan.artifact.precision,
        _parameter_tier(plan.model_identity.parameter_count),
    )


def _count_matching(
    plans: Sequence[ExecutionPlan],
    predicate: object,
) -> int:
    match = predicate
    assert callable(match)
    return sum(1 for plan in plans if match(plan))


def _position_for(
    ranked: RankedPlan,
    best: RankedPlan,
    index: int,
) -> RecommendationPosition:
    if index == 0:
        return RecommendationPosition.BEST_OVERALL
    if ranked.assessment.measured_generation_tokens_per_second is not None:
        return RecommendationPosition.FASTER
    current_memory = (
        ranked.assessment.measured_memory_bytes
        or ranked.assessment.estimated_memory_bytes
    )
    best_memory = (
        best.assessment.measured_memory_bytes
        or best.assessment.estimated_memory_bytes
    )
    if (
        current_memory is not None
        and best_memory is not None
        and current_memory <= int(best_memory * 0.75)
    ):
        return RecommendationPosition.LIGHTER
    current_params = ranked.plan.model_identity.parameter_count
    best_params = best.plan.model_identity.parameter_count
    if (
        current_params is not None
        and best_params is not None
        and current_params >= int(best_params * 1.25)
    ):
        return RecommendationPosition.HIGHER_QUALITY
    return RecommendationPosition.ALTERNATIVE


__all__ = [
    "DiversifiedRecommendation",
    "diversify_ranked_plans",
]
