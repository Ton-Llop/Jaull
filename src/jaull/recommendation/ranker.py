"""Turn scored candidates into a ranked, deterministic shortlist."""

from __future__ import annotations

from collections.abc import Callable

from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation import policies
from jaull.recommendation.actionability import (
    ActionabilityLevel,
    actionability_penalty,
    assess_actionability,
)
from jaull.recommendation.models import ScoreBreakdown

PenaltyFn = Callable[[EvaluatedCandidate], float]
UnmetFn = Callable[[EvaluatedCandidate], list[str]]


def weights_for(priority: RecommendationPriority) -> dict[str, float]:
    """Base weights adjusted for the priority, renormalised back to 1.0.

    Renormalising matters: without it, boosting one component would quietly
    inflate every total and make scores from different priorities incomparable.
    """
    modifiers = policies.PRIORITY_MODIFIERS[priority]
    adjusted = {
        key: value * modifiers.get(key, 1.0)
        for key, value in policies.BASE_WEIGHTS.items()
    }
    total = sum(adjusted.values())
    if total <= 0:
        return dict(policies.BASE_WEIGHTS)
    return {key: value / total for key, value in adjusted.items()}


def score_breakdown(
    evaluated: EvaluatedCandidate,
    priority: RecommendationPriority,
    hard_penalty: float = 1.0,
    unmet_requirements: list[str] | None = None,
) -> ScoreBreakdown:
    weights = weights_for(priority)
    parts = {
        "memory_fit": evaluated.memory_fit_score,
        "concurrency_fit": evaluated.concurrency_fit_score,
        "capability": evaluated.capability_score,
        "task_match": evaluated.task_match_score,
        "language_match": evaluated.language_match_score,
        "license": evaluated.license_score,
        "metadata_quality": evaluated.metadata_quality_score,
        "popularity": evaluated.popularity_score,
    }
    total = sum(parts[key] * weights[key] for key in weights)
    # Confidence scales the whole result: an answer we could not measure well
    # must not outrank one we could.
    total *= policies.CONFIDENCE_MULTIPLIER[_confidence(evaluated)]
    # Hard-requirement penalties come last so they can veto a top pick.
    total *= max(0.0, min(1.0, hard_penalty))
    # Actionability gate: a speculative candidate must not out-rank a
    # confirmed-and-executable one of similar composite quality.
    total *= actionability_penalty(assess_actionability(evaluated))
    return ScoreBreakdown(
        total=max(0.0, min(1.0, total)),
        weights=weights,
        hardware_fit=min(evaluated.memory_fit_score, evaluated.concurrency_fit_score),
        hard_penalty=hard_penalty,
        unmet_requirements=unmet_requirements or [],
        **parts,
    )


def _confidence(evaluated: EvaluatedCandidate) -> EstimationConfidence:
    if evaluated.compatibility is not None:
        return evaluated.compatibility.confidence
    return evaluated.candidate.metadata_confidence


def status_of(evaluated: EvaluatedCandidate) -> CompatibilityStatus:
    if evaluated.compatibility is None:
        return CompatibilityStatus.UNKNOWN
    return evaluated.compatibility.status


def can_be_primary(evaluated: EvaluatedCandidate) -> bool:
    """A model must be at least ``WORST_PRIMARY_STATUS`` to lead the results.

    Excluded from the primary slot:

    - ``insufficient`` / ``unknown`` compatibility status;
    - ``BLOCKED`` actionability (unsupported runtime, or hard veto).
    """
    status = status_of(evaluated)
    if status is CompatibilityStatus.UNKNOWN:
        return False
    if assess_actionability(evaluated) is ActionabilityLevel.BLOCKED:
        return False
    return policies.STATUS_RANK[status] <= policies.STATUS_RANK[
        policies.WORST_PRIMARY_STATUS
    ]


def sort_candidates(
    evaluated: list[EvaluatedCandidate],
    priority: RecommendationPriority,
    penalty_of: PenaltyFn | None = None,
    unmet_of: UnmetFn | None = None,
) -> list[tuple[EvaluatedCandidate, ScoreBreakdown]]:
    """Rank candidates with a fully deterministic ordering.

    Ties are broken by status, then confidence, then downloads, then repo_id, so
    the same inputs always produce the same output — important for a tool whose
    results end up in a report.
    """
    scored = [
        (
            item,
            score_breakdown(
                item,
                priority,
                hard_penalty=penalty_of(item) if penalty_of else 1.0,
                unmet_requirements=unmet_of(item) if unmet_of else [],
            ),
        )
        for item in evaluated
    ]

    def key(pair: tuple[EvaluatedCandidate, ScoreBreakdown]) -> tuple[object, ...]:
        item, breakdown = pair
        return (
            -round(breakdown.total, 6),
            policies.STATUS_RANK[status_of(item)],
            -policies.CONFIDENCE_SCORE[
                item.compatibility.confidence
                if item.compatibility is not None
                else item.candidate.metadata_confidence
            ],
            -item.candidate.downloads,
            item.candidate.repo_id,
        )

    return sorted(scored, key=key)


def select_ranked(
    evaluated: list[EvaluatedCandidate],
    priority: RecommendationPriority,
    limit: int,
    penalty_of: PenaltyFn | None = None,
    unmet_of: UnmetFn | None = None,
) -> list[tuple[EvaluatedCandidate, ScoreBreakdown]]:
    """Pick up to ``limit`` results with a valid primary at the front.

    Filters out ``INSUFFICIENT`` memory and ``BLOCKED`` actionability — the
    latter catches "AWQ recommended on a CPU-only machine" and similar
    runtime/hardware mismatches surfaced by the executability rules.
    """
    ordered = [
        pair
        for pair in sort_candidates(evaluated, priority, penalty_of, unmet_of)
        if status_of(pair[0]) is not CompatibilityStatus.INSUFFICIENT
        and assess_actionability(pair[0]) is not ActionabilityLevel.BLOCKED
    ]
    if not ordered:
        return []

    # Promote the best candidate that is allowed to lead. If none qualifies we
    # return the list unchanged: the caller renders it as "closest matches"
    # rather than as a recommendation.
    for index, pair in enumerate(ordered):
        if can_be_primary(pair[0]):
            if index:
                ordered.insert(0, ordered.pop(index))
            break

    return ordered[:limit]


# Minimum capability-score gap before we are willing to say "higher capability".
# 0.05 is deliberately small: the score is bounded [0,1] and 5 % is roughly one
# meaningful rung on the size curve without demanding a huge delta.
_CAPABILITY_GAP = 0.05
# Companion memory-fit threshold: the alternative is only "tighter" when it
# is measurably harder to fit than the primary.
_MEMORY_GAP = 0.05


def alternative_label(
    primary: EvaluatedCandidate, other: EvaluatedCandidate
) -> str | None:
    """Describe how an alternative differs — only when the difference is real.

    Uses ``capability_score`` (a real family/parameter-count signal) rather
    than raw memory footprint. That is what stops the previous behaviour
    where a bigger *estimate* was silently reported as "higher quality"
    regardless of the underlying model.
    """
    primary_total = _total_bytes(primary)
    other_total = _total_bytes(other)

    higher_capability = (
        other.capability_score >= primary.capability_score + _CAPABILITY_GAP
    )
    tighter_fit = (
        other.memory_fit_score + _MEMORY_GAP <= primary.memory_fit_score
    )
    safer_fit = (
        other.memory_fit_score >= primary.memory_fit_score + _MEMORY_GAP
    )

    if higher_capability and tighter_fit:
        return "Higher estimated capability, tighter fit"
    if higher_capability:
        return "Higher estimated capability"
    if safer_fit and other.capability_score <= primary.capability_score:
        return "Safer option"

    if primary_total is not None and other_total is not None:
        if other_total < primary_total * 0.9:
            return "Smaller and faster option"
        if other_total > primary_total * 1.1:
            # Larger footprint without any evidence of higher capability →
            # honest label rather than an inferred quality claim.
            return "Larger footprint, similar capability"

    return "Alternative"


def _total_bytes(evaluated: EvaluatedCandidate) -> int | None:
    if evaluated.memory_estimate is None:
        return None
    return evaluated.memory_estimate.total_bytes


__all__ = [
    "alternative_label",
    "can_be_primary",
    "score_breakdown",
    "select_ranked",
    "sort_candidates",
    "status_of",
    "weights_for",
]
