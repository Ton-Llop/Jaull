"""Selection preferences applied within the base order's viability boundaries."""

from collections.abc import Sequence

from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.engine_v2 import RankedPlan, _runnability_rank
from jaull.recommendation.policies import has_confirmed_memory_compatibility


def _viability_group(item: RankedPlan) -> tuple[bool, bool, int]:
    """Smaller is more viable: not rejected, memory fit confirmed, then runnability."""
    prediction = item.plan.memory_prediction
    confirmed = prediction is not None and has_confirmed_memory_compatibility(
        prediction.assessment.status
    )
    return (item.assessment.rejected, not confirmed, _runnability_rank(item.assessment))


def weak_plans_last(
    plans: Sequence[RankedPlan],
    priority: RecommendationPriority,
) -> list[RankedPlan]:
    """Balanced: a plan without known capability goes behind every capable plan that is at
    least as viable as it, and nowhere else.

    Only the plans that are not capable move, and only backwards; capable plans keep their
    relative order, so the licence-first and runnable-before-capable orders still decide
    between them. A capable plan that is *less* viable than the weak one (it needs offloading
    while the weak one runs) does not hold the weak one back: that plan stays ahead of the
    capable ones that run. Unknown capability cannot leapfrog weak capability merely
    because it is not labelled weak.
    """
    if priority is not RecommendationPriority.BALANCED:
        return list(plans)
    capable_levels = {AssessmentLevel.STRONG, AssessmentLevel.ADEQUATE}
    capable = [p for p in plans if p.assessment.capability in capable_levels]
    classes = [_viability_group(p) for p in capable]
    # slots[i] holds the plans that come right after capable[i - 1].
    slots: list[list[RankedPlan]] = [[] for _ in range(len(capable) + 1)]
    seen_capable = 0
    for plan in plans:
        if plan.assessment.capability in capable_levels:
            seen_capable += 1
            continue
        own = _viability_group(plan)
        anchor = max(
            (i + 1 for i, other in enumerate(classes) if other <= own),
            default=0,
        )
        slots[max(seen_capable, anchor)].append(plan)
    result: list[RankedPlan] = list(slots[0])
    for index, plan in enumerate(capable):
        result.append(plan)
        result.extend(slots[index + 1])
    return result
