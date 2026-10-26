"""Selection preferences applied within the base order's viability boundaries."""

from collections.abc import Sequence
from itertools import groupby

from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.engine_v2 import RankedPlan, _runnability_rank
from jaull.recommendation.policies import has_confirmed_memory_compatibility


def _viability_group(item: RankedPlan) -> tuple[bool, bool, int]:
    prediction = item.plan.memory_prediction
    return (
        item.assessment.rejected,
        prediction is not None and has_confirmed_memory_compatibility(prediction.assessment.status),
        _runnability_rank(item.assessment),
    )


def weak_plans_last(
    plans: Sequence[RankedPlan],
    priority: RecommendationPriority,
) -> list[RankedPlan]:
    """Balanced prefers known capability only inside contiguous viability groups.

    No plan crosses a confirmed-fit, rejection or runnability boundary. Preserve
    order inside each preference class; unknown capability cannot leapfrog weak
    capability merely because it is not labelled weak.
    """
    if priority is not RecommendationPriority.BALANCED:
        return list(plans)
    result: list[RankedPlan] = []
    capable = {AssessmentLevel.STRONG, AssessmentLevel.ADEQUATE}
    for _, members in groupby(plans, key=_viability_group):
        group = list(members)
        result.extend(p for p in group if p.assessment.capability in capable)
        result.extend(p for p in group if p.assessment.capability not in capable)
    return result
