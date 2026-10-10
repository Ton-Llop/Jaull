"""Quality/Fastest/Balanced orders computed beside the active ranking, never instead of it.

Policy shadow-v1 (docs/recommendation-vfinal-plan.md, section 3):

1. Start from the active order before diversity.
2. Evidence never crosses a stratum: confirmed placement, rejection, commercial confirmation, task
   suitability, license and language. Groups form inside contiguous runs of one
   stratum, so a reordered plan cannot pass a plan of another stratum.
3. A group is the plans of one run sharing a comparable cohort. Only the
   positions its members already held are reordered; a plan without applicable
   evidence keeps its place, and a one-member group cannot move.
4. Quality and Fastest order by their metric (higher is better), ties in base
   order. Balanced needs both axes in the same cohorts and orders Pareto fronts,
   each front in base order. Memory keeps the active order.
5. Existing diversity then picks the top 5 from each order, and the two kinds
   of change - moves and top-5 entries/exits - are reported separately.

Audited full IFEval can apply to English chat under its controlled-artifact
protocol, not the proposed execution configuration. Speed still falls back.
`PlanEvidence` is the boundary; neither projection replaces the active order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from json import dumps
from typing import Any

from pydantic import BaseModel, ConfigDict

from jaull.domain.recommendation import PlanAssessment
from jaull.domain.requirements import RecommendationPriority, UserRequirements
from jaull.recommendation.diversity import diversify_ranked_plans
from jaull.recommendation.engine_v2 import RankedPlan, _requirement_confirmation_rank
from jaull.recommendation.policies import has_confirmed_memory_compatibility
from jaull.recommendation.quality import assess_quality, index_quality_records

POLICY_VERSION = "shadow-v1"


@dataclass(frozen=True)
class Measured:
    """One applicable figure. Equal ``cohort`` must mean directly comparable."""

    cohort: str
    value: float
    label: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanEvidence:
    quality: Measured | None = None
    speed: Measured | None = None
    fallback_reasons: tuple[str, ...] = ()


class ShadowMove(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    repo_id: str
    base_index: int
    shadow_index: int
    group: str
    rule: str
    evidence_ids: tuple[str, ...] = ()


class ShadowGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    plan_ids: tuple[str, ...]


class ShadowFallback(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    reasons: tuple[str, ...]
    repo_id: str = ""
    artifact_label: str = ""
    artifact_sha256: str | None = None


class ShadowTop5Change(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    change: str  # "entered" or "left"
    # False means the plan did not move: diversity or a neighbour's move did it.
    moved: bool


class ShadowReport(BaseModel):
    """What the proposed policy would show, built from the same ranked pool."""

    model_config = ConfigDict(frozen=True)

    policy_version: str = POLICY_VERSION
    priority: RecommendationPriority
    applied: bool
    base_order: tuple[str, ...]
    shadow_order: tuple[str, ...]
    groups: tuple[ShadowGroup, ...] = ()
    moves: tuple[ShadowMove, ...] = ()
    fallback: tuple[ShadowFallback, ...] = ()
    base_top5: tuple[str, ...] = ()
    shadow_top5: tuple[str, ...] = ()
    top5_changes: tuple[ShadowTop5Change, ...] = ()


def production_evidence(assessment: PlanAssessment) -> PlanEvidence:
    """Applicable evidence from what ranking already attached, with why where there is none.

    Quality applies only as ``controlled_artifact``: one complete audited run of
    the requested profile's suite on these exact bytes. Speed never applies yet.
    """
    reasons = []
    quality = None
    measured = assessment.quality.applicable
    if measured is not None:
        quality = Measured(
            cohort=measured.cohort, value=measured.value,
            label=f"{measured.suite} {measured.metric}",
            evidence_ids=(measured.record_sha256,),
        )
        reasons.append(
            "Quality: measured, but it orders only against other measured plans in its "
            "comparable cohort and stratum."
        )
    elif assessment.quality.applicability == "absent":
        reasons.append("Quality: no evidence for this artifact.")
    else:
        reasons.append("Quality: references only; none passes task/protocol applicability.")
    if assessment.speed.applicability == "absent":
        reasons.append("Speed: no benchmark of this exact artifact.")
    else:
        newest = assessment.speed.references[0]
        reasons.append(f"Speed: {newest.benchmark_id} is blocked: {newest.blockers[0]}"
                       if newest.blockers else f"Speed: {newest.benchmark_id} is reference only.")
    return PlanEvidence(quality=quality, fallback_reasons=tuple(reasons))


def apply_quality_policy(
    ranked: Sequence[RankedPlan], requirements: UserRequirements,
) -> list[RankedPlan]:
    """The active order for the Quality priority: shadow-v1 Quality, activated in Step 6.

    Only plans with applicable measured quality, in one comparable cohort and one
    contiguous stratum run, swap the positions they already held; every other
    plan keeps its base place, so with no applicable evidence this is the base
    order. Other priorities are returned unchanged: Speed and Balanced stay in
    shadow until speed evidence applies.
    """
    if requirements.priority is not RecommendationPriority.QUALITY:
        return list(ranked)
    order, _, _ = _shadow_order(
        ranked, requirements.priority, requirements.commercial_use_required is True,
        lambda item: production_evidence(item.assessment),
    )
    return order


def build_shadow_report(
    ranked: Sequence[RankedPlan],
    requirements: UserRequirements,
    *,
    limit: int,
    evidence: Mapping[str, PlanEvidence] | None = None,
) -> ShadowReport:
    priority = requirements.priority
    commercial = requirements.commercial_use_required is True
    found = evidence if evidence is not None else {
        item.plan.plan_id: production_evidence(item.assessment) for item in ranked
    }
    blank = PlanEvidence()
    shadow, groups, moves = _shadow_order(
        ranked, priority, commercial, lambda item: found.get(item.plan.plan_id, blank),
    )
    base_top = [i.primary.plan.plan_id for i in diversify_ranked_plans(ranked, limit=limit)]
    shadow_top = [i.primary.plan.plan_id for i in diversify_ranked_plans(shadow, limit=limit)]
    moved = {move.plan_id for move in moves}
    in_group = {plan_id for group in groups for plan_id in group.plan_ids}
    return ShadowReport(
        priority=priority,
        applied=_policy(priority) is not None,
        base_order=tuple(item.plan.plan_id for item in ranked),
        shadow_order=tuple(item.plan.plan_id for item in shadow),
        groups=tuple(groups),
        moves=tuple(moves),
        fallback=tuple(
            ShadowFallback(plan_id=item.plan.plan_id,
                           repo_id=item.plan.artifact.repo_id,
                           artifact_label=item.plan.artifact.label,
                           artifact_sha256=item.plan.artifact.sha256,
                           reasons=_fallback_reasons(item, ranked, found, priority))
            for item in ranked
            if item.plan.plan_id not in in_group and not item.assessment.rejected
        ),
        base_top5=tuple(base_top),
        shadow_top5=tuple(shadow_top),
        top5_changes=tuple(
            [ShadowTop5Change(plan_id=p, change="entered", moved=p in moved)
             for p in shadow_top if p not in base_top]
            + [ShadowTop5Change(plan_id=p, change="left", moved=p in moved)
               for p in base_top if p not in shadow_top]
        ),
    )


def _fallback_reasons(
    item: RankedPlan, ranked: Sequence[RankedPlan],
    evidence: Mapping[str, PlanEvidence], priority: RecommendationPriority,
) -> tuple[str, ...]:
    found = evidence.get(item.plan.plan_id, PlanEvidence())
    reasons = list(found.fallback_reasons)
    if item.plan.artifact.sha256 is None:
        reasons.insert(0,
            "Quality: artifact SHA256 unavailable; exact-artifact matching is impossible."
        )
    quality = item.assessment.quality
    if found.quality is None:
        if quality.local_references:
            references = ", ".join(dict.fromkeys(
                f"{ref.suite}: {ref.classification}, "
                f"{ref.samples_used}/{ref.samples_available} samples"
                for ref in quality.local_references
            ))
            reasons.append(f"Quality references for this SHA256: {references}.")
        reasons.extend(quality.limitations)
    policy = _policy(priority)
    if policy is None:
        reasons.insert(0, "This priority does not reorder plans using measured evidence.")
    elif (cohort := _cohort(policy, found)) is not None:
        peers = [
            evidence.get(other.plan.plan_id, PlanEvidence()) for other in ranked
            if other.plan.plan_id != item.plan.plan_id and not other.assessment.rejected
        ]
        if any(_cohort(policy, peer) == cohort for peer in peers):
            # This plan has no group despite a comparable peer: a stratum boundary separates them.
            reasons.insert(0,
                "Comparable evidence exists, but a stratum boundary separates the plans "
                "(confirmed fit, commercial confirmation, suitability, license or language)."
            )
        elif any(_cohort(policy, peer) is not None for peer in peers):
            reasons.insert(0,
                "Other eligible plans have evidence, but no matching comparison protocol "
                "(suite, dataset/samples, evaluator, runtime or generation settings)."
            )
        else:
            reasons.insert(0, "No second eligible plan has applicable evidence for this priority.")
    elif policy == "balanced":
        reasons.insert(0,
            "Balanced requires applicable quality and speed; an unknown axis cannot rank."
        )
    return tuple(dict.fromkeys(reasons))


def recompare_quality(
    ranked: Sequence[RankedPlan], requirements: UserRequirements,
    records: Sequence[dict[str, Any]], *, limit: int,
) -> ShadowReport:
    """Refresh only quality evidence on the saved pool; never re-plan or re-rank it.

    Hardware, readiness, task/eligibility strata, speed and the fallback order stay
    as observed during Search. The proposal is separate from that original result.
    """
    index = index_quality_records(records)
    refreshed = [
        replace(item, assessment=item.assessment.model_copy(update={
            "quality": assess_quality(
                item.plan, requirements, item.assessment.quality.metadata_prior, index,
                published=item.assessment.external_evaluations,
            ),
        }))
        for item in ranked
    ]
    return build_shadow_report(refreshed, requirements, limit=limit)


def _policy(priority: RecommendationPriority) -> str | None:
    return {
        RecommendationPriority.QUALITY: "quality",
        RecommendationPriority.SPEED: "fastest",
        RecommendationPriority.BALANCED: "balanced",
    }.get(priority)


def _shadow_order(
    ranked: Sequence[RankedPlan],
    priority: RecommendationPriority,
    commercial: bool,
    evidence_of: Callable[[RankedPlan], PlanEvidence],
) -> tuple[list[RankedPlan], list[ShadowGroup], list[ShadowMove]]:
    policy = _policy(priority)
    order = list(ranked)
    if policy is None:
        return order, [], []

    # Contiguous runs of one stratum; a group never spans two runs.
    members: dict[tuple[int, tuple[str, ...]], list[int]] = {}
    run = 0
    previous = None
    for index, item in enumerate(ranked):
        stratum = _stratum(item, commercial)
        if index and stratum != previous:
            run += 1
        previous = stratum
        if item.assessment.rejected:
            continue
        cohort = _cohort(policy, evidence_of(item))
        if cohort is not None:
            members.setdefault((run, cohort), []).append(index)

    position = {item.plan.plan_id: index for index, item in enumerate(ranked)}
    groups: list[ShadowGroup] = []
    moves: list[ShadowMove] = []
    for (run_index, cohort), indexes in members.items():
        if len(indexes) < 2:
            continue  # A lone measurement is not promoted for existing.
        key = f"run{run_index}|{dumps(cohort)}"
        groups.append(ShadowGroup(key=key, plan_ids=tuple(ranked[i].plan.plan_id for i in indexes)))
        ordered, rules = _order_group(policy, [ranked[i] for i in indexes], evidence_of)
        for slot, item in zip(indexes, ordered, strict=True):
            order[slot] = item
            base_index = position[item.plan.plan_id]
            if base_index != slot:
                found = evidence_of(item)
                used = (
                    (found.quality,) if policy == "quality" else
                    (found.speed,) if policy == "fastest" else
                    (found.quality, found.speed)
                )
                ids = tuple(evidence_id for measured in used if measured is not None
                            for evidence_id in measured.evidence_ids)
                moves.append(ShadowMove(
                    plan_id=item.plan.plan_id, repo_id=item.evaluated.repo_id,
                    base_index=base_index, shadow_index=slot, group=key,
                    rule=rules[item.plan.plan_id], evidence_ids=ids,
                ))
    moves.sort(key=lambda move: move.shadow_index)
    return order, groups, moves


def _stratum(item: RankedPlan, commercial: bool) -> tuple[object, ...]:
    a = item.assessment
    prediction = item.plan.memory_prediction
    confirmed = (
        not a.rejected and prediction is not None
        and has_confirmed_memory_compatibility(prediction.assessment.status)
    )
    return (
        confirmed,
        a.rejected,
        _requirement_confirmation_rank(item) if commercial else 0,
        a.suitability, a.license_fit, a.language_fit,
    )


def _cohort(policy: str, evidence: PlanEvidence) -> tuple[str, ...] | None:
    if policy == "quality":
        return (evidence.quality.cohort,) if evidence.quality else None
    if policy == "fastest":
        return (evidence.speed.cohort,) if evidence.speed else None
    if evidence.quality is None or evidence.speed is None:
        return None  # An unknown axis infers no dominance.
    return evidence.quality.cohort, evidence.speed.cohort


def _order_group(
    policy: str, items: list[RankedPlan], evidence_of: Callable[[RankedPlan], PlanEvidence],
) -> tuple[list[RankedPlan], dict[str, str]]:
    """Items arrive in base order; Python's stable sort keeps it for ties."""
    if policy in {"quality", "fastest"}:
        def figure(item: RankedPlan) -> Measured:
            found = evidence_of(item)
            measured = found.quality if policy == "quality" else found.speed
            assert measured is not None
            return measured

        ordered = sorted(items, key=lambda item: -figure(item).value)
        return ordered, {
            item.plan.plan_id: f"{figure(item).label} {figure(item).value:g}" for item in items
        }

    def axes(item: RankedPlan) -> tuple[float, float]:
        found = evidence_of(item)
        assert found.quality is not None and found.speed is not None
        return found.quality.value, found.speed.value

    def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return a[0] >= b[0] and a[1] >= b[1] and a != b

    fronts: dict[str, int] = {}
    remaining = list(items)
    front = 0
    while remaining:
        current = [x for x in remaining
                   if not any(dominates(axes(y), axes(x)) for y in remaining if y is not x)]
        for item in current:
            fronts[item.plan.plan_id] = front
        placed = {item.plan.plan_id for item in current}
        remaining = [x for x in remaining if x.plan.plan_id not in placed]
        front += 1
    ordered = sorted(items, key=lambda item: fronts[item.plan.plan_id])
    rules = {}
    for item in items:
        quality, speed = axes(item)
        found = evidence_of(item)
        assert found.quality is not None and found.speed is not None
        rules[item.plan.plan_id] = (
            f"Pareto front {fronts[item.plan.plan_id] + 1}: "
            f"{found.quality.label} {quality:g}, {found.speed.label} {speed:g}"
        )
    return ordered, rules


__all__ = [
    "POLICY_VERSION",
    "Measured",
    "PlanEvidence",
    "ShadowReport",
    "apply_quality_policy",
    "build_shadow_report",
    "production_evidence",
    "recompare_quality",
]
