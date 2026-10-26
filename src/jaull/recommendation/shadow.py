"""Shared measured policies for the active ranking and Recompare proposals.

Policy shadow-v2 (docs/recommendation.md):

1. Start from the active order before diversity.
2. Evidence never crosses a stratum: confirmed placement, rejection, commercial confirmation, task
   suitability, license and language. Groups form inside contiguous runs of one
   stratum, so a reordered plan cannot pass a plan of another stratum.
3. A group is the plans of one run sharing a comparable cohort. Only the
   positions its members already held are reordered; a plan without applicable
   evidence keeps its place, and a one-member group cannot move.
4. Quality orders by its metric. Fastest and Balanced compare each pair directly:
   speed only when observed ranges are disjoint, quality (Balanced) only when the
   95% Wilson intervals are. Balanced puts a plan first when no axis is
   demonstrably worse and one is better, so clearly better quality decides a
   speed tie and a trade-off decides nothing. Chained overlaps are never merged.
   The order is the closest to the base that honours every such pair. Memory is
   unchanged.
5. Existing diversity then picks the top 5 from each order, and the two kinds
   of change - moves and top-5 entries/exits - are reported separately.

Audited full IFEval can apply to English chat under its controlled-artifact
protocol, not the proposed execution configuration. Speed requires at least
three distinct valid calls per plan; its observed range is not a confidence interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from json import dumps
from math import sqrt
from typing import Any

from pydantic import BaseModel, ConfigDict

from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.recommendation import PlanAssessment
from jaull.domain.requirements import RecommendationPriority, UserRequirements
from jaull.recommendation.diversity import diversify_ranked_plans, tie_key_for
from jaull.recommendation.engine_v2 import RankedPlan, _requirement_confirmation_rank
from jaull.recommendation.local_evidence import assess_speed
from jaull.recommendation.policies import has_confirmed_memory_compatibility
from jaull.recommendation.quality import assess_quality, index_quality_records
from jaull.recommendation.selection import weak_plans_last

POLICY_VERSION = "shadow-v2"


@dataclass(frozen=True)
class Measured:
    """One applicable figure. Equal ``cohort`` must mean directly comparable."""

    cohort: str
    value: float
    label: str
    evidence_ids: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    comparison_ready: bool = True


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
    limitations: tuple[str, ...] = ()


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
    the requested profile's suite on these exact bytes. Speed applies as the plan's
    own llama_bench_v2 tg128 (``SpeedAssessment.applicable``); its cohort keeps
    figures from other machines, builds or protocols apart.
    """
    reasons = []
    quality = None
    measured = assessment.quality.applicable
    if measured is not None:
        low, high = _wilson(measured.correct, measured.samples)
        quality = Measured(
            cohort=measured.cohort, value=measured.value,
            label=f"{measured.suite} {measured.metric}",
            evidence_ids=(measured.record_sha256,),
            # Read only by Balanced: quality must be clearly better to decide a speed tie.
            minimum=low, maximum=high,
        )
        reasons.append(
            "Quality: measured, but it orders only against other measured plans in its "
            "comparable cohort and stratum."
        )
    elif assessment.quality.applicability == "absent":
        reasons.append("Quality: no evidence for this artifact.")
    else:
        reasons.append("Quality: references only; none passes task/protocol applicability.")
    speed = None
    applicable_speed = assessment.speed.applicable
    if applicable_speed is not None:
        speed = Measured(
            cohort=applicable_speed.cohort, value=applicable_speed.tokens_per_second,
            label=f"{applicable_speed.metric} tok/s",
            evidence_ids=applicable_speed.benchmark_ids or (applicable_speed.benchmark_id,),
            minimum=applicable_speed.observed_min, maximum=applicable_speed.observed_max,
            comparison_ready=applicable_speed.ranking_ready,
        )
        reasons.append(
            "Speed: measured, but it orders only against other measured plans in its "
            "comparable cohort and stratum."
            if applicable_speed.ranking_ready else
            f"Speed: {applicable_speed.run_count} stored calls; at least three distinct "
            "calls with an observed range are required to order plans."
        )
    elif assessment.speed.applicability == "absent":
        reasons.append("Speed: no benchmark of this exact artifact.")
    else:
        newest = assessment.speed.references[0]
        reasons.append(f"Speed: {newest.benchmark_id} is blocked: {newest.blockers[0]}"
                       if newest.blockers else f"Speed: {newest.benchmark_id} is reference only.")
    return PlanEvidence(quality=quality, speed=speed, fallback_reasons=tuple(reasons))


def apply_measured_policy(
    ranked: Sequence[RankedPlan], requirements: UserRequirements,
) -> list[RankedPlan]:
    """The active order: shadow-v2 for the request's priority.

    Quality reads applicable measured quality, Fastest applicable llama_bench_v2
    tg128 ranges, Balanced both (pairwise, a speed tie decided by clear quality). Only plans
    in one comparable cohort and one contiguous stratum run swap the positions
    they already held; every other plan keeps its base place, so with no
    applicable evidence this is the base order. Memory has no measured policy.
    """
    if _policy(requirements.priority) is None:
        return list(ranked)
    order, _, _ = _shadow_order(
        ranked, requirements.priority, requirements.commercial_use_required is True,
        lambda item: production_evidence(item.assessment),
    )
    return order


def apply_active_policy(
    ranked: Sequence[RankedPlan], requirements: UserRequirements,
) -> list[RankedPlan]:
    """Shared active order: selection preference, then comparable measured evidence."""
    selected = weak_plans_last(ranked, requirements.priority)
    return apply_measured_policy(selected, requirements)


def build_shadow_report(
    ranked: Sequence[RankedPlan],
    requirements: UserRequirements,
    *,
    limit: int,
    evidence: Mapping[str, PlanEvidence] | None = None,
) -> ShadowReport:
    priority = requirements.priority
    ranked = weak_plans_last(ranked, priority)
    commercial = requirements.commercial_use_required is True
    found = evidence if evidence is not None else {
        item.plan.plan_id: production_evidence(item.assessment) for item in ranked
    }
    blank = PlanEvidence()
    shadow, groups, moves = _shadow_order(
        ranked, priority, commercial, lambda item: found.get(item.plan.plan_id, blank),
    )
    tie_key = tie_key_for(requirements)
    base_top = [
        i.primary.plan.plan_id
        for i in diversify_ranked_plans(ranked, limit=limit, tie_key=tie_key)
    ]
    shadow_top = [
        i.primary.plan.plan_id
        for i in diversify_ranked_plans(shadow, limit=limit, tie_key=tie_key)
    ]
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
    benchmark_records: Sequence[BenchmarkRecord] | None = None,
) -> ShadowReport:
    """Refresh stored quality (and speed, given benchmarks) on the saved pool only.

    Hardware, readiness, task/eligibility strata and the fallback order stay as
    observed during Search; the pool itself is never re-planned or re-ranked.
    Without ``benchmark_records`` speed stays as Search saw it. The proposal is
    separate from that original result.
    """
    index = index_quality_records(records)
    refreshed = []
    for item in ranked:
        update: dict[str, Any] = {"quality": assess_quality(
            item.plan, requirements, item.assessment.quality.metadata_prior, index,
            published=item.assessment.external_evaluations,
        )}
        if benchmark_records is not None:
            update["speed"] = assess_speed(item.plan, requirements, benchmark_records)
        refreshed.append(replace(item, assessment=item.assessment.model_copy(update=update)))
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
        items = [ranked[i] for i in indexes]
        limitations: tuple[str, ...] = ()
        if policy != "quality" and _speed_overlaps(items, evidence_of):
            limitations = (
                "Some observed speed ranges overlap: those pairs are not ordered by speed "
                "(Balanced lets clearly better quality decide them). Ranges are min-max of "
                "the stored calls, not confidence intervals; one atypical call can keep two "
                "plans indistinguishable.",
            )
        groups.append(ShadowGroup(key=key, plan_ids=tuple(item.plan.plan_id for item in items),
                                  limitations=limitations))
        ordered, rules = _order_group(policy, items, evidence_of)
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
    if evidence.speed is None or not evidence.speed.comparison_ready:
        return None
    if policy == "fastest":
        return (evidence.speed.cohort,)
    if evidence.quality is None or evidence.speed is None:
        return None  # An unknown axis infers no dominance.
    return evidence.quality.cohort, evidence.speed.cohort


def _bounds(measured: Measured | None) -> tuple[float, float]:
    assert measured is not None
    return (measured.minimum if measured.minimum is not None else measured.value,
            measured.maximum if measured.maximum is not None else measured.value)


def _compare(a: Measured | None, b: Measured | None) -> int:
    """1 or -1 only when the two intervals are disjoint; 0 means indistinguishable."""
    (a_low, a_high), (b_low, b_high) = _bounds(a), _bounds(b)
    return 1 if a_low > b_high else -1 if a_high < b_low else 0


def _precedes(policy: str, a: PlanEvidence, b: PlanEvidence) -> bool:
    """Whether ``a`` demonstrably belongs before ``b``, judged on this pair alone.

    Fastest: disjoint speed ranges. Balanced: no axis demonstrably worse and one
    demonstrably better; overlapping speed is an operational tie, so clearly
    better quality decides it, and a real trade-off (faster vs better) decides
    nothing. Chained overlaps are never merged: A~B and B~C say nothing of A, C.
    """
    speed = _compare(a.speed, b.speed)
    if policy == "fastest":
        return speed > 0
    quality = _compare(a.quality, b.quality)
    return speed >= 0 and quality >= 0 and (speed > 0 or quality > 0)


def _demonstrable_order(
    items: list[RankedPlan], precedes: Callable[[RankedPlan, RankedPlan], bool],
) -> list[RankedPlan]:
    """Closest order to the base that honours every demonstrable pair.

    Filled from the back: the last slot goes to the latest base plan that must
    not precede any plan still unplaced. A winner therefore rises to the loser's
    place instead of the loser sinking below unrelated plans.
    """
    remaining = list(items)
    placed: list[RankedPlan] = []
    while remaining:
        free = [x for x in remaining
                if not any(precedes(x, y) for y in remaining if y is not x)]
        # ponytail: a cycle (not expected from interval comparisons) keeps base order.
        pick = free[-1] if free else remaining[-1]
        remaining.remove(pick)
        placed.append(pick)
    return placed[::-1]


def _order_group(
    policy: str, items: list[RankedPlan], evidence_of: Callable[[RankedPlan], PlanEvidence],
) -> tuple[list[RankedPlan], dict[str, str]]:
    """Items arrive in base order; Python's stable sort keeps it for ties."""
    if policy == "quality":
        def figure(item: RankedPlan) -> Measured:
            measured = evidence_of(item).quality
            assert measured is not None
            return measured

        ordered = sorted(items, key=lambda item: -figure(item).value)
        return ordered, {
            item.plan.plan_id: f"{figure(item).label} {figure(item).value:g}" for item in items
        }

    ordered = _demonstrable_order(
        items, lambda a, b: _precedes(policy, evidence_of(a), evidence_of(b)),
    )
    rules = {}
    for item in items:
        found = evidence_of(item)
        assert found.speed is not None
        low, high = _bounds(found.speed)
        rule = f"tg128 median {found.speed.value:g}, observed range {low:g}-{high:g}"
        if policy == "balanced":
            assert found.quality is not None
            q_low, q_high = _bounds(found.quality)
            rule += (f"; {found.quality.label} {found.quality.value:g} "
                     f"(95% interval {q_low:.3f}-{q_high:.3f})")
        rules[item.plan.plan_id] = rule + "; ordered only by pairwise distinguishable differences"
    return ordered, rules


def _speed_overlaps(items: list[RankedPlan],
                    evidence_of: Callable[[RankedPlan], PlanEvidence]) -> bool:
    return any(_compare(evidence_of(a).speed, evidence_of(b).speed) == 0
               for index, a in enumerate(items) for b in items[index + 1:])


def _wilson(correct: int, samples: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a proportion."""
    p = correct / samples
    denominator = 1 + z * z / samples
    centre = (p + z * z / (2 * samples)) / denominator
    half = z * sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


__all__ = [
    "POLICY_VERSION",
    "Measured",
    "PlanEvidence",
    "ShadowReport",
    "apply_active_policy",
    "apply_measured_policy",
    "build_shadow_report",
    "production_evidence",
    "recompare_quality",
]
