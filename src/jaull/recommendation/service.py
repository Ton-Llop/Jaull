"""Compose scoring, family collapsing, ranking and explanations."""

from __future__ import annotations

from jaull.discovery import series
from jaull.discovery.grouping import collapse_families
from jaull.discovery.models import EvaluatedCandidate
from jaull.domain.estimation import CompatibilityStatus
from jaull.recommendation import (
    capability,
    explanations,
    policies,
    ranker,
    requirements_gate,
    scoring,
    tier,
)
from jaull.recommendation.models import (
    ModelRecommendation,
    SeriesSibling,
)
from jaull.workflow import policies as workflow_policies
from jaull.workflow.models import UserRequirements


def score_all(
    evaluated: list[EvaluatedCandidate],
    requirements: UserRequirements,
    capability_analyzer: capability.CapabilityAnalyzer | None = None,
) -> list[EvaluatedCandidate]:
    """Populate every sub-score, normalising popularity across the whole set."""
    usable = [item for item in evaluated if not item.failed]
    max_log = scoring.max_log_downloads([item.candidate for item in usable])
    analyzer = capability_analyzer or capability.MetadataCapabilityAnalyzer()
    scored = [
        scoring.score_candidate(
            item,
            requirements,
            max_log,
            capability_signal=analyzer.analyze(item.candidate, item.analysis).score,
        )
        for item in usable
    ]

    # Second pass: evaluate hard requirements once the sub-scores exist (so
    # concurrency_fit_score is available to the gate).
    with_gate: list[EvaluatedCandidate] = []
    for item in scored:
        satisfaction = requirements_gate.evaluate_requirements(item, requirements)
        with_gate.append(
            item.model_copy(
                update={
                    "requirement_penalty": satisfaction.penalty_multiplier,
                    "unmet_requirement_labels": satisfaction.unmet_labels,
                }
            )
        )
    return with_gate


def recommend(
    evaluated: list[EvaluatedCandidate],
    requirements: UserRequirements,
    limit: int = workflow_policies.MAX_RECOMMENDATIONS,
    capability_analyzer: capability.CapabilityAnalyzer | None = None,
) -> list[ModelRecommendation]:
    """Produce at most ``limit`` explained recommendations."""
    analyzer = capability_analyzer or capability.MetadataCapabilityAnalyzer()
    scored = score_all(evaluated, requirements, capability_analyzer=analyzer)
    if not scored:
        return []

    # Collapse the original repo and its GGUF conversions into one entry, using
    # the composite score to decide which member represents the family.
    provisional = {
        item.repo_id: ranker.score_breakdown(
            item,
            requirements.priority,
            hard_penalty=item.requirement_penalty,
            unmet_requirements=item.unmet_requirement_labels,
        ).total
        for item in scored
    }
    collapsed, siblings = collapse_families(scored, provisional)

    # Series grouping is applied on top of family collapsing: same-family models
    # of different sizes (Qwen2.5-0.5B / 1.5B / 3B / 7B) become one primary +
    # SeriesSiblings so we recommend the best size instead of three of them.
    series_map = series.group_by_series(collapsed)
    series_winners: list[EvaluatedCandidate] = []
    series_others: dict[str, list[EvaluatedCandidate]] = {}
    for members in series_map.values():
        if len(members) <= 1:
            series_winners.append(members[0])
            continue
        best = max(
            members, key=lambda item: provisional.get(item.repo_id, 0.0)
        )
        series_winners.append(best)
        series_others[best.repo_id] = [
            m for m in members if m.repo_id != best.repo_id
        ]

    ranked = ranker.select_ranked(
        series_winners,
        requirements.priority,
        limit,
        penalty_of=lambda item: item.requirement_penalty,
        unmet_of=lambda item: item.unmet_requirement_labels,
    )
    if not ranked:
        return []

    primary = ranked[0][0]
    recommendations: list[ModelRecommendation] = []

    for index, (item, breakdown) in enumerate(ranked, start=1):
        label = (
            None if index == 1 else ranker.alternative_label(primary, item)
        )
        status = ranker.status_of(item)
        confidence = scoring.confidence_of(item)
        rec_tier = tier.choose_tier(status, confidence, breakdown.hard_penalty)
        recommendations.append(
            ModelRecommendation(
                rank=index,
                evaluated=item,
                score=breakdown,
                status=status,
                confidence=confidence,
                license_category=policies.classify_license(item.candidate.license),
                tier=rec_tier.value,
                reasons=explanations.build_reasons(item, requirements),
                warnings=explanations.build_warnings(item, requirements),
                related_repositories=siblings.get(item.repo_id, []),
                alternative_label=label,
                series_alternatives=_build_series_siblings(
                    series_others.get(item.repo_id, []),
                    analyzer,
                ),
            )
        )

    return recommendations


def _build_series_siblings(
    others: list[EvaluatedCandidate],
    capability_analyzer: capability.CapabilityAnalyzer,
) -> list[SeriesSibling]:
    """Turn the losing members of a series into displayable siblings."""
    siblings: list[SeriesSibling] = []
    for item in others:
        params = capability_analyzer.analyze(
            item.candidate, item.analysis
        ).parameter_count
        label = series.parameter_label(params)
        status = (
            item.compatibility.status
            if item.compatibility is not None
            else CompatibilityStatus.UNKNOWN
        )
        total = (
            item.memory_estimate.total_bytes if item.memory_estimate else None
        )
        note = _sibling_note(label, status)
        siblings.append(
            SeriesSibling(
                repo_id=item.repo_id,
                parameter_count=params,
                parameter_label=label,
                status=status,
                total_bytes=total,
                note=note,
            )
        )
    # Order: smallest first, then by status quality.
    siblings.sort(
        key=lambda s: (s.parameter_count or 0, s.status.value)
    )
    return siblings


def _sibling_note(label: str, status: CompatibilityStatus) -> str:
    words = {
        CompatibilityStatus.COMFORTABLE: "comfortable",
        CompatibilityStatus.COMPATIBLE: "fits",
        CompatibilityStatus.TIGHT: "tight fit",
        CompatibilityStatus.OFFLOADING_REQUIRED: "needs offloading",
        CompatibilityStatus.INSUFFICIENT: "does not fit",
        CompatibilityStatus.UNKNOWN: "unknown",
    }
    return f"{label} — {words[status]}"


__all__ = ["recommend", "score_all"]
