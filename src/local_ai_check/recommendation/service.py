"""Compose scoring, family collapsing, ranking and explanations."""

from __future__ import annotations

from local_ai_check.discovery.grouping import collapse_families
from local_ai_check.discovery.models import EvaluatedCandidate
from local_ai_check.recommendation import explanations, policies, ranker, scoring
from local_ai_check.recommendation.models import ModelRecommendation
from local_ai_check.workflow import policies as workflow_policies
from local_ai_check.workflow.models import UserRequirements


def score_all(
    evaluated: list[EvaluatedCandidate], requirements: UserRequirements
) -> list[EvaluatedCandidate]:
    """Populate every sub-score, normalising popularity across the whole set."""
    usable = [item for item in evaluated if not item.failed]
    max_log = scoring.max_log_downloads([item.candidate for item in usable])
    return [scoring.score_candidate(item, requirements, max_log) for item in usable]


def recommend(
    evaluated: list[EvaluatedCandidate],
    requirements: UserRequirements,
    limit: int = workflow_policies.MAX_RECOMMENDATIONS,
) -> list[ModelRecommendation]:
    """Produce at most ``limit`` explained recommendations."""
    scored = score_all(evaluated, requirements)
    if not scored:
        return []

    # Collapse the original repo and its GGUF conversions into one entry, using
    # the composite score to decide which member represents the family.
    provisional = {
        item.repo_id: ranker.score_breakdown(item, requirements.priority).total
        for item in scored
    }
    collapsed, siblings = collapse_families(scored, provisional)

    ranked = ranker.select_ranked(collapsed, requirements.priority, limit)
    if not ranked:
        return []

    primary = ranked[0][0]
    recommendations: list[ModelRecommendation] = []

    for index, (item, breakdown) in enumerate(ranked, start=1):
        label = (
            None if index == 1 else ranker.alternative_label(primary, item)
        )
        recommendations.append(
            ModelRecommendation(
                rank=index,
                evaluated=item,
                score=breakdown,
                status=ranker.status_of(item),
                confidence=scoring.confidence_of(item),
                license_category=policies.classify_license(item.candidate.license),
                reasons=explanations.build_reasons(item, requirements),
                warnings=explanations.build_warnings(item, requirements),
                related_repositories=siblings.get(item.repo_id, []),
                alternative_label=label,
            )
        )

    return recommendations


__all__ = ["recommend", "score_all"]
