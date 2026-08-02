"""The six sub-scores that feed the composite ranking.

Every function returns a value in ``[0.0, 1.0]`` so the weights in
:mod:`local_ai_check.recommendation.policies` are the only thing controlling
relative importance.
"""

from __future__ import annotations

import math

from local_ai_check.discovery.models import EvaluatedCandidate, ModelCandidate
from local_ai_check.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimationConfidence,
)
from local_ai_check.recommendation import policies
from local_ai_check.workflow import policies as workflow_policies
from local_ai_check.workflow.models import UseCase, UserRequirements

# Keywords that suggest a repository targets a given use case. Matched against
# the repo id and the tag list, both lowercased.
_USE_CASE_KEYWORDS: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("instruct", "chat", "assistant", "it", "conversational"),
    UseCase.CODING: ("coder", "code", "programming", "starcoder", "codellama", "dev"),
    UseCase.DOCUMENT_QA: ("instruct", "long-context", "longcontext", "qa", "rag", "chat"),
    UseCase.WRITING_TRANSLATION: (
        "instruct",
        "translation",
        "translate",
        "multilingual",
        "writing",
        "nllb",
        "opus-mt",
    ),
}

# Keywords that count double because they are the defining signal for the case.
_STRONG_KEYWORDS: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("instruct", "chat"),
    UseCase.CODING: ("coder", "code"),
    UseCase.DOCUMENT_QA: ("long-context", "longcontext", "rag"),
    UseCase.WRITING_TRANSLATION: ("translation", "multilingual"),
}

# A base-instruct model is a poor fit for a coding question and vice versa;
# these mark repositories that clearly target a *different* case.
_NEGATIVE_KEYWORDS: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("coder", "starcoder"),
    UseCase.CODING: (),
    UseCase.DOCUMENT_QA: ("coder",),
    UseCase.WRITING_TRANSLATION: ("coder",),
}


def task_match(candidate: ModelCandidate, requirements: UserRequirements) -> float:
    """How well the repository's advertised purpose matches the use case."""
    haystack = " ".join(
        [candidate.repo_id.lower(), *(tag.lower() for tag in candidate.tags)]
    )

    keywords = _USE_CASE_KEYWORDS[requirements.use_case]
    strong = _STRONG_KEYWORDS[requirements.use_case]
    negative = _NEGATIVE_KEYWORDS[requirements.use_case]

    hits = sum(1 for word in keywords if word in haystack)
    strong_hits = sum(1 for word in strong if word in haystack)

    # Saturating: three matching keywords is already a confident signal.
    score = (
        0.25 if hits == 0 else min(1.0, 0.45 + 0.18 * hits + 0.15 * strong_hits)
    )

    if candidate.pipeline_tag == workflow_policies.TEXT_GENERATION_PIPELINE:
        score = min(1.0, score + 0.1)

    if any(word in haystack for word in negative):
        score *= 0.5

    return _clamp(score)


def language_match(
    candidate: ModelCandidate, requirements: UserRequirements
) -> float:
    """Fraction of the requested languages the model card actually declares.

    An undeclared language list is not scored as zero: most cards omit it, so
    that would punish honest metadata gaps harder than a genuine mismatch.
    """
    wanted = [code.lower() for code in requirements.languages]
    if not wanted:
        return 1.0

    declared = {code.lower().split("-")[0] for code in candidate.languages}
    if not declared:
        # Unconfirmed rather than absent. Multilingual tags are weak evidence.
        tags = {tag.lower() for tag in candidate.tags}
        if "multilingual" in tags or declared & set(wanted):
            return 0.6
        return 0.45

    matched = sum(1 for code in wanted if code.split("-")[0] in declared)
    return _clamp(matched / len(wanted))


def hardware_fit(
    assessment: CompatibilityAssessment | None, requirements: UserRequirements
) -> float:
    """Status-driven fit, reduced when concurrency demands more headroom."""
    if assessment is None:
        return policies.STATUS_FIT_SCORE[CompatibilityStatus.UNKNOWN]

    score = policies.STATUS_FIT_SCORE[assessment.status]

    # Concurrency is a headroom signal, never a throughput claim: a model that
    # barely fits for one user is a worse answer for ten.
    if requirements.concurrent_users > 1 and assessment.ratio is not None:
        extra = min(
            workflow_policies.MAX_CONCURRENCY_HEADROOM,
            1.0
            + workflow_policies.CONCURRENCY_HEADROOM_PER_USER
            * (requirements.concurrent_users - 1),
        )
        projected = assessment.ratio * extra
        if projected > 1.0:
            score *= 0.5
        elif projected > 0.9:
            score *= 0.75

    return _clamp(score)


def license_score(
    candidate: ModelCandidate, requirements: UserRequirements
) -> float:
    category = policies.classify_license(candidate.license)
    score = policies.LICENSE_SCORE[category]
    if (
        requirements.commercial_use_required
        and category is policies.LicenseCategory.UNKNOWN
    ):
        # Kept, but clearly worse than a license we can actually vouch for.
        score *= 0.7
    return _clamp(score)


def metadata_quality(
    candidate: ModelCandidate, assessment: CompatibilityAssessment | None
) -> float:
    """Confidence in the numbers, from card completeness and estimate confidence."""
    score = policies.CONFIDENCE_SCORE[candidate.metadata_confidence]
    if assessment is not None:
        score = (score + policies.CONFIDENCE_SCORE[assessment.confidence]) / 2
    # Each recorded gap costs a little, floored so a thin card is never zero.
    score -= 0.05 * len(candidate.penalties)
    return _clamp(score)


def popularity(candidate: ModelCandidate, max_log_downloads: float) -> float:
    """Log-compressed downloads plus a small likes contribution.

    Downloads span six orders of magnitude, so the raw count would swamp every
    other signal. Combined with a 5 % weight this stays a tie-breaker.
    """
    if max_log_downloads <= 0:
        return 0.0
    downloads = math.log1p(max(0, candidate.downloads)) / max_log_downloads
    likes = math.log1p(max(0, candidate.likes)) / max_log_downloads
    combined = (
        downloads * (1 - policies.POPULARITY_LIKES_WEIGHT)
        + likes * policies.POPULARITY_LIKES_WEIGHT
    )
    return _clamp(combined)


def max_log_downloads(candidates: list[ModelCandidate]) -> float:
    """Normalisation base so popularity is relative to this candidate set."""
    if not candidates:
        return 0.0
    return max(math.log1p(max(0, c.downloads)) for c in candidates) or 1.0


def score_candidate(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
    max_log: float,
) -> EvaluatedCandidate:
    """Return a copy with all six sub-scores populated."""
    candidate = evaluated.candidate
    assessment = evaluated.compatibility
    return evaluated.model_copy(
        update={
            "task_match_score": task_match(candidate, requirements),
            "language_match_score": language_match(candidate, requirements),
            "hardware_fit_score": hardware_fit(assessment, requirements),
            "license_score": license_score(candidate, requirements),
            "metadata_quality_score": metadata_quality(candidate, assessment),
            "popularity_score": popularity(candidate, max_log),
        }
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def confidence_of(evaluated: EvaluatedCandidate) -> EstimationConfidence:
    """The confidence a recommendation should advertise."""
    if evaluated.compatibility is not None:
        return evaluated.compatibility.confidence
    return evaluated.candidate.metadata_confidence


__all__ = [
    "confidence_of",
    "hardware_fit",
    "language_match",
    "license_score",
    "max_log_downloads",
    "metadata_quality",
    "popularity",
    "score_candidate",
    "task_match",
]
