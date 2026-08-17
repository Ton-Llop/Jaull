"""The six sub-scores that feed the composite ranking.

Every function returns a value in ``[0.0, 1.0]`` so the weights in
:mod:`jaull.recommendation.policies` are the only thing controlling
relative importance.
"""

from __future__ import annotations

import math

from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.domain.policies import TEXT_GENERATION_PIPELINE
from jaull.domain.requirements import UseCase, UserRequirements
from jaull.recommendation import policies

# Keywords that suggest a repository targets a given use case. Matched against
# the repo id and the tag list, both lowercased.
_USE_CASE_KEYWORDS: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("instruct", "chat", "assistant", "it", "conversational"),
    UseCase.CODING: ("coder", "code", "programming", "starcoder", "codellama", "dev"),
    UseCase.DOCUMENT_QA: ("instruct", "long-context", "longcontext", "qa", "rag", "chat"),
    UseCase.SUMMARIZATION_EXTRACTION: (
        "summarization",
        "summary",
        "extraction",
        "extract",
        "instruct",
    ),
    UseCase.REASONING: ("reasoning", "math", "logic", "problem", "instruct"),
    UseCase.BATCH_PROCESSING: ("fast", "small", "efficient", "instruct"),
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
    UseCase.SUMMARIZATION_EXTRACTION: ("summarization", "extraction"),
    UseCase.REASONING: ("reasoning", "math"),
    UseCase.BATCH_PROCESSING: ("fast", "small", "efficient"),
    UseCase.WRITING_TRANSLATION: ("translation", "multilingual"),
}

# A base-instruct model is a poor fit for a coding question and vice versa;
# these mark repositories that clearly target a *different* case.
_NEGATIVE_KEYWORDS: dict[UseCase, tuple[str, ...]] = {
    UseCase.GENERAL_CHAT: ("coder", "starcoder"),
    UseCase.CODING: (),
    UseCase.DOCUMENT_QA: ("coder",),
    UseCase.SUMMARIZATION_EXTRACTION: ("coder",),
    UseCase.REASONING: (),
    UseCase.BATCH_PROCESSING: (),
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

    if candidate.pipeline_tag == TEXT_GENERATION_PIPELINE:
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


def memory_fit(assessment: CompatibilityAssessment | None) -> float:
    """Pure "does one user fit on this hardware?" signal.

    The estimator has already sized the memory for the requested number of
    concurrent sessions (Fase 5), so we do not fold concurrency in again here.
    """
    if assessment is None:
        return policies.STATUS_FIT_SCORE[CompatibilityStatus.UNKNOWN]
    return policies.STATUS_FIT_SCORE[assessment.status]


def concurrency_fit(
    assessment: CompatibilityAssessment | None, requirements: UserRequirements
) -> float:
    """Does the estimate still fit at the requested concurrency?

    The KV cache has already been multiplied by ``concurrent_users`` in the
    estimate, so ``assessment.ratio`` is the projected utilisation. This score
    is deliberately harsher than memory_fit above the 1.0 ratio: at that point
    the machine cannot serve the requested workload without spilling.
    """
    if requirements.concurrent_users <= 1:
        return 1.0
    if assessment is None or assessment.ratio is None:
        return policies.STATUS_FIT_SCORE[CompatibilityStatus.UNKNOWN]

    ratio = assessment.ratio
    if ratio <= 0.75:
        return 1.0
    if ratio <= 0.9:
        return 0.8
    if ratio <= 1.0:
        return 0.5
    if ratio <= 1.5:
        return 0.25
    return 0.0


def artifact_realism(evaluated: EvaluatedCandidate) -> float:
    """Down-weight recommendations that rely on theoretical quantisation.

    A real GGUF artifact or a first-class quant tag (bnb-4bit, gptq, awq, ...)
    scores full. A Transformers repo loaded at its native dtype is worth 0.75.
    An int8/int4 selection with no confirmed artifact drops to 0.4 — the
    theoretical footprint is still shown, but it stops competing with
    repositories where the same size actually exists on disk.
    """
    config = evaluated.selected_configuration
    tags = {tag.lower() for tag in evaluated.candidate.tags}
    if config is None:
        return policies.ARTIFACT_REALISM_SCORES["unknown"]
    if config.quantization is not None:
        return policies.ARTIFACT_REALISM_SCORES["real_gguf"]
    if tags & policies.REAL_QUANT_TAGS:
        return policies.ARTIFACT_REALISM_SCORES["real_quant_tag"]
    if config.precision in policies.THEORETICAL_DTYPES:
        return policies.ARTIFACT_REALISM_SCORES["theoretical"]
    return policies.ARTIFACT_REALISM_SCORES["native_dtype"]


def hardware_fit(
    assessment: CompatibilityAssessment | None, requirements: UserRequirements
) -> float:
    """Legacy single-axis fit: kept for consumers written before the split."""
    return min(memory_fit(assessment), concurrency_fit(assessment, requirements))


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
    capability_signal: float | None = None,
) -> EvaluatedCandidate:
    """Return a copy with all sub-scores populated.

    ``capability_signal`` is passed in from the caller so the recommendation
    service can source it from ``recommendation.capability`` without this
    module importing an analyser it does not otherwise need.
    """
    candidate = evaluated.candidate
    assessment = evaluated.compatibility
    memory = memory_fit(assessment)
    concurrency = concurrency_fit(assessment, requirements)
    realism = artifact_realism(evaluated)
    # Memory-fit is scaled by realism so a real artifact beats a theoretical
    # bytes-per-parameter estimate of the same nominal footprint.
    scaled_memory = _clamp(memory * realism)
    return evaluated.model_copy(
        update={
            "task_match_score": task_match(candidate, requirements),
            "language_match_score": language_match(candidate, requirements),
            "memory_fit_score": scaled_memory,
            "concurrency_fit_score": concurrency,
            "capability_score": (
                capability_signal if capability_signal is not None else 0.5
            ),
            "artifact_realism_score": realism,
            "hardware_fit_score": min(scaled_memory, concurrency),
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
    "artifact_realism",
    "concurrency_fit",
    "confidence_of",
    "hardware_fit",
    "language_match",
    "license_score",
    "max_log_downloads",
    "memory_fit",
    "metadata_quality",
    "popularity",
    "score_candidate",
    "task_match",
]
