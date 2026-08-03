"""Every weight, threshold and table the recommender uses.

Centralised so the ranking is auditable: no scoring rule anywhere else in the
package is allowed to hard-code a number.
"""

from __future__ import annotations

from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence

# Re-export from domain so existing callers keep working without an extra hop.
from jaull.domain.policies import STATUS_FIT_SCORE, STATUS_RANK
from jaull.domain.requirements import RecommendationPriority

# --------------------------------------------------------------------------
# Composite score weights. They sum to 1.0 and are renormalised after the
# per-priority modifiers below, so tweaking one never silently rescales the
# total.
# --------------------------------------------------------------------------
# Fase 5 split hardware_fit into two axes: memory (does it fit at all with one
# user) and concurrency (does it fit with the requested number of concurrent
# sessions). `capability` is new — a family + parameter-count signal so a
# stronger model wins ties even when the hardware fit is identical.
BASE_WEIGHTS: dict[str, float] = {
    "memory_fit": 0.25,
    "concurrency_fit": 0.10,
    "capability": 0.15,
    "task_match": 0.20,
    "language_match": 0.12,
    "license": 0.08,
    "metadata_quality": 0.07,
    "popularity": 0.03,
}

# Multipliers applied to BASE_WEIGHTS before renormalisation.
PRIORITY_MODIFIERS: dict[RecommendationPriority, dict[str, float]] = {
    # Quality: model capability matters more than every last MiB of headroom.
    RecommendationPriority.QUALITY: {
        "task_match": 1.4,
        "capability": 1.4,
        "memory_fit": 0.85,
    },
    RecommendationPriority.BALANCED: {},
    # Speed and memory both reward headroom on memory and on concurrency.
    RecommendationPriority.SPEED: {
        "memory_fit": 1.3,
        "concurrency_fit": 1.3,
        "popularity": 0.8,
    },
    RecommendationPriority.MEMORY: {
        "memory_fit": 1.5,
        "concurrency_fit": 1.3,
        "task_match": 0.85,
    },
}

# Artifact realism scales memory_fit so a real quantised artifact beats a
# theoretical bytes-per-parameter estimate at the same nominal footprint.
ARTIFACT_REALISM_SCORES: dict[str, float] = {
    "real_gguf": 1.0,
    "real_quant_tag": 1.0,
    "native_dtype": 0.75,
    "theoretical": 0.4,
    "unknown": 0.6,
}

# Tags that confirm a real quantised artifact exists in the repository.
REAL_QUANT_TAGS: frozenset[str] = frozenset(
    {
        "bnb-4bit",
        "bnb-8bit",
        "gptq",
        "awq",
        "compressed-tensors",
        "int4",
        "4-bit",
        "8-bit",
    }
)

# STATUS_RANK and STATUS_FIT_SCORE now live in ``jaull.domain.policies`` and
# are re-exported from the top of this module for backwards compatibility.

CONFIDENCE_SCORE: dict[EstimationConfidence, float] = {
    EstimationConfidence.HIGH: 1.0,
    EstimationConfidence.MEDIUM: 0.7,
    EstimationConfidence.LOW: 0.4,
    EstimationConfidence.UNKNOWN: 0.15,
}

# Applied to the whole composite score, on top of the metadata_quality weight.
#
# Without this, a model we could barely measure competes on equal terms with one
# we measured properly: a tiny repository with no license, no declared languages
# and half a config.json scores well on hardware fit precisely *because* so
# little is known about it. Scaling the total keeps an uncertain answer from
# outranking a well-understood one, which is the whole point of an explainable
# recommender.
CONFIDENCE_MULTIPLIER: dict[EstimationConfidence, float] = {
    EstimationConfidence.HIGH: 1.0,
    EstimationConfidence.MEDIUM: 0.92,
    EstimationConfidence.LOW: 0.78,
    EstimationConfidence.UNKNOWN: 0.55,
}

# A model may not be recommended as the primary pick below this status.
WORST_PRIMARY_STATUS = CompatibilityStatus.OFFLOADING_REQUIRED


# --------------------------------------------------------------------------
# Licensing.
#
# This is a conservative lookup, not legal advice: it answers "is this license
# widely understood to permit commercial use?" and defaults to `unknown`
# whenever it is not certain.
# --------------------------------------------------------------------------
# License categorisation lives in ``jaull.domain.licenses`` so discovery can
# reach it without importing recommendation. The names below are re-exported
# for backwards compatibility with callers that still read them from here.
from jaull.domain.licenses import (  # noqa: E402
    CUSTOM_LICENSE_PREFIXES,
    LEGAL_DISCLAIMER,
    LicenseCategory,
    classify_license,
)

LICENSE_SCORE: dict[LicenseCategory, float] = {
    LicenseCategory.COMMERCIAL_ALLOWED: 1.0,
    LicenseCategory.UNKNOWN: 0.45,
    LicenseCategory.COMMERCIAL_RESTRICTED: 0.1,
}


# The configuration ladders now live in ``estimator.policies`` (they are
# used by ``estimator.configuration``). Re-exported here so callers that
# still read them via ``recommendation.policies`` keep working.
from jaull.estimator.policies import (  # noqa: E402
    AGGRESSIVE_QUANTIZATIONS,
    QUANTIZATION_LADDERS,
    THEORETICAL_DTYPES,
    TRANSFORMERS_DTYPE_LADDER,
)

# --------------------------------------------------------------------------
# Popularity. Downloads span many orders of magnitude, so the raw number is
# log-compressed and then normalised across the candidate set. Combined with a
# 5 % weight this keeps popularity a tie-breaker rather than a ranking.
# --------------------------------------------------------------------------
POPULARITY_LIKES_WEIGHT = 0.3

__all__ = [
    "AGGRESSIVE_QUANTIZATIONS",
    "ARTIFACT_REALISM_SCORES",
    "BASE_WEIGHTS",
    "CONFIDENCE_SCORE",
    "CUSTOM_LICENSE_PREFIXES",
    "LEGAL_DISCLAIMER",
    "LICENSE_SCORE",
    "POPULARITY_LIKES_WEIGHT",
    "PRIORITY_MODIFIERS",
    "QUANTIZATION_LADDERS",
    "REAL_QUANT_TAGS",
    "STATUS_FIT_SCORE",
    "STATUS_RANK",
    "THEORETICAL_DTYPES",
    "TRANSFORMERS_DTYPE_LADDER",
    "WORST_PRIMARY_STATUS",
    "LicenseCategory",
    "classify_license",
]
