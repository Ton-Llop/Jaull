"""Every weight, threshold and table the recommender uses.

Centralised so the ranking is auditable: no scoring rule anywhere else in the
package is allowed to hard-code a number.
"""

from __future__ import annotations

from enum import StrEnum

from local_ai_check.domain.estimation import CompatibilityStatus, EstimationConfidence
from local_ai_check.domain.inference import WeightPrecision
from local_ai_check.workflow.models import RecommendationPriority

# --------------------------------------------------------------------------
# Composite score weights. They sum to 1.0 and are renormalised after the
# per-priority modifiers below, so tweaking one never silently rescales the
# total.
# --------------------------------------------------------------------------
BASE_WEIGHTS: dict[str, float] = {
    "hardware_fit": 0.35,
    "task_match": 0.25,
    "language_match": 0.15,
    "license": 0.10,
    "metadata_quality": 0.10,
    "popularity": 0.05,
}

# Multipliers applied to BASE_WEIGHTS before renormalisation.
PRIORITY_MODIFIERS: dict[RecommendationPriority, dict[str, float]] = {
    # Quality: what the model is good at matters more than how comfortably it fits.
    RecommendationPriority.QUALITY: {"task_match": 1.4, "hardware_fit": 0.85},
    RecommendationPriority.BALANCED: {},
    # Speed and memory both reward headroom, which hardware_fit already measures.
    RecommendationPriority.SPEED: {"hardware_fit": 1.35, "popularity": 0.8},
    RecommendationPriority.MEMORY: {"hardware_fit": 1.5, "task_match": 0.85},
}

# --------------------------------------------------------------------------
# Compatibility ordering. Lower rank is better. `insufficient` can never be a
# primary recommendation and `unknown` only ever appears as a flagged
# alternative — both rules are enforced in the ranker, not here.
# --------------------------------------------------------------------------
STATUS_RANK: dict[CompatibilityStatus, int] = {
    CompatibilityStatus.COMFORTABLE: 0,
    CompatibilityStatus.COMPATIBLE: 1,
    CompatibilityStatus.TIGHT: 2,
    CompatibilityStatus.OFFLOADING_REQUIRED: 3,
    CompatibilityStatus.UNKNOWN: 4,
    CompatibilityStatus.INSUFFICIENT: 5,
}

# How each status maps onto the hardware-fit sub-score.
STATUS_FIT_SCORE: dict[CompatibilityStatus, float] = {
    CompatibilityStatus.COMFORTABLE: 1.0,
    CompatibilityStatus.COMPATIBLE: 0.85,
    CompatibilityStatus.TIGHT: 0.55,
    CompatibilityStatus.OFFLOADING_REQUIRED: 0.30,
    CompatibilityStatus.UNKNOWN: 0.20,
    CompatibilityStatus.INSUFFICIENT: 0.0,
}

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
class LicenseCategory(StrEnum):
    COMMERCIAL_ALLOWED = "commercial_allowed"
    COMMERCIAL_RESTRICTED = "commercial_restricted"
    UNKNOWN = "unknown"


PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc0-1.0",
        "gemma",
        "isc",
        "mpl-2.0",
        "openrail",
        "unlicense",
        "zlib",
    }
)

# Non-commercial or otherwise restricted for business use.
RESTRICTED_LICENSES: frozenset[str] = frozenset(
    {
        "cc-by-nc-2.0",
        "cc-by-nc-3.0",
        "cc-by-nc-4.0",
        "cc-by-nc-nd-4.0",
        "cc-by-nc-sa-4.0",
        "creativeml-openrail-m",
        "gpl-3.0",
        "agpl-3.0",
    }
)

# Prefixes for bespoke vendor licenses. These frequently *do* allow commercial
# use below a user threshold, which is exactly why they are not "allowed":
# the tool cannot verify the threshold, so it reports and warns instead.
CUSTOM_LICENSE_PREFIXES: tuple[str, ...] = (
    "llama",
    "other",
    "deepseek",
    "qwen",
    "yi-",
    "tongyi",
    "falcon",
)

LICENSE_SCORE: dict[LicenseCategory, float] = {
    LicenseCategory.COMMERCIAL_ALLOWED: 1.0,
    LicenseCategory.UNKNOWN: 0.45,
    LicenseCategory.COMMERCIAL_RESTRICTED: 0.1,
}

LEGAL_DISCLAIMER = (
    "License information is reported from model metadata and is not legal advice; "
    "check the model's license yourself before commercial use."
)


def classify_license(license_value: str | None) -> LicenseCategory:
    """Bucket a declared license string. Unknown is the safe default."""
    if not license_value:
        return LicenseCategory.UNKNOWN
    normalised = license_value.strip().lower()
    if normalised in PERMISSIVE_LICENSES:
        return LicenseCategory.COMMERCIAL_ALLOWED
    if normalised in RESTRICTED_LICENSES:
        return LicenseCategory.COMMERCIAL_RESTRICTED
    if normalised.startswith("cc-by-nc"):
        return LicenseCategory.COMMERCIAL_RESTRICTED
    if normalised.startswith(CUSTOM_LICENSE_PREFIXES):
        return LicenseCategory.UNKNOWN
    return LicenseCategory.UNKNOWN


# --------------------------------------------------------------------------
# Automatic configuration selection.
#
# Ladders are tried in order against the quantizations a repository actually
# publishes; nothing here assumes a given name exists.
# --------------------------------------------------------------------------
QUANTIZATION_LADDERS: dict[RecommendationPriority, tuple[str, ...]] = {
    RecommendationPriority.QUALITY: ("Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M"),
    RecommendationPriority.BALANCED: ("Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M"),
    RecommendationPriority.SPEED: ("Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S"),
    RecommendationPriority.MEMORY: ("Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"),
}

# Below this the quality loss is severe enough that it is only worth offering
# when nothing else fits at all.
AGGRESSIVE_QUANTIZATIONS: frozenset[str] = frozenset({"Q2_K", "Q3_K_S", "IQ1_S", "IQ2_XS"})

# dtype ladder for Transformers repositories. int8/int4 are theoretical: no
# quantized artifact is confirmed to exist, so picking them lowers confidence.
TRANSFORMERS_DTYPE_LADDER: tuple[WeightPrecision, ...] = (
    WeightPrecision.FLOAT16,
    WeightPrecision.INT8,
    WeightPrecision.INT4,
)

THEORETICAL_DTYPES: frozenset[WeightPrecision] = frozenset(
    {WeightPrecision.INT8, WeightPrecision.INT4}
)

# --------------------------------------------------------------------------
# Popularity. Downloads span many orders of magnitude, so the raw number is
# log-compressed and then normalised across the candidate set. Combined with a
# 5 % weight this keeps popularity a tie-breaker rather than a ranking.
# --------------------------------------------------------------------------
POPULARITY_LIKES_WEIGHT = 0.3

__all__ = [
    "AGGRESSIVE_QUANTIZATIONS",
    "BASE_WEIGHTS",
    "CONFIDENCE_SCORE",
    "CUSTOM_LICENSE_PREFIXES",
    "LEGAL_DISCLAIMER",
    "LICENSE_SCORE",
    "POPULARITY_LIKES_WEIGHT",
    "PRIORITY_MODIFIERS",
    "QUANTIZATION_LADDERS",
    "STATUS_FIT_SCORE",
    "STATUS_RANK",
    "THEORETICAL_DTYPES",
    "TRANSFORMERS_DTYPE_LADDER",
    "WORST_PRIMARY_STATUS",
    "LicenseCategory",
    "classify_license",
]
