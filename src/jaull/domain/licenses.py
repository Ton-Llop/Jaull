"""License categorisation shared by discovery filtering and recommendation scoring.

Kept in ``domain`` because both layers need to answer the same question — "is
this license widely understood to permit commercial use?" — and neither should
depend on the other. Deliberately conservative: unknown is the safe default.
This is metadata classification, not legal advice.
"""

from __future__ import annotations

from enum import StrEnum


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


__all__ = [
    "CUSTOM_LICENSE_PREFIXES",
    "LEGAL_DISCLAIMER",
    "PERMISSIVE_LICENSES",
    "RESTRICTED_LICENSES",
    "LicenseCategory",
    "classify_license",
]
