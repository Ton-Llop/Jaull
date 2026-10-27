"""Hard-requirement checks that penalise a model's score.

The scoring machinery is deliberately continuous — every sub-score returns a
value in ``[0, 1]``. That is the right shape for "how well does this fit?" but
it is a poor shape for "you asked for commercial use and this license forbids
it". This module produces a multiplier that the ranker applies **after** the
weighted sum, so a hard failure can either eliminate a candidate outright
(``penalty = 1.0``) or knock its total down enough to demote it below models
that satisfy every hard requirement (``penalty < 1.0``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.licenses import (
    LicenseCategory,
    classify_license,
)
from jaull.domain.requirements import UserRequirements


class RequirementCheck(BaseModel):
    """One requirement, its outcome and its penalty weight."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    required: bool
    satisfied: bool
    penalty: float = Field(ge=0.0, le=1.0)


class RequirementSatisfaction(BaseModel):
    """The full set of hard checks for one candidate."""

    model_config = ConfigDict(frozen=True)

    checks: list[RequirementCheck] = Field(default_factory=list)

    @property
    def unmet(self) -> list[RequirementCheck]:
        """Every check that came back unsatisfied — hard or soft.

        The ``required`` flag is a display distinction (hard fail vs soft
        miss), not a filter, so a soft miss still contributes its penalty.
        """
        return [c for c in self.checks if not c.satisfied]

    @property
    def penalty_multiplier(self) -> float:
        """Product of ``1 - penalty`` for every unmet check.

        Clamped to ``[0, 1]`` so callers can multiply the total safely.
        """
        multiplier = 1.0
        for check in self.unmet:
            multiplier *= max(0.0, 1.0 - check.penalty)
        return max(0.0, min(1.0, multiplier))

    @property
    def unmet_labels(self) -> list[str]:
        return [c.description for c in self.unmet]


def evaluate_requirements(
    evaluated: EvaluatedCandidate, requirements: UserRequirements
) -> RequirementSatisfaction:
    checks: list[RequirementCheck] = []

    # Commercial use ---------------------------------------------------------
    if requirements.commercial_use_required:
        category = classify_license(evaluated.candidate.license)
        if category is LicenseCategory.COMMERCIAL_RESTRICTED:
            checks.append(
                RequirementCheck(
                    name="commercial_use",
                    description=(
                        f"License {evaluated.candidate.license!r} restricts commercial use."
                    ),
                    required=True,
                    satisfied=False,
                    penalty=1.0,
                )
            )
        elif category is LicenseCategory.UNKNOWN:
            checks.append(
                RequirementCheck(
                    name="commercial_use",
                    description=(
                        "License terms could not be confirmed as permitting commercial use."
                    ),
                    required=True,
                    satisfied=False,
                    penalty=0.4,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="commercial_use",
                    description="License is generally suitable for commercial use.",
                    required=True,
                    satisfied=True,
                    penalty=0.0,
                )
            )

    # Language coverage ------------------------------------------------------
    for language in requirements.languages:
        support = _language_support(language, evaluated)
        label = language.upper()
        if support is not _LanguageSupport.OTHER_ONLY:
            # Confirmed, or undeclared: most cards omit the field, so that is unknown, which is
            # shown as a notice and never costs anything.
            description, penalty = f"Support for {label} not confirmed in metadata.", 0.0
        else:
            description = (
                f"Support for {label} not confirmed: the card declares only other languages."
            )
            penalty = 0.15
        checks.append(
            RequirementCheck(
                name=f"language:{language}",
                description=description,
                required=False,
                satisfied=support is _LanguageSupport.CONFIRMED,
                penalty=penalty,
            )
        )

    # Concurrency fit --------------------------------------------------------
    if requirements.concurrent_users > 1 and evaluated.concurrency_fit_score <= 0.0:
        checks.append(
            RequirementCheck(
                name="concurrency",
                description=(
                    f"Cannot serve {requirements.concurrency_range.lower()} without "
                    "exceeding available memory."
                ),
                required=False,
                satisfied=False,
                penalty=0.35,
            )
        )

    return RequirementSatisfaction(checks=checks)


class _LanguageSupport(StrEnum):
    CONFIRMED = "confirmed"
    UNDECLARED = "undeclared"
    OTHER_ONLY = "other_only"


def _language_support(language: str, evaluated: EvaluatedCandidate) -> _LanguageSupport:
    wanted = language.lower().split("-")[0]
    declared = {code.lower().split("-")[0] for code in evaluated.candidate.languages}
    tags = {tag.lower() for tag in evaluated.candidate.tags}
    if wanted in declared or "multilingual" in tags:
        return _LanguageSupport.CONFIRMED
    return _LanguageSupport.OTHER_ONLY if declared else _LanguageSupport.UNDECLARED


__all__ = [
    "RequirementCheck",
    "RequirementSatisfaction",
    "evaluate_requirements",
]
