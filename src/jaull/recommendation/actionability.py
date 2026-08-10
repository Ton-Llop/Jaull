"""Would this candidate actually run for the user?

A recommendation earns the top slot only when the whole chain — artifact,
runtime, hardware, memory — is at least plausible. This module aggregates the
existing signals (``ArtifactProfile``, ``RuntimeAssessment``,
``CompatibilityAssessment``) into a single small helper the ranker and tier
selector can consult without duplicating the rules.

Deliberately not a new weighted score: memory fit, artifact realism and
capability already contribute to the composite total. Actionability is a
*gate* — it excludes truly unusable candidates from the primary slot and
downgrades the tier of "runs, but only theoretically" picks.
"""

from __future__ import annotations

from enum import StrEnum

from jaull.domain.artifact_profile import ArtifactConfirmation
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.runtime import RuntimeExecutability


class ActionabilityLevel(StrEnum):
    # Runs today: confirmed artifact + executable runtime + hardware fits.
    ACTIONABLE = "actionable"
    # Probably runs: minor gaps (potentially-executable runtime, tight fit).
    LIKELY_ACTIONABLE = "likely_actionable"
    # Requires a leap of faith: theoretical artifact and/or unknown runtime.
    SPECULATIVE = "speculative"
    # Cannot be recommended: unsupported runtime or insufficient memory.
    BLOCKED = "blocked"


def assess_actionability(evaluated: EvaluatedCandidate) -> ActionabilityLevel:
    status = (
        evaluated.compatibility.status
        if evaluated.compatibility is not None
        else CompatibilityStatus.UNKNOWN
    )
    if status is CompatibilityStatus.INSUFFICIENT:
        return ActionabilityLevel.BLOCKED

    runtime = evaluated.runtime_assessment
    artifact = evaluated.artifact_profile

    if runtime is not None and runtime.executability is RuntimeExecutability.UNSUPPORTED:
        return ActionabilityLevel.BLOCKED

    if artifact is not None and artifact.confirmation is ArtifactConfirmation.THEORETICAL:
        # A theoretical artifact never earns "actionable", even with a
        # potentially-executable runtime.
        return ActionabilityLevel.SPECULATIVE

    if runtime is None or runtime.executability is RuntimeExecutability.UNKNOWN:
        # Missing runtime evaluation is treated as speculative: the caller
        # decides whether to still show it.
        return ActionabilityLevel.SPECULATIVE

    if status is CompatibilityStatus.UNKNOWN:
        return ActionabilityLevel.SPECULATIVE

    artifact_confirmed = (
        artifact is not None
        and artifact.confirmation
        in (ArtifactConfirmation.CONFIRMED, ArtifactConfirmation.INFERRED)
    )

    if (
        artifact_confirmed
        and runtime.executability is RuntimeExecutability.EXECUTABLE
        and status
        in (CompatibilityStatus.COMFORTABLE, CompatibilityStatus.COMPATIBLE)
    ):
        return ActionabilityLevel.ACTIONABLE

    if runtime.executability in (
        RuntimeExecutability.EXECUTABLE,
        RuntimeExecutability.POTENTIALLY_EXECUTABLE,
    ):
        return ActionabilityLevel.LIKELY_ACTIONABLE

    return ActionabilityLevel.SPECULATIVE


# Actionability multiplier applied to the composite score. Keeps the same
# 0…1 scale as ``requirement_penalty`` and, like it, stays deliberately gentle
# so a strong but speculative candidate can still surface as an alternative.
_ACTIONABILITY_PENALTY: dict[ActionabilityLevel, float] = {
    ActionabilityLevel.ACTIONABLE: 1.0,
    ActionabilityLevel.LIKELY_ACTIONABLE: 1.0,
    ActionabilityLevel.SPECULATIVE: 0.7,
    ActionabilityLevel.BLOCKED: 0.0,
}


def actionability_penalty(level: ActionabilityLevel) -> float:
    return _ACTIONABILITY_PENALTY[level]


__all__ = [
    "ActionabilityLevel",
    "actionability_penalty",
    "assess_actionability",
]
