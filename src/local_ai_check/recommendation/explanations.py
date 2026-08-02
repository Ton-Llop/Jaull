"""Rule-based reasons and warnings for a recommendation.

No language model is involved and none should be: every sentence here is
derived from a value the pipeline actually computed, so a reason can always be
traced back to a number. Keeping this in one module is what stops explanation
strings from scattering into the Textual screens as ad-hoc conditionals.
"""

from __future__ import annotations

from local_ai_check.discovery.models import EvaluatedCandidate
from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from local_ai_check.recommendation import policies
from local_ai_check.workflow.models import UseCase, UserRequirements

_USE_CASE_LABEL: dict[UseCase, str] = {
    UseCase.GENERAL_CHAT: "general chat and assistant use",
    UseCase.CODING: "programming tasks",
    UseCase.DOCUMENT_QA: "working with documents",
    UseCase.WRITING_TRANSLATION: "writing and translation",
}

# Sub-score above which a component is worth calling out as a strength.
_STRONG = 0.7
# Below this, the same component becomes a caveat.
_WEAK = 0.4


def build_reasons(
    evaluated: EvaluatedCandidate, requirements: UserRequirements
) -> list[str]:
    """Positive, user-facing reasons this model was picked."""
    reasons: list[str] = []
    candidate = evaluated.candidate

    if evaluated.task_match_score >= _STRONG:
        reasons.append(
            f"Strong match for {_USE_CASE_LABEL[requirements.use_case]}."
        )

    if evaluated.language_match_score >= _STRONG and candidate.languages:
        listed = ", ".join(sorted({lang.upper() for lang in candidate.languages})[:4])
        reasons.append(f"Model metadata lists {listed}.")

    config = evaluated.selected_configuration
    if config is not None:
        if config.quantization:
            reasons.append(
                f"{config.quantization} variant fits in the detected memory."
            )
        elif config.precision:
            reasons.append(
                f"Fits in the detected memory at {config.precision.value} precision."
            )

    assessment = evaluated.compatibility
    if assessment is not None:
        if assessment.status is CompatibilityStatus.COMFORTABLE:
            reasons.append("Leaves comfortable free memory after loading.")
        elif assessment.status is CompatibilityStatus.COMPATIBLE:
            reasons.append("Fits the detected memory with room to spare.")

    category = policies.classify_license(candidate.license)
    if category is policies.LicenseCategory.COMMERCIAL_ALLOWED and candidate.license:
        reasons.append(
            f"{candidate.license} license is generally suitable for commercial use."
        )

    if config is not None:
        reasons.append(f"Suggested context of {config.context_length} tokens.")

    return reasons


def build_warnings(
    evaluated: EvaluatedCandidate, requirements: UserRequirements
) -> list[str]:
    """Limitations the user needs to see before trusting the recommendation."""
    warnings: list[str] = list(evaluated.warnings)
    candidate = evaluated.candidate
    assessment = evaluated.compatibility

    if assessment is None:
        warnings.append(
            "Compatibility could not be determined; not enough metadata to estimate memory."
        )
    else:
        if assessment.status is CompatibilityStatus.TIGHT:
            warnings.append(
                "Estimated memory leaves limited free VRAM; other applications may not fit."
            )
        elif assessment.status is CompatibilityStatus.OFFLOADING_REQUIRED:
            warnings.append(
                "Model does not fit in VRAM alone and would need CPU/GPU offloading, "
                "which is significantly slower."
            )
        elif assessment.status is CompatibilityStatus.UNKNOWN:
            warnings.append(
                "Memory estimate is incomplete; treat this as a low-confidence suggestion."
            )

        if assessment.confidence in (
            EstimationConfidence.LOW,
            EstimationConfidence.UNKNOWN,
        ):
            warnings.append(
                "Confidence is low because part of the model metadata was missing."
            )

    category = policies.classify_license(candidate.license)
    if category is policies.LicenseCategory.UNKNOWN:
        if candidate.license:
            warnings.append(
                f"License {candidate.license!r} is a custom or unrecognised license; "
                "review its terms before commercial use."
            )
        else:
            warnings.append("No license is declared for this repository.")
    elif category is policies.LicenseCategory.COMMERCIAL_RESTRICTED:
        warnings.append(
            f"License {candidate.license!r} generally restricts commercial use."
        )

    if evaluated.language_match_score < _STRONG and requirements.languages:
        wanted = ", ".join(code.upper() for code in requirements.languages)
        warnings.append(
            f"Support for {wanted} is not confirmed in the model metadata."
        )

    if requirements.concurrent_users > 1:
        warnings.append(
            f"Sized for a single session. {requirements.concurrency_range} would need "
            "more memory per concurrent request; this tool does not model throughput."
        )

    if evaluated.metadata_quality_score < _WEAK:
        warnings.append("Model card metadata is sparse, so figures are approximate.")

    if (
        evaluated.analysis is not None
        and evaluated.analysis.classification.primary_type is RepositoryType.TRANSFORMERS
        and evaluated.selected_configuration is not None
        and evaluated.selected_configuration.precision in policies.THEORETICAL_DTYPES
    ):
        warnings.append(
            "The selected precision is a theoretical estimate; no pre-quantized "
            "artifact was found in this repository."
        )

    return _dedupe(warnings)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def no_results_explanation(
    evaluated: list[EvaluatedCandidate],
) -> list[str]:
    """Explain why nothing was compatible, using what the closest candidates needed."""
    if not evaluated:
        return [
            "No candidate models were found for these requirements.",
            "Try a broader use case, or check your network connection.",
        ]

    lines = ["No fully compatible models were found.", "", "Closest candidates require:"]
    needs: list[str] = []

    if any(
        item.compatibility is not None
        and item.compatibility.status is CompatibilityStatus.OFFLOADING_REQUIRED
        for item in evaluated
    ):
        needs.append("- CPU/GPU offloading")
    if any(
        item.compatibility is not None
        and item.compatibility.status is CompatibilityStatus.INSUFFICIENT
        for item in evaluated
    ):
        needs.append("- More RAM or VRAM than this machine reports")
    if any(item.compatibility is None for item in evaluated):
        needs.append("- More complete model metadata to estimate reliably")
    needs.append("- A smaller context length")

    return lines + needs


__all__ = ["build_reasons", "build_warnings", "no_results_explanation"]
