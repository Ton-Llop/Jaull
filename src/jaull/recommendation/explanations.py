"""Rule-based reasons and warnings for a recommendation.

No language model is involved and none should be: every sentence here is
derived from a value the pipeline actually computed, so a reason can always be
traced back to a number. Keeping this in one module is what stops explanation
strings from scattering into the Textual screens as ad-hoc conditionals.
"""

from __future__ import annotations

from jaull.domain.artifact_profile import ArtifactConfirmation
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.requirements import UseCase, UserRequirements
from jaull.domain.runtime import RuntimeExecutability
from jaull.recommendation import policies

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

    assessment = evaluated.compatibility
    config = evaluated.selected_configuration
    configuration_reason = _configuration_memory_reason(config, assessment)
    if configuration_reason is not None:
        reasons.append(configuration_reason)

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


def _configuration_memory_reason(
    config: InferenceConfiguration | None,
    assessment: CompatibilityAssessment | None,
) -> str | None:
    """Describe a chosen representation without claiming unproven memory fit."""
    if config is None:
        return None
    if config.quantization:
        label = f"{config.quantization} variant"
    elif config.precision:
        label = f"{config.precision.value} precision"
    else:
        return None

    display_label = label[:1].upper() + label[1:]

    if assessment is None or assessment.status is CompatibilityStatus.UNKNOWN:
        return f"{display_label} selected for evaluation."
    if assessment.status is CompatibilityStatus.COMFORTABLE:
        return f"{display_label} fits comfortably in the detected memory."
    if assessment.status is CompatibilityStatus.COMPATIBLE:
        return f"{display_label} fits in the detected memory."
    if assessment.status is CompatibilityStatus.TIGHT:
        return f"{display_label} fits with limited memory headroom."
    if assessment.status is CompatibilityStatus.OFFLOADING_REQUIRED:
        return f"{display_label} requires an offloaded memory placement."
    return None


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
        device = assessment.effective_device
        has_gpu = assessment.available_vram_bytes is not None
        if assessment.status is CompatibilityStatus.TIGHT:
            warnings.append(_tight_warning(device, has_gpu))
        elif assessment.status is CompatibilityStatus.OFFLOADING_REQUIRED:
            # The configuration ladder already explains the offload when it
            # exhausted itself, and its wording is the more precise of the two
            # (it says which option is being reported, and that a placement
            # estimate is not a verified run). `_dedupe` cannot catch this: the
            # two sentences differ, they just say the same thing.
            if not any(_is_offloading_memory_warning(warning) for warning in warnings):
                warnings.append(_offloading_warning(has_gpu))
        elif assessment.status is CompatibilityStatus.UNKNOWN:
            warnings.append(
                "Memory estimate is incomplete; treat this as a low-confidence suggestion."
            )

        if assessment.confidence in (
            EstimationConfidence.LOW,
            EstimationConfidence.UNKNOWN,
        ):
            warnings.append(_confidence_warning(evaluated))

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

    # Only claim "no pre-quantized artifact" when the artifact analyser
    # actually agrees. An AWQ/GPTQ/bnb repo whose config.json declares a
    # quantization block is a confirmed artifact even when the estimator
    # renders it at an int4/int8 precision — misdescribing that in the
    # warning list was one of the reported regressions.
    if (
        evaluated.analysis is not None
        and evaluated.analysis.classification.primary_type is RepositoryType.TRANSFORMERS
        and evaluated.selected_configuration is not None
        and evaluated.selected_configuration.precision in policies.THEORETICAL_DTYPES
        and (
            evaluated.artifact_profile is None
            or evaluated.artifact_profile.confirmation is ArtifactConfirmation.THEORETICAL
        )
    ):
        warnings.append(
            "The selected precision is a theoretical estimate; no pre-quantized "
            "artifact was found in this repository."
        )

    # Runtime executability — surface hard incompatibilities so nobody spends
    # time trying to load an AWQ artifact on a CPU-only laptop.
    runtime = evaluated.runtime_assessment
    if runtime is not None:
        if runtime.executability is RuntimeExecutability.UNSUPPORTED:
            for reason in runtime.reasons:
                warnings.append(
                    f"Runtime {runtime.runtime.value} is not usable here: {reason}"
                )
        warnings.extend(runtime.warnings)

    estimate = evaluated.memory_estimate
    if estimate is not None and estimate.runtime_recommendation is not None:
        warnings.extend(estimate.runtime_recommendation.warnings)

    return _dedupe(warnings)


def _confidence_warning(evaluated: EvaluatedCandidate) -> str:
    """Name the component that actually capped the confidence.

    The old text blamed missing model-card metadata for every low-confidence
    estimate. That is almost never the whole story and often not true at all:
    the combined confidence is the weakest of the components, and
    ``runtime_overhead`` is tagged ASSUMED unconditionally, so a model with a
    complete card and an exact parameter count still lands on LOW. Blaming the
    card sent readers to fix something that was not the cause.
    """
    estimate = evaluated.memory_estimate
    if estimate is None:
        return (
            "Confidence is low: no memory estimate was produced for this "
            "configuration."
        )

    weakest = [
        name
        for name, component in (
            ("weights", estimate.weights.component),
            ("KV cache", estimate.kv_cache.component),
            ("runtime overhead", estimate.runtime_overhead.component),
        )
        if component.confidence
        in (EstimationConfidence.LOW, EstimationConfidence.UNKNOWN)
    ]
    assumed = [
        name
        for name, component in (
            ("weights", estimate.weights.component),
            ("KV cache", estimate.kv_cache.component),
            ("runtime overhead", estimate.runtime_overhead.component),
        )
        if component.source is EstimateSource.ASSUMED
    ]

    if not weakest:
        return (
            "Confidence is low: the overall estimate is capped by the "
            "compatibility assessment rather than by any single component."
        )
    detail = ", ".join(weakest)
    heuristic = [name for name in weakest if name in assumed]
    if heuristic:
        plural = "s are" if len(heuristic) > 1 else " is"
        return (
            f"Confidence is low because the estimate rests on {detail}. The "
            f"{', '.join(heuristic)} figure{plural} a documented heuristic, "
            "not a measurement."
        )
    return f"Confidence is low because of the {detail} estimate."


def _tight_warning(device: TargetDevice, has_gpu: bool) -> str:
    """Word the "tight fit" caveat around the device we would actually use."""
    if device is TargetDevice.CPU or not has_gpu:
        return (
            "Estimated memory leaves limited free system RAM; other "
            "applications may not fit alongside the model."
        )
    if device is TargetDevice.GPU:
        return (
            "Estimated memory leaves limited free VRAM; other GPU workloads "
            "may not fit alongside the model."
        )
    return (
        "Estimated memory leaves limited free headroom for other workloads."
    )


def _offloading_warning(has_gpu: bool) -> str:
    if not has_gpu:
        # No GPU: talking about VRAM/RAM offloading is a category error.
        return (
            "Model does not fit in system RAM alone; a smaller variant is "
            "recommended for this machine."
        )
    return (
        "Model does not fit in VRAM alone and would need CPU/GPU offloading, "
        "which is significantly slower."
    )


def _is_offloading_memory_warning(value: str) -> bool:
    """Recognize the ladder's memory warning without hiding unrelated warnings.

    A backend warning may mention offloading too, but it must not suppress the
    compatibility warning explaining that the selected placement needs it.
    """
    normalized = value.lower()
    return "fits in vram alone" in normalized and "offloading" in normalized


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
