"""Pick an inference configuration automatically.

In guided mode the user never chooses a quantization or a dtype. This module
walks the priority-specific ladder from
:mod:`local_ai_check.recommendation.policies`, estimates each rung with the
existing estimator, and keeps the first one that actually fits — recording the
rungs it rejected so the report can show its work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from local_ai_check.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from local_ai_check.domain.model import ModelAnalysis
from local_ai_check.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from local_ai_check.recommendation import policies
from local_ai_check.workflow.models import UserRequirements

# Statuses we are willing to stop the ladder on. Anything worse means "keep
# looking for a smaller rung".
_ACCEPTABLE = (
    CompatibilityStatus.COMFORTABLE,
    CompatibilityStatus.COMPATIBLE,
    CompatibilityStatus.TIGHT,
)

EstimateFn = Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]


@dataclass
class ConfigurationChoice:
    """The configuration we settled on, plus everything we tried to get there."""

    configuration: InferenceConfiguration | None
    estimate: MemoryEstimate | None
    reason: str
    considered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def select_configuration(
    analysis: ModelAnalysis,
    requirements: UserRequirements,
    estimate_fn: EstimateFn,
) -> ConfigurationChoice:
    """Choose the best configuration this repository can offer for these needs."""
    if analysis.classification.primary_type is RepositoryType.GGUF:
        return _select_gguf(analysis, requirements, estimate_fn)
    return _select_transformers(analysis, requirements, estimate_fn)


# --------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------
def _select_gguf(
    analysis: ModelAnalysis,
    requirements: UserRequirements,
    estimate_fn: EstimateFn,
) -> ConfigurationChoice:
    available = {
        variant.quantization.upper(): variant.quantization
        for variant in analysis.classification.gguf_variants
    }
    if not available:
        return ConfigurationChoice(
            configuration=None,
            estimate=None,
            reason="Repository is GGUF but publishes no recognisable quantization.",
            warnings=["No GGUF variants could be identified in this repository."],
        )

    ladder = policies.QUANTIZATION_LADDERS[requirements.priority]
    # Anything the repo publishes that is not on the ladder is still usable as a
    # last resort, ordered so the least aggressive comes first.
    extras = sorted(set(available) - set(ladder))
    ordered = [rung for rung in ladder if rung in available] + extras

    considered: list[str] = []
    warnings: list[str] = []
    best_effort: tuple[InferenceConfiguration, MemoryEstimate] | None = None

    for rung in ordered:
        quantization = available[rung]
        config = _build_config(requirements, quantization=quantization)
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{quantization}: {status.value}")

        if best_effort is None:
            best_effort = (config, estimate)

        if status in _ACCEPTABLE:
            if rung in policies.AGGRESSIVE_QUANTIZATIONS:
                warnings.append(
                    f"{quantization} is an aggressive quantization; it was chosen "
                    "because nothing larger fits, and quality loss may be noticeable."
                )
            return ConfigurationChoice(
                configuration=config,
                estimate=estimate,
                reason=(
                    f"{quantization} is the first variant on the "
                    f"{requirements.priority.value} ladder that fits the detected memory."
                ),
                considered=considered,
                warnings=warnings,
            )

    warnings.append("No GGUF variant fits comfortably in the detected memory.")
    return ConfigurationChoice(
        configuration=best_effort[0] if best_effort else None,
        estimate=best_effort[1] if best_effort else None,
        reason="No variant fits; reporting the closest option.",
        considered=considered,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------
def _select_transformers(
    analysis: ModelAnalysis,
    requirements: UserRequirements,
    estimate_fn: EstimateFn,
) -> ConfigurationChoice:
    considered: list[str] = []
    warnings: list[str] = []
    best_effort: tuple[InferenceConfiguration, MemoryEstimate] | None = None

    for precision in policies.TRANSFORMERS_DTYPE_LADDER:
        config = _build_config(requirements, precision=precision)
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{precision.value}: {status.value}")

        if best_effort is None:
            best_effort = (config, estimate)

        if status in _ACCEPTABLE:
            if precision in policies.THEORETICAL_DTYPES:
                warnings.append(
                    f"{precision.value} is a theoretical estimate: this repository "
                    "ships no confirmed quantized artifact, so loading it at that "
                    "precision requires a runtime that quantizes on the fly."
                )
                estimate = _lower_confidence(estimate)
            return ConfigurationChoice(
                configuration=config,
                estimate=estimate,
                reason=(
                    f"{precision.value} is the highest precision that fits the "
                    "detected memory."
                ),
                considered=considered,
                warnings=warnings,
            )

    warnings.append("No precision fits the detected memory, including int4.")
    return ConfigurationChoice(
        configuration=best_effort[0] if best_effort else None,
        estimate=best_effort[1] if best_effort else None,
        reason="No precision fits; reporting the closest option.",
        considered=considered,
        warnings=warnings,
    )


def _build_config(
    requirements: UserRequirements,
    *,
    quantization: str | None = None,
    precision: WeightPrecision | None = None,
) -> InferenceConfiguration:
    return InferenceConfiguration(
        context_length=requirements.desired_context,
        # batch_size stays 1: concurrency is handled as a headroom signal in
        # scoring, not by pretending one user equals one batch slot.
        batch_size=1,
        target_device=TargetDevice.AUTO,
        precision=precision,
        quantization=quantization,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
    )


def _lower_confidence(estimate: MemoryEstimate) -> MemoryEstimate:
    """Drop an estimate's confidence by one rung, floor at LOW."""
    order = (
        EstimationConfidence.HIGH,
        EstimationConfidence.MEDIUM,
        EstimationConfidence.LOW,
    )
    current = estimate.assessment.confidence
    if current not in order:
        return estimate
    index = min(order.index(current) + 1, len(order) - 1)
    assessment = estimate.assessment.model_copy(update={"confidence": order[index]})
    return estimate.model_copy(update={"assessment": assessment})


__all__ = ["ConfigurationChoice", "EstimateFn", "select_configuration"]
