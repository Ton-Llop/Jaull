"""Pick an inference configuration automatically.

In guided mode the user never chooses a quantization or a dtype. This module
walks the priority-specific ladder from
:mod:`jaull.recommendation.policies`, estimates each rung with the
existing estimator, and keeps the first one that actually fits — recording the
rungs it rejected so the report can show its work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import UserRequirements
from jaull.estimator import policies
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)

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
    seen: list[CompatibilityStatus] = []
    best_effort: tuple[InferenceConfiguration, MemoryEstimate] | None = None

    for rung in ordered:
        quantization = available[rung]
        config = _build_config(requirements, quantization=quantization)
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{quantization}: {status.value}")
        seen.append(status)

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

    message = _exhausted_ladder_message(
        seen,
        best_effort[1].assessment.status if best_effort else None,
        unit="GGUF variant",
        aggressive=", even the most aggressive variant",
    )
    warnings.append(message)
    return ConfigurationChoice(
        configuration=best_effort[0] if best_effort else None,
        estimate=best_effort[1] if best_effort else None,
        reason=f"{message} Reporting the closest option.",
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
    seen: list[CompatibilityStatus] = []

    for precision in policies.TRANSFORMERS_DTYPE_LADDER:
        config = _build_config(requirements, precision=precision)
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{precision.value}: {status.value}")
        seen.append(status)

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

    message = _exhausted_ladder_message(
        seen,
        best_effort[1].assessment.status if best_effort else None,
        unit="precision",
        aggressive=", including int4",
    )
    warnings.append(message)
    return ConfigurationChoice(
        configuration=best_effort[0] if best_effort else None,
        estimate=best_effort[1] if best_effort else None,
        reason=message,
        considered=considered,
        warnings=warnings,
    )


def _exhausted_ladder_message(
    seen: list[CompatibilityStatus],
    chosen: CompatibilityStatus | None,
    *,
    unit: str,
    aggressive: str,
) -> str:
    """Say what the ladder found, for the option it is actually reporting.

    The ladder stops only on a *resident* fit, so falling off the end covers
    three different outcomes and they are not interchangeable:

    * the reported option would run with offloading - it fits, just not in VRAM
      alone;
    * nothing could be confirmed - an unsupported architecture, for example a
      MoE whose per-expert KV footprint is not modelled. Unknown means "cannot
      confirm", never "shown not to fit";
    * everything was measured as too large.

    ``chosen`` is the status of the configuration this call returns, which is
    the *first* rung tried, not the best one seen. Describing the set instead of
    the returned option claimed offloading viability for a configuration the
    estimator had marked INSUFFICIENT.
    """
    others = [status for status in seen if status is not chosen]

    if chosen is CompatibilityStatus.OFFLOADING_REQUIRED:
        return (
            f"No {unit} fits in VRAM alone. The reported option is estimated to "
            "fit with CPU/GPU offloading, which is significantly slower; that is "
            "a placement estimate, not a verified run."
        )
    if chosen is CompatibilityStatus.UNKNOWN:
        return (
            f"No {unit} could be confirmed to fit: the memory model could not "
            "produce an estimate for the reported option. This is an unconfirmed "
            "result, not a demonstration that it does not fit."
        )

    suffix = ""
    if any(status is CompatibilityStatus.OFFLOADING_REQUIRED for status in others):
        suffix = (
            " Other options on the ladder are estimated to fit with CPU/GPU "
            "offloading."
        )
    elif any(status is CompatibilityStatus.UNKNOWN for status in others):
        suffix = " Other options could not be confirmed either way."

    if chosen is None:
        return f"No {unit} was evaluated for the detected memory.{suffix}"

    # "No X fits" is a claim about the whole ladder, so it may only be made when
    # the whole ladder came back INSUFFICIENT. Saying it while `suffix` reports
    # that other rungs do fit with offloading contradicts itself in successive
    # sentences.
    if all(status is CompatibilityStatus.INSUFFICIENT for status in seen):
        return f"No {unit} fits the detected memory{aggressive}."
    return (
        "The reported configuration is estimated not to fit the detected "
        f"memory.{suffix}"
    )


def _build_config(
    requirements: UserRequirements,
    *,
    quantization: str | None = None,
    precision: WeightPrecision | None = None,
) -> InferenceConfiguration:
    return InferenceConfiguration(
        context_length=requirements.desired_context,
        # batch_size stays 1: that is parallel tokens per session. concurrency is
        # a separate axis and multiplies the KV cache directly in the estimate.
        batch_size=1,
        target_device=TargetDevice.AUTO,
        precision=precision,
        quantization=quantization,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        concurrent_users=requirements.concurrent_users,
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
