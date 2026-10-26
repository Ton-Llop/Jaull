"""Pick an inference configuration automatically.

In guided mode the user never chooses a quantization or a dtype. This module
walks the priority-specific ladder from
:mod:`jaull.estimator.policies`, estimates each rung with the
existing estimator, and keeps the first one that actually fits — recording the
rungs it rejected so the report can show its work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jaull.domain.artifact_profile import packed_transformers_quantization_label
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.gguf_variants import preferred_gguf_variants
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelAnalysis
from jaull.domain.requirements import RecommendationPriority, UserRequirements
from jaull.domain.runtime_support import (
    QUANTIZED_CPU_LIMITATION,
    available_compute_backends,
    quantized_transformers_supported,
)
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
    *,
    hardware: HardwareProfile | None = None,
) -> ConfigurationChoice:
    """Choose the best configuration this repository can offer for these needs."""
    if analysis.classification.primary_type is RepositoryType.GGUF:
        return _select_gguf(analysis, requirements, estimate_fn, hardware=hardware)
    return _select_transformers(analysis, requirements, estimate_fn, hardware=hardware)


# --------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------
def _select_gguf(
    analysis: ModelAnalysis,
    requirements: UserRequirements,
    estimate_fn: EstimateFn,
    *,
    hardware: HardwareProfile | None,
) -> ConfigurationChoice:
    available = {
        variant.quantization.upper(): variant.quantization
        for variant in preferred_gguf_variants(analysis.classification.gguf_variants)
    }
    if not available:
        return ConfigurationChoice(
            configuration=None,
            estimate=None,
            reason="No unambiguous complete GGUF representation could be selected.",
            warnings=["GGUF representations are missing, incomplete or ambiguous."],
        )

    ladder = policies.QUANTIZATION_LADDERS[requirements.priority]
    # Anything the repo publishes that is not on the ladder is still usable as a
    # last resort, ordered so the least aggressive comes first.
    extras = sorted(
        set(available) - set(ladder),
        key=lambda rung: (rung in policies.AGGRESSIVE_QUANTIZATIONS, rung),
    )
    ordered = [rung for rung in ladder if rung in available] + extras

    considered: list[str] = []
    warnings: list[str] = []
    seen: list[CompatibilityStatus] = []
    fallback_options: list[
        tuple[CompatibilityStatus, InferenceConfiguration, MemoryEstimate]
    ] = []
    tight_fallback: tuple[str, InferenceConfiguration, MemoryEstimate] | None = None

    for rung in ordered:
        quantization = available[rung]
        config = _build_config(requirements, quantization=quantization)
        if hardware is not None and available_compute_backends(hardware) == {ComputeBackend.CPU}:
            config = config.model_copy(update={"target_device": TargetDevice.CPU})
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{quantization}: {status.value}")
        seen.append(status)

        fallback_options.append((status, config, estimate))

        if status in _ACCEPTABLE:
            if (
                status is CompatibilityStatus.TIGHT
                and requirements.priority is RecommendationPriority.BALANCED
                and rung not in policies.AGGRESSIVE_QUANTIZATIONS
            ):
                # Balanced trades a little fidelity for headroom: remember the tight
                # rung and only use it if no later reasonable rung fits comfortably.
                tight_fallback = tight_fallback or (rung, config, estimate)
                continue
            if tight_fallback is not None and status is CompatibilityStatus.TIGHT:
                continue
            return _gguf_choice(
                rung, config, estimate, requirements, considered, warnings,
            )

    if tight_fallback is not None:
        rung, config, estimate = tight_fallback
        warnings.append(
            f"{available[rung]} is a tight fit; no reasonable smaller variant fits "
            "with more headroom."
        )
        return _gguf_choice(rung, config, estimate, requirements, considered, warnings)

    message = _exhausted_ladder_message(
        seen,
        _fallback_status(fallback_options),
        unit="GGUF variant",
        aggressive=", even the most aggressive variant",
    )
    warnings.append(message)
    return ConfigurationChoice(
        configuration=(
            _fallback_option(fallback_options)[1]
            if fallback_options
            else None
        ),
        estimate=(
            _fallback_option(fallback_options)[2]
            if fallback_options
            else None
        ),
        reason=f"{message} Reporting the closest option.",
        considered=considered,
        warnings=warnings,
    )


def _gguf_choice(
    rung: str,
    config: InferenceConfiguration,
    estimate: MemoryEstimate,
    requirements: UserRequirements,
    considered: list[str],
    warnings: list[str],
) -> ConfigurationChoice:
    quantization = config.quantization
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


# --------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------
def _select_transformers(
    analysis: ModelAnalysis,
    requirements: UserRequirements,
    estimate_fn: EstimateFn,
    *,
    hardware: HardwareProfile | None,
) -> ConfigurationChoice:
    considered: list[str] = []
    warnings: list[str] = []
    fallback_options: list[
        tuple[CompatibilityStatus, InferenceConfiguration, MemoryEstimate]
    ] = []
    seen: list[CompatibilityStatus] = []

    packed = packed_transformers_quantization_label(analysis.config) is not None
    for precision in transformers_precision_options(analysis):
        if (
            hardware is not None and precision in policies.THEORETICAL_DTYPES
            and not quantized_transformers_supported(hardware)
        ):
            considered.append(f"{precision.value}: unsupported configuration")
            if QUANTIZED_CPU_LIMITATION not in warnings:
                warnings.append(QUANTIZED_CPU_LIMITATION)
            continue
        config = _build_config(requirements, precision=precision)
        if hardware is not None and not quantized_transformers_supported(hardware):
            config = config.model_copy(update={"target_device": TargetDevice.CPU})
        estimate = estimate_fn(analysis, config)
        status = estimate.assessment.status
        considered.append(f"{precision.value}: {status.value}")
        seen.append(status)

        fallback_options.append((status, config, estimate))

        if packed and hardware is not None and not quantized_transformers_supported(hardware):
            return ConfigurationChoice(
                configuration=config, estimate=estimate, reason=QUANTIZED_CPU_LIMITATION,
                considered=considered, warnings=[QUANTIZED_CPU_LIMITATION],
            )

        if status in _ACCEPTABLE:
            if precision in policies.THEORETICAL_DTYPES:
                warnings.append(
                    f"{precision.value} is a theoretical estimate: this repository "
                    "ships no confirmed quantized artifact, so loading it at that "
                    "precision requires a runtime that quantizes on the fly."
                )
                estimate = lower_theoretical_confidence(estimate)
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
        _fallback_status(fallback_options),
        unit=("technically supported precision" if hardware is not None
              and not quantized_transformers_supported(hardware) else "precision"),
        aggressive=("" if hardware is not None and not quantized_transformers_supported(hardware)
                    else ", including int4"),
    )
    warnings.append(message)
    return ConfigurationChoice(
        configuration=(
            _fallback_option(fallback_options)[1]
            if fallback_options
            else None
        ),
        estimate=(
            _fallback_option(fallback_options)[2]
            if fallback_options
            else None
        ),
        reason=message,
        considered=considered,
        warnings=warnings,
    )


def transformers_precision_options(analysis: ModelAnalysis) -> tuple[WeightPrecision, ...]:
    """A bounded native-dtype plus existing quantization ladder."""
    declared = analysis.config.torch_dtype if analysis.config is not None else None
    native = {
        "float32": WeightPrecision.FLOAT32,
        "float16": WeightPrecision.FLOAT16,
        "bfloat16": WeightPrecision.BFLOAT16,
    }.get((declared or "").removeprefix("torch."))
    ladder = tuple(dict.fromkeys([
        *([native] if native else []), *policies.TRANSFORMERS_DTYPE_LADDER,
    ]))
    return ladder[:1] if packed_transformers_quantization_label(analysis.config) else ladder


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
      model with multi-head latent attention. Unknown means "cannot
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


def _fallback_option(
    options: list[tuple[CompatibilityStatus, InferenceConfiguration, MemoryEstimate]],
) -> tuple[CompatibilityStatus, InferenceConfiguration, MemoryEstimate]:
    """Choose the most actionable result after the ladder is exhausted."""
    priority = (
        CompatibilityStatus.OFFLOADING_REQUIRED,
        CompatibilityStatus.UNKNOWN,
        CompatibilityStatus.INSUFFICIENT,
    )
    for status in priority:
        for option in options:
            if option[0] is status:
                return option
    return options[0]


def _fallback_status(
    options: list[tuple[CompatibilityStatus, InferenceConfiguration, MemoryEstimate]],
) -> CompatibilityStatus | None:
    return _fallback_option(options)[0] if options else None


def _build_config(
    requirements: UserRequirements,
    *,
    quantization: str | None = None,
    precision: WeightPrecision | None = None,
) -> InferenceConfiguration:
    return InferenceConfiguration(
        context_length=requirements.desired_context,
        # batch_size stays 1: that is sequences per batch. concurrency is
        # a separate axis and multiplies the KV cache directly in the estimate.
        batch_size=1,
        target_device=TargetDevice.AUTO,
        precision=precision,
        quantization=quantization,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        concurrent_users=requirements.concurrent_users,
    )


def lower_theoretical_confidence(estimate: MemoryEstimate) -> MemoryEstimate:
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
