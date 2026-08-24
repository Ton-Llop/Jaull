"""Compare a memory requirement against detected local resources."""

from __future__ import annotations

from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimationConfidence,
    HardwareFitMode,
    HardwareFitPlacementMethod,
    HardwareFitResult,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import TargetDevice
from jaull.estimator import hardware_fit
from jaull.estimator.policies import (
    COMFORTABLE_MAX,
    COMPATIBLE_MAX,
    TIGHT_MAX,
)


def assess(
    total_bytes: int | None,
    hardware: HardwareProfile,
    device_target: TargetDevice,
) -> CompatibilityAssessment:
    vram = _available_vram(hardware)
    ram = hardware.memory.available_bytes

    if total_bytes is None:
        return CompatibilityAssessment(
            status=CompatibilityStatus.UNKNOWN,
            confidence=EstimationConfidence.UNKNOWN,
            target_device=device_target,
            effective_device=device_target,
            available_vram_bytes=vram,
            available_ram_bytes=ram,
            ratio=None,
            reasons=["Total memory could not be estimated."],
        )

    if device_target is TargetDevice.CPU:
        return _assess_single(
            device=TargetDevice.CPU,
            capacity=ram,
            need=total_bytes,
            missing_reason="No RAM information available.",
        )

    if device_target is TargetDevice.GPU:
        if vram is None:
            return CompatibilityAssessment(
                status=CompatibilityStatus.INSUFFICIENT,
                confidence=EstimationConfidence.HIGH,
                target_device=device_target,
                effective_device=TargetDevice.GPU,
                available_vram_bytes=None,
                available_ram_bytes=ram,
                ratio=None,
                reasons=["GPU requested but no NVIDIA GPU detected."],
            )
        return _assess_single(
            device=TargetDevice.GPU,
            capacity=vram,
            need=total_bytes,
            missing_reason="No VRAM information available.",
        )

    # AUTO ---------------------------------------------------------------
    if vram is not None and total_bytes <= vram:
        return _assess_single(
            device=TargetDevice.GPU,
            capacity=vram,
            need=total_bytes,
            missing_reason="No VRAM information available.",
            reason_prefix="Fits in VRAM",
        )

    if vram is not None and total_bytes <= (vram + ram):
        return CompatibilityAssessment(
            status=CompatibilityStatus.OFFLOADING_REQUIRED,
            confidence=EstimationConfidence.LOW,
            target_device=device_target,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=vram,
            available_ram_bytes=ram,
            ratio=total_bytes / (vram + ram),
            reasons=[
                "Model does not fit entirely in available VRAM.",
                "Combined RAM+VRAM would cover it, but per-layer offloading is not modelled.",
            ],
            warnings=["CPU/GPU offloading depends on the runtime and is approximate."],
        )

    if total_bytes <= ram:
        prefix = (
            "No GPU capacity for this model" if vram is not None else "No GPU detected"
        )
        return _assess_single(
            device=TargetDevice.CPU,
            capacity=ram,
            need=total_bytes,
            missing_reason="No RAM information available.",
            reason_prefix=prefix,
        )

    return CompatibilityAssessment(
        status=CompatibilityStatus.INSUFFICIENT,
        confidence=EstimationConfidence.HIGH,
        target_device=device_target,
        effective_device=TargetDevice.CPU,
        available_vram_bytes=vram,
        available_ram_bytes=ram,
        ratio=total_bytes / (ram if ram else 1),
        reasons=[
            "Estimated memory exceeds combined available RAM and VRAM.",
        ],
    )


def _assess_single(
    device: TargetDevice,
    capacity: int | None,
    need: int,
    missing_reason: str,
    reason_prefix: str | None = None,
) -> CompatibilityAssessment:
    if capacity is None or capacity <= 0:
        return CompatibilityAssessment(
            status=CompatibilityStatus.UNKNOWN,
            confidence=EstimationConfidence.UNKNOWN,
            target_device=device,
            effective_device=device,
            available_vram_bytes=None,
            available_ram_bytes=None,
            ratio=None,
            reasons=[missing_reason],
        )

    ratio = need / capacity
    status = _status_from_ratio(ratio)
    reasons: list[str] = []
    if reason_prefix:
        reasons.append(reason_prefix)
    reasons.append(f"Uses {ratio * 100:.1f}% of the available {device.value.upper()}.")

    return CompatibilityAssessment(
        status=status,
        confidence=EstimationConfidence.HIGH,
        target_device=device,
        effective_device=device,
        available_vram_bytes=capacity if device is TargetDevice.GPU else None,
        available_ram_bytes=capacity if device is TargetDevice.CPU else None,
        ratio=ratio,
        reasons=reasons,
    )


def _status_from_ratio(ratio: float) -> CompatibilityStatus:
    if ratio <= COMFORTABLE_MAX:
        return CompatibilityStatus.COMFORTABLE
    if ratio <= COMPATIBLE_MAX:
        return CompatibilityStatus.COMPATIBLE
    if ratio <= TIGHT_MAX:
        return CompatibilityStatus.TIGHT
    return CompatibilityStatus.INSUFFICIENT


def _available_vram(hardware: HardwareProfile) -> int | None:
    if not hardware.gpus:
        return None
    # Use the GPU with most VRAM available (typical for a single-GPU setup).
    return max(gpu.vram_available_bytes for gpu in hardware.gpus)


def assess_components(
    *,
    weights_bytes: int | None,
    kv_cache_bytes: int | None,
    overhead_bytes: int | None,
    device_reserve_bytes: int,
    safety_margin_bytes: int,
    total_bytes: int | None,
    total_layers: int | None,
    hardware: HardwareProfile,
    device_target: TargetDevice,
) -> CompatibilityAssessment:
    """Assess placement using separated memory components when available."""

    if device_target is not TargetDevice.AUTO:
        return assess(total_bytes, hardware, device_target)

    if weights_bytes is None or kv_cache_bytes is None or overhead_bytes is None:
        return assess(total_bytes, hardware, device_target)

    fit = hardware_fit.analyze_components(
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        device_reserve_bytes=device_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
        total_layers=total_layers,
        hardware=hardware,
    )
    return _assessment_from_fit(fit, device_target)


def _assessment_from_fit(
    fit: HardwareFitResult,
    target_device: TargetDevice,
) -> CompatibilityAssessment:
    vram = fit.available_vram_bytes
    ram = fit.available_ram_bytes

    if fit.mode is HardwareFitMode.GPU_RESIDENT:
        assert fit.gpu_required_bytes is not None
        ratio = _ratio(fit.gpu_required_bytes, vram)
        return CompatibilityAssessment(
            status=_status_from_ratio(ratio),
            confidence=EstimationConfidence.HIGH,
            target_device=target_device,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=vram,
            available_ram_bytes=ram,
            ratio=ratio,
            reasons=[fit.reason],
            warnings=fit.warnings,
        )

    if fit.mode is HardwareFitMode.GPU_OFFLOAD:
        offload_ratio = _max_known_ratio(
            (fit.gpu_required_bytes, vram),
            (fit.ram_required_bytes, ram),
        )
        confidence = (
            EstimationConfidence.MEDIUM
            if fit.placement_method is HardwareFitPlacementMethod.LAYERS
            else EstimationConfidence.LOW
        )
        return CompatibilityAssessment(
            status=CompatibilityStatus.OFFLOADING_REQUIRED,
            confidence=confidence,
            target_device=target_device,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=vram,
            available_ram_bytes=ram,
            ratio=offload_ratio,
            reasons=[fit.reason],
            warnings=fit.warnings,
        )

    if fit.mode is HardwareFitMode.CPU_RAM:
        assert fit.ram_required_bytes is not None
        ratio = _ratio(fit.ram_required_bytes, ram)
        return CompatibilityAssessment(
            status=_status_from_ratio(ratio),
            confidence=EstimationConfidence.HIGH,
            target_device=target_device,
            effective_device=TargetDevice.CPU,
            available_vram_bytes=vram,
            available_ram_bytes=ram,
            ratio=ratio,
            reasons=[fit.reason],
            warnings=fit.warnings,
        )

    return CompatibilityAssessment(
        status=CompatibilityStatus.INSUFFICIENT,
        confidence=EstimationConfidence.HIGH,
        target_device=target_device,
        effective_device=TargetDevice.CPU,
        available_vram_bytes=vram,
        available_ram_bytes=ram,
        ratio=None,
        reasons=[fit.reason],
        warnings=fit.warnings,
    )


def _ratio(need: int, capacity: int | None) -> float:
    if capacity is None or capacity <= 0:
        return 1.0
    return need / capacity


def _max_known_ratio(*pairs: tuple[int | None, int | None]) -> float | None:
    ratios = [_ratio(need, capacity) for need, capacity in pairs if need is not None]
    if not ratios:
        return None
    return max(ratios)


__all__ = ["assess", "assess_components"]
