"""The Transformers launch policy: a generic memory situation -> device_map.

Sibling of :mod:`jaull.runtime.llama_cpp_launch_policy`. Verbatim port of
``runtime.transformers._pick_device_map``. Must not import ``hardware_fit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaull.domain.estimation import CompatibilityStatus, MemoryEstimate
from jaull.domain.inference import TargetDevice


@dataclass(frozen=True)
class TransformersDeviceMapPlan:
    device_map: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pick_device_map(estimate: MemoryEstimate) -> TransformersDeviceMapPlan:
    assessment = estimate.assessment
    device = assessment.effective_device
    status = assessment.status

    if device is TargetDevice.CPU:
        return TransformersDeviceMapPlan(
            device_map="cpu",
            reasons=["Effective device is CPU; loading the whole model into RAM."],
        )
    if status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }:
        return TransformersDeviceMapPlan(
            device_map="cuda",
            reasons=[
                f"Model fits in VRAM ({status.value}); placing everything on the GPU."
            ],
        )
    return TransformersDeviceMapPlan(
        device_map="auto",
        reasons=["Model does not fit fully in VRAM; letting Accelerate split layers."],
        warnings=[
            "device_map='auto' requires the `accelerate` package. Actual placement "
            "depends on Accelerate's heuristics and available disk swap."
        ],
    )


__all__ = ["TransformersDeviceMapPlan", "pick_device_map"]
