"""Pure technical support rules; installed packages and binaries are not inputs."""

from __future__ import annotations

from jaull.domain.hardware import (
    AcceleratorType,
    BackendAvailability,
    ComputeBackend,
    HardwareProfile,
)
from jaull.domain.inference import WeightPrecision
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation

QUANTIZED_TRANSFORMERS_BACKENDS = frozenset({ComputeBackend.CUDA, ComputeBackend.HIP})
QUANTIZED_CPU_LIMITATION = (
    "This quantized Transformers configuration is not supported on CPU by "
    "Jaull's current loader; it requires an available CUDA/HIP backend."
)


def available_compute_backends(hardware: HardwareProfile) -> set[ComputeBackend]:
    backends = {ComputeBackend.CPU}
    for accelerator in hardware.accelerators:
        if accelerator.type is AcceleratorType.SOFTWARE:
            continue
        backends.update(
            item.backend
            for item in accelerator.backends
            if item.availability is BackendAvailability.AVAILABLE and not item.software_renderer
        )
    # Old snapshots only have NVML GPUs. Explicit generic probe results take
    # precedence over that legacy fallback, including unavailable/unknown CUDA.
    if hardware.gpus and not hardware.accelerators:
        backends.add(ComputeBackend.CUDA)
    return backends


def quantized_transformers_supported(hardware: HardwareProfile) -> bool:
    return bool(available_compute_backends(hardware) & QUANTIZED_TRANSFORMERS_BACKENDS)


def technical_support_issue(
    runtime: RuntimeRecommendation,
    *,
    backend: ComputeBackend,
    hardware: HardwareProfile,
    quantization: str | None = None,
    precision: WeightPrecision | None = None,
) -> str | None:
    """Return a demonstrated technical blocker, independent of readiness."""
    if backend not in available_compute_backends(hardware):
        return f"The {backend.value} backend is not observed as available in this environment."
    if runtime.runtime is RuntimeName.TRANSFORMERS:
        if backend is ComputeBackend.VULKAN:
            return "Jaull's Transformers loader does not support the Vulkan backend."
        flags = {flag.name: flag.value for flag in runtime.flags}
        quantized = (
            quantization is not None
            or precision in {WeightPrecision.INT4, WeightPrecision.INT8}
            or flags.get("quantization") in {"4bit", "8bit", "int4", "int8"}
        )
        if quantized and backend not in QUANTIZED_TRANSFORMERS_BACKENDS:
            return QUANTIZED_CPU_LIMITATION
        if (
            quantization
            and quantization.upper().startswith(("AWQ", "GPTQ"))
            and backend is not ComputeBackend.CUDA
        ):
            return "Jaull's AWQ/GPTQ loader requires an available CUDA backend."
    return None
