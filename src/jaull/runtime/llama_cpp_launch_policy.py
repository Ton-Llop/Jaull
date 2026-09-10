"""The llama.cpp launch policy: a generic memory situation -> --n-gpu-layers.

This is the heuristic that used to live, unnamed, inside
``runtime.llama_cpp._pick_layers``. It is a *backend policy*, deliberately
separate from :mod:`jaull.estimator.hardware_fit`: the HardwareFitAnalyzer
answers "what placement is viable on this hardware" in runtime-agnostic terms
(transformer blocks, byte budgets); this module answers "what integer does
llama.cpp's ``--n-gpu-layers`` flag take". The two are not the same unit and are
not expected to agree -- see
docs/qwen2.5-tests/docs/README_qwen2.5_tests_actualizado.md.

The formula is unchanged from the pre-unification heuristic. It must not import
``hardware_fit`` or read ``gpu_transformer_blocks`` (guarded by
tests/test_architecture_dependencies.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.inference import TargetDevice
from jaull.runtime.policies import (
    LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN,
    LLAMA_CPP_HEADROOM_BYTES,
)


@dataclass(frozen=True)
class LlamaCppLayerPlan:
    """The ``--n-gpu-layers`` value and why. ``-1`` means "all layers"."""

    n_gpu_layers: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: EstimationConfidence = EstimationConfidence.MEDIUM


def pick_gpu_layers(estimate: MemoryEstimate) -> LlamaCppLayerPlan:
    """Choose llama.cpp's ``--n-gpu-layers`` from an already-built estimate.

    Verbatim port of ``runtime.llama_cpp._pick_layers`` (the dead ``hardware``
    parameter is dropped). Behaviour is pinned by
    tests/test_llama_cpp_launch_policy.py and tests/test_runtime_llama_cpp.py.
    """

    assessment = estimate.assessment
    device = assessment.effective_device
    status = assessment.status

    if device is TargetDevice.CPU:
        return LlamaCppLayerPlan(
            n_gpu_layers=0,
            reasons=["Effective device is CPU, so no layers are offloaded to GPU."],
            confidence=EstimationConfidence.HIGH,
        )

    if status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }:
        return LlamaCppLayerPlan(
            n_gpu_layers=-1,
            reasons=[
                f"Model fits in VRAM ({status.value}); using --n-gpu-layers -1 (all)."
            ],
            confidence=EstimationConfidence.HIGH,
        )

    weights_bytes = estimate.weights.component.bytes
    block_count = estimate.kv_cache.layers
    if weights_bytes and block_count and block_count > 0:
        bytes_per_layer = weights_bytes // block_count
        vram = assessment.available_vram_bytes or 0
        reserve = estimate.inference_configuration.device_reserve_bytes
        kv_bytes = estimate.kv_cache.component.bytes or 0
        available_for_layers = vram - reserve - LLAMA_CPP_HEADROOM_BYTES - kv_bytes
        if available_for_layers <= 0 or bytes_per_layer <= 0:
            n = 0
        else:
            n = min(block_count, available_for_layers // bytes_per_layer)
        return LlamaCppLayerPlan(
            n_gpu_layers=int(n),
            reasons=[
                f"Offloading required: {n} of {block_count} layers "
                f"({bytes_per_layer / (1024**3):.2f} GiB/layer) fit in VRAM."
            ],
            confidence=EstimationConfidence.MEDIUM,
        )

    return LlamaCppLayerPlan(
        n_gpu_layers=LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN,
        reasons=["Offloading required but layer count is unknown."],
        warnings=[
            "Block count unavailable; using a conservative fallback of "
            f"{LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN} layers. "
            "Increase manually if VRAM allows."
        ],
        confidence=EstimationConfidence.LOW,
    )


__all__ = ["LlamaCppLayerPlan", "pick_gpu_layers"]
