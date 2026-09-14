"""The llama.cpp launch policy: a generic memory situation -> --n-gpu-layers.

This is the heuristic that used to live, unnamed, inside
``runtime.llama_cpp._pick_layers``. It is a *backend policy*, deliberately
separate from :mod:`jaull.estimator.hardware_fit`: the HardwareFitAnalyzer
answers "what placement is viable on this hardware" in runtime-agnostic terms
(transformer blocks, byte budgets); this module answers "what integer does
llama.cpp's ``--n-gpu-layers`` flag take". The two are not the same unit and are
not expected to agree -- see
docs/qwen2.5-tests/docs/README_qwen2.5_tests_actualizado.md.

It must not import ``hardware_fit`` or read ``gpu_transformer_blocks`` (guarded
by tests/test_architecture_dependencies.py). It may read the estimator's weight
decomposition: that says what the artifact is made of, not where any analyzer
decided to put it.

The offload branch costs a block at the price of a block and scales the KV cache
with the requested offload. A config-derived block/non-block decomposition does
not establish tensor placement. Until a backend-specific tensor-placement
contract exists, the complete non-block aggregate is charged as a fixed GPU cost:
conservative, but it avoids turning an architectural estimate into an OOM-prone
runtime assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.inference import TargetDevice
from jaull.runtime.llama_cpp_tensor_policy import (
    VERIFIED_BUILD,
    LlamaCppTensorContext,
    select_tensor_budget,
)
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


def pick_gpu_layers(
    estimate: MemoryEstimate, *, tensor_context: LlamaCppTensorContext | None = None,
) -> LlamaCppLayerPlan:
    """Choose llama.cpp's ``--n-gpu-layers`` from an already-built estimate.

    Optional verified local descriptors refine the backend heuristic. Without
    them the existing aggregate budget remains the fallback.
    """

    if tensor_context is not None and estimate.assessment.effective_device is not TargetDevice.CPU:
        refined = select_tensor_budget(estimate, tensor_context)
        if not isinstance(refined, str):
            index = tensor_context.index
            assert index is not None
            return LlamaCppLayerPlan(
                n_gpu_layers=refined.requested_units,
                reasons=[
                    f"Local GGUF tensor budget; llama.cpp build {VERIFIED_BUILD}: "
                    f"{refined.requested_units} runtime units, {refined.gpu_blocks}/"
                    f"{refined.total_blocks} repeating blocks plus output when enabled.",
                    f"GPU planning bytes: blocks={refined.gpu_block_bytes}, "
                    f"output={refined.gpu_output_bytes}, KV={refined.gpu_kv_bytes}, "
                    f"reserve={refined.reserve_bytes}, headroom={refined.headroom_bytes}; "
                    f"required={refined.required_bytes}, available={refined.available_vram_bytes}. "
                    f"Input tensor budget on host={refined.host_input_bytes}.",
                    f"Artifact: {index.artifact.repo_id}@{index.artifact.revision}/"
                    f"{index.artifact.filename}; sha256={index.artifact.sha256 or 'unrecorded'}. "
                    "Stored tensor bytes and estimated KV are not measured CUDA allocations.",
                ],
            )
        fallback = _pick_aggregate_gpu_layers(estimate)
        fallback.warnings.append(f"Local tensor refinement not applied: {refined}")
        return fallback
    return _pick_aggregate_gpu_layers(estimate)


def _pick_aggregate_gpu_layers(estimate: MemoryEstimate) -> LlamaCppLayerPlan:

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

    split = _weight_split(estimate)
    if split is not None:
        vram = assessment.available_vram_bytes or 0
        reserve = estimate.inference_configuration.device_reserve_bytes
        kv_bytes = estimate.kv_cache.component.bytes or 0
        # The device reserve, llama.cpp's own headroom and the non-block weight
        # aggregate are fixed costs, paid before the first block. The aggregate
        # is deliberately conservative: config-derived components do not prove
        # their backend placement.
        #
        # The KV cache is not fixed: only the offloaded model portion keeps its
        # cache in VRAM, so charging the whole cache before knowing ``n`` would
        # understate the budget for every launch unit.
        budget = (
            vram
            - reserve
            - LLAMA_CPP_HEADROOM_BYTES
            - split.fixed_gpu_weight_bytes
        )
        if budget <= 0 or split.bytes_per_block <= 0:
            n = 0
        else:
            # n·bytes_per_block + kv·n/T ≤ budget, solved for n so the KV share
            # scales with the answer instead of being guessed beforehand.
            n = min(
                split.block_count,
                (split.block_count * budget)
                // (split.block_count * split.bytes_per_block + kv_bytes),
            )

        reasons = [
            f"Offloading required: requesting {n} llama.cpp GPU units from a "
            f"conservative {split.block_count}-transformer-block budget "
            f"({split.bytes_per_block / (1024**3):.2f} GiB/block)."
        ]
        if split.fixed_gpu_weight_bytes > 0:
            reasons.append(
                f"The {split.fixed_gpu_weight_bytes / (1024**3):.2f} GiB non-block "
                "weight aggregate is charged to GPU as a conservative bound; "
                "config-derived embedding/output estimates do not establish "
                "llama.cpp placement."
            )
        return LlamaCppLayerPlan(
            n_gpu_layers=int(n),
            reasons=reasons,
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


@dataclass(frozen=True)
class _WeightSplit:
    """What one launch unit costs, and what is paid before any of them.

    ``fixed_gpu_weight_bytes`` is the non-block aggregate charged before any
    launch unit. It is a conservative planning bound, not a claim that every
    non-block tensor is physically resident on the device.
    """

    block_count: int
    bytes_per_block: int
    fixed_gpu_weight_bytes: int


def _weight_split(estimate: MemoryEstimate) -> _WeightSplit | None:
    """Turn the estimator's weight decomposition into launch-unit costs.

    The estimator separates transformer-block from non-block artifact bytes.
    Optional embedding/output estimates remain projections of parameter counts,
    so this policy intentionally consumes only the aggregate until it has
    backend-specific tensor-placement evidence for the exact artifact.

    The block count comes from the KV estimate rather than the decomposition,
    and the published per-block figure is rounded up — both err towards charging
    slightly too much, the safe direction here.
    """

    block_count = estimate.kv_cache.layers
    if not block_count or block_count <= 0:
        return None

    decomposition = estimate.weights.transformer_block_decomposition
    if decomposition is not None:
        return _WeightSplit(
            block_count=block_count,
            bytes_per_block=decomposition.estimated_bytes_per_transformer_block,
            fixed_gpu_weight_bytes=decomposition.estimated_non_block_weight_bytes,
        )

    weights_bytes = estimate.weights.component.bytes
    if weights_bytes:
        # No decomposition at all: every byte is charged uniformly to a block,
        # exactly as the pre-decomposition heuristic did.
        return _WeightSplit(
            block_count=block_count,
            bytes_per_block=weights_bytes // block_count,
            fixed_gpu_weight_bytes=0,
        )
    return None
__all__ = ["LlamaCppLayerPlan", "pick_gpu_layers"]
