"""llama.cpp recommendation builder."""

from __future__ import annotations

from pathlib import PurePosixPath

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.runtime.policies import (
    LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN,
    LLAMA_CPP_HEADROOM_BYTES,
)


def build(estimate: MemoryEstimate, hardware: HardwareProfile) -> RuntimeRecommendation:
    cfg = estimate.inference_configuration
    variant = estimate.weights.gguf_variant or "model"
    file_hint = _first_gguf_filename(estimate) or f"{variant}.gguf"

    n_gpu_layers, reasons, warnings, confidence = _pick_layers(estimate, hardware)

    flags = [
        RuntimeFlag(
            name="--model",
            value=file_hint,
            source=RuntimeFlagSource.USER_INPUT,
            explanation=(
                "Path to the .gguf file. Download it first with "
                "`huggingface-cli download` and pass the resulting local path."
            ),
        ),
        RuntimeFlag(
            name="--ctx-size",
            value=str(cfg.context_length),
            source=RuntimeFlagSource.ESTIMATE,
            explanation="Context length used for the memory estimate.",
        ),
        RuntimeFlag(
            name="--n-gpu-layers",
            value=str(n_gpu_layers),
            source=RuntimeFlagSource.HARDWARE,
            explanation=(
                "Number of transformer layers to place on GPU. -1 means all."
            ),
        ),
        RuntimeFlag(
            name="--batch-size",
            value=str(cfg.batch_size),
            source=RuntimeFlagSource.ESTIMATE,
            explanation="Prompt batch size (llama.cpp calls this -b).",
        ),
    ]

    command = (
        f"llama-server --model {file_hint} --ctx-size {cfg.context_length} "
        f"--n-gpu-layers {n_gpu_layers} --batch-size {cfg.batch_size}"
    )

    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        command_preview=command,
        python_snippet=None,
        flags=flags,
        reasons=reasons,
        warnings=warnings,
        confidence=confidence,
    )


def _pick_layers(
    estimate: MemoryEstimate, hardware: HardwareProfile
) -> tuple[int, list[str], list[str], EstimationConfidence]:
    assessment = estimate.assessment
    device = assessment.effective_device
    status = assessment.status

    if device is TargetDevice.CPU:
        return (
            0,
            ["Effective device is CPU, so no layers are offloaded to GPU."],
            [],
            EstimationConfidence.HIGH,
        )

    if status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }:
        return (
            -1,
            [f"Model fits in VRAM ({status.value}); using --n-gpu-layers -1 (all)."],
            [],
            EstimationConfidence.HIGH,
        )

    # Offloading required.
    weights_bytes = estimate.weights.component.bytes
    block_count = estimate.kv_cache.layers
    if weights_bytes and block_count and block_count > 0:
        bytes_per_layer = weights_bytes // block_count
        vram = assessment.available_vram_bytes or 0
        reserve = estimate.inference_configuration.device_reserve_bytes
        kv_bytes = estimate.kv_cache.component.bytes or 0
        available_for_layers = (
            vram - reserve - LLAMA_CPP_HEADROOM_BYTES - kv_bytes
        )
        if available_for_layers <= 0 or bytes_per_layer <= 0:
            n = 0
        else:
            n = min(block_count, available_for_layers // bytes_per_layer)
        return (
            int(n),
            [
                f"Offloading required: {n} of {block_count} layers "
                f"({bytes_per_layer / (1024**3):.2f} GiB/layer) fit in VRAM."
            ],
            [],
            EstimationConfidence.MEDIUM,
        )

    return (
        LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN,
        ["Offloading required but layer count is unknown."],
        [
            "Block count unavailable; using a conservative fallback of "
            f"{LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN} layers. "
            "Increase manually if VRAM allows."
        ],
        EstimationConfidence.LOW,
    )


def _first_gguf_filename(estimate: MemoryEstimate) -> str | None:
    variant = estimate.weights.gguf_variant
    if variant is None:
        return None
    # Walk the analysis? We don't have it here — the caller passes only the estimate.
    # Fall back to a stable placeholder including the variant name.
    return f"{PurePosixPath(estimate.repository.repo_id).name}.{variant}.gguf"


__all__ = ["build"]
