"""llama.cpp recommendation builder."""

from __future__ import annotations

from pathlib import PurePosixPath

from jaull.domain.estimation import MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.runtime.llama_cpp_launch_policy import pick_gpu_layers


def build(estimate: MemoryEstimate, hardware: HardwareProfile) -> RuntimeRecommendation:
    del hardware  # the launch policy reads everything it needs from the estimate
    cfg = estimate.inference_configuration
    variant = estimate.weights.gguf_variant or "model"
    file_hint = _first_gguf_filename(estimate) or f"{variant}.gguf"

    layer_plan = pick_gpu_layers(estimate)
    n_gpu_layers = layer_plan.n_gpu_layers
    reasons = layer_plan.reasons
    warnings = layer_plan.warnings
    confidence = layer_plan.confidence

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


def _first_gguf_filename(estimate: MemoryEstimate) -> str | None:
    variant = estimate.weights.gguf_variant
    if variant is None:
        return None
    # Walk the analysis? We don't have it here — the caller passes only the estimate.
    # Fall back to a stable placeholder including the variant name.
    return f"{PurePosixPath(estimate.repository.repo_id).name}.{variant}.gguf"


__all__ = ["build"]
