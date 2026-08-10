"""Validate a runtime recommendation against the detected hardware.

The estimator says "here is what runs Q4_K_M under 6 GiB". The runtime
selector says "the recommended runtime is llama.cpp". Neither answers the
question the user actually cares about: *can I run this on the machine
in front of me?* An AWQ artifact loaded through Transformers needs CUDA —
suggesting it on a laptop with no NVIDIA GPU is worse than saying nothing.

This module fills that gap with pure rules over what the pipeline already
computed. No process is launched and no binary is probed.
"""

from __future__ import annotations

from jaull.domain.artifact_profile import (
    ArtifactConfirmation,
    ArtifactFormat,
    ArtifactProfile,
)
from jaull.domain.estimation import CompatibilityStatus, MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    RuntimeAssessment,
    RuntimeExecutability,
    RuntimeName,
    RuntimeRecommendation,
)

# Runtimes that need a CUDA-capable GPU to load AWQ/GPTQ weights.
_CUDA_ONLY_QUANT_FORMATS = {ArtifactFormat.AWQ, ArtifactFormat.GPTQ}


def assess_runtime(
    recommendation: RuntimeRecommendation | None,
    hardware: HardwareProfile,
    artifact: ArtifactProfile | None,
    estimate: MemoryEstimate | None,
) -> RuntimeAssessment:
    """Grade the *ejecutabilidad* of a runtime recommendation."""
    runtime = recommendation.runtime if recommendation else RuntimeName.UNKNOWN
    has_gpu = bool(hardware.gpus)

    if recommendation is None or runtime is RuntimeName.UNKNOWN:
        return RuntimeAssessment(
            runtime=runtime,
            executability=RuntimeExecutability.UNKNOWN,
            reasons=["No runtime recommendation was produced."],
        )

    if estimate is not None and estimate.assessment.status is CompatibilityStatus.INSUFFICIENT:
        return RuntimeAssessment(
            runtime=runtime,
            executability=RuntimeExecutability.UNSUPPORTED,
            reasons=["Insufficient memory to load the model at all."],
        )

    if runtime is RuntimeName.LLAMA_CPP:
        return _assess_llama_cpp(has_gpu, artifact)
    if runtime is RuntimeName.TRANSFORMERS:
        return _assess_transformers(has_gpu, artifact)
    if runtime is RuntimeName.VLLM:
        return _assess_vllm(has_gpu, artifact)

    return RuntimeAssessment(
        runtime=runtime,
        executability=RuntimeExecutability.UNKNOWN,
        reasons=[f"No executability rules for runtime {runtime.value!r}."],
    )


def _assess_llama_cpp(
    has_gpu: bool, artifact: ArtifactProfile | None
) -> RuntimeAssessment:
    if artifact is not None and artifact.format is ArtifactFormat.GGUF:
        detail = "GGUF + GPU" if has_gpu else "GGUF + CPU"
        return RuntimeAssessment(
            runtime=RuntimeName.LLAMA_CPP,
            executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
            reasons=[
                f"{detail}: llama.cpp is designed for this pairing.",
                "Binary installation is not verified by this tool.",
            ],
        )
    return RuntimeAssessment(
        runtime=RuntimeName.LLAMA_CPP,
        executability=RuntimeExecutability.UNKNOWN,
        reasons=["Runtime is llama.cpp but no GGUF artifact was confirmed."],
    )


def _assess_transformers(
    has_gpu: bool, artifact: ArtifactProfile | None
) -> RuntimeAssessment:
    warnings: list[str] = []
    reasons: list[str] = []

    if artifact is None:
        return RuntimeAssessment(
            runtime=RuntimeName.TRANSFORMERS,
            executability=RuntimeExecutability.UNKNOWN,
            reasons=["Artifact profile is unavailable."],
        )

    if artifact.format in _CUDA_ONLY_QUANT_FORMATS:
        if not has_gpu:
            return RuntimeAssessment(
                runtime=RuntimeName.TRANSFORMERS,
                executability=RuntimeExecutability.UNSUPPORTED,
                reasons=[
                    f"{artifact.format.value.upper()} kernels require a CUDA GPU; "
                    "the detected hardware has no NVIDIA GPU."
                ],
            )
        return RuntimeAssessment(
            runtime=RuntimeName.TRANSFORMERS,
            executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
            reasons=[
                f"{artifact.format.value.upper()} loads via Transformers on the "
                "detected NVIDIA GPU."
            ],
            warnings=["Requires the matching AWQ/GPTQ Python extras installed."],
        )

    if artifact.format is ArtifactFormat.BITSANDBYTES:
        if not has_gpu:
            return RuntimeAssessment(
                runtime=RuntimeName.TRANSFORMERS,
                executability=RuntimeExecutability.UNSUPPORTED,
                reasons=[
                    "bitsandbytes int4/int8 kernels require a CUDA GPU; "
                    "the detected hardware has no NVIDIA GPU."
                ],
            )
        return RuntimeAssessment(
            runtime=RuntimeName.TRANSFORMERS,
            executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
            reasons=["bitsandbytes 4/8-bit loading available on the detected GPU."],
        )

    if artifact.confirmation is ArtifactConfirmation.THEORETICAL:
        # No confirmed weights at the chosen precision → without CUDA there is
        # no realistic on-the-fly quantization path in Transformers.
        if not has_gpu:
            return RuntimeAssessment(
                runtime=RuntimeName.TRANSFORMERS,
                executability=RuntimeExecutability.UNSUPPORTED,
                reasons=[
                    "Theoretical quantization has no confirmed artifact and no "
                    "CPU quantization path in Transformers."
                ],
            )
        warnings.append(
            "Theoretical quantization: requires in-flight quantization on load."
        )
        reasons.append("Runnable via Transformers with in-flight quantization on GPU.")
        return RuntimeAssessment(
            runtime=RuntimeName.TRANSFORMERS,
            executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
            reasons=reasons,
            warnings=warnings,
        )

    # Native dtype (FP16/BF16/FP32) — CPU works but is very slow, GPU is
    # comfortable.
    if has_gpu:
        reasons.append("Native precision on the detected NVIDIA GPU.")
        return RuntimeAssessment(
            runtime=RuntimeName.TRANSFORMERS,
            executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
            reasons=reasons,
        )
    reasons.append(
        "Native precision on CPU only — Transformers can load it but throughput "
        "will be significantly lower."
    )
    warnings.append("No GPU detected; consider a llama.cpp GGUF conversion.")
    return RuntimeAssessment(
        runtime=RuntimeName.TRANSFORMERS,
        executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
        reasons=reasons,
        warnings=warnings,
    )


def _assess_vllm(has_gpu: bool, artifact: ArtifactProfile | None) -> RuntimeAssessment:
    if not has_gpu:
        return RuntimeAssessment(
            runtime=RuntimeName.VLLM,
            executability=RuntimeExecutability.UNSUPPORTED,
            reasons=["vLLM requires a CUDA GPU; none was detected."],
        )
    del artifact
    return RuntimeAssessment(
        runtime=RuntimeName.VLLM,
        executability=RuntimeExecutability.POTENTIALLY_EXECUTABLE,
        reasons=["vLLM available on the detected NVIDIA GPU."],
    )


__all__ = ["assess_runtime"]
