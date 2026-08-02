"""vLLM recommendation builder (only for supported architectures + GPU-fits scenarios)."""

from __future__ import annotations

from local_ai_check.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from local_ai_check.domain.hardware import HardwareProfile
from local_ai_check.domain.inference import TargetDevice, WeightPrecision
from local_ai_check.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from local_ai_check.runtime.policies import VLLM_SUPPORTED_ARCHITECTURES


def is_viable(estimate: MemoryEstimate, hardware: HardwareProfile) -> bool:
    """True when vLLM is a reasonable alternative to the primary Transformers rec."""
    del hardware
    if estimate.repository.pipeline_tag not in {
        "text-generation",
        "conversational",
        None,
    }:
        return False
    arch = _architecture_hint(estimate)
    if arch not in VLLM_SUPPORTED_ARCHITECTURES:
        return False
    if estimate.assessment.effective_device is not TargetDevice.GPU:
        return False
    return estimate.assessment.status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }


def build(estimate: MemoryEstimate, hardware: HardwareProfile) -> RuntimeRecommendation:
    del hardware
    precision = estimate.weights.precision or WeightPrecision.BFLOAT16
    dtype = _cli_dtype(precision)
    cfg = estimate.inference_configuration

    command = (
        f"vllm serve {estimate.repository.repo_id} "
        f"--dtype {dtype} --max-model-len {cfg.context_length} "
        f"--gpu-memory-utilization 0.9"
    )

    flags = [
        RuntimeFlag(
            name="--dtype",
            value=dtype,
            source=RuntimeFlagSource.ESTIMATE,
            explanation=f"Derived from the estimated weight precision ({precision.value}).",
        ),
        RuntimeFlag(
            name="--max-model-len",
            value=str(cfg.context_length),
            source=RuntimeFlagSource.ESTIMATE,
            explanation="Context length from the estimate.",
        ),
        RuntimeFlag(
            name="--gpu-memory-utilization",
            value="0.9",
            source=RuntimeFlagSource.POLICY,
            explanation="Leaves ~10 % of VRAM as headroom for the KV cache growth.",
        ),
    ]

    return RuntimeRecommendation(
        runtime=RuntimeName.VLLM,
        command_preview=command,
        python_snippet=None,
        flags=flags,
        reasons=[
            "Architecture is in vLLM's supported list and the model fits in VRAM.",
        ],
        warnings=[
            "vLLM's real support matrix is broader; consult its docs for architectures "
            "not in the shortlist."
        ],
        confidence=EstimationConfidence.MEDIUM,
    )


def _architecture_hint(estimate: MemoryEstimate) -> str | None:
    if estimate.architecture:
        return estimate.architecture.lower()
    return None


def _cli_dtype(precision: WeightPrecision) -> str:
    return {
        WeightPrecision.FLOAT32: "float32",
        WeightPrecision.FLOAT16: "float16",
        WeightPrecision.BFLOAT16: "bfloat16",
        WeightPrecision.INT8: "int8",
        WeightPrecision.INT4: "int8",  # vLLM cli maps 4-bit through --quantization
    }[precision]


__all__ = ["build", "is_viable"]
