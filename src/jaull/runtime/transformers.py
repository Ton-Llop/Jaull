"""HF Transformers recommendation builder."""

from __future__ import annotations

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import TargetDevice, WeightPrecision
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)


def build(estimate: MemoryEstimate, hardware: HardwareProfile) -> RuntimeRecommendation:
    del hardware  # Not needed once assessment is done — flags come from the estimate.
    precision = estimate.weights.precision or WeightPrecision.FLOAT16
    dtype_expr = _torch_dtype_expr(precision)
    device_map, reasons, warnings = _pick_device_map(estimate)
    repo_id = estimate.repository.repo_id

    snippet = (
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "import torch\n\n"
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
        f'model = AutoModelForCausalLM.from_pretrained(\n'
        f'    "{repo_id}",\n'
        f"    torch_dtype={dtype_expr},\n"
        f'    device_map="{device_map}",\n'
        f")"
    )

    flags = [
        RuntimeFlag(
            name="torch_dtype",
            value=dtype_expr,
            source=RuntimeFlagSource.ESTIMATE,
            explanation=(
                f"Weight precision derived from the estimate ({precision.value})."
            ),
        ),
        RuntimeFlag(
            name="device_map",
            value=device_map,
            source=RuntimeFlagSource.HARDWARE,
            explanation="How Accelerate should place the layers across devices.",
        ),
    ]

    confidence = EstimationConfidence.HIGH if precision else EstimationConfidence.MEDIUM

    return RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        command_preview=None,
        python_snippet=snippet,
        flags=flags,
        reasons=reasons,
        warnings=warnings,
        confidence=confidence,
    )


def _pick_device_map(
    estimate: MemoryEstimate,
) -> tuple[str, list[str], list[str]]:
    assessment = estimate.assessment
    device = assessment.effective_device
    status = assessment.status

    if device is TargetDevice.CPU:
        return (
            "cpu",
            ["Effective device is CPU; loading the whole model into RAM."],
            [],
        )
    if status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
        CompatibilityStatus.TIGHT,
    }:
        return (
            "cuda",
            [f"Model fits in VRAM ({status.value}); placing everything on the GPU."],
            [],
        )
    # Offloading
    return (
        "auto",
        ["Model does not fit fully in VRAM; letting Accelerate split layers."],
        [
            "device_map='auto' requires the `accelerate` package. Actual placement "
            "depends on Accelerate's heuristics and available disk swap."
        ],
    )


def _torch_dtype_expr(precision: WeightPrecision) -> str:
    return {
        WeightPrecision.FLOAT32: "torch.float32",
        WeightPrecision.FLOAT16: "torch.float16",
        WeightPrecision.BFLOAT16: "torch.bfloat16",
        WeightPrecision.INT8: "torch.int8",
        # int4 needs a BitsAndBytes quantization_config in HF; the torch_dtype below
        # is a coarse approximation used only for the estimate.
        WeightPrecision.INT4: "torch.int8",
    }[precision]


__all__ = ["build"]
