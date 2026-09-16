"""HF Transformers recommendation builder."""

from __future__ import annotations

from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import WeightPrecision
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.runtime.transformers_launch_policy import pick_device_map
from jaull.runtime.transformers_quantization import COMPUTE_DTYPE, quantization_mode


def build(estimate: MemoryEstimate, hardware: HardwareProfile) -> RuntimeRecommendation:
    del hardware  # Not needed once assessment is done — flags come from the estimate.
    precision = estimate.weights.precision or WeightPrecision.FLOAT16
    device_plan = pick_device_map(estimate)
    device_map = device_plan.device_map
    reasons = device_plan.reasons
    warnings = list(device_plan.warnings)
    repo_id = estimate.repository.repo_id

    # Exactly one flag carries the precision, and it names the mechanism that
    # actually produces it: a dtype for float weights, a quantization config for
    # int4/int8. Emitting `torch_dtype=torch.int8` for a quantized estimate used
    # to look like a precision and load like nothing of the sort.
    mode = quantization_mode(precision)
    if mode is not None:
        snippet = (
            "from transformers import (\n"
            "    AutoModelForCausalLM,\n"
            "    AutoTokenizer,\n"
            "    BitsAndBytesConfig,\n"
            ")\n"
            "import torch\n\n"
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            "model = AutoModelForCausalLM.from_pretrained(\n"
            f'    "{repo_id}",\n'
            "    quantization_config=BitsAndBytesConfig(\n"
            f"        load_in_{mode}=True,\n"
            + (
                f"        bnb_4bit_compute_dtype={COMPUTE_DTYPE},\n"
                if mode == "4bit"
                else ""
            )
            + "    ),\n"
            f'    device_map="{device_map}",\n'
            ")"
        )
        precision_flag = RuntimeFlag(
            name="quantization",
            value=mode,
            source=RuntimeFlagSource.ESTIMATE,
            explanation=(
                f"The estimate is {precision.value}, which Transformers loads "
                f"through BitsAndBytesConfig(load_in_{mode}=True), not through a "
                "torch dtype."
            ),
        )
        warnings.append(
            "Running this configuration requires the 'bitsandbytes' package, "
            "which Jaull does not install."
        )
        warnings.append(
            "The estimate assumes every weight is quantized. bitsandbytes keeps "
            "some modules at a higher precision, so the loaded footprint is "
            "larger than parameters x bits/8."
        )
    else:
        dtype_expr = _torch_dtype_expr(precision)
        snippet = (
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "import torch\n\n"
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            "model = AutoModelForCausalLM.from_pretrained(\n"
            f'    "{repo_id}",\n'
            f"    torch_dtype={dtype_expr},\n"
            f'    device_map="{device_map}",\n'
            ")"
        )
        precision_flag = RuntimeFlag(
            name="torch_dtype",
            value=dtype_expr,
            source=RuntimeFlagSource.ESTIMATE,
            explanation=(
                f"Weight precision derived from the estimate ({precision.value})."
            ),
        )

    flags = [
        precision_flag,
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


def _torch_dtype_expr(precision: WeightPrecision) -> str:
    """The dtype for a precision that loads as one. Quantized never reaches here."""
    return {
        WeightPrecision.FLOAT32: "torch.float32",
        WeightPrecision.FLOAT16: "torch.float16",
        WeightPrecision.BFLOAT16: "torch.bfloat16",
    }[precision]


__all__ = ["build"]
