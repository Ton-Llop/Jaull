"""Human-facing labels for models, artifacts and execution plans.

Every screen used to grow its own copy of these: ``_backend_hint`` existed
three times, plan labels five times, and byte formatting was reimplemented in
four places while :func:`jaull.presentation.console.format_bytes` already
existed. One divergent copy is how "GGUF Q4_K_M" ends up written three
different ways on three screens.

Nothing here derives a fact the domain does not already hold. Where a value is
missing the label says so rather than guessing — a plan with no readiness
evaluation reads "Ready when prepared", never "Ready".
"""

from __future__ import annotations

import re

from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    PlanCompatibilityStatus,
)
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    ComputeBackend,
    HardwareProfile,
)
from jaull.domain.runtime import (
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.presentation.console import format_bytes
from jaull.runtime.device_matching import match_runtime_device

# Repository names carry the artifact format as a suffix (`-GGUF`, `-AWQ`).
# That is a fact about the repository, not about the logical model, and it is
# already shown next to the artifact — so it does not belong in the title.
_FORMAT_SUFFIX = re.compile(r"[-_](gguf|awq|gptq|exl2|mlx)$", re.IGNORECASE)
_SEPARATORS = re.compile(r"[-_]+")

# The quantizations worth putting in front of someone who has not asked for a
# specific one. Everything else is available, just one action further away.
SUGGESTED_QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q6_K")


def model_display_name(identity: ModelIdentity, *, fallback: str | None = None) -> str:
    """A readable title for a logical model.

    Derived from ``model_name`` by dropping the artifact-format suffix and
    turning separators into spaces, so ``Qwen2.5-7B-Instruct-GGUF`` reads as
    ``Qwen2.5 7B Instruct``. The exact repository id is preserved verbatim in
    technical details; this is only the title.
    """
    raw = identity.model_name or fallback or identity.canonical_repo_id or ""
    name = _FORMAT_SUFFIX.sub("", raw.split("/")[-1])
    words = [_recase(word) for word in _SEPARATORS.split(name) if word]
    return " ".join(words) or raw or "unknown model"


def _recase(word: str) -> str:
    """Capitalise a word only when nothing about its case was deliberate.

    ``Qwen2.5``, ``7B`` and ``v0.3`` already carry meaning in their casing and
    are left exactly as they are. A word that is purely lower-case letters —
    ``gemma``, ``it`` — carries none, and reads as a name once capitalised.
    """
    if word.isalpha() and word.islower():
        return word.capitalize()
    return word


def hardware_summary(profile: HardwareProfile | None) -> str:
    """The machine in one line, for the persistent context bar.

    Deliberately three facts at most. The full profile belongs on the hardware
    screen and in Doctor; repeating it on every screen is what pushed the
    actual content below the fold.
    """
    if profile is None:
        return ""
    parts: list[str] = []
    if profile.cpu.model:
        parts.append(_short_cpu(profile.cpu.model))
    parts.append(f"{_round_gib(profile.memory.total_bytes)} RAM")
    if profile.gpus:
        gpu = profile.gpus[0]
        parts.append(f"{_short_gpu(gpu.name)} {_round_gib(gpu.vram_total_bytes)}")
    elif (card := _dedicated_accelerator(profile)) is not None:
        parts.append(f"{_short_gpu(card.name)} {_round_gib(card.dedicated_memory_bytes or 0)}")
    return " · ".join(parts)


def _dedicated_accelerator(profile: HardwareProfile) -> AcceleratorProfile | None:
    """The largest card with its own memory, for machines NVML does not describe.

    A shared-memory iGPU adds nothing to the RAM figure already shown, so only a
    dedicated device is worth a place in a three-fact line.
    """
    cards = [
        accelerator
        for accelerator in profile.accelerators
        if accelerator.type is AcceleratorType.DEDICATED
        and not accelerator.shared_memory
        and accelerator.dedicated_memory_bytes
    ]
    return max(cards, key=lambda card: card.dedicated_memory_bytes or 0, default=None)


def _round_gib(value: int) -> str:
    """GiB without decimals that carry no information — "32 GiB", not "32.00"."""
    gib = value / 1024**3
    return f"{gib:.0f} GiB" if abs(gib - round(gib)) < 0.05 else f"{gib:.1f} GiB"


# Vendor boilerplate is the same on every machine, so it carries no
# information in a one-line summary.
_CPU_NOISE = re.compile(
    r"\b(\d+-Core Processor|Processor|CPU|\(R\)|\(TM\)|@.*)\b|\s{2,}",
    re.IGNORECASE,
)
_GPU_NOISE = re.compile(r"\b(NVIDIA|AMD|Intel|GeForce|Radeon)\s*", re.IGNORECASE)


def _short_cpu(model: str) -> str:
    return _CPU_NOISE.sub(" ", model).strip() or model


_TRADEMARK_MARKS = re.compile(r"\((?:R|TM)\)", re.IGNORECASE)


def _short_gpu(name: str) -> str:
    without_marks = _TRADEMARK_MARKS.sub("", name)
    return _GPU_NOISE.sub("", without_marks).strip() or name


def backend_hint(runtime: RuntimeRecommendation) -> str:
    """Where this runtime intends to execute, read off its own flags."""
    if runtime.runtime is RuntimeName.TRANSFORMERS:
        return next(
            (flag.value for flag in runtime.flags if flag.name == "device_map"),
            "cpu",
        )
    gpu_layers = next(
        (flag.value for flag in runtime.flags if flag.name == "--n-gpu-layers"),
        None,
    )
    if gpu_layers in {None, "0"}:
        return "CPU"
    return "accelerated"


def plan_backend(plan: ExecutionPlan) -> str:
    """The selected backend when one was chosen, else the runtime's intent."""
    if plan.backend_selection is not None:
        return _backend_display(plan.backend_selection.selected_backend.value)
    return backend_hint(plan.runtime)


def is_gguf_plan(plan: ExecutionPlan) -> bool:
    return plan.artifact.format is ArtifactVariantFormat.GGUF


def artifact_label(variant: ArtifactVariant) -> str:
    """Format plus quantization/precision, e.g. ``GGUF Q4_K_M``.

    Delegates to :attr:`ArtifactVariant.label` and only upper-cases the format
    token, which the domain stores lower-case.
    """
    label = variant.label
    parts = label.split(" ", 1)
    parts[0] = parts[0].upper() if parts[0] in {"gguf"} else parts[0]
    return " ".join(parts)


def artifact_display(plan: ExecutionPlan) -> str:
    """The artifact as a user picks it, with `·` between format and variant."""
    if is_gguf_plan(plan):
        return f"GGUF · {plan.artifact.quantization or 'unknown'}"
    if plan.artifact.quantization:
        return f"{plan.artifact.format.value} · {plan.artifact.quantization}"
    if plan.artifact.format is ArtifactVariantFormat.SAFETENSORS:
        return "safetensors"
    if plan.artifact.format is ArtifactVariantFormat.TRANSFORMERS:
        return "safetensors" if plan.runtime.runtime is RuntimeName.TRANSFORMERS else "transformers"
    if plan.artifact.precision:
        return f"{plan.artifact.format.value} · {plan.artifact.precision}"
    return plan.artifact.format.value


def execution_option_label(plan: ExecutionPlan) -> str:
    """The runtime option a plan belongs to — the level above quantization."""
    backend = plan_backend(plan)
    if plan.runtime.runtime is RuntimeName.TRANSFORMERS:
        return f"Transformers · PyTorch · {backend}"
    if is_gguf_plan(plan):
        return f"llama.cpp · {backend}"
    return f"{plan.runtime.runtime.value} · {backend}"


def plan_summary_line(plan: ExecutionPlan) -> str:
    """One line naming the whole execution path: runtime, artifact, backend."""
    return f"{execution_option_label(plan)} · {artifact_display(plan)}"


def plan_variant(plan: ExecutionPlan) -> str:
    """The setting chosen within a runtime option — a quantization, a format."""
    if is_gguf_plan(plan):
        return plan.artifact.quantization or "GGUF"
    return artifact_display(plan)


def selected_plan_label(plan: ExecutionPlan) -> str:
    bits = [execution_option_label(plan), plan_variant(plan)]
    return " · ".join(bit for bit in bits if bit)


def readiness_label(plan: ExecutionPlan) -> str:
    """Readiness in plain words, without claiming a check that never ran."""
    if plan.memory_prediction is not None and (
        plan.memory_prediction.assessment.status is CompatibilityStatus.INSUFFICIENT
    ):
        return "Insufficient memory"
    if (
        plan.compatibility is PlanCompatibilityStatus.NOT_COMPATIBLE
        and plan.runtime_family is not RuntimeName.UNKNOWN
    ):
        return "Unsupported configuration"
    if plan.execution_readiness is None:
        return "Readiness unknown"
    status = plan.execution_readiness.status.value.replace("_", " ").capitalize()
    return {"Not ready": "Blocked", "Preparable": "Preparable"}.get(status, status)


def readiness_detail(plan: ExecutionPlan) -> str:
    label = readiness_label(plan)
    if label in {"Unsupported configuration", "Insufficient memory"}:
        return f"{label} — {next(reversed(plan.warnings), 'Configuration cannot be executed.')}"
    readiness = plan.execution_readiness
    if readiness is None:
        return "Readiness has not been checked."
    label = readiness_label(plan)
    if readiness.status is ExecutionReadinessStatus.READY:
        return label
    return f"{label} — {readiness.message or readiness.reason.value.replace('_', ' ')}"


_DOES_NOT_FIT_LABELS = frozenset({"Insufficient memory", "Unsupported configuration"})


def fits_here(plan: ExecutionPlan) -> bool:
    """Whether the plan could work on this machine once its runtime is sorted out.

    A missing or unprepared runtime is fixable and stays a candidate; too little
    memory or a configuration Jaull cannot load here is not, so those lists apart.
    """
    return readiness_label(plan) not in _DOES_NOT_FIT_LABELS


def is_ready_plan(plan: ExecutionPlan) -> bool:
    return (
        plan.compatibility is not PlanCompatibilityStatus.NOT_COMPATIBLE
        and plan.execution_readiness is not None
        and plan.execution_readiness.status
        in {ExecutionReadinessStatus.READY, ExecutionReadinessStatus.PREPARABLE}
    )


def runtime_block_reason(plan: ExecutionPlan) -> str | None:
    """Why this plan cannot be acted on here, or ``None`` when it can be tried.

    Recommending a plan and acting on it are separate questions, so this answers
    only the second one. The wording names the plan rather than any one verb, so
    Run, Validate and Benchmark can all report the same condition without each
    inventing its own sentence. Vulkan requires confirmed readiness: otherwise
    an attempt could let the runtime choose an unidentified GPU. Other backends
    preserve the existing policy for unknown readiness.
    """

    if plan.compatibility is PlanCompatibilityStatus.NOT_COMPATIBLE:
        return readiness_detail(plan)
    readiness = plan.execution_readiness
    selection = plan.backend_selection or (readiness.selection if readiness is not None else None)
    vulkan = selection is not None and selection.selected_backend is ComputeBackend.VULKAN
    readiness_confirms_vulkan = readiness is not None and readiness.status in {
        ExecutionReadinessStatus.READY,
        ExecutionReadinessStatus.PREPARABLE,
    }
    vulkan_unconfirmed = (
        vulkan
        and not readiness_confirms_vulkan
    )
    if readiness is None:
        return "Vulkan device readiness has not been confirmed; recheck the runtime." if (
            vulkan_unconfirmed
        ) else None
    if vulkan and readiness_confirms_vulkan:
        capability = readiness.selected_backend_capability
        if (
            selection is None or readiness.selection != selection
            or not isinstance(capability, LlamaCppBackendCapability)
            or capability.backend is not ComputeBackend.VULKAN
            or capability.state is not LlamaCppBackendCapabilityState.CONFIRMED
            or match_runtime_device(selection, capability).device is None
        ):
            return "Vulkan device cannot be uniquely identified; recheck the runtime."
    device_unidentified = (
        readiness.status is ExecutionReadinessStatus.UNKNOWN
        and readiness.reason is ExecutionReadinessReason.SELECTED_DEVICE_NOT_IDENTIFIED
    )
    if (
        readiness.status is not ExecutionReadinessStatus.NOT_READY
        and not device_unidentified and not vulkan_unconfirmed
    ):
        return None
    detail = readiness.message or readiness.reason.value.replace("_", " ")
    required = plan.runtime_family.value
    if plan.backend_selection is not None:
        backend = _backend_display(plan.backend_selection.selected_backend.value)
        required += f" with {backend} support"
    return (
        "This execution plan is not ready on this machine. "
        f"Required runtime: {required}. Execution blocked: {detail} "
        "Open Paths to inspect or select another plan."
    )


def _backend_display(backend: str) -> str:
    normalized = backend.lower()
    if normalized in {"cpu", "cuda", "hip"}:
        return normalized.upper()
    if normalized == "vulkan":
        return "Vulkan"
    return backend


def format_gib(value: int | None) -> str:
    """Fixed GiB, for columns of memory figures that must line up."""
    if value is None:
        return "unknown"
    return f"{value / 1024**3:.2f} GiB"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f} s"


def format_throughput(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.1f} tok/s"


__all__ = [
    "SUGGESTED_QUANTIZATIONS",
    "artifact_display",
    "artifact_label",
    "backend_hint",
    "execution_option_label",
    "format_bytes",
    "format_gib",
    "format_seconds",
    "format_throughput",
    "hardware_summary",
    "is_gguf_plan",
    "is_ready_plan",
    "model_display_name",
    "plan_backend",
    "plan_summary_line",
    "plan_variant",
    "readiness_label",
    "runtime_block_reason",
    "selected_plan_label",
]
