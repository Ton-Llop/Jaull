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

from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.presentation.console import format_bytes

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
    return " · ".join(parts)


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


def _short_gpu(name: str) -> str:
    return _GPU_NOISE.sub("", name).strip() or name


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
    if plan.execution_readiness is None:
        return "Ready when prepared"
    status = plan.execution_readiness.status.value.replace("_", " ").capitalize()
    return "Ready" if status == "Ready" else status


def is_ready_plan(plan: ExecutionPlan) -> bool:
    return (
        plan.execution_readiness is not None
        and plan.execution_readiness.status.value == "ready"
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
    "selected_plan_label",
]
