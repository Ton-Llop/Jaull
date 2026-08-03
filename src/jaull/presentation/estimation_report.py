from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryComponent,
    MemoryEstimate,
)
from jaull.presentation.console import format_bytes

SCHEMA_VERSION = 1


def render_estimate(estimate: MemoryEstimate, console: Console) -> None:
    console.print(_header_table(estimate))
    console.print()
    console.print(_breakdown_table(estimate))
    console.print()
    console.print(_resources_table(estimate))
    console.print()
    console.print(_assessment_table(estimate))

    if estimate.runtime_recommendation is not None:
        console.print()
        _render_runtime(estimate.runtime_recommendation, console)

    if estimate.assumptions:
        console.print()
        console.print("[bold]Assumptions[/bold]")
        for line in estimate.assumptions:
            console.print(f"  - {line}")

    if estimate.warnings:
        console.print()
        console.print("[bold]Warnings[/bold]")
        for warning in estimate.warnings:
            console.print(f"[yellow]  ! {warning}[/yellow]")


def _render_runtime(recommendation: object, console: Console) -> None:
    # Local import to avoid pulling runtime into the domain of estimation.py.
    from jaull.domain.runtime import RuntimeName, RuntimeRecommendation

    assert isinstance(recommendation, RuntimeRecommendation)

    header = Table(
        title="Recommended runtime", show_header=False, box=None, pad_edge=False
    )
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value")
    header.add_row("Runtime", recommendation.runtime.value)
    header.add_row("Confidence", recommendation.confidence.value)
    if recommendation.alternatives:
        header.add_row(
            "Alternatives",
            ", ".join(alt.value for alt in recommendation.alternatives),
        )
    console.print(header)

    if recommendation.command_preview:
        console.print()
        console.print("[bold]Command[/bold]")
        console.print(f"  $ {recommendation.command_preview}")

    if recommendation.python_snippet:
        console.print()
        console.print("[bold]Python snippet[/bold]")
        for line in recommendation.python_snippet.splitlines():
            console.print(f"  {line}")

    if recommendation.flags:
        console.print()
        flag_table = Table(title="Flags", show_lines=False)
        flag_table.add_column("Flag", style="cyan")
        flag_table.add_column("Value")
        flag_table.add_column("Source")
        flag_table.add_column("Explanation")
        for flag in recommendation.flags:
            flag_table.add_row(
                flag.name, flag.value, flag.source.value, flag.explanation
            )
        console.print(flag_table)

    for reason in recommendation.reasons:
        console.print(f"  [dim]· {reason}[/dim]")
    for warning in recommendation.warnings:
        console.print(f"[yellow]  ! {warning}[/yellow]")

    if recommendation.runtime is RuntimeName.UNKNOWN and not recommendation.warnings:
        console.print("  [dim]No runtime recommendation available.[/dim]")


def _header_table(estimate: MemoryEstimate) -> Table:
    cfg = estimate.inference_configuration
    table = Table(title="Model estimation", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Repository", estimate.repository.repo_id)
    table.add_row("Repository type", estimate.repository_type.value)
    if estimate.weights.gguf_variant:
        table.add_row("Selected variant", estimate.weights.gguf_variant)
    if estimate.weights.precision:
        table.add_row("Weight precision", estimate.weights.precision.value)

    base = estimate.base_model_resolution
    if base and base.repo_id:
        table.add_row("Base model", base.repo_id)
        table.add_row("Base model source", base.source.value)
    if estimate.configuration_sources:
        distinct = sorted({s.value for s in estimate.configuration_sources.values()})
        table.add_row("Configuration source", " + ".join(distinct))

    table.add_row("Context length", str(cfg.context_length))
    table.add_row("Batch size", str(cfg.batch_size))
    table.add_row("Target device", cfg.target_device.value)
    table.add_row("KV cache dtype", cfg.kv_cache_dtype.value)
    return table


def _breakdown_table(estimate: MemoryEstimate) -> Table:
    table = Table(title="Memory breakdown", show_lines=False)
    table.add_column("Component", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Source")
    table.add_column("Explanation")

    for component in _components(estimate):
        table.add_row(
            component.name,
            format_bytes(component.bytes),
            component.source.value,
            component.explanation,
        )

    if estimate.total_bytes is not None:
        table.add_row("Total required", format_bytes(estimate.total_bytes), "", "")

    return table


def _resources_table(estimate: MemoryEstimate) -> Table:
    assessment = estimate.assessment
    table = Table(title="Local resources", show_header=False, box=None, pad_edge=False)
    table.add_column("Resource", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row(
        "Available VRAM",
        format_bytes(assessment.available_vram_bytes)
        if assessment.available_vram_bytes is not None
        else "no NVIDIA GPU detected",
    )
    table.add_row(
        "Available RAM",
        format_bytes(assessment.available_ram_bytes)
        if assessment.available_ram_bytes is not None
        else "unknown",
    )
    return table


def _assessment_table(estimate: MemoryEstimate) -> Table:
    assessment = estimate.assessment
    table = Table(title="Assessment", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Result", _pretty_status(assessment.status))
    table.add_row("Confidence", assessment.confidence.value)
    table.add_row("Effective device", assessment.effective_device.value)
    if assessment.ratio is not None:
        table.add_row("Utilisation", f"{assessment.ratio * 100:.1f}%")
    if assessment.reasons:
        table.add_row("Reasons", "\n".join(f"- {r}" for r in assessment.reasons))
    return table


def _components(estimate: MemoryEstimate) -> list[MemoryComponent]:
    parts = [
        estimate.weights.component,
        estimate.kv_cache.component,
        estimate.runtime_overhead.component,
        estimate.device_reserve,
    ]
    if estimate.safety_margin is not None:
        parts.append(estimate.safety_margin)
    return parts


def _pretty_status(status: CompatibilityStatus) -> str:
    styles = {
        CompatibilityStatus.COMFORTABLE: "[green]Comfortable[/green]",
        CompatibilityStatus.COMPATIBLE: "[green]Compatible[/green]",
        CompatibilityStatus.TIGHT: "[yellow]Tight fit[/yellow]",
        CompatibilityStatus.OFFLOADING_REQUIRED: "[yellow]Offloading required[/yellow]",
        CompatibilityStatus.INSUFFICIENT: "[red]Insufficient[/red]",
        CompatibilityStatus.UNKNOWN: "Unknown",
    }
    return styles[status]


# ---------------------------------------------------------------------------
# JSON emitter
# ---------------------------------------------------------------------------


def estimate_to_json_dict(estimate: MemoryEstimate) -> dict[str, Any]:
    cfg = estimate.inference_configuration
    assessment = estimate.assessment

    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "repo_id": estimate.repository.repo_id,
            "repository_type": estimate.repository_type.value,
            "author": estimate.repository.author,
            "pipeline_tag": estimate.repository.pipeline_tag,
            "library_name": estimate.repository.library_name,
            "license": estimate.repository.license,
            "gated": estimate.repository.gated,
        },
        "inference_configuration": {
            "context_length": cfg.context_length,
            "batch_size": cfg.batch_size,
            "target_device": cfg.target_device.value,
            "precision": cfg.precision.value if cfg.precision else None,
            "quantization": cfg.quantization,
            "kv_cache_dtype": cfg.kv_cache_dtype.value,
            "safety_margin_percent": cfg.safety_margin_percent,
            "device_reserve_bytes": cfg.device_reserve_bytes,
        },
        "memory": {
            "weights_bytes": estimate.weights.component.bytes,
            "kv_cache_bytes": estimate.kv_cache.component.bytes,
            "runtime_overhead_bytes": estimate.runtime_overhead.component.bytes,
            "device_reserve_bytes": estimate.device_reserve.bytes,
            "safety_margin_bytes": (
                estimate.safety_margin.bytes if estimate.safety_margin else 0
            ),
            "total_bytes": estimate.total_bytes,
            "components": [
                _component_to_dict(c) for c in _components(estimate)
            ],
        },
        "weights": {
            "num_parameters": estimate.weights.num_parameters,
            "bits_per_parameter": estimate.weights.bits_per_parameter,
            "precision": (
                estimate.weights.precision.value
                if estimate.weights.precision
                else None
            ),
            "gguf_variant": estimate.weights.gguf_variant,
        },
        "kv_cache": {
            "layers": estimate.kv_cache.layers,
            "kv_heads": estimate.kv_cache.kv_heads,
            "head_dim": estimate.kv_cache.head_dim,
            "context_length": estimate.kv_cache.context_length,
            "batch_size": estimate.kv_cache.batch_size,
            "dtype_bytes": estimate.kv_cache.dtype_bytes,
            "formula": estimate.kv_cache.formula,
            "notes": list(estimate.kv_cache.notes),
        },
        "hardware": {
            "available_vram_bytes": assessment.available_vram_bytes,
            "available_ram_bytes": assessment.available_ram_bytes,
        },
        "assessment": {
            "status": assessment.status.value,
            "confidence": _confidence_to_string(assessment.confidence),
            "target_device": assessment.target_device.value,
            "effective_device": assessment.effective_device.value,
            "utilisation_ratio": assessment.ratio,
            "reasons": list(assessment.reasons),
            "warnings": list(assessment.warnings),
        },
        "assumptions": list(estimate.assumptions),
        "warnings": list(estimate.warnings),
        "base_model_resolution": _base_resolution_to_dict(
            estimate.base_model_resolution
        ),
        "configuration_sources": {
            field: source.value
            for field, source in estimate.configuration_sources.items()
        },
        "architecture": estimate.architecture,
        "runtime_recommendation": _runtime_to_dict(estimate.runtime_recommendation),
    }


def _runtime_to_dict(recommendation: object) -> dict[str, Any] | None:
    if recommendation is None:
        return None
    from jaull.domain.runtime import RuntimeRecommendation

    assert isinstance(recommendation, RuntimeRecommendation)
    return {
        "runtime": recommendation.runtime.value,
        "command_preview": recommendation.command_preview,
        "python_snippet": recommendation.python_snippet,
        "flags": [
            {
                "name": flag.name,
                "value": flag.value,
                "source": flag.source.value,
                "explanation": flag.explanation,
            }
            for flag in recommendation.flags
        ],
        "reasons": list(recommendation.reasons),
        "warnings": list(recommendation.warnings),
        "confidence": recommendation.confidence.value,
        "alternatives": [alt.value for alt in recommendation.alternatives],
    }


def _base_resolution_to_dict(resolution: object) -> dict[str, Any] | None:
    if resolution is None:
        return None
    return {
        "repo_id": getattr(resolution, "repo_id", None),
        "source": getattr(getattr(resolution, "source", None), "value", None),
        "confidence": getattr(getattr(resolution, "confidence", None), "value", None),
        "evidence": list(getattr(resolution, "evidence", []) or []),
        "warnings": list(getattr(resolution, "warnings", []) or []),
        "candidates": list(getattr(resolution, "candidates", []) or []),
    }


def _component_to_dict(component: MemoryComponent) -> dict[str, Any]:
    return {
        "name": component.name,
        "bytes": component.bytes,
        "source": component.source.value,
        "confidence": _confidence_to_string(component.confidence),
        "explanation": component.explanation,
    }


def _confidence_to_string(confidence: EstimationConfidence) -> str:
    return confidence.value


__all__ = ["SCHEMA_VERSION", "estimate_to_json_dict", "render_estimate"]
