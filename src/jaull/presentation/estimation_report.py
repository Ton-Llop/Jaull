"""Rich rendering of the estimation view.

The JSON emitter lives in ``jaull.reporting.estimation``. It is re-exported
here for backwards compatibility — callers that only need the dict form
should import from ``jaull.reporting.estimation`` directly.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from jaull.domain.estimation import (
    CompatibilityStatus,
    MemoryComponent,
    MemoryEstimate,
)
from jaull.presentation.console import format_bytes

# Re-export for callers that still import from here.
from jaull.reporting.estimation import (
    SCHEMA_VERSION,
    estimate_to_json_dict,
)


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


__all__ = [
    "SCHEMA_VERSION",
    "estimate_to_json_dict",
    "render_estimate",
]
