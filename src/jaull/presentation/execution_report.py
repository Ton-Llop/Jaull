"""Presentation helpers for real execution observations."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from jaull.domain.execution import ExecutionObservation, InferenceResult
from jaull.presentation.console import format_bytes


def observation_rows(
    observation: ExecutionObservation,
    *,
    include_failure_reason: bool = True,
    include_measurement: bool = False,
) -> list[tuple[str, str]]:
    """Return stable human-facing rows for an execution observation."""
    rows = [
        ("Success", "yes" if observation.success else "no"),
        ("Duration", f"{observation.duration_seconds:.2f} s"),
        ("Peak RAM", _format_observed_bytes(observation.peak_ram_bytes)),
        ("Peak VRAM", _format_observed_bytes(observation.peak_vram_bytes)),
        (
            "Exit code",
            str(observation.exit_code)
            if observation.exit_code is not None
            else "unavailable",
        ),
    ]
    if include_failure_reason and observation.failure_reason is not None:
        rows.append(("Reason", observation.failure_reason.value))
    if include_measurement:
        rows.extend(
            [
                ("RAM measurement", observation.measurement.ram_measurement),
                ("VRAM measurement", observation.measurement.vram_measurement),
                (
                    "Sample interval",
                    f"{observation.measurement.sample_interval_seconds:.3f} s",
                ),
            ]
        )
    return rows


def render_inference_result(console: Console, result: InferenceResult) -> None:
    """Render model text and its execution observation as separate blocks."""
    console.rule("Model response")
    console.print(result.text or "[dim](empty response)[/dim]")
    console.print()
    render_execution_observation(console, result.observation)


def render_execution_observation(
    console: Console,
    observation: ExecutionObservation,
    *,
    title: str = "Execution observation",
    include_measurement: bool = False,
) -> None:
    console.rule(title)
    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column()
    for label, value in observation_rows(
        observation,
        include_measurement=include_measurement,
    ):
        table.add_row(label, value)
    console.print(table)


def inline_observation_summary(
    observation: ExecutionObservation,
    *,
    omit_missing: bool = False,
) -> str:
    """Compact summary for TUI history metadata.

    ``omit_missing`` drops the metrics that were not measured. A report wants
    the full shape — "VRAM unavailable" documents that the field exists and was
    not captured — but under a chat response it is noise: two of the four facts
    on the line would say nothing at all.
    """
    parts = [f"{observation.duration_seconds:.2f} s"]
    for label, value in (
        ("RAM", observation.peak_ram_bytes),
        ("VRAM", observation.peak_vram_bytes),
    ):
        if value is None and omit_missing:
            continue
        parts.append(f"{label} {_format_observed_bytes(value)}")
    return " · ".join(parts)


def _format_observed_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return format_bytes(value)


__all__ = [
    "inline_observation_summary",
    "observation_rows",
    "render_execution_observation",
    "render_inference_result",
]
