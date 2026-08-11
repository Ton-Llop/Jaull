"""Rich rendering for prediction-vs-observation comparisons."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from jaull.domain.comparison import MetricComparison, PredictionComparison
from jaull.presentation.console import format_bytes


def render_prediction_comparison(
    console: Console,
    comparison: PredictionComparison,
    *,
    title: str = "Prediction validation",
) -> None:
    console.rule(title)
    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column()
    table.add_column()
    table.add_column()

    _add_metric_rows(table, "RAM", comparison.ram)
    _add_metric_rows(table, "VRAM", comparison.vram)

    compat = comparison.compatibility
    table.add_row("", "", "", "")
    table.add_row("Compatibility", "", "", "")
    table.add_row("Predicted", _runnable(compat.predicted_runnable), "", "")
    table.add_row("Observed", "success" if compat.observed_success else "failure", "", "")
    table.add_row("Result", compat.outcome.value, "", "")
    if compat.failure_reason is not None:
        table.add_row("Reason", compat.failure_reason.value, "", "")

    console.print(table)


def _add_metric_rows(table: Table, label: str, metric: MetricComparison) -> None:
    table.add_row(label, "", "", "")
    table.add_row("Estimated", _metric_bytes(metric.predicted_bytes), "", "")
    table.add_row("Measured", _metric_bytes(metric.measured_bytes), "", "")
    table.add_row("Difference", _metric_signed_bytes(metric.error_bytes), "", "")
    table.add_row("Error", _metric_percent(metric.error_percent), "", "")
    if metric.unavailable_reason is not None:
        table.add_row("Availability", metric.availability.value, "", "")


def _metric_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return format_bytes(value)


def _metric_percent(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:+.1f}%"


def _metric_signed_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value == 0:
        return format_bytes(0)
    sign = "+" if value > 0 else "-"
    return f"{sign}{format_bytes(abs(value))}"


def _runnable(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "runnable" if value else "not runnable"


__all__ = ["render_prediction_comparison"]
