"""Experiment inspection and offline re-evaluation commands."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.domain.comparison import MetricComparison
from jaull.domain.experiments import ExperimentReevaluationResult
from jaull.experiments.errors import ExperimentStoreError
from jaull.presentation.console import format_bytes, make_console


@dataclass(frozen=True)
class ReevaluateOptions:
    as_json: bool = False


def run_reevaluate(
    experiment_id: str,
    options: ReevaluateOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        result = resolved.reevaluate_experiment(experiment_id)
    except ExperimentStoreError as exc:
        _print_error(console, exc, options.as_json)
        return 3

    if options.as_json:
        sys.stdout.write(json.dumps(result.model_dump(mode="json"), indent=2))
        sys.stdout.write("\n")
        return 0

    _render_reevaluation(result, console)
    return 0


def _render_reevaluation(result: ExperimentReevaluationResult, console: Console) -> None:
    console.print(f"[bold]Experiment[/bold] {result.experiment_id}")
    console.print(f"Mode: {result.mode.value}")
    console.print(f"Replayability: {result.replayability.status.value}")
    for reason in result.replayability.reasons:
        console.print(f"- {reason}")
    for warning in result.replayability.warnings:
        console.print(f"Warning: {warning}")

    if result.current_prediction is None or result.current_comparison is None:
        return

    console.print("\n[bold]Original prediction[/bold]")
    if result.original_prediction is None:
        console.print("Unavailable")
    else:
        console.print(f"Memory: {format_bytes(result.original_prediction.total_bytes)}")
        console.print(
            f"Compatibility: {result.original_prediction.assessment.status.value}"
        )

    console.print("\n[bold]Current prediction[/bold]")
    console.print(f"Memory: {format_bytes(result.current_prediction.total_bytes)}")
    console.print(f"Compatibility: {result.current_prediction.assessment.status.value}")

    console.print("\n[bold]Recorded observation[/bold]")
    observation = result.observation
    if observation is not None:
        console.print(f"Success: {'yes' if observation.success else 'no'}")
        console.print(f"Peak RAM: {format_bytes(observation.peak_ram_bytes)}")
        console.print(f"Peak VRAM: {format_bytes(observation.peak_vram_bytes)}")

    console.print("\n[bold]Current comparison[/bold]")
    console.print(f"RAM: {_metric_summary(result.current_comparison.ram)}")
    console.print(f"VRAM: {_metric_summary(result.current_comparison.vram)}")
    console.print(
        "Compatibility: "
        f"{result.current_comparison.compatibility.outcome.value}"
    )


def _metric_summary(metric: MetricComparison) -> str:
    if metric.error_percent is None:
        return metric.availability.value
    return (
        f"{metric.error_percent:+.1f}% "
        f"({format_bytes(metric.predicted_bytes)} predicted, "
        f"{format_bytes(metric.measured_bytes)} measured)"
    )


def _print_error(console: Console, exc: Exception, as_json: bool) -> None:
    message = str(exc)
    if as_json:
        sys.stderr.write(json.dumps({"schema_version": 1, "error": message}) + "\n")
    else:
        console.print(f"[red]{message}[/red]")


__all__ = ["ReevaluateOptions", "run_reevaluate"]
