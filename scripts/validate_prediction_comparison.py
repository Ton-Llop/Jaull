#!/usr/bin/env python
"""Run one CPU-only TinyLlama prediction/observation/comparison validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from jaull.advisor.service import AdvisorService
from jaull.domain.comparison import PredictionComparison
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import InferenceResult
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.runtime import RuntimeRecommendation
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from jaull.evaluation.comparison import (
    assert_prediction_runtime_matches,
    compare_prediction,
)
from jaull.presentation.comparison_report import render_prediction_comparison
from jaull.presentation.console import format_bytes, make_console
from jaull.presentation.execution_report import render_execution_observation

DEFAULT_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_CONTEXT = 4096
DEFAULT_PROMPT = "In one short sentence, explain what GGUF is."


def main() -> int:
    args = _parse_args()
    console = make_console()
    advisor = AdvisorService.default(
        llama_cli_path=args.llama_cli,
        llama_cli_timeout_seconds=args.timeout_seconds,
    )

    console.print("[bold]Preparing prediction[/bold]")
    hardware = advisor.scan_hardware()
    analysis = advisor.inspect_model(args.model)
    inference_cfg = InferenceConfiguration(
        context_length=args.context,
        batch_size=1,
        target_device=TargetDevice.CPU,
        quantization=args.quantization,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
    )
    estimate = advisor.estimate_model(
        analysis,
        hardware,
        inference_cfg,
        recommend_runtime=True,
    )
    runtime = _runtime_or_raise(estimate)
    assert_prediction_runtime_matches(estimate=estimate, runtime=runtime)
    _assert_cpu_only_runtime(runtime)

    console.print("[bold]Preparing artifact[/bold]")
    artifact = advisor.resolve_artifact(args.model, quantization=args.quantization)
    if not artifact.is_downloaded:
        artifact = advisor.download_artifact(artifact)
    artifact = advisor.verify_artifact(artifact, full=args.full_verify)

    console.print("[bold]Running model[/bold]")
    result = advisor.run_artifact(
        artifact=artifact,
        prompt=args.prompt,
        runtime=runtime,
    )
    assert_prediction_runtime_matches(estimate=estimate, runtime=runtime)
    comparison = compare_prediction(
        estimate=estimate,
        observation=result.observation,
        runtime=runtime,
    )

    _render_result(
        console=console,
        estimate=estimate,
        runtime=runtime,
        result=result,
        comparison=comparison,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PredictionComparison with a real CPU-only GGUF run."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--full-verify", action="store_true")
    return parser.parse_args()


def _runtime_or_raise(estimate: MemoryEstimate) -> RuntimeRecommendation:
    runtime = estimate.runtime_recommendation
    if runtime is None:
        raise RuntimeError("MemoryEstimate did not produce a RuntimeRecommendation.")
    return runtime


def _assert_cpu_only_runtime(runtime: RuntimeRecommendation) -> None:
    n_gpu_layers = _runtime_flag(runtime, "--n-gpu-layers")
    if n_gpu_layers != "0":
        raise RuntimeError(
            "Expected CPU-only RuntimeRecommendation with --n-gpu-layers 0, "
            f"got {n_gpu_layers!r}."
        )


def _runtime_flag(runtime: RuntimeRecommendation, name: str) -> str | None:
    for flag in runtime.flags:
        if flag.name == name:
            return flag.value
    return None


def _render_result(
    *,
    console: Console,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation,
    result: InferenceResult,
    comparison: PredictionComparison,
) -> None:
    console.rule("Prediction validation")
    _render_model(console, estimate, runtime)
    _render_prediction(console, estimate)
    render_execution_observation(
        console,
        result.observation,
        title="Observation",
        include_measurement=True,
    )
    console.print()
    render_prediction_comparison(console, comparison)
    console.print()
    console.rule("Model response")
    console.print(result.text or "[dim](empty response)[/dim]")


def _render_model(
    console: Console,
    estimate: MemoryEstimate,
    runtime: RuntimeRecommendation,
) -> None:
    table = Table(title="Model", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Repository", estimate.repository.repo_id)
    table.add_row("Quantization", estimate.inference_configuration.quantization or "")
    table.add_row("Context", str(estimate.inference_configuration.context_length))
    table.add_row("Target", estimate.inference_configuration.target_device.value)
    table.add_row("Runtime", runtime.runtime.value)
    table.add_row("GPU layers", _runtime_flag(runtime, "--n-gpu-layers") or "default")
    console.print(table)
    console.print()


def _render_prediction(console: Console, estimate: MemoryEstimate) -> None:
    process_ram = _predicted_process_ram(estimate)
    table = Table(title="Prediction", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Weights", format_bytes(estimate.weights.component.bytes))
    table.add_row("KV cache", format_bytes(estimate.kv_cache.component.bytes))
    table.add_row(
        "Runtime overhead",
        format_bytes(estimate.runtime_overhead.component.bytes),
    )
    table.add_row("Process RAM", format_bytes(process_ram))
    table.add_row("Device reserve", format_bytes(estimate.device_reserve.bytes))
    table.add_row(
        "Safety margin",
        format_bytes(estimate.safety_margin.bytes if estimate.safety_margin else 0),
    )
    table.add_row("Total required", format_bytes(estimate.total_bytes))
    console.print(table)
    console.print(
        "[dim]Device reserve, safety margin and total required are shown for "
        "context, but are excluded from the RSS comparison.[/dim]"
    )
    console.print()


def _predicted_process_ram(estimate: MemoryEstimate) -> int | None:
    parts = [
        estimate.weights.component.bytes,
        estimate.kv_cache.component.bytes,
        estimate.runtime_overhead.component.bytes,
    ]
    if any(part is None for part in parts):
        return None
    return sum(part for part in parts if part is not None)


if __name__ == "__main__":
    raise SystemExit(main())
