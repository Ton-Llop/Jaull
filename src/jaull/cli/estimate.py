"""``estimate`` subcommand: memory footprint + compatibility for a model."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.application.model_reference import normalize_repo_id
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelAnalysis
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from jaull.exceptions import (
    EstimationError,
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    ModelAccessDeniedError,
    ModelNotFoundError,
    QuantizationNotFoundError,
)
from jaull.presentation.console import make_console
from jaull.presentation.estimation_report import render_context_sweep, render_estimate
from jaull.reporting.context_sweep import context_sweep_to_json_dict
from jaull.reporting.estimation import estimate_to_json_dict

_DEFAULT_CONTEXT_FALLBACK = 4096
_GIB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class EstimateOptions:
    quantization: str | None = None
    dtype: WeightPrecision | None = None
    context: int | None = None
    batch_size: int = 1
    device: TargetDevice = TargetDevice.AUTO
    kv_dtype: WeightPrecision = WeightPrecision.FLOAT16
    safety_margin_percent: float = SAFETY_MARGIN_DEFAULT_PERCENT
    device_reserve_gib: float = DEVICE_RESERVE_DEFAULT_BYTES / _GIB
    as_json: bool = False
    resolve_base_model: bool = True
    recommend_runtime: bool = True
    context_sweep: bool = False


def run_estimate(
    reference: str,
    options: EstimateOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    if options.context_sweep and options.context is not None:
        _print_error(
            console,
            ValueError(
                "--context-sweep tries several contexts: do not combine it with --context."
            ),
            options.as_json,
        )
        return 2
    resolved = advisor or AdvisorService.default()

    try:
        repo_id = normalize_repo_id(reference)
    except InvalidModelReferenceError as exc:
        _print_error(console, exc, options.as_json)
        return 2

    try:
        analysis = resolved.inspect_model(repo_id)
    except ModelNotFoundError as exc:
        _print_error(console, exc, options.as_json)
        return 3
    except ModelAccessDeniedError as exc:
        _print_error(console, exc, options.as_json)
        if not options.as_json:
            console.print(
                "[dim]Set HF_TOKEN in your environment if you have been granted access.[/dim]"
            )
        return 4
    except HuggingFaceUnavailableError as exc:
        _print_error(console, exc, options.as_json)
        return 5

    inference_cfg = _build_configuration(
        analysis_config_context=_config_context(analysis),
        options=options,
    )

    hardware = resolved.scan_hardware()

    if options.context_sweep:
        return _run_context_sweep(
            resolved, repo_id, analysis, hardware, inference_cfg, options, console
        )

    try:
        estimate = resolved.estimate_model(
            analysis=analysis,
            hardware=hardware,
            inference_cfg=inference_cfg,
            resolve_base_model=options.resolve_base_model,
            recommend_runtime=options.recommend_runtime,
        )
    except QuantizationNotFoundError as exc:
        _print_error(console, exc, options.as_json)
        return 7
    except EstimationError as exc:
        _print_error(console, exc, options.as_json)
        return 8

    if options.as_json:
        payload = estimate_to_json_dict(estimate)
        sys.stdout.write(json.dumps(payload, indent=2, default=str))
        sys.stdout.write("\n")
        return 0

    render_estimate(estimate, console)
    return 0


def _run_context_sweep(
    advisor: AdvisorService,
    repo_id: str,
    analysis: ModelAnalysis,
    hardware: HardwareProfile,
    inference_cfg: InferenceConfiguration,
    options: EstimateOptions,
    console: Console,
) -> int:
    try:
        sweep = advisor.sweep_context_for_model(
            analysis,
            hardware,
            inference_cfg,
            resolve_base_model=options.resolve_base_model,
        )
    except QuantizationNotFoundError as exc:
        _print_error(console, exc, options.as_json)
        return 7
    except EstimationError as exc:
        _print_error(console, exc, options.as_json)
        return 8

    if options.as_json:
        sys.stdout.write(json.dumps(context_sweep_to_json_dict(repo_id, sweep), indent=2))
        sys.stdout.write("\n")
        return 0

    render_context_sweep(repo_id, sweep, console)
    return 0


def _config_context(analysis: ModelAnalysis) -> int | None:
    if analysis.config and analysis.config.max_position_embeddings:
        return int(analysis.config.max_position_embeddings)
    return None


def _build_configuration(
    analysis_config_context: int | None, options: EstimateOptions
) -> InferenceConfiguration:
    context = options.context or analysis_config_context or _DEFAULT_CONTEXT_FALLBACK
    reserve_bytes = int(options.device_reserve_gib * _GIB)
    return InferenceConfiguration(
        context_length=context,
        batch_size=options.batch_size,
        target_device=options.device,
        precision=options.dtype,
        quantization=options.quantization,
        kv_cache_dtype=options.kv_dtype,
        safety_margin_percent=options.safety_margin_percent,
        device_reserve_bytes=reserve_bytes,
    )


def _print_error(console: Console, exc: Exception, as_json: bool) -> None:
    message = str(exc)
    if as_json:
        payload = {"schema_version": 1, "error": {"message": message}}
        sys.stderr.write(json.dumps(payload) + "\n")
    else:
        console.print(f"[red]{message}[/red]")


__all__ = ["EstimateOptions", "run_estimate"]
