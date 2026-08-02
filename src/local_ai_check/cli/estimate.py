"""``estimate`` subcommand: memory footprint + compatibility for a model."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from rich.console import Console

from local_ai_check.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from local_ai_check.domain.model import ModelAnalysis
from local_ai_check.estimator import service
from local_ai_check.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from local_ai_check.exceptions import (
    EstimationError,
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    ModelAccessDeniedError,
    ModelNotFoundError,
    QuantizationNotFoundError,
)
from local_ai_check.hardware.detector import detect_hardware
from local_ai_check.huggingface.client import HfClient, HfClientProtocol
from local_ai_check.huggingface.repository import inspect_model
from local_ai_check.huggingface.url_parser import normalize_repo_id
from local_ai_check.metadata.range_reader import HttpRangeClient, HttpxRangeClient
from local_ai_check.presentation.console import make_console
from local_ai_check.presentation.estimation_report import (
    estimate_to_json_dict,
    render_estimate,
)

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


def run_estimate(
    reference: str,
    options: EstimateOptions,
    client: HfClientProtocol | None = None,
    range_client: HttpRangeClient | None = None,
) -> int:
    console = make_console()

    try:
        repo_id = normalize_repo_id(reference)
    except InvalidModelReferenceError as exc:
        _print_error(console, exc, options.as_json)
        return 2

    hf_client = client if client is not None else HfClient()

    try:
        analysis = inspect_model(repo_id, client=hf_client)
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

    hardware = detect_hardware()

    effective_range_client: HttpRangeClient | None = range_client
    if effective_range_client is None and options.resolve_base_model:
        effective_range_client = HttpxRangeClient()

    try:
        estimate = service.estimate_memory(
            analysis=analysis,
            hardware=hardware,
            inference_cfg=inference_cfg,
            client=hf_client,
            resolve_base_model=options.resolve_base_model,
            range_client=effective_range_client,
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
