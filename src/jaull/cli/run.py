"""``run`` subcommand: execute a verified GGUF artifact with llama-cli."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionOverrides, ExecutionPlanningError
from jaull.artifacts.errors import ArtifactError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
)
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import RuntimeName
from jaull.exceptions import (
    InvalidModelReferenceError,
    JaullError,
    QuantizationNotFoundError,
)
from jaull.execution.errors import ExecutionError
from jaull.huggingface.url_parser import normalize_repo_id
from jaull.presentation.console import make_console
from jaull.presentation.execution_report import (
    render_execution_observation,
    render_inference_result,
)

_DEFAULT_CONTEXT_SIZE = 4096


@dataclass(frozen=True)
class RunOptions:
    quantization: str | None
    prompt: str
    revision: str | None = None
    llama_cli_path: str | Path | None = None
    #: ``None`` => use the automatic default (4096) for estimation and never
    #: mark it as a user input. An explicit value is preserved and flagged
    #: ``USER_INPUT``.
    context_size: int | None = None
    #: ``None`` => AUTO (run the launch policy). ``0`` => explicit CPU-only.
    #: ``N`` / ``-1`` => explicit override, preserved exactly.
    n_gpu_layers: int | None = None
    timeout_seconds: float = 300.0
    full_verify: bool = False


def run_model(
    reference: str,
    options: RunOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default(
        llama_cli_path=options.llama_cli_path,
        llama_cli_timeout_seconds=options.timeout_seconds,
    )

    try:
        repo_id = normalize_repo_id(reference)
        artifact = resolved.resolve_artifact(
            repo_id,
            quantization=options.quantization,
            revision=options.revision,
        )
        if not artifact.is_downloaded:
            artifact = resolved.download_artifact(artifact)
        artifact = resolved.verify_artifact(artifact, full=options.full_verify)

        plan = _plan_for_run(resolved, repo_id, artifact, options)
        result = resolved.run_artifact(
            artifact=artifact,
            prompt=options.prompt,
            runtime=plan.runtime,
        )
    except InvalidModelReferenceError as exc:
        _print_error(console, exc)
        return 2
    except QuantizationNotFoundError as exc:
        _print_error(console, exc)
        return 3
    except ArtifactError as exc:
        _print_error(console, exc)
        return 4
    except ExecutionError as exc:
        _print_error(console, exc)
        if exc.observation is not None:
            console.print()
            render_execution_observation(
                console,
                exc.observation,
                title="Execution failed",
            )
        return 5
    except ExecutionPlanningError as exc:
        _print_error(console, exc)
        return 6

    render_inference_result(console, result)
    return 0


def _plan_for_run(
    advisor: AdvisorService,
    repo_id: str,
    artifact: ModelArtifact,
    options: RunOptions,
) -> ExecutionPlan:
    identity = ModelIdentity(model_name=repo_id.split("/")[-1])
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id=repo_id,
        revision=artifact.revision,
        format=ArtifactVariantFormat.GGUF,
        filename=artifact.filename,
        size_bytes=artifact.size_bytes,
        quantization=artifact.quantization,
        source="cli",
        compatible_runtimes=[RuntimeName.LLAMA_CPP],
    )

    if options.n_gpu_layers is not None:
        # Explicit override: keep today's estimation-free fast path.
        return advisor.plan_execution(
            model_identity=identity,
            artifact=variant,
            runtime=RuntimeName.LLAMA_CPP,
            estimate=None,
            hardware=None,
            overrides=ExecutionOverrides(
                context_size=options.context_size,
                n_gpu_layers=options.n_gpu_layers,
            ),
        )

    # AUTO: scan, inspect, estimate, then run the launch policy. A failure here
    # is a hard error -- CPU-only must be an explicit user choice, never the
    # fallback for planning that broke.
    try:
        hardware = advisor.scan_hardware()
        analysis = advisor.inspect_model(repo_id)
        estimate = advisor.estimate_model(
            analysis,
            hardware,
            InferenceConfiguration(
                context_length=options.context_size or _DEFAULT_CONTEXT_SIZE,
                quantization=options.quantization,
            ),
            recommend_runtime=True,
        )
    except JaullError as exc:
        raise ExecutionPlanningError(
            f"Could not resolve automatic GPU placement: {exc}. Re-run with "
            "--n-gpu-layers 0 for CPU-only, or --n-gpu-layers N to choose the "
            "offload explicitly."
        ) from exc

    return advisor.plan_execution(
        model_identity=identity,
        artifact=variant,
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=hardware,
        overrides=ExecutionOverrides(context_size=options.context_size),
    )


def _print_error(console: Console, exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")


__all__ = ["RunOptions", "run_model"]
