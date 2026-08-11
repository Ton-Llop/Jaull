"""``run`` subcommand: execute a verified GGUF artifact with llama-cli."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.artifacts.errors import ArtifactError
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.runtime import (
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutionError
from jaull.huggingface.url_parser import normalize_repo_id
from jaull.presentation.console import make_console
from jaull.presentation.execution_report import (
    render_execution_observation,
    render_inference_result,
)


@dataclass(frozen=True)
class RunOptions:
    quantization: str | None
    prompt: str
    revision: str | None = None
    llama_cli_path: str | Path | None = None
    context_size: int = 4096
    n_gpu_layers: int = 0
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

        result = resolved.run_artifact(
            artifact=artifact,
            prompt=options.prompt,
            runtime=_runtime_from_options(options),
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

    render_inference_result(console, result)
    return 0


def _runtime_from_options(options: RunOptions) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        command_preview=None,
        python_snippet=None,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(options.context_size),
                source=RuntimeFlagSource.USER_INPUT,
                explanation="Context length passed to llama-cli.",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(options.n_gpu_layers),
                source=RuntimeFlagSource.USER_INPUT,
                explanation="Number of model layers to offload to GPU.",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _print_error(console: Console, exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")


__all__ = ["RunOptions", "run_model"]
