"""Runtime runner for a verified local GGUF artifact through llama-cli."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import ExecutionRequest, InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.exceptions import JaullError
from jaull.execution.errors import ExecutableNotFoundError
from jaull.execution.ports import ExecutionBackendProtocol

_DEFAULT_CONTEXT_SIZE = 4096
_DEFAULT_GPU_LAYERS = 0
_LLAMA_CLI = "llama-cli"


class LlamaCppRunnerError(JaullError):
    """Base class for llama.cpp runner failures."""


class InvalidLlamaCppArtifactError(LlamaCppRunnerError):
    """The supplied artifact cannot be executed by this runner."""


class InvalidPromptError(LlamaCppRunnerError):
    """The supplied inference prompt is empty."""


@dataclass(frozen=True)
class LlamaCppRunner:
    backend: ExecutionBackendProtocol
    llama_cli_path: str | Path | None = None
    timeout_seconds: float = 300.0
    _llama_cli: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_llama_cli", _resolve_llama_cli(self.llama_cli_path))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")

    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        model_path = _validate_artifact(artifact)
        if not prompt.strip():
            raise InvalidPromptError("Prompt must not be empty.")

        ctx_size = _int_flag(runtime, "--ctx-size", default=_DEFAULT_CONTEXT_SIZE)
        n_gpu_layers = _int_flag(
            runtime, "--n-gpu-layers", default=_DEFAULT_GPU_LAYERS
        )
        command = (
            self._llama_cli,
            "--model",
            str(model_path),
            "--ctx-size",
            str(ctx_size),
            "--n-gpu-layers",
            str(n_gpu_layers),
            "--prompt",
            prompt,
        )

        result = self.backend.execute(
            ExecutionRequest(command=command, timeout_seconds=self.timeout_seconds)
        )
        return InferenceResult(
            text=result.stdout,
            duration_seconds=result.duration_seconds,
            exit_code=result.exit_code,
            runtime=RuntimeName.LLAMA_CPP.value,
            model_path=model_path,
        )


def _resolve_llama_cli(configured: str | Path | None) -> str:
    if configured is None:
        found = shutil.which(_LLAMA_CLI)
        if found is None:
            raise ExecutableNotFoundError(
                "llama-cli executable not found. Install or build llama.cpp and "
                "ensure llama-cli is available in PATH."
            )
        return found

    configured_str = str(configured)
    configured_path = Path(configured).expanduser()
    if configured_path.parent != Path("."):
        if not configured_path.is_file():
            raise ExecutableNotFoundError(
                f"llama-cli executable not found at {configured_path}."
            )
        return str(configured_path)

    found = shutil.which(configured_str)
    if found is None:
        raise ExecutableNotFoundError(
            f"llama-cli executable not found: {configured_str!r}. Install or build "
            "llama.cpp and ensure llama-cli is available in PATH."
        )
    return found


def _validate_artifact(artifact: ModelArtifact) -> Path:
    if artifact.format.lower() != "gguf":
        raise InvalidLlamaCppArtifactError(
            f"llama.cpp runner only supports GGUF artifacts, got {artifact.format!r}."
        )
    if artifact.local_path is None:
        raise InvalidLlamaCppArtifactError("Artifact has no local_path.")
    if not artifact.is_downloaded:
        raise InvalidLlamaCppArtifactError("Artifact must be downloaded before run.")
    if not artifact.is_verified:
        raise InvalidLlamaCppArtifactError("Artifact must be verified before run.")
    if not artifact.local_path.is_file():
        raise InvalidLlamaCppArtifactError(
            f"Artifact file does not exist: {artifact.local_path}."
        )
    return artifact.local_path


def _int_flag(
    runtime: RuntimeRecommendation | None, name: str, *, default: int
) -> int:
    if runtime is None:
        return default
    if runtime.runtime is not RuntimeName.LLAMA_CPP:
        raise LlamaCppRunnerError(
            f"Runtime recommendation must be {RuntimeName.LLAMA_CPP.value!r}, "
            f"got {runtime.runtime.value!r}."
        )
    raw = next((flag.value for flag in runtime.flags if flag.name == name), None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LlamaCppRunnerError(
            f"Runtime flag {name} must be an integer, got {raw!r}."
        ) from exc


__all__ = [
    "InvalidLlamaCppArtifactError",
    "InvalidPromptError",
    "LlamaCppRunner",
    "LlamaCppRunnerError",
]
