"""Runtime runner for Transformers/PyTorch through an isolated Python worker."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import ExecutionRequest, InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
)
from jaull.execution.ports import ExecutionBackendProtocol

_DEFAULT_DEVICE_MAP = "cpu"
_DEFAULT_MAX_NEW_TOKENS = 64


class TransformersRunnerError(ExecutionError):
    """Base class for Transformers runner failures."""


class InvalidTransformersArtifactError(TransformersRunnerError):
    """The supplied model reference cannot be executed by this runner."""


class InvalidTransformersPromptError(TransformersRunnerError):
    """The supplied inference prompt is empty."""


@dataclass(frozen=True)
class TransformersRunner:
    backend: ExecutionBackendProtocol
    python_executable: str | Path | None = None
    timeout_seconds: float = 300.0
    _python: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_python", _resolve_python(self.python_executable))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")

    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        if not prompt.strip():
            raise InvalidTransformersPromptError("Prompt must not be empty.")
        model_ref = _model_ref(artifact)
        device_map = _flag(runtime, "device_map", default=_DEFAULT_DEVICE_MAP)
        command: tuple[str, ...] = (
            self._python,
            "-m",
            "jaull.runtime.transformers_worker",
            "--model-ref",
            model_ref,
            "--prompt",
            prompt,
            "--device-map",
            device_map or _DEFAULT_DEVICE_MAP,
            "--max-new-tokens",
            str(_int_flag(runtime, "max_new_tokens", default=_DEFAULT_MAX_NEW_TOKENS)),
        )
        command = _append_optional(command, "--revision", artifact.revision)
        torch_dtype = _flag(runtime, "torch_dtype", default=None)
        if torch_dtype is not None:
            command = _append_optional(command, "--torch-dtype", torch_dtype)

        try:
            result = self.backend.execute(
                ExecutionRequest(command=command, timeout_seconds=self.timeout_seconds)
            )
        except ExecutionFailedError as exc:
            raise _worker_failed_error(exc) from exc
        payload = _parse_worker_payload(result.stdout)
        if payload.get("success") is False:
            error = str(payload.get("error") or "Transformers worker failed.")
            raise TransformersRunnerError(error, result.observation)
        text = str(payload.get("text") or "")
        return InferenceResult(
            text=text.strip(),
            runtime=RuntimeName.TRANSFORMERS.value,
            model_path=Path(model_ref),
            observation=result.observation,
        )


def _resolve_python(configured: str | Path | None) -> str:
    if configured is None:
        return sys.executable

    configured_str = str(configured)
    configured_path = Path(configured).expanduser()
    if configured_path.parent != Path("."):
        if not configured_path.is_file():
            raise ExecutableNotFoundError(
                f"Python executable not found at {configured_path}."
            )
        return str(configured_path)

    found = shutil.which(configured_str)
    if found is None:
        raise ExecutableNotFoundError(
            f"Python executable not found: {configured_str!r}."
        )
    return found


def _model_ref(artifact: ModelArtifact) -> str:
    if artifact.format.lower() in {"gguf", "onnx", "adapter"}:
        raise InvalidTransformersArtifactError(
            "Transformers runner requires a Transformers/PyTorch model reference, "
            f"got format {artifact.format!r}."
        )
    if artifact.local_path is not None:
        if artifact.local_path.is_dir():
            return str(artifact.local_path)
        if artifact.local_path.exists():
            raise InvalidTransformersArtifactError(
                "Transformers runner needs a repository or local model directory, "
                f"not a single file: {artifact.local_path}."
            )
    if not artifact.repo_id.strip():
        raise InvalidTransformersArtifactError("Artifact repo_id must not be empty.")
    return artifact.repo_id


def _flag(
    runtime: RuntimeRecommendation | None,
    name: str,
    *,
    default: str | None,
) -> str | None:
    if runtime is None:
        return default
    if runtime.runtime is not RuntimeName.TRANSFORMERS:
        raise TransformersRunnerError(
            f"Runtime recommendation must be {RuntimeName.TRANSFORMERS.value!r}, "
            f"got {runtime.runtime.value!r}."
        )
    return next((flag.value for flag in runtime.flags if flag.name == name), default)


def _int_flag(
    runtime: RuntimeRecommendation | None, name: str, *, default: int
) -> int:
    raw = _flag(runtime, name, default=str(default))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise TransformersRunnerError(
            f"Runtime flag {name} must be an integer, got {raw!r}."
        ) from exc


def _append_optional(
    command: tuple[str, ...],
    name: str,
    value: str | None,
) -> tuple[str, ...]:
    if value is None or not value.strip():
        return command
    return (*command, name, value)


def _parse_worker_payload(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return dict(payload)
    raise TransformersRunnerError("Transformers worker did not emit JSON output.")


def _worker_failed_error(exc: ExecutionFailedError) -> TransformersRunnerError:
    payload = _parse_optional_worker_payload(exc.result.stdout)
    if payload is not None and payload.get("success") is False:
        detail = str(payload.get("error") or "").strip()
        if detail:
            return TransformersRunnerError(
                f"Transformers worker failed: {detail}",
                exc.observation,
            )

    output_detail = _last_nonempty_line(exc.result.stderr) or _last_nonempty_line(
        exc.result.stdout
    )
    if output_detail:
        return TransformersRunnerError(
            f"Transformers worker failed: {output_detail}",
            exc.observation,
        )
    return TransformersRunnerError(str(exc), exc.observation)


def _parse_optional_worker_payload(stdout: str) -> dict[str, object] | None:
    try:
        return _parse_worker_payload(stdout)
    except TransformersRunnerError:
        return None


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


__all__ = [
    "InvalidTransformersArtifactError",
    "InvalidTransformersPromptError",
    "TransformersRunner",
    "TransformersRunnerError",
]
