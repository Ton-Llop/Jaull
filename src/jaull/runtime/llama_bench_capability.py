"""Inspect whether a llama-bench binary is available."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from jaull.domain.benchmarks import (
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol

_LLAMA_BENCH = "llama-bench"
_VERSION_SOURCE = "llama-bench --version"


def inspect_llama_bench(
    *,
    backend: ExecutionBackendProtocol,
    llama_bench_path: str | Path | None = None,
    timeout_seconds: float = 10.0,
) -> LlamaBenchCapability:
    resolved = resolve_llama_bench_binary(llama_bench_path)
    if resolved.binary_status is not LlamaBenchBinaryStatus.UNKNOWN:
        return resolved
    assert resolved.binary_path is not None

    try:
        result = backend.execute(
            ExecutionRequest(
                command=(resolved.binary_path, "--version"),
                timeout_seconds=timeout_seconds,
            )
        )
    except ExecutableNotFoundError as exc:
        return LlamaBenchCapability(
            binary_path=resolved.binary_path,
            binary_status=LlamaBenchBinaryStatus.MISSING,
            message=str(exc),
        )
    except (ExecutionFailedError, ExecutionTimeoutError) as exc:
        # Some llama-bench builds do not support --version even though the
        # benchmark command itself runs correctly. Path resolution proves the
        # binary is present; keep version/probe failure as metadata instead of
        # blocking a real benchmark.
        stdout = exc.result.stdout if exc.result is not None else ""
        stderr = exc.result.stderr if exc.result is not None else ""
        return LlamaBenchCapability(
            binary_path=resolved.binary_path,
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            probe_source=_VERSION_SOURCE,
            message=_probe_failure_message(str(exc), stdout=stdout, stderr=stderr),
        )
    except ExecutionError as exc:
        status = (
            LlamaBenchBinaryStatus.NOT_EXECUTABLE
            if exc.observation is not None
            and exc.observation.failure_reason is ExecutionFailureReason.SPAWN_ERROR
            else LlamaBenchBinaryStatus.PROBE_FAILED
        )
        return LlamaBenchCapability(
            binary_path=resolved.binary_path,
            binary_status=status,
            probe_source=_VERSION_SOURCE,
            message=str(exc),
        )

    version = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )
    return LlamaBenchCapability(
        binary_path=resolved.binary_path,
        binary_status=LlamaBenchBinaryStatus.AVAILABLE,
        version_text=version or None,
        probe_source=_VERSION_SOURCE,
    )


def resolve_llama_bench_binary(
    path: str | Path | None,
) -> LlamaBenchCapability:
    if path is None:
        resolved = shutil.which(_LLAMA_BENCH)
        if resolved is None:
            return LlamaBenchCapability(
                binary_status=LlamaBenchBinaryStatus.MISSING,
                message=(
                    "llama-bench executable not found. Install or build llama.cpp "
                    "and ensure llama-bench is available in PATH."
                ),
            )
        return LlamaBenchCapability(
            binary_path=resolved,
            binary_status=LlamaBenchBinaryStatus.UNKNOWN,
        )

    configured = str(path)
    candidate = Path(path).expanduser()
    if candidate.parent == Path("."):
        resolved = shutil.which(configured)
        if resolved is None:
            return LlamaBenchCapability(
                binary_status=LlamaBenchBinaryStatus.MISSING,
                message=(
                    f"llama-bench executable not found: {configured!r}. Install "
                    "or build llama.cpp and ensure llama-bench is available in PATH."
                ),
            )
        return LlamaBenchCapability(
            binary_path=resolved,
            binary_status=LlamaBenchBinaryStatus.UNKNOWN,
        )

    if not candidate.exists():
        return LlamaBenchCapability(
            binary_path=str(candidate),
            binary_status=LlamaBenchBinaryStatus.MISSING,
            message=f"llama-bench executable not found at {candidate}.",
        )
    if candidate.is_dir() or (os.name != "nt" and not os.access(candidate, os.X_OK)):
        return LlamaBenchCapability(
            binary_path=str(candidate),
            binary_status=LlamaBenchBinaryStatus.NOT_EXECUTABLE,
            message=f"llama-bench is not executable at {candidate}.",
        )
    return LlamaBenchCapability(
        binary_path=str(candidate),
        binary_status=LlamaBenchBinaryStatus.UNKNOWN,
    )


def _probe_failure_message(message: str, *, stdout: str, stderr: str) -> str:
    parts = [message]
    stderr_line = _last_useful_line(stderr)
    stdout_line = _last_useful_line(stdout)
    if stderr_line:
        parts.append(f"stderr: {stderr_line}")
    if stdout_line:
        parts.append(f"stdout: {stdout_line}")
    return "\n".join(parts)


def _last_useful_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


__all__ = ["inspect_llama_bench", "resolve_llama_bench_binary"]
