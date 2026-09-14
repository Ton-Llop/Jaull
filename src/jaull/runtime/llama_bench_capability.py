"""Inspect whether a llama-bench binary is available."""

from __future__ import annotations

import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from jaull.domain.benchmarks import (
    BenchmarkObservation,
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.domain.runtime import RuntimeResolutionStatus
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol
from jaull.runtime.locator import RuntimeLocator, RuntimeLocatorConfig

_LLAMA_BENCH = "llama-bench"
_VERSION_SOURCE = "llama-bench --version"
_BUILD_LINE = re.compile(r"^build:\s*(.+?)\s*$", re.MULTILINE)


def inspect_llama_bench(
    *,
    backend: ExecutionBackendProtocol,
    llama_bench_path: str | Path | None = None,
    timeout_seconds: float = 10.0,
    allow_empty_workload_probe: bool = False,
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
        capability = LlamaBenchCapability(
            binary_path=resolved.binary_path,
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            probe_source=_VERSION_SOURCE,
            message=_probe_failure_message(str(exc), stdout=stdout, stderr=stderr),
        )
        if allow_empty_workload_probe and isinstance(exc, ExecutionFailedError):
            return _probe_empty_workload_build(capability, backend, timeout_seconds)
        return capability
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

    version = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return LlamaBenchCapability(
        binary_path=resolved.binary_path,
        binary_status=LlamaBenchBinaryStatus.AVAILABLE,
        version_text=version or None,
        probe_source=_VERSION_SOURCE,
    )


def resolve_llama_bench_binary(
    path: str | Path | None,
) -> LlamaBenchCapability:
    installation = RuntimeLocator(
        config=RuntimeLocatorConfig(llama_bench_path=path),
    ).resolve_llama_cpp()
    if installation.llama_bench is None:
        return LlamaBenchCapability(
            binary_status=LlamaBenchBinaryStatus.MISSING,
            message=installation.discovery.message or "llama-bench executable was not found.",
        )
    candidate = Path(installation.llama_bench)
    if installation.status in {
        RuntimeResolutionStatus.CONFIGURED_RUNTIME_MISSING,
        RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
    }:
        return LlamaBenchCapability(
            binary_path=str(candidate),
            binary_status=LlamaBenchBinaryStatus.MISSING,
            message=installation.discovery.message
            or f"llama-bench executable not found at {candidate}.",
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


def _probe_empty_workload_build(
    capability: LlamaBenchCapability,
    backend: ExecutionBackendProtocol,
    timeout_seconds: float,
) -> LlamaBenchCapability:
    """Request only the build footer, with no prompt/generation/combined tasks.

    A nonexistent local model protects against accidental loading if an older
    binary does not honor the empty workload. No Hub arguments are supplied.
    Backends may be initialized, but no model or context is needed for this probe.
    """
    assert capability.binary_path is not None
    try:
        with TemporaryDirectory(prefix="jaull-bench-version-") as directory:
            result = backend.execute(
                ExecutionRequest(
                    command=(
                        capability.binary_path,
                        "-m",
                        str(Path(directory) / "absent.gguf"),
                        "-p",
                        "0",
                        "-n",
                        "0",
                        "-pg",
                        "0,0",
                        "-o",
                        "md",
                    ),
                    timeout_seconds=timeout_seconds,
                )
            )
    except (ExecutionError, OSError):
        return capability
    if result.exit_code != 0:
        return capability
    matches = _BUILD_LINE.findall("\n".join((result.stdout, result.stderr)))
    if not matches:
        return capability
    return capability.model_copy(
        update={
            "version_text": f"build: {matches[-1]}",
            "probe_source": "llama-bench empty workload build footer",
        }
    )


def enrich_capability_from_benchmark_output(
    capability: LlamaBenchCapability,
    observation: BenchmarkObservation,
) -> LlamaBenchCapability:
    """Preserve a build emitted by successful llama-bench output.

    Some llama-bench builds reject ``--version`` yet append a stable
    ``build: <hash> (<number>)`` line to every successful benchmark. That is
    stronger provenance than a failed version probe and needs no second command.
    """

    if capability.version_text is not None:
        return capability
    matches = _BUILD_LINE.findall("\n".join((observation.raw_stdout, observation.raw_stderr)))
    if not matches:
        return capability
    build = matches[-1]
    return capability.model_copy(
        update={
            "version_text": f"build: {build}",
            "probe_source": "llama-bench benchmark output",
        }
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


__all__ = [
    "enrich_capability_from_benchmark_output",
    "inspect_llama_bench",
    "resolve_llama_bench_binary",
]
