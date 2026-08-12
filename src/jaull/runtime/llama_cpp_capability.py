"""Inspect observable capabilities of an existing ``llama-cli`` binary."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
)
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol

_LIST_DEVICES_SOURCE = "llama-cli --list-devices"
_VERSION_SOURCE = "llama-cli --version"
_DEVICE_LINE_RE = re.compile(
    r"^\s*(?P<prefix>CUDA|Vulkan|HIP|ROCm)(?P<index>\d+)\s*:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"^(?P<name>.*?)\s*\((?P<total>\d+)\s*MiB,\s*"
    r"(?P<free>\d+)\s*MiB\s+free\)\s*$",
    re.IGNORECASE,
)


def inspect_llama_cpp_runtime(
    *,
    backend: ExecutionBackendProtocol,
    llama_cli_path: str | Path | None = None,
    timeout_seconds: float = 10.0,
) -> LlamaCppRuntimeCapability:
    """Probe a configured ``llama-cli`` without executing a model.

    Backend capability is strictly observable: a CUDA/Vulkan/HIP capability is
    confirmed only when ``--list-devices`` exposes a matching runtime device.
    Missing devices are reported as not observed, not as proof that the binary
    was compiled without that backend.
    """

    resolved = _resolve_llama_cli(llama_cli_path)
    if resolved.status is not LlamaCppBinaryStatus.UNKNOWN:
        return LlamaCppRuntimeCapability(
            binary_path=resolved.path,
            binary_status=resolved.status,
            message=resolved.message,
        )

    assert resolved.path is not None
    command = (resolved.path, "--list-devices")
    try:
        result = backend.execute(
            ExecutionRequest(command=command, timeout_seconds=timeout_seconds)
        )
    except ExecutableNotFoundError as exc:
        return _failed_capability(
            path=resolved.path,
            status=LlamaCppBinaryStatus.MISSING,
            message=str(exc),
        )
    except (ExecutionFailedError, ExecutionTimeoutError) as exc:
        return _failed_capability(
            path=resolved.path,
            status=LlamaCppBinaryStatus.PROBE_FAILED,
            message=str(exc),
        )
    except ExecutionError as exc:
        status = (
            LlamaCppBinaryStatus.NOT_EXECUTABLE
            if exc.observation is not None
            and exc.observation.failure_reason is ExecutionFailureReason.SPAWN_ERROR
            else LlamaCppBinaryStatus.PROBE_FAILED
        )
        return _failed_capability(path=resolved.path, status=status, message=str(exc))

    devices = parse_llama_cpp_devices("\n".join((result.stdout, result.stderr)))
    version_text = _probe_version(
        backend=backend,
        binary_path=resolved.path,
        timeout_seconds=timeout_seconds,
    )
    return LlamaCppRuntimeCapability(
        binary_path=resolved.path,
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text=version_text,
        backend_capabilities=_build_backend_capabilities(devices),
        probe_source=_LIST_DEVICES_SOURCE,
    )


def evaluate_execution_readiness(
    *,
    selection: RuntimeBackendSelection,
    runtime_capability: LlamaCppRuntimeCapability,
) -> ExecutionReadiness:
    """Return a conservative preflight decision for the selected backend.

    ``READY`` means no blocking mismatch was found between the selected
    hardware backend and the devices exposed by this ``llama-cli`` probe. It
    does not demonstrate that an actual inference will succeed.
    """

    if runtime_capability.binary_status is LlamaCppBinaryStatus.MISSING:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.RUNTIME_MISSING,
            selection=selection,
            runtime_capability=runtime_capability,
            message="llama-cli executable was not found",
        )
    if runtime_capability.binary_status is LlamaCppBinaryStatus.NOT_EXECUTABLE:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.RUNTIME_NOT_EXECUTABLE,
            selection=selection,
            runtime_capability=runtime_capability,
            message="llama-cli exists but could not be executed",
        )
    if runtime_capability.binary_status is LlamaCppBinaryStatus.PROBE_FAILED:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.PROBE_FAILED,
            selection=selection,
            runtime_capability=runtime_capability,
            message="llama-cli capability probe failed",
        )
    if runtime_capability.binary_status is not LlamaCppBinaryStatus.AVAILABLE:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.CAPABILITY_UNKNOWN,
            selection=selection,
            runtime_capability=runtime_capability,
            message="llama-cli capability is unknown",
        )

    if selection.selected_backend is ComputeBackend.CPU:
        capability = _find_capability(runtime_capability, ComputeBackend.CPU)
        return _readiness(
            status=ExecutionReadinessStatus.READY,
            reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message="CPU execution is available because llama-cli probe succeeded",
        )

    capability = _find_capability(runtime_capability, selection.selected_backend)
    if capability is None:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.CAPABILITY_UNKNOWN,
            selection=selection,
            runtime_capability=runtime_capability,
            message=f"{selection.selected_backend.value} capability was not reported",
        )
    if capability.state is LlamaCppBackendCapabilityState.CONFIRMED:
        return _readiness(
            status=ExecutionReadinessStatus.READY,
            reason=ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message=f"llama-cli exposes {selection.selected_backend.value} device(s)",
        )
    if capability.state is LlamaCppBackendCapabilityState.NOT_OBSERVED:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.SELECTED_BACKEND_NOT_EXPOSED,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message=(
                f"llama-cli did not expose {selection.selected_backend.value} "
                "devices in --list-devices"
            ),
        )
    return _readiness(
        status=ExecutionReadinessStatus.UNKNOWN,
        reason=ExecutionReadinessReason.CAPABILITY_UNKNOWN,
        selection=selection,
        runtime_capability=runtime_capability,
        selected_backend_capability=capability,
        message=f"{selection.selected_backend.value} capability is unknown",
    )


def parse_llama_cpp_devices(output: str) -> list[LlamaCppRuntimeDevice]:
    """Parse the minimal device identifiers Jaull needs from ``--list-devices``.

    Current llama.cpp outputs expose CUDA and Vulkan as ``CUDA0``/``Vulkan0``.
    HIP/ROCm builds are treated as Jaull's HIP backend when they expose a
    ``ROCm`` device prefix; ``HIP`` is accepted only as a tolerant fallback.
    """

    devices: list[LlamaCppRuntimeDevice] = []
    for line in output.splitlines():
        match = _DEVICE_LINE_RE.match(line)
        if match is None:
            continue
        backend = _backend_from_prefix(match.group("prefix"))
        if backend is None:
            continue
        runtime_id = f"{match.group('prefix')}{match.group('index')}"
        body = match.group("body").strip()
        name, total, free = _parse_memory_suffix(body)
        devices.append(
            LlamaCppRuntimeDevice(
                backend=backend,
                runtime_id=runtime_id,
                name=name or None,
                memory_total_bytes=total,
                memory_free_bytes=free,
            )
        )
    return devices


class _ResolvedBinary:
    def __init__(
        self,
        *,
        path: str | None,
        status: LlamaCppBinaryStatus,
        message: str | None = None,
    ) -> None:
        self.path = path
        self.status = status
        self.message = message


def _resolve_llama_cli(path: str | Path | None) -> _ResolvedBinary:
    if path is None:
        resolved = shutil.which("llama-cli")
        if resolved is None:
            return _ResolvedBinary(
                path=None,
                status=LlamaCppBinaryStatus.MISSING,
                message=(
                    "llama-cli executable not found. Install or build llama.cpp "
                    "and ensure llama-cli is available in PATH."
                ),
            )
        return _ResolvedBinary(path=resolved, status=LlamaCppBinaryStatus.UNKNOWN)

    configured_str = str(path)
    candidate = Path(path).expanduser()
    if candidate.parent == Path("."):
        resolved = shutil.which(configured_str)
        if resolved is None:
            return _ResolvedBinary(
                path=None,
                status=LlamaCppBinaryStatus.MISSING,
                message=(
                    f"llama-cli executable not found: {configured_str!r}. Install "
                    "or build llama.cpp and ensure llama-cli is available in PATH."
                ),
            )
        return _ResolvedBinary(path=resolved, status=LlamaCppBinaryStatus.UNKNOWN)

    if not candidate.exists():
        return _ResolvedBinary(
            path=str(candidate),
            status=LlamaCppBinaryStatus.MISSING,
            message=f"llama-cli executable not found at {candidate}",
        )
    if candidate.is_dir() or (os.name != "nt" and not os.access(candidate, os.X_OK)):
        return _ResolvedBinary(
            path=str(candidate),
            status=LlamaCppBinaryStatus.NOT_EXECUTABLE,
            message=f"llama-cli is not executable at {candidate}",
        )
    return _ResolvedBinary(path=str(candidate), status=LlamaCppBinaryStatus.UNKNOWN)


def _failed_capability(
    *,
    path: str | None,
    status: LlamaCppBinaryStatus,
    message: str,
) -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path=path,
        binary_status=status,
        message=message,
    )


def _probe_version(
    *,
    backend: ExecutionBackendProtocol,
    binary_path: str,
    timeout_seconds: float,
) -> str | None:
    try:
        result = backend.execute(
            ExecutionRequest(
                command=(binary_path, "--version"),
                timeout_seconds=timeout_seconds,
            )
        )
    except ExecutionError:
        return None
    text = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return text or None


def _build_backend_capabilities(
    devices: list[LlamaCppRuntimeDevice],
) -> list[LlamaCppBackendCapability]:
    capabilities: list[LlamaCppBackendCapability] = [
        LlamaCppBackendCapability(
            backend=ComputeBackend.CPU,
            state=LlamaCppBackendCapabilityState.CONFIRMED,
            reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
            source=_LIST_DEVICES_SOURCE,
            detail=(
                "CPU execution is treated as usable when llama-cli itself "
                "successfully responds; CPU may not appear in --list-devices."
            ),
        )
    ]
    for backend in (ComputeBackend.CUDA, ComputeBackend.VULKAN, ComputeBackend.HIP):
        backend_devices = [device for device in devices if device.backend is backend]
        if backend_devices:
            capabilities.append(
                LlamaCppBackendCapability(
                    backend=backend,
                    state=LlamaCppBackendCapabilityState.CONFIRMED,
                    devices=backend_devices,
                    reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
                    source=_LIST_DEVICES_SOURCE,
                )
            )
            continue
        capabilities.append(
            LlamaCppBackendCapability(
                backend=backend,
                state=LlamaCppBackendCapabilityState.NOT_OBSERVED,
                reason=LlamaCppCapabilityReason.BACKEND_NOT_OBSERVED,
                source=_LIST_DEVICES_SOURCE,
                detail=(
                    "No runtime device for this backend was observed; this is "
                    "not proof that the binary was compiled without it."
                ),
            )
        )
    return capabilities


def _backend_from_prefix(prefix: str) -> ComputeBackend | None:
    normalized = prefix.casefold()
    if normalized == "cuda":
        return ComputeBackend.CUDA
    if normalized == "vulkan":
        return ComputeBackend.VULKAN
    if normalized in {"hip", "rocm"}:
        return ComputeBackend.HIP
    return None


def _parse_memory_suffix(body: str) -> tuple[str, int | None, int | None]:
    match = _MEMORY_RE.match(body)
    if match is None:
        return body, None, None
    mib = 1024 * 1024
    return (
        match.group("name").strip(),
        int(match.group("total")) * mib,
        int(match.group("free")) * mib,
    )


def _find_capability(
    runtime_capability: LlamaCppRuntimeCapability,
    backend: ComputeBackend,
) -> LlamaCppBackendCapability | None:
    return next(
        (
            capability
            for capability in runtime_capability.backend_capabilities
            if capability.backend is backend
        ),
        None,
    )


def _readiness(
    *,
    status: ExecutionReadinessStatus,
    reason: ExecutionReadinessReason,
    selection: RuntimeBackendSelection,
    runtime_capability: LlamaCppRuntimeCapability,
    selected_backend_capability: LlamaCppBackendCapability | None = None,
    message: str | None = None,
) -> ExecutionReadiness:
    return ExecutionReadiness(
        status=status,
        reason=reason,
        selection=selection,
        runtime_capability=runtime_capability,
        selected_backend_capability=selected_backend_capability,
        message=message,
    )


__all__ = [
    "evaluate_execution_readiness",
    "inspect_llama_cpp_runtime",
    "parse_llama_cpp_devices",
]
