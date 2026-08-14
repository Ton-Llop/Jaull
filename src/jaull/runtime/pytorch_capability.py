"""Inspect observable capabilities of a Python/PyTorch/Transformers runtime."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jaull.domain.execution import ExecutionFailureReason, ExecutionRequest
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    PyTorchBackendCapability,
    PyTorchBackendCapabilityState,
    PyTorchCapabilityReason,
    PyTorchRuntimeCapability,
    PyTorchRuntimeDevice,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
)
from jaull.execution.errors import (
    ExecutableNotFoundError,
    ExecutionError,
    ExecutionFailedError,
    ExecutionTimeoutError,
)
from jaull.execution.ports import ExecutionBackendProtocol

_PROBE_SOURCE = "python -c pytorch_runtime_probe"

_PROBE_SCRIPT = r"""
import importlib
import json
import traceback


def emit(payload):
    print(json.dumps(payload, sort_keys=True))


def import_module(name):
    try:
        return importlib.import_module(name), None
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None, {"status": f"{name}_missing", "message": str(exc)}
        return None, {
            "status": "import_failed",
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }
    except Exception as exc:
        return None, {
            "status": "import_failed",
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


torch, error = import_module("torch")
if error is not None:
    emit({"runtime_status": error["status"], "message": error["message"]})
    raise SystemExit(0)

transformers, error = import_module("transformers")
if error is not None:
    status = error["status"]
    if status == "transformers_missing":
        status = "transformers_missing"
    emit({"runtime_status": status, "message": error["message"]})
    raise SystemExit(0)

cuda_available = None
cuda_device_count = None
cuda_error = None
devices = []

try:
    cuda_available = bool(torch.cuda.is_available())
except Exception as exc:
    cuda_available = None
    cuda_error = str(exc)

try:
    cuda_device_count = int(torch.cuda.device_count())
except Exception as exc:
    cuda_device_count = None
    cuda_error = str(exc)

if cuda_device_count:
    for index in range(cuda_device_count):
        device = {"index": index}
        try:
            device["name"] = torch.cuda.get_device_name(index)
        except Exception as exc:
            device["name_error"] = str(exc)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            device["memory_free_bytes"] = int(free_bytes)
            device["memory_total_bytes"] = int(total_bytes)
        except Exception as exc:
            device["memory_error"] = str(exc)
        devices.append(device)

emit(
    {
        "runtime_status": "available",
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_error": cuda_error,
        "devices": devices,
    }
)
"""


def inspect_pytorch_runtime(
    *,
    backend: ExecutionBackendProtocol,
    python_executable: str | Path | None = None,
    timeout_seconds: float = 10.0,
) -> PyTorchRuntimeCapability:
    """Probe Python, PyTorch and Transformers without loading a model."""

    resolved = _resolve_python(python_executable)
    if resolved.status is PyTorchRuntimeStatus.PYTHON_MISSING:
        return PyTorchRuntimeCapability(
            python_executable=resolved.path,
            runtime_status=resolved.status,
            message=resolved.message,
        )

    assert resolved.path is not None
    try:
        result = backend.execute(
            ExecutionRequest(
                command=(resolved.path, "-c", _PROBE_SCRIPT),
                timeout_seconds=timeout_seconds,
            )
        )
    except ExecutableNotFoundError as exc:
        return PyTorchRuntimeCapability(
            python_executable=resolved.path,
            runtime_status=PyTorchRuntimeStatus.PYTHON_MISSING,
            message=str(exc),
        )
    except (ExecutionFailedError, ExecutionTimeoutError) as exc:
        return PyTorchRuntimeCapability(
            python_executable=resolved.path,
            runtime_status=PyTorchRuntimeStatus.PROBE_FAILED,
            probe_source=_PROBE_SOURCE,
            message=str(exc),
        )
    except ExecutionError as exc:
        status = (
            PyTorchRuntimeStatus.PYTHON_MISSING
            if exc.observation is not None
            and exc.observation.failure_reason
            is ExecutionFailureReason.EXECUTABLE_NOT_FOUND
            else PyTorchRuntimeStatus.PROBE_FAILED
        )
        return PyTorchRuntimeCapability(
            python_executable=resolved.path,
            runtime_status=status,
            probe_source=_PROBE_SOURCE,
            message=str(exc),
        )

    capability = parse_pytorch_probe_json(
        result.stdout,
        python_executable=resolved.path,
    )
    return capability.model_copy(update={"probe_source": _PROBE_SOURCE})


def parse_pytorch_probe_json(
    stdout: str,
    *,
    python_executable: str | None = None,
) -> PyTorchRuntimeCapability:
    """Parse the structured JSON emitted by the PyTorch runtime probe."""

    try:
        raw = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        return _unknown_capability(
            python_executable=python_executable,
            message=f"PyTorch probe did not emit valid JSON: {exc}",
        )
    if not isinstance(raw, dict):
        return _unknown_capability(
            python_executable=python_executable,
            message="PyTorch probe JSON must be an object",
        )

    try:
        status = PyTorchRuntimeStatus(str(raw.get("runtime_status", "unknown")))
    except ValueError:
        status = PyTorchRuntimeStatus.UNKNOWN

    if status is not PyTorchRuntimeStatus.AVAILABLE:
        return PyTorchRuntimeCapability(
            python_executable=python_executable,
            runtime_status=status,
            probe_source=_PROBE_SOURCE,
            message=_str_or_none(raw.get("message")),
            raw_probe=dict(raw),
        )

    try:
        return PyTorchRuntimeCapability(
            python_executable=python_executable,
            runtime_status=PyTorchRuntimeStatus.AVAILABLE,
            torch_version=_str_or_none(raw.get("torch_version")),
            transformers_version=_str_or_none(raw.get("transformers_version")),
            torch_cuda_version=_str_or_none(raw.get("torch_cuda_version")),
            torch_hip_version=_str_or_none(raw.get("torch_hip_version")),
            backend_capabilities=_build_backend_capabilities(raw),
            probe_source=_PROBE_SOURCE,
            message=_str_or_none(raw.get("cuda_error")),
            raw_probe=dict(raw),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        return _unknown_capability(
            python_executable=python_executable,
            message=f"PyTorch probe JSON could not be interpreted: {exc}",
            raw_probe=dict(raw),
        )


def evaluate_pytorch_execution_readiness(
    *,
    selection: RuntimeBackendSelection,
    runtime_capability: PyTorchRuntimeCapability,
) -> ExecutionReadiness:
    """Return a conservative preflight decision for PyTorch/Transformers."""

    status = runtime_capability.runtime_status
    if status in {
        PyTorchRuntimeStatus.PYTHON_MISSING,
        PyTorchRuntimeStatus.TORCH_MISSING,
        PyTorchRuntimeStatus.TRANSFORMERS_MISSING,
    }:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.RUNTIME_MISSING,
            selection=selection,
            runtime_capability=runtime_capability,
            message=_missing_message(runtime_capability),
        )
    if status is PyTorchRuntimeStatus.IMPORT_FAILED:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.RUNTIME_IMPORT_FAILED,
            selection=selection,
            runtime_capability=runtime_capability,
            message=runtime_capability.message
            or "PyTorch/Transformers runtime import failed",
        )
    if status is PyTorchRuntimeStatus.PROBE_FAILED:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.PROBE_FAILED,
            selection=selection,
            runtime_capability=runtime_capability,
            message="PyTorch capability probe failed",
        )
    if status is not PyTorchRuntimeStatus.AVAILABLE:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.CAPABILITY_UNKNOWN,
            selection=selection,
            runtime_capability=runtime_capability,
            message="PyTorch runtime capability is unknown",
        )

    capability = _find_capability(runtime_capability, selection.selected_backend)
    if selection.selected_backend is ComputeBackend.CPU:
        return _readiness(
            status=ExecutionReadinessStatus.READY,
            reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message=(
                "CPU execution is available because Python, PyTorch and "
                "Transformers import successfully"
            ),
        )

    if capability is None:
        return _readiness(
            status=ExecutionReadinessStatus.UNKNOWN,
            reason=ExecutionReadinessReason.CAPABILITY_UNKNOWN,
            selection=selection,
            runtime_capability=runtime_capability,
            message=f"{selection.selected_backend.value} capability was not reported",
        )
    if capability.state is PyTorchBackendCapabilityState.CONFIRMED:
        return _readiness(
            status=ExecutionReadinessStatus.READY,
            reason=ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message=(
                "PyTorch exposes "
                f"{selection.selected_backend.value} device(s) for this runtime"
            ),
        )
    if capability.state is PyTorchBackendCapabilityState.NOT_OBSERVED:
        return _readiness(
            status=ExecutionReadinessStatus.NOT_READY,
            reason=ExecutionReadinessReason.SELECTED_BACKEND_NOT_EXPOSED,
            selection=selection,
            runtime_capability=runtime_capability,
            selected_backend_capability=capability,
            message=(
                "PyTorch did not expose "
                f"{selection.selected_backend.value} device(s)"
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


class _ResolvedPython:
    def __init__(
        self,
        *,
        path: str | None,
        status: PyTorchRuntimeStatus,
        message: str | None = None,
    ) -> None:
        self.path = path
        self.status = status
        self.message = message


def _resolve_python(path: str | Path | None) -> _ResolvedPython:
    if path is None:
        executable = sys.executable or shutil.which("python3") or shutil.which("python")
        if executable is None:
            return _ResolvedPython(
                path=None,
                status=PyTorchRuntimeStatus.PYTHON_MISSING,
                message="Python executable was not found.",
            )
        return _ResolvedPython(path=executable, status=PyTorchRuntimeStatus.UNKNOWN)

    configured = str(path)
    candidate = Path(path).expanduser()
    if candidate.parent == Path("."):
        resolved = shutil.which(configured)
        if resolved is None:
            return _ResolvedPython(
                path=None,
                status=PyTorchRuntimeStatus.PYTHON_MISSING,
                message=f"Python executable was not found: {configured!r}.",
            )
        return _ResolvedPython(path=resolved, status=PyTorchRuntimeStatus.UNKNOWN)

    if not candidate.exists():
        return _ResolvedPython(
            path=str(candidate),
            status=PyTorchRuntimeStatus.PYTHON_MISSING,
            message=f"Python executable was not found at {candidate}.",
        )
    if candidate.is_dir() or (os.name != "nt" and not os.access(candidate, os.X_OK)):
        return _ResolvedPython(
            path=str(candidate),
            status=PyTorchRuntimeStatus.PYTHON_MISSING,
            message=f"Python executable is not runnable at {candidate}.",
        )
    return _ResolvedPython(path=str(candidate), status=PyTorchRuntimeStatus.UNKNOWN)


def _build_backend_capabilities(
    raw: dict[str, Any],
) -> list[PyTorchBackendCapability]:
    cuda_available = bool(raw.get("cuda_available"))
    device_count = _int_or_none(raw.get("cuda_device_count")) or 0
    torch_cuda_version = _str_or_none(raw.get("torch_cuda_version"))
    torch_hip_version = _str_or_none(raw.get("torch_hip_version"))
    raw_devices = raw.get("devices", [])
    if not isinstance(raw_devices, list):
        raw_devices = []

    logical_backend: ComputeBackend | None = None
    if cuda_available and device_count > 0:
        logical_backend = (
            ComputeBackend.HIP if torch_hip_version else ComputeBackend.CUDA
        )

    devices = [
        _device_from_probe(device, logical_backend)
        for device in raw_devices
        if isinstance(device, dict) and logical_backend is not None
    ]

    return [
        PyTorchBackendCapability(
            backend=ComputeBackend.CPU,
            state=PyTorchBackendCapabilityState.CONFIRMED,
            reason=PyTorchCapabilityReason.RUNTIME_AVAILABLE,
            source=_PROBE_SOURCE,
            detail=(
                "CPU execution is treated as usable when Python, PyTorch and "
                "Transformers import successfully."
            ),
        ),
        _cuda_capability(
            logical_backend=logical_backend,
            devices=devices,
            torch_cuda_version=torch_cuda_version,
        ),
        PyTorchBackendCapability(
            backend=ComputeBackend.VULKAN,
            state=PyTorchBackendCapabilityState.NOT_OBSERVED,
            reason=PyTorchCapabilityReason.BACKEND_NOT_OBSERVED,
            source=_PROBE_SOURCE,
            detail="PyTorch/Transformers Vulkan execution is not probed by Jaull.",
        ),
        _hip_capability(
            logical_backend=logical_backend,
            devices=devices,
            torch_hip_version=torch_hip_version,
        ),
    ]


def _cuda_capability(
    *,
    logical_backend: ComputeBackend | None,
    devices: list[PyTorchRuntimeDevice],
    torch_cuda_version: str | None,
) -> PyTorchBackendCapability:
    cuda_devices = [
        device for device in devices if device.backend is ComputeBackend.CUDA
    ]
    if logical_backend is ComputeBackend.CUDA and cuda_devices:
        return PyTorchBackendCapability(
            backend=ComputeBackend.CUDA,
            state=PyTorchBackendCapabilityState.CONFIRMED,
            devices=cuda_devices,
            reason=PyTorchCapabilityReason.BACKEND_EXPOSED,
            source=_PROBE_SOURCE,
            detail=(
                f"torch.version.cuda={torch_cuda_version}"
                if torch_cuda_version
                else "torch.cuda exposed device(s)"
            ),
        )
    return PyTorchBackendCapability(
        backend=ComputeBackend.CUDA,
        state=PyTorchBackendCapabilityState.NOT_OBSERVED,
        reason=PyTorchCapabilityReason.BACKEND_NOT_OBSERVED,
        source=_PROBE_SOURCE,
        detail=(
            "No CUDA runtime device was observed; this does not prove PyTorch "
            "was compiled without CUDA."
        ),
    )


def _hip_capability(
    *,
    logical_backend: ComputeBackend | None,
    devices: list[PyTorchRuntimeDevice],
    torch_hip_version: str | None,
) -> PyTorchBackendCapability:
    hip_devices = [device for device in devices if device.backend is ComputeBackend.HIP]
    if logical_backend is ComputeBackend.HIP and hip_devices:
        return PyTorchBackendCapability(
            backend=ComputeBackend.HIP,
            state=PyTorchBackendCapabilityState.CONFIRMED,
            devices=hip_devices,
            reason=PyTorchCapabilityReason.BACKEND_EXPOSED,
            source=_PROBE_SOURCE,
            detail=f"torch.version.hip={torch_hip_version}",
        )
    if torch_hip_version:
        return PyTorchBackendCapability(
            backend=ComputeBackend.HIP,
            state=PyTorchBackendCapabilityState.UNKNOWN,
            reason=PyTorchCapabilityReason.CAPABILITY_UNKNOWN,
            source=_PROBE_SOURCE,
            detail=(
                "ROCm/HIP build metadata was observed, but no usable HIP "
                "device was exposed by the runtime probe."
            ),
        )
    return PyTorchBackendCapability(
        backend=ComputeBackend.HIP,
        state=PyTorchBackendCapabilityState.NOT_OBSERVED,
        reason=PyTorchCapabilityReason.BACKEND_NOT_OBSERVED,
        source=_PROBE_SOURCE,
    )


def _device_from_probe(
    raw: dict[str, Any],
    backend: ComputeBackend,
) -> PyTorchRuntimeDevice:
    index = _int_or_none(raw.get("index"))
    runtime_id = f"{backend.value}:{index}" if index is not None else backend.value
    return PyTorchRuntimeDevice(
        backend=backend,
        runtime_id=runtime_id,
        index=index,
        name=_str_or_none(raw.get("name")),
        memory_total_bytes=_int_or_none(raw.get("memory_total_bytes")),
        memory_free_bytes=_int_or_none(raw.get("memory_free_bytes")),
    )


def _unknown_capability(
    *,
    python_executable: str | None,
    message: str,
    raw_probe: dict[str, object] | None = None,
) -> PyTorchRuntimeCapability:
    return PyTorchRuntimeCapability(
        python_executable=python_executable,
        runtime_status=PyTorchRuntimeStatus.UNKNOWN,
        probe_source=_PROBE_SOURCE,
        message=message,
        raw_probe=raw_probe,
    )


def _find_capability(
    runtime_capability: PyTorchRuntimeCapability,
    backend: ComputeBackend,
) -> PyTorchBackendCapability | None:
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
    runtime_capability: PyTorchRuntimeCapability,
    selected_backend_capability: PyTorchBackendCapability | None = None,
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


def _missing_message(capability: PyTorchRuntimeCapability) -> str:
    if capability.message:
        return capability.message
    return {
        PyTorchRuntimeStatus.PYTHON_MISSING: "Python executable was not found",
        PyTorchRuntimeStatus.TORCH_MISSING: "PyTorch is not installed",
        PyTorchRuntimeStatus.TRANSFORMERS_MISSING: "Transformers is not installed",
    }.get(capability.runtime_status, "Runtime dependency is missing")


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


__all__ = [
    "evaluate_pytorch_execution_readiness",
    "inspect_pytorch_runtime",
    "parse_pytorch_probe_json",
]
