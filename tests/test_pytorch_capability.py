from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaull.advisor.service import AdvisorService
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    PyTorchBackendCapabilityState,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
)
from jaull.execution.errors import ExecutionFailedError
from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness
from jaull.runtime.pytorch_capability import (
    evaluate_pytorch_execution_readiness,
    inspect_pytorch_runtime,
    parse_pytorch_probe_json,
)
from jaull.workflow.container import ServiceContainer


class _FakeExecutionBackend:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.requests: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if isinstance(self.outcome, ExecutionResult):
            return self.outcome
        return _result(stdout=str(self.outcome))


def test_valid_cpu_runtime_confirms_cpu_and_metadata(tmp_path: Path) -> None:
    binary = _fake_python(tmp_path)
    backend = _FakeExecutionBackend(
        _probe_json(
            torch_version="2.8.0",
            transformers_version="4.55.0",
            torch_cuda_version=None,
            cuda_available=False,
            cuda_device_count=0,
        )
    )

    capability = inspect_pytorch_runtime(backend=backend, python_executable=binary)

    assert capability.runtime_status is PyTorchRuntimeStatus.AVAILABLE
    assert capability.python_executable == str(binary)
    assert capability.torch_version == "2.8.0"
    assert capability.transformers_version == "4.55.0"
    assert _capability_state(capability, ComputeBackend.CPU) is (
        PyTorchBackendCapabilityState.CONFIRMED
    )
    assert _capability_state(capability, ComputeBackend.CUDA) is (
        PyTorchBackendCapabilityState.NOT_OBSERVED
    )


def test_torch_missing_is_preserved() -> None:
    capability = parse_pytorch_probe_json(
        json.dumps({"runtime_status": "torch_missing", "message": "no torch"}),
        python_executable="/venv/bin/python",
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.TORCH_MISSING
    assert capability.message == "no torch"
    assert capability.backend_capabilities == []


def test_transformers_missing_is_preserved() -> None:
    capability = parse_pytorch_probe_json(
        json.dumps(
            {"runtime_status": "transformers_missing", "message": "no transformers"}
        )
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.TRANSFORMERS_MISSING


def test_import_failure_is_not_collapsed_to_missing() -> None:
    capability = parse_pytorch_probe_json(
        json.dumps({"runtime_status": "import_failed", "message": "bad extension"})
    )

    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.IMPORT_FAILED
    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.RUNTIME_IMPORT_FAILED


def test_probe_process_failure_yields_probe_failed(tmp_path: Path) -> None:
    binary = _fake_python(tmp_path)
    failed = ExecutionFailedError(
        "python probe failed",
        _result(
            stdout="",
            stderr="boom",
            success=False,
            exit_code=1,
            failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
        ),
    )

    capability = inspect_pytorch_runtime(
        backend=_FakeExecutionBackend(failed),
        python_executable=binary,
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.PROBE_FAILED


def test_malformed_json_yields_unknown_capability() -> None:
    capability = parse_pytorch_probe_json("not-json")

    assert capability.runtime_status is PyTorchRuntimeStatus.UNKNOWN
    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )
    assert readiness.status is ExecutionReadinessStatus.UNKNOWN
    assert readiness.reason is ExecutionReadinessReason.CAPABILITY_UNKNOWN


def test_missing_python_does_not_call_backend(tmp_path: Path) -> None:
    backend = _FakeExecutionBackend(_probe_json())

    capability = inspect_pytorch_runtime(
        backend=backend,
        python_executable=tmp_path / "missing-python",
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.PYTHON_MISSING
    assert backend.requests == []


def test_cpu_selection_ready_when_runtime_is_available() -> None:
    capability = parse_pytorch_probe_json(_probe_json())

    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.reason is ExecutionReadinessReason.RUNTIME_AVAILABLE


def test_cuda_selection_ready_when_cuda_device_is_observable() -> None:
    capability = parse_pytorch_probe_json(
        _probe_json(
            torch_cuda_version="12.4",
            cuda_available=True,
            cuda_device_count=1,
            devices=[
                {
                    "index": 0,
                    "name": "NVIDIA RTX 2060",
                    "memory_total_bytes": 6 * 1024**3,
                    "memory_free_bytes": 5 * 1024**3,
                }
            ],
        )
    )

    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CUDA),
        runtime_capability=capability,
    )

    cuda = _capability(capability, ComputeBackend.CUDA)
    assert cuda.state is PyTorchBackendCapabilityState.CONFIRMED
    assert cuda.devices[0].runtime_id == "cuda:0"
    assert cuda.devices[0].name == "NVIDIA RTX 2060"
    assert cuda.devices[0].memory_total_bytes == 6 * 1024**3
    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED


def test_cuda_selection_not_ready_when_cuda_is_not_observed() -> None:
    capability = parse_pytorch_probe_json(
        _probe_json(cuda_available=False, cuda_device_count=0)
    )

    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CUDA),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_BACKEND_NOT_EXPOSED


def test_torch_cuda_build_metadata_is_preserved_without_inferring_cause() -> None:
    capability = parse_pytorch_probe_json(
        _probe_json(
            torch_cuda_version="12.1",
            cuda_available=False,
            cuda_device_count=0,
        )
    )

    cuda = _capability(capability, ComputeBackend.CUDA)

    assert capability.torch_cuda_version == "12.1"
    assert cuda.state is PyTorchBackendCapabilityState.NOT_OBSERVED
    assert "does not prove" in (cuda.detail or "")


def test_hip_metadata_does_not_become_cuda_or_ready_without_devices() -> None:
    capability = parse_pytorch_probe_json(
        _probe_json(
            torch_hip_version="6.2",
            cuda_available=False,
            cuda_device_count=0,
        )
    )

    cuda = _capability(capability, ComputeBackend.CUDA)
    hip = _capability(capability, ComputeBackend.HIP)
    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.HIP),
        runtime_capability=capability,
    )

    assert capability.torch_hip_version == "6.2"
    assert cuda.state is PyTorchBackendCapabilityState.NOT_OBSERVED
    assert hip.state is PyTorchBackendCapabilityState.UNKNOWN
    assert readiness.status is ExecutionReadinessStatus.UNKNOWN


def test_rocm_runtime_devices_map_to_hip_backend() -> None:
    capability = parse_pytorch_probe_json(
        _probe_json(
            torch_hip_version="6.2",
            cuda_available=True,
            cuda_device_count=1,
            devices=[{"index": 0, "name": "AMD Radeon"}],
        )
    )

    hip = _capability(capability, ComputeBackend.HIP)

    assert hip.state is PyTorchBackendCapabilityState.CONFIRMED
    assert hip.devices[0].backend is ComputeBackend.HIP
    assert _capability_state(capability, ComputeBackend.CUDA) is (
        PyTorchBackendCapabilityState.NOT_OBSERVED
    )


def test_vulkan_selection_is_not_ready_for_pytorch_runtime() -> None:
    capability = parse_pytorch_probe_json(_probe_json())

    readiness = evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.VULKAN),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY


def test_execution_backend_is_used_for_probe(tmp_path: Path) -> None:
    binary = _fake_python(tmp_path)
    backend = _FakeExecutionBackend(_probe_json())

    inspect_pytorch_runtime(backend=backend, python_executable=binary)

    assert len(backend.requests) == 1
    assert backend.requests[0].command[0] == str(binary)
    assert backend.requests[0].command[1] == "-c"
    assert "importlib" in backend.requests[0].command[2]


def test_advisor_service_exposes_pytorch_probe_and_readiness(tmp_path: Path) -> None:
    binary = _fake_python(tmp_path)
    backend = _FakeExecutionBackend(_probe_json())
    advisor = AdvisorService(
        services=_services(),
        python_executable=binary,
    )

    capability = advisor.inspect_pytorch_runtime(backend=backend)
    readiness = advisor.evaluate_pytorch_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert capability.runtime_status is PyTorchRuntimeStatus.AVAILABLE
    assert readiness.status is ExecutionReadinessStatus.READY


def test_llama_cpp_readiness_still_accepts_llama_cpp_capability() -> None:
    capability = LlamaCppRuntimeCapability(
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
            )
        ],
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.runtime_capability is capability


def _probe_json(**updates: object) -> str:
    payload: dict[str, object] = {
        "runtime_status": "available",
        "torch_version": "2.8.0",
        "transformers_version": "4.55.0",
        "torch_cuda_version": None,
        "torch_hip_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "devices": [],
    }
    payload.update(updates)
    return json.dumps(payload)


def _selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK
        if backend is ComputeBackend.CPU
        else RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def _capability(capability: Any, backend: ComputeBackend) -> Any:
    return next(
        item
        for item in capability.backend_capabilities
        if item.backend is backend
    )


def _capability_state(capability: Any, backend: ComputeBackend) -> object:
    return _capability(capability, backend).state


def _fake_python(tmp_path: Path) -> Path:
    binary = tmp_path / "python"
    binary.write_text("fake", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _result(
    *,
    stdout: str,
    stderr: str = "",
    success: bool = True,
    exit_code: int | None = 0,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        observation=ExecutionObservation(
            success=success,
            duration_seconds=0.01,
            exit_code=exit_code,
            failure_reason=failure_reason,
        ),
    )


def _services() -> ServiceContainer:
    return ServiceContainer(
        hf_client=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
        detect_hardware=lambda: None,  # type: ignore[arg-type]
        inspect_model=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        estimate_memory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
