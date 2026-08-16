from __future__ import annotations

from pathlib import Path
from typing import Any

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
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
)
from jaull.execution.errors import ExecutionError, ExecutionFailedError
from jaull.runtime.llama_cpp_capability import (
    evaluate_execution_readiness,
    inspect_llama_cpp_runtime,
    parse_llama_cpp_devices,
)


class _FakeExecutionBackend:
    def __init__(self, outputs: dict[tuple[str, ...], object]) -> None:
        self.outputs = outputs
        self.requests: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        outcome = self.outputs.get(request.command)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, ExecutionResult):
            return outcome
        if isinstance(outcome, tuple):
            stdout, stderr = outcome
            return _result(stdout=str(stdout), stderr=str(stderr))
        if isinstance(outcome, str):
            return _result(stdout=outcome)
        if request.command[-1] == "--version":
            return _result(stdout="llama-cli build test")
        return _result(stdout="")


def test_missing_binary_returns_missing_without_backend_call(tmp_path: Path) -> None:
    missing = tmp_path / "missing-llama-cli"
    backend = _FakeExecutionBackend({})

    capability = inspect_llama_cpp_runtime(
        backend=backend,
        llama_cli_path=missing,
    )

    assert capability.binary_status is LlamaCppBinaryStatus.MISSING
    assert capability.binary_path == str(missing)
    assert backend.requests == []


def test_not_executable_spawn_error_is_reported(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): ExecutionError(
                "spawn failed",
                observation=_observation(
                    success=False,
                    exit_code=None,
                    failure_reason=ExecutionFailureReason.SPAWN_ERROR,
                ),
            )
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert capability.binary_status is LlamaCppBinaryStatus.NOT_EXECUTABLE


def test_cpu_only_runtime_confirms_cpu_but_no_gpu_backends(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): "Available devices:\n",
            (str(binary), "--version"): "llama-cli b1234",
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert capability.binary_status is LlamaCppBinaryStatus.AVAILABLE
    assert capability.version_text == "llama-cli b1234"
    assert _capability_state(capability, ComputeBackend.CPU) is (
        LlamaCppBackendCapabilityState.CONFIRMED
    )
    assert _capability_state(capability, ComputeBackend.CUDA) is (
        LlamaCppBackendCapabilityState.NOT_OBSERVED
    )
    assert _capability_state(capability, ComputeBackend.VULKAN) is (
        LlamaCppBackendCapabilityState.NOT_OBSERVED
    )


def test_cuda_runtime_devices_are_confirmed(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): (
                "startup log\n"
                "Available devices:\n"
                "  CUDA0: NVIDIA GeForce RTX 2060 (6144 MiB, 6000 MiB free)\n"
            )
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)
    cuda = _capability(capability, ComputeBackend.CUDA)

    assert cuda.state is LlamaCppBackendCapabilityState.CONFIRMED
    assert cuda.devices[0].runtime_id == "CUDA0"
    assert cuda.devices[0].name == "NVIDIA GeForce RTX 2060"
    assert cuda.devices[0].memory_total_bytes == 6144 * 1024 * 1024
    assert cuda.devices[0].memory_free_bytes == 6000 * 1024 * 1024


def test_vulkan_runtime_devices_are_confirmed_from_stderr(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): (
                "",
                "Available devices:\n"
                "Vulkan0: AMD Radeon(TM) Graphics (29696 MiB, 28598 MiB free)\n",
            )
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)
    vulkan = _capability(capability, ComputeBackend.VULKAN)

    assert vulkan.state is LlamaCppBackendCapabilityState.CONFIRMED
    assert vulkan.devices[0].runtime_id == "Vulkan0"
    assert vulkan.devices[0].name == "AMD Radeon(TM) Graphics"


def test_cuda_and_vulkan_runtime_devices_can_coexist(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): (
                "Available devices:\n"
                "CUDA0: NVIDIA GeForce RTX 2060\n"
                "Vulkan0: NVIDIA GeForce RTX 2060\n"
            )
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert _capability_state(capability, ComputeBackend.CUDA) is (
        LlamaCppBackendCapabilityState.CONFIRMED
    )
    assert _capability_state(capability, ComputeBackend.VULKAN) is (
        LlamaCppBackendCapabilityState.CONFIRMED
    )


def test_malformed_successful_output_keeps_gpu_backends_not_observed(
    tmp_path: Path,
) -> None:
    binary = _fake_binary(tmp_path)
    backend = _FakeExecutionBackend(
        {(str(binary), "--list-devices"): "not the expected format"}
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert capability.binary_status is LlamaCppBinaryStatus.AVAILABLE
    assert _capability_state(capability, ComputeBackend.CPU) is (
        LlamaCppBackendCapabilityState.CONFIRMED
    )
    assert _capability_state(capability, ComputeBackend.CUDA) is (
        LlamaCppBackendCapabilityState.NOT_OBSERVED
    )


def test_non_zero_list_devices_marks_probe_failed(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    failure_result = _result(
        stdout="",
        stderr="bad option",
        success=False,
        exit_code=1,
        failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
    )
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): ExecutionFailedError(
                "llama-cli --list-devices failed",
                failure_result,
            )
        }
    )

    capability = inspect_llama_cpp_runtime(backend=backend, llama_cli_path=binary)

    assert capability.binary_status is LlamaCppBinaryStatus.PROBE_FAILED
    assert capability.backend_capabilities == []


def test_parser_maps_rocm_prefix_to_hip_backend() -> None:
    devices = parse_llama_cpp_devices(
        "Available devices:\n"
        "CUDA0: NVIDIA RTX\n"
        "Vulkan1: AMD Radeon\n"
        "ROCm0: AMD GPU\n"
        "Metal0: Apple GPU\n"
    )

    assert [device.backend for device in devices] == [
        ComputeBackend.CUDA,
        ComputeBackend.VULKAN,
        ComputeBackend.HIP,
    ]


def test_parser_tolerates_hip_prefix_without_requiring_it() -> None:
    devices = parse_llama_cpp_devices("Available devices:\nHIP0: AMD GPU\n")

    assert len(devices) == 1
    assert devices[0].backend is ComputeBackend.HIP


def test_cpu_selection_ready_when_runtime_probe_succeeds(tmp_path: Path) -> None:
    capability = _runtime_capability(tmp_path, "Available devices:\n")

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.reason is ExecutionReadinessReason.RUNTIME_AVAILABLE


def test_missing_runtime_is_not_ready_for_any_selection(tmp_path: Path) -> None:
    capability = inspect_llama_cpp_runtime(
        backend=_FakeExecutionBackend({}),
        llama_cli_path=tmp_path / "missing",
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.VULKAN),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.RUNTIME_MISSING


def test_cuda_selection_ready_when_cuda_device_is_exposed(tmp_path: Path) -> None:
    capability = _runtime_capability(
        tmp_path,
        "Available devices:\nCUDA0: NVIDIA GeForce RTX 2060\n",
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CUDA),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED


def test_cuda_selection_not_ready_with_cpu_only_runtime(tmp_path: Path) -> None:
    capability = _runtime_capability(tmp_path, "Available devices:\n")

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CUDA),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY
    assert readiness.reason is ExecutionReadinessReason.SELECTED_BACKEND_NOT_EXPOSED


def test_vulkan_selection_ready_when_vulkan_device_is_exposed(tmp_path: Path) -> None:
    capability = _runtime_capability(
        tmp_path,
        "Available devices:\nVulkan0: AMD Radeon(TM) Graphics\n",
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.VULKAN),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.READY


def test_vulkan_selection_never_ready_when_only_cuda_is_exposed(
    tmp_path: Path,
) -> None:
    capability = _runtime_capability(
        tmp_path,
        "Available devices:\nCUDA0: NVIDIA GeForce RTX 2060\n",
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.VULKAN),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY


def test_mixed_runtime_can_satisfy_cuda_or_vulkan_selection(tmp_path: Path) -> None:
    capability = _runtime_capability(
        tmp_path,
        "Available devices:\nCUDA0: NVIDIA RTX\nVulkan0: NVIDIA RTX\n",
    )

    cuda = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CUDA),
        runtime_capability=capability,
    )
    vulkan = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.VULKAN),
        runtime_capability=capability,
    )

    assert cuda.status is ExecutionReadinessStatus.READY
    assert vulkan.status is ExecutionReadinessStatus.READY


def test_probe_failure_yields_unknown_readiness(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    failure_result = _result(
        stdout="",
        stderr="timeout",
        success=False,
        exit_code=None,
        failure_reason=ExecutionFailureReason.TIMEOUT,
    )
    capability = inspect_llama_cpp_runtime(
        backend=_FakeExecutionBackend(
            {
                (str(binary), "--list-devices"): ExecutionFailedError(
                    "probe failed",
                    failure_result,
                )
            }
        ),
        llama_cli_path=binary,
    )

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.CPU),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.UNKNOWN
    assert readiness.reason is ExecutionReadinessReason.PROBE_FAILED


def test_hip_is_ready_only_with_positive_runtime_evidence(tmp_path: Path) -> None:
    capability = _runtime_capability(tmp_path, "Available devices:\n")

    readiness = evaluate_execution_readiness(
        selection=_selection(ComputeBackend.HIP),
        runtime_capability=capability,
    )

    assert readiness.status is ExecutionReadinessStatus.NOT_READY


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "llama-cli"
    binary.write_text("fake", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _runtime_capability(tmp_path: Path, list_devices_output: str):
    binary = _fake_binary(tmp_path)
    return inspect_llama_cpp_runtime(
        backend=_FakeExecutionBackend(
            {
                (str(binary), "--list-devices"): list_devices_output,
                (str(binary), "--version"): "llama-cli test build",
            }
        ),
        llama_cli_path=binary,
    )


def _selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK
        if backend is ComputeBackend.CPU
        else RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
    )


def _capability(capability: Any, backend: ComputeBackend):
    return next(
        item
        for item in capability.backend_capabilities
        if item.backend is backend
    )


def _capability_state(capability: Any, backend: ComputeBackend):
    return _capability(capability, backend).state


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
        observation=_observation(
            success=success,
            exit_code=exit_code,
            failure_reason=failure_reason,
        ),
    )


def _observation(
    *,
    success: bool,
    exit_code: int | None,
    failure_reason: ExecutionFailureReason | None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=0.01,
        exit_code=exit_code,
        failure_reason=failure_reason,
    )
