"""Synthetic regressions: no model, GPU, Docker or user evidence is used."""

from pathlib import Path

import pytest

from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import TargetDevice
from jaull.domain.runtime import (
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    RuntimeFlag,
    RuntimeFlagSource,
)
from jaull.experiments.errors import ExperimentConfigurationError, ExperimentRunnerError
from jaull.presentation.plan_labels import runtime_block_reason
from jaull.runtime.device_matching import match_runtime_device, runtime_device_name_matches
from jaull.runtime.llama_cpp_runner import _observed_backend
from tests._intel_fixtures import WINDOWS_OS, scan
from tests.test_experiment_runner import (
    _experiment_runner,
    _FakeArtifactRunner,
    _request,
    _runtime,
)
from tests.test_intel_execution_gates import READY, _plan, _readiness, _with_readiness
from tests.test_runtime_device_matching import B60, IGPU, _capability, _device, _selection


@pytest.mark.parametrize("device", ["Vulkan0", None, "Vulkan0,Vulkan1", "Vulkan1"])
def test_validate_checks_prepared_device_against_fresh_enumeration(tmp_path: Path, device) -> None:
    artifact_runner = _FakeArtifactRunner(raw_stderr=_execution_log("Vulkan1", B60))
    runner = _experiment_runner(
        tmp_path, devices=f"Vulkan0: {IGPU}\nVulkan1: {B60}\n",
        artifact_runner=artifact_runner,
    )
    runtime = _runtime(n_gpu_layers=-1)
    if device is not None:
        runtime = runtime.model_copy(update={"flags": [*runtime.flags, RuntimeFlag(
            name="--device", value=device, source=RuntimeFlagSource.ESTIMATE,
            explanation="Prepared before the device enumeration changed",
        )]})
    request = _request(
        tmp_path, backend=ComputeBackend.VULKAN, runtime=runtime,
        hardware=scan(WINDOWS_OS), prediction_device=TargetDevice.GPU, persist=False,
    ).model_copy(update={"backend_selection": _selection(B60), "runtime": runtime})
    if device == "Vulkan1":
        runner.run(request)
        assert len(artifact_runner.calls) == 1
        assert [flag.value for flag in artifact_runner.calls[0][2].flags
                if flag.name == "--device"] == ["Vulkan1"]
    else:
        with pytest.raises(ExperimentConfigurationError, match="device"):
            runner.run(request)
        assert not artifact_runner.calls


def _execution_log(device: str, name: str) -> str:
    return (
        f"llama_prepare_model_devices: using device {device} ({name}) (0000:08:00.0)\n"
        "load_tensors: offloaded 29/29 layers to GPU\n"
        f"load_tensors: {device} model buffer size = 100.00 MiB\n"
        f"llama_context: {device} compute buffer size = 10.00 MiB\n"
    )


@pytest.mark.parametrize("output", ["", _execution_log("Vulkan0", IGPU),
                                     _execution_log("CUDA0", B60)])
def test_validate_never_saves_unproven_or_wrong_device_as_success(tmp_path, output):
    artifact_runner = _FakeArtifactRunner(raw_stderr=output)
    runner = _experiment_runner(tmp_path, devices=f"Vulkan0: {B60}\n",
                                artifact_runner=artifact_runner)
    request = _request(tmp_path, backend=ComputeBackend.VULKAN,
                       prediction_device=TargetDevice.GPU).model_copy(
        update={"backend_selection": _selection(B60)}
    )
    with pytest.raises(ExperimentRunnerError, match="no validated experiment record"):
        runner.run(request)
    assert len(artifact_runner.calls) == 1
    assert runner.store.list_ids() == []


def test_execution_enumeration_must_uniquely_name_the_selected_device():
    log = _execution_log("Vulkan1", B60)
    assert runtime_device_name_matches(log, "Vulkan1", B60)
    assert not runtime_device_name_matches(log, "Vulkan0", B60)
    assert not runtime_device_name_matches(log, "Vulkan1", IGPU)
    assert not runtime_device_name_matches(
        log + _execution_log("Vulkan0", B60), "Vulkan1", B60,
    )
    assert not runtime_device_name_matches(
        log + _execution_log("Vulkan1", IGPU), "Vulkan1", B60,
    )
    assert runtime_device_name_matches(
        log + "ggml_cuda_init: Device 0: NVIDIA RTX 2060, compute capability 7.5\n",
        "Vulkan1", B60,
    )


def test_vulkan_requires_a_named_selected_device() -> None:
    assert match_runtime_device(_selection(B60), _capability(_device(0, None))).device is None
    assert match_runtime_device(_selection(None), _capability(_device(0, B60))).device is None
    match = match_runtime_device(_selection(B60), _capability(_device(1, B60)))
    assert match.device.runtime_id == (
        "Vulkan1"
    )


@pytest.mark.parametrize("backend", [ComputeBackend.CUDA, ComputeBackend.HIP])
def test_non_vulkan_single_device_policy_is_preserved(backend) -> None:
    selection = _selection(B60).model_copy(update={"selected_backend": backend})
    capability = _capability(_device(0, None)).model_copy(update={"backend": backend})
    assert match_runtime_device(selection, capability).device is not None


def test_vulkan_probe_failure_blocks_tui_without_changing_cpu_unknown_policy() -> None:
    readiness = _readiness(
        ExecutionReadinessStatus.UNKNOWN, ExecutionReadinessReason.PROBE_FAILED,
        "Vulkan capability probe timed out",
    )
    assert runtime_block_reason(_with_readiness(_plan(), readiness)) is not None
    cpu = readiness.selection.model_copy(update={"selected_backend": ComputeBackend.CPU})
    assert runtime_block_reason(_with_readiness(
        _plan(), readiness.model_copy(update={"selection": cpu}),
    )) is None


@pytest.mark.parametrize("capability", [None, _capability(_device(0, None)),
                                      _capability(_device(0, IGPU)),
                                      _capability(_device(0, B60), _device(1, B60))])
def test_stale_vulkan_ready_without_unique_device_identity_blocks_tui(capability):
    ready = READY.model_copy(update={"selected_backend_capability": capability})
    assert runtime_block_reason(_with_readiness(_plan(), ready)) is not None


def test_vulkan_unknown_is_blocked_even_if_plan_selection_is_missing():
    ready = _readiness(ExecutionReadinessStatus.UNKNOWN, ExecutionReadinessReason.PROBE_FAILED,
                       "Vulkan probe failed")
    plan = _with_readiness(_plan(), ready).model_copy(update={"backend_selection": None})
    assert runtime_block_reason(plan) is not None


@pytest.mark.parametrize("initialization", [
    "ggml_vulkan: Found 1 device", "ggml_cuda_init: found 1",
])
def test_backend_initialization_is_not_gpu_execution(initialization) -> None:
    log = initialization + "\nload_tensors: offloaded 0/29 layers to GPU\n" + (
        "load_tensors: CPU_Mapped model buffer size = 100.00 MiB\n"
        "llama_context: CPU compute buffer size = 10.00 MiB\n"
    )
    assert _observed_backend(log) is ComputeBackend.CPU
    assert _observed_backend(initialization) is None
    # Zero offloaded layers can still reserve GPU compute; that is not CPU-only proof.
    assert _observed_backend(log.replace("CPU compute", "CUDA0 compute")) is None


@pytest.mark.parametrize("device,backend", [
    ("CUDA0", ComputeBackend.CUDA), ("Vulkan1", ComputeBackend.VULKAN),
    ("ROCm0", ComputeBackend.HIP),
    ("HIP0", ComputeBackend.HIP),
])
def test_observed_backend_uses_assignments_not_loaded_libraries(device, backend) -> None:
    log = (
        "ggml_cuda_init: found 1 device\nggml_vulkan: Found 1 device\n"
        "load_tensors: offloaded 29/29 layers to GPU\n"
        f"load_tensors: {device} model buffer size = 100.00 MiB\n"
        f"llama_context: {device} compute buffer size = 10.00 MiB\n"
    )
    assert _observed_backend(log) is backend
    split = log + "load_tensors: Vulkan2 model buffer size = 20.00 MiB\n"
    assert _observed_backend(split) is None
