from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
    KvCacheEstimate,
    MemoryComponent,
    MemoryEstimate,
    RuntimeOverheadEstimate,
    WeightEstimate,
)
from jaull.domain.execution import (
    ExecutionFailureReason,
    ExecutionObservation,
    ExecutionRequest,
    ExecutionResult,
    InferenceResult,
)
from jaull.domain.experiments import (
    ExperimentRequest,
    ExperimentWorkload,
    RequestedComputeBackend,
)
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelAnalysis, ModelRepositoryInfo, SafetensorsSummary
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    PyTorchRuntimeStatus,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import ExecutionFailedError
from jaull.experiments.errors import (
    ExperimentConfigurationError,
    ExperimentNotReadyError,
    ExperimentRunnerError,
)
from jaull.experiments.runner import ExperimentRunner
from jaull.experiments.storage import ExperimentStore

GIB = 1024**3


def test_successful_cpu_experiment_produces_and_persists_record(
    tmp_path: Path,
) -> None:
    runner = _experiment_runner(tmp_path, devices="Available devices:\n")
    request = _request(tmp_path, backend=ComputeBackend.CPU, persist=True)

    result = runner.run(request)

    assert result.record.workload == request.workload
    assert result.record.backend_trace.requested_backend is RequestedComputeBackend.CPU
    assert result.record.backend_trace.observed_backend is None
    assert result.record.observation.success is True
    assert result.record.preflight.execution_readiness.status is (
        ExecutionReadinessStatus.READY
    )
    assert result.persisted_path is not None
    assert runner.store is not None
    assert runner.store.load(result.record.identity.experiment_id) == result.record


def test_successful_accelerator_experiment_produces_record(tmp_path: Path) -> None:
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\nVulkan0: AMD Radeon(TM) Graphics\n",
    )
    request = _request(
        tmp_path,
        backend=ComputeBackend.VULKAN,
        runtime=_runtime(n_gpu_layers=-1),
        hardware=_hardware_with_vulkan(),
        prediction_device=TargetDevice.GPU,
        persist=False,
    )

    result = runner.run(request)

    assert result.record.preflight.execution_readiness.status is (
        ExecutionReadinessStatus.READY
    )
    assert result.record.preflight.execution_readiness.reason.value == (
        "selected_backend_exposed"
    )
    assert (
        result.record.backend_trace.requested_backend
        is RequestedComputeBackend.VULKAN
    )
    assert result.record.backend_trace.observed_backend is None
    assert result.persisted_path is None


def test_successful_transformers_cpu_experiment_produces_record(tmp_path: Path) -> None:
    llama_runner = _FakeArtifactRunner()
    transformers_runner = _FakeArtifactRunner()
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        artifact_runner=llama_runner,
        transformers_runner=transformers_runner,
        pytorch_probe=_pytorch_probe_json(),
    )
    request = _request(
        tmp_path,
        backend=ComputeBackend.CPU,
        runtime=_transformers_runtime(),
        artifact=_transformers_artifact(),
        prediction=_transformers_estimate(_transformers_runtime()),
        persist=True,
    )

    result = runner.run(request)

    assert llama_runner.calls == []
    assert len(transformers_runner.calls) == 1
    assert result.record.runtime.runtime is RuntimeName.TRANSFORMERS
    assert result.record.artifact.format == "safetensors"
    assert result.record.preflight.runtime_capability.runtime_status is (
        PyTorchRuntimeStatus.AVAILABLE
    )
    assert result.record.preflight.execution_readiness.status is (
        ExecutionReadinessStatus.READY
    )
    assert result.persisted_path is not None


def test_failed_execution_still_produces_experiment_record(tmp_path: Path) -> None:
    failed_observation = _observation(
        success=False,
        exit_code=1,
        failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
    )
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        artifact_runner=_FakeArtifactRunner(error_observation=failed_observation),
    )

    result = runner.run(_request(tmp_path, backend=ComputeBackend.CPU, persist=False))

    assert result.record.observation.success is False
    assert result.record.comparison.compatibility.observed_success is False


def test_transformers_not_ready_does_not_invoke_runner(tmp_path: Path) -> None:
    transformers_runner = _FakeArtifactRunner()
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        transformers_runner=transformers_runner,
        pytorch_probe=_pytorch_probe_json(runtime_status="torch_missing"),
    )
    runtime = _transformers_runtime()

    with pytest.raises(ExperimentNotReadyError) as captured:
        runner.run(
            _request(
                tmp_path,
                backend=ComputeBackend.CPU,
                runtime=runtime,
                artifact=_transformers_artifact(),
                prediction=_transformers_estimate(runtime),
            )
        )

    assert transformers_runner.calls == []
    assert captured.value.readiness.status is ExecutionReadinessStatus.NOT_READY


def test_transformers_experiment_requires_configured_transformers_runner(
    tmp_path: Path,
) -> None:
    runtime = _transformers_runtime()
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        transformers_runner=None,
        pytorch_probe=_pytorch_probe_json(),
    )

    with pytest.raises(ExperimentRunnerError, match="no TransformersRunner"):
        runner.run(
            _request(
                tmp_path,
                backend=ComputeBackend.CPU,
                runtime=runtime,
                artifact=_transformers_artifact(),
                prediction=_transformers_estimate(runtime),
            )
        )


def test_not_ready_does_not_invoke_llama_runner(tmp_path: Path) -> None:
    artifact_runner = _FakeArtifactRunner()
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        artifact_runner=artifact_runner,
    )
    request = _request(
        tmp_path,
        backend=ComputeBackend.VULKAN,
        runtime=_runtime(n_gpu_layers=-1),
        hardware=_hardware_with_vulkan(),
        prediction_device=TargetDevice.GPU,
    )

    with pytest.raises(ExperimentNotReadyError) as captured:
        runner.run(request)

    assert artifact_runner.calls == []
    assert captured.value.readiness.status is ExecutionReadinessStatus.NOT_READY


def test_unknown_readiness_does_not_invoke_llama_runner(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    artifact_runner = _FakeArtifactRunner()
    backend = _FakeExecutionBackend(
        {
            (str(binary), "--list-devices"): ExecutionFailedError(
                "probe failed",
                _execution_result(
                    success=False,
                    exit_code=1,
                    failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
                ),
            )
        }
    )
    runner = ExperimentRunner(
        execution_backend=backend,
        llama_cpp_runner=artifact_runner,
        store=ExperimentStore(root=tmp_path / "experiments"),
        llama_cli_path=binary,
    )

    with pytest.raises(ExperimentNotReadyError) as captured:
        runner.run(_request(tmp_path, backend=ComputeBackend.CPU))

    assert artifact_runner.calls == []
    assert captured.value.readiness.status is ExecutionReadinessStatus.UNKNOWN


def test_comparison_is_generated_by_runner(tmp_path: Path) -> None:
    runner = _experiment_runner(tmp_path, devices="Available devices:\n")

    result = runner.run(_request(tmp_path, backend=ComputeBackend.CPU, persist=False))

    assert result.record.comparison.ram.predicted_bytes == 1000
    assert result.record.comparison.ram.measured_bytes == 1000


def test_no_persistence_leaves_store_untouched(tmp_path: Path) -> None:
    runner = _experiment_runner(tmp_path, devices="Available devices:\n")

    result = runner.run(_request(tmp_path, backend=ComputeBackend.CPU, persist=False))

    assert result.persisted_path is None
    assert runner.store is not None
    assert runner.store.list_ids() == []


def test_persistence_requested_without_store_raises_after_record_build(
    tmp_path: Path,
) -> None:
    binary = _fake_binary(tmp_path)
    runner = ExperimentRunner(
        execution_backend=_FakeExecutionBackend(
            {(str(binary), "--list-devices"): "Available devices:\n"}
        ),
        llama_cpp_runner=_FakeArtifactRunner(),
        store=None,
        llama_cli_path=binary,
    )

    with pytest.raises(ExperimentRunnerError, match="no ExperimentStore"):
        runner.run(_request(tmp_path, backend=ComputeBackend.CPU, persist=True))


def test_runtime_configuration_mismatch_fails_before_execution(tmp_path: Path) -> None:
    artifact_runner = _FakeArtifactRunner()
    runner = _experiment_runner(
        tmp_path,
        devices="Available devices:\n",
        artifact_runner=artifact_runner,
    )
    runtime = _runtime(context_size=2048, n_gpu_layers=0)
    request = _request(
        tmp_path,
        backend=ComputeBackend.CPU,
        runtime=runtime,
        prediction=_estimate(_runtime(context_size=4096, n_gpu_layers=0)),
    )

    with pytest.raises(ExperimentConfigurationError, match="ctx-size"):
        runner.run(request)

    assert artifact_runner.calls == []


def test_advisor_run_experiment_delegates_to_experiment_runner(tmp_path: Path) -> None:
    experiment_runner = _experiment_runner(tmp_path, devices="Available devices:\n")
    advisor = AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=_hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
        experiment_runner=experiment_runner,
    )

    result = advisor.run_experiment(
        _request(tmp_path, backend=ComputeBackend.CPU, persist=False)
    )

    assert result.record.workload is not None
    assert result.record.workload.prompt == "Hello experimental world"


class _FakeExecutionBackend:
    def __init__(self, outputs: dict[tuple[str, ...], object]) -> None:
        self.outputs = outputs
        self.requests: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        outcome = self.outputs.get(request.command)
        if outcome is None and _is_pytorch_probe(request.command):
            outcome = self.outputs.get(("PYTORCH_PROBE",))
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return _execution_result(stdout=outcome)
        return _execution_result(stdout="llama-cli test")


class _FakeArtifactRunner:
    def __init__(
        self,
        *,
        error_observation: ExecutionObservation | None = None,
    ) -> None:
        self.error_observation = error_observation
        self.calls: list[tuple[ModelArtifact, str, RuntimeRecommendation | None]] = []

    def run(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        self.calls.append((artifact, prompt, runtime))
        if self.error_observation is not None:
            raise ExecutionFailedError(
                "llama-cli failed",
                ExecutionResult(
                    stdout="",
                    stderr="failure",
                    observation=self.error_observation,
                ),
            )
        return InferenceResult(
            text="response",
            runtime=runtime.runtime.value if runtime is not None else RuntimeName.LLAMA_CPP.value,
            model_path=artifact.local_path or Path("/tmp/model.gguf"),
            observation=_observation(),
        )


class _FakeHfClient:
    def model_info(self, repo_id: str) -> object:
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


def _experiment_runner(
    tmp_path: Path,
    *,
    devices: str,
    artifact_runner: _FakeArtifactRunner | None = None,
    transformers_runner: _FakeArtifactRunner | None = None,
    pytorch_probe: str | None = None,
    store: ExperimentStore | None = None,
) -> ExperimentRunner:
    binary = _fake_binary(tmp_path)
    outputs: dict[tuple[str, ...], object] = {
        (str(binary), "--list-devices"): devices,
        (str(binary), "--version"): "llama-cli b123",
    }
    if pytorch_probe is not None:
        outputs[("PYTORCH_PROBE",)] = pytorch_probe
    return ExperimentRunner(
        execution_backend=_FakeExecutionBackend(outputs),
        llama_cpp_runner=artifact_runner or _FakeArtifactRunner(),
        transformers_runner=transformers_runner,
        store=store if store is not None else ExperimentStore(root=tmp_path / "store"),
        llama_cli_path=binary,
    )


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "llama-cli"
    binary.write_text("fake", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _request(
    tmp_path: Path,
    *,
    backend: ComputeBackend,
    runtime: RuntimeRecommendation | None = None,
    prediction: MemoryEstimate | None = None,
    artifact: ModelArtifact | None = None,
    hardware: HardwareProfile | None = None,
    prediction_device: TargetDevice = TargetDevice.CPU,
    persist: bool = True,
) -> ExperimentRequest:
    effective_runtime = runtime or _runtime(n_gpu_layers=0)
    return ExperimentRequest(
        hardware=hardware or _hardware(),
        artifact=artifact or _artifact(tmp_path),
        runtime=effective_runtime,
        prediction=prediction
        or _estimate(effective_runtime, effective_device=prediction_device),
        backend_selection=RuntimeBackendSelection(
            selected_backend=backend,
            reason=RuntimeBackendSelectionReason.CPU_FALLBACK
            if backend is ComputeBackend.CPU
            else RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
        ),
        requested_backend=_requested_backend(backend),
        workload=ExperimentWorkload(prompt="Hello experimental world"),
        persist=persist,
    )


def _requested_backend(backend: ComputeBackend) -> RequestedComputeBackend:
    return RequestedComputeBackend(backend.value)


def _artifact(tmp_path: Path) -> ModelArtifact:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fake")
    return ModelArtifact(
        repo_id="owner/repo",
        revision="main",
        filename="model.gguf",
        format="gguf",
        quantization="Q4_K_M",
        local_path=path,
        sha256="1" * 64,
        is_downloaded=True,
        is_verified=True,
    )


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU"),
        memory=MemoryInfo(total_bytes=16 * GIB, available_bytes=8 * GIB),
    )


def _hardware_with_vulkan() -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        arch="AMD64",
        cpu=CpuInfo(model="Ryzen 5 5500U"),
        memory=MemoryInfo(total_bytes=16 * GIB, available_bytes=8 * GIB),
        accelerators=[
            AcceleratorProfile(
                name="AMD Radeon(TM) Graphics",
                vendor=AcceleratorVendor.AMD,
                type=AcceleratorType.INTEGRATED,
                shared_memory=True,
                backends=[
                    ComputeBackendInfo(
                        backend=ComputeBackend.VULKAN,
                        availability=BackendAvailability.AVAILABLE,
                    )
                ],
            )
        ],
    )


def _runtime(
    *,
    context_size: int = 4096,
    n_gpu_layers: int,
) -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(context_size),
                source=RuntimeFlagSource.USER_INPUT,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(n_gpu_layers),
                source=RuntimeFlagSource.USER_INPUT,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _transformers_runtime() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        flags=[
            RuntimeFlag(
                name="device_map",
                value="cpu",
                source=RuntimeFlagSource.USER_INPUT,
                explanation="test",
            ),
            RuntimeFlag(
                name="torch_dtype",
                value="torch.float32",
                source=RuntimeFlagSource.USER_INPUT,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _estimate(
    runtime: RuntimeRecommendation,
    *,
    effective_device: TargetDevice = TargetDevice.CPU,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="owner/repo"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=effective_device,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=500,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            )
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV",
                bytes=300,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=1,
            kv_heads=1,
            head_dim=1,
            context_length=4096,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=200,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=200,
            weight_fraction=0.1,
            minimum_bytes=100,
        ),
        device_reserve=MemoryComponent(
            name="Reserve",
            bytes=100,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=1100,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMPATIBLE,
            confidence=EstimationConfidence.HIGH,
            target_device=effective_device,
            effective_device=effective_device,
            available_ram_bytes=8 * GIB,
            available_vram_bytes=2 * GIB
            if effective_device is TargetDevice.GPU
            else None,
        ),
        runtime_recommendation=runtime,
    )


def _transformers_artifact() -> ModelArtifact:
    return ModelArtifact(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="main",
        filename="Qwen/Qwen2.5-0.5B-Instruct",
        format="safetensors",
    )


def _transformers_estimate(runtime: RuntimeRecommendation) -> MemoryEstimate:
    return _estimate(runtime).model_copy(
        update={
            "repository": ModelRepositoryInfo(repo_id="Qwen/Qwen2.5-0.5B-Instruct"),
            "repository_type": RepositoryType.TRANSFORMERS,
            "inference_configuration": InferenceConfiguration(
                context_length=4096,
                target_device=TargetDevice.CPU,
                precision=None,
            ),
        }
    )


def _observation(
    *,
    success: bool = True,
    exit_code: int = 0,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=1.0,
        peak_ram_bytes=1000,
        peak_vram_bytes=None,
        exit_code=exit_code,
        failure_reason=failure_reason,
    )


def _pytorch_probe_json(*, runtime_status: str = "available") -> str:
    payload: dict[str, object] = {
        "runtime_status": runtime_status,
    }
    if runtime_status == "available":
        payload.update(
            {
                "torch_version": "2.8.0",
                "transformers_version": "4.55.0",
                "torch_cuda_version": None,
                "torch_hip_version": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "devices": [],
            }
        )
    return json.dumps(payload)


def _is_pytorch_probe(command: tuple[str, ...]) -> bool:
    return len(command) == 3 and command[1] == "-c" and "importlib" in command[2]


def _execution_result(
    *,
    stdout: str = "",
    success: bool = True,
    exit_code: int | None = 0,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr="",
        observation=ExecutionObservation(
            success=success,
            duration_seconds=0.01,
            exit_code=exit_code,
            failure_reason=failure_reason,
        ),
    )


def _unused_inspect(repo_id: str, client: object | None = None) -> ModelAnalysis:
    del repo_id, client
    raise NotImplementedError


def _unused_estimate(**kwargs: Any) -> MemoryEstimate:
    del kwargs
    raise NotImplementedError
