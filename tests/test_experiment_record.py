from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.comparison import PredictionComparison
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
)
from jaull.domain.experiments import (
    EXPERIMENT_RECORD_SCHEMA_VERSION,
    ExperimentBackendTrace,
    ExperimentEnvironment,
    ExperimentIdentity,
    ExperimentPredictionInput,
    ExperimentRecord,
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
from jaull.domain.model import (
    ModelAnalysis,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.experiments import build_experiment_record

GIB = 1024**3


def test_complete_record_can_be_built() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(runtime=runtime, prediction=_estimate(runtime=runtime))

    assert record.identity.experiment_id.startswith("exp-")
    assert record.schema_version == EXPERIMENT_RECORD_SCHEMA_VERSION
    assert record.hardware.cpu.model == "Ryzen 5 5500U"
    assert record.artifact.filename == "tinyllama.Q4_K_M.gguf"
    assert record.runtime is runtime
    assert record.prediction_engine == "memory-estimator-v1"
    assert record.comparison_engine == "prediction-comparison-v1"
    assert isinstance(record.comparison, PredictionComparison)
    assert record.observation.success is True


def test_two_experiments_get_different_ids() -> None:
    runtime = _runtime(n_gpu_layers=0)

    first = _record(runtime=runtime, prediction=_estimate(runtime=runtime))
    second = _record(runtime=runtime, prediction=_estimate(runtime=runtime))

    assert first.identity.experiment_id != second.identity.experiment_id


def test_created_at_is_timezone_aware() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(runtime=runtime, prediction=_estimate(runtime=runtime))

    assert record.identity.created_at.tzinfo is not None
    assert record.identity.created_at.utcoffset() is not None


def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentIdentity(
            experiment_id="exp-test",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
        )


def test_serialization_roundtrip_preserves_semantics() -> None:
    runtime = _runtime(n_gpu_layers=0)
    record = _record(runtime=runtime, prediction=_estimate(runtime=runtime))

    payload = record.model_dump(mode="json")
    restored = ExperimentRecord.model_validate(payload)

    assert restored == record
    assert payload["schema_version"] == EXPERIMENT_RECORD_SCHEMA_VERSION


def test_cpu_experiment_ready_success_is_valid() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(
        runtime=runtime,
        prediction=_estimate(runtime=runtime, target_device=TargetDevice.CPU),
        selection=_selection(ComputeBackend.CPU),
        capability=_capability(ComputeBackend.CPU),
    )

    assert record.preflight.execution_readiness.status is ExecutionReadinessStatus.READY
    assert record.observation.success is True


def test_accelerator_experiment_ready_success_is_valid() -> None:
    runtime = _runtime(n_gpu_layers=-1)
    capability = _capability(ComputeBackend.VULKAN)

    record = _record(
        hardware=_hardware_with_accelerator(),
        runtime=runtime,
        prediction=_estimate(
            runtime=runtime,
            target_device=TargetDevice.GPU,
            effective_device=TargetDevice.GPU,
        ),
        selection=_selection(ComputeBackend.VULKAN),
        capability=capability,
    )

    assert record.preflight.execution_readiness.status is ExecutionReadinessStatus.READY
    assert record.preflight.runtime_capability is capability
    assert record.observation.success is True


def test_failed_execution_is_valid_experimental_evidence() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(
        runtime=runtime,
        prediction=_estimate(runtime=runtime),
        observation=_observation(success=False),
    )

    assert record.observation.success is False
    assert record.comparison.compatibility.observed_success is False


def test_readiness_ready_execution_failure_is_valid() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(
        runtime=runtime,
        prediction=_estimate(runtime=runtime),
        observation=_observation(
            success=False,
            failure_reason=ExecutionFailureReason.NON_ZERO_EXIT,
        ),
    )

    assert record.preflight.execution_readiness.status is ExecutionReadinessStatus.READY
    assert record.observation.success is False


def test_missing_optional_environment_provenance_is_valid() -> None:
    runtime = _runtime(n_gpu_layers=0)

    record = _record(
        runtime=runtime,
        prediction=_estimate(runtime=runtime),
        environment=ExperimentEnvironment(),
    )

    assert record.environment.jaull_version is None
    assert record.environment.git_commit is None


def test_backend_trace_separates_requested_selected_and_observed() -> None:
    runtime = _runtime(n_gpu_layers=-1)
    record = _record(
        hardware=_hardware_with_accelerator(),
        runtime=runtime,
        prediction=_estimate(
            runtime=runtime,
            target_device=TargetDevice.GPU,
            effective_device=TargetDevice.GPU,
        ),
        selection=_selection(ComputeBackend.VULKAN),
        capability=_capability(ComputeBackend.VULKAN),
        backend_trace=ExperimentBackendTrace(
            requested_backend=RequestedComputeBackend.AUTO,
            observed_backend=None,
            observed_source=None,
        ),
    )

    assert record.backend_trace.requested_backend is RequestedComputeBackend.AUTO
    assert (
        record.preflight.execution_readiness.selection.selected_backend
        is ComputeBackend.VULKAN
    )
    assert record.backend_trace.observed_backend is None


def test_prediction_input_snapshot_is_preserved() -> None:
    runtime = _runtime(n_gpu_layers=0)
    prediction = _estimate(runtime=runtime)
    snapshot = ExperimentPredictionInput(
        analysis=_analysis(),
        inference_configuration=prediction.inference_configuration,
    )
    capability = _capability(ComputeBackend.CPU)

    record = build_experiment_record(
        hardware=_hardware(),
        artifact=_artifact(),
        runtime=runtime,
        prediction=prediction,
        prediction_input=snapshot,
        runtime_capability=capability,
        execution_readiness=_readiness(
            selection=_selection(ComputeBackend.CPU),
            capability=capability,
        ),
        observation=_observation(),
    )

    assert record.prediction_input == snapshot


def test_advisor_build_experiment_record_delegates_to_builder() -> None:
    runtime = _runtime(n_gpu_layers=0)
    prediction = _estimate(runtime=runtime)
    capability = _capability(ComputeBackend.CPU)
    advisor = AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=_hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
    )

    record = advisor.build_experiment_record(
        hardware=_hardware(),
        artifact=_artifact(),
        runtime=runtime,
        prediction=prediction,
        runtime_capability=capability,
        execution_readiness=_readiness(
            selection=_selection(ComputeBackend.CPU),
            capability=capability,
        ),
        observation=_observation(),
    )

    assert record.runtime is runtime
    assert record.prediction is prediction


def test_prediction_runtime_mismatch_is_rejected() -> None:
    runtime = _runtime(n_gpu_layers=0)
    mismatched = _estimate(runtime=_runtime(context_size=2048, n_gpu_layers=0))

    with pytest.raises(ValueError, match="Experiment runtime must match"):
        _record(runtime=runtime, prediction=mismatched)


def test_record_rejects_preflight_capability_mismatch() -> None:
    runtime = _runtime(n_gpu_layers=0)
    prediction = _estimate(runtime=runtime)
    capability = _capability(ComputeBackend.CPU)
    other_capability = _capability(ComputeBackend.VULKAN)
    readiness = _readiness(
        selection=_selection(ComputeBackend.CPU),
        capability=other_capability,
    )

    with pytest.raises(ValidationError, match="same runtime capability"):
        ExperimentRecord.create(
            hardware=_hardware(),
            artifact=_artifact(),
            runtime=runtime,
            prediction=prediction,
            runtime_capability=capability,
            execution_readiness=readiness,
            observation=_observation(),
            comparison=build_experiment_record(
                hardware=_hardware(),
                artifact=_artifact(),
                runtime=runtime,
                prediction=prediction,
                runtime_capability=capability,
                execution_readiness=_readiness(
                    selection=_selection(ComputeBackend.CPU),
                    capability=capability,
                ),
                observation=_observation(),
            ).comparison,
    )


class _FakeHfClient:
    def model_info(self, repo_id: str) -> object:
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


def _unused_inspect(repo_id: str, client: object | None = None) -> ModelAnalysis:
    del repo_id, client
    raise NotImplementedError


def _analysis() -> ModelAnalysis:
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF"),
        classification=RepositoryClassification(primary_type=RepositoryType.TRANSFORMERS),
    )


def _unused_estimate(**kwargs: object) -> MemoryEstimate:
    del kwargs
    raise NotImplementedError


def _record(
    *,
    runtime: RuntimeRecommendation,
    prediction: MemoryEstimate,
    hardware: HardwareProfile | None = None,
    selection: RuntimeBackendSelection | None = None,
    capability: LlamaCppRuntimeCapability | None = None,
    backend_trace: ExperimentBackendTrace | None = None,
    observation: ExecutionObservation | None = None,
    environment: ExperimentEnvironment | None = None,
) -> ExperimentRecord:
    selected = selection or _selection(ComputeBackend.CPU)
    runtime_capability = capability or _capability(selected.selected_backend)
    return build_experiment_record(
        hardware=hardware or _hardware(),
        artifact=_artifact(),
        backend_trace=backend_trace,
        runtime=runtime,
        prediction=prediction,
        runtime_capability=runtime_capability,
        execution_readiness=_readiness(
            selection=selected,
            capability=runtime_capability,
        ),
        observation=observation or _observation(),
        environment=environment,
    )


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF",
        revision="main",
        filename="tinyllama.Q4_K_M.gguf",
        format="gguf",
        quantization="Q4_K_M",
        size_bytes=669 * 1024 * 1024,
        local_path=Path("/models/tinyllama.Q4_K_M.gguf"),
        sha256="0" * 64,
        is_downloaded=True,
        is_verified=True,
    )


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Ryzen 5 5500U", physical_cores=6, logical_cores=12),
        memory=MemoryInfo(total_bytes=16 * GIB, available_bytes=8 * GIB),
    )


def _hardware_with_accelerator() -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        arch="AMD64",
        cpu=CpuInfo(model="Ryzen 5 5500U", physical_cores=6, logical_cores=12),
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
        command_preview=None,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(context_size),
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value=str(n_gpu_layers),
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _estimate(
    *,
    runtime: RuntimeRecommendation,
    target_device: TargetDevice = TargetDevice.CPU,
    effective_device: TargetDevice = TargetDevice.CPU,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=target_device,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=500,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            gguf_variant="Q4_K_M",
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=300,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=22,
            kv_heads=4,
            head_dim=64,
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
            name="Device reserve",
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
            target_device=target_device,
            effective_device=effective_device,
            available_ram_bytes=8 * GIB,
            available_vram_bytes=2 * GIB
            if effective_device is TargetDevice.GPU
            else None,
        ),
        runtime_recommendation=runtime,
    )


def _selection(backend: ComputeBackend) -> RuntimeBackendSelection:
    return RuntimeBackendSelection(
        selected_backend=backend,
        reason=RuntimeBackendSelectionReason.CPU_FALLBACK
        if backend is ComputeBackend.CPU
        else RuntimeBackendSelectionReason.VULKAN_BACKEND_AVAILABLE,
    )


def _capability(backend: ComputeBackend) -> LlamaCppRuntimeCapability:
    devices = []
    if backend is not ComputeBackend.CPU:
        devices.append(
            LlamaCppRuntimeDevice(
                backend=backend,
                runtime_id=f"{backend.value.title()}0",
                name="test accelerator",
            )
        )
    backend_capability = LlamaCppBackendCapability(
        backend=backend,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        devices=devices,
        reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE
        if backend is ComputeBackend.CPU
        else LlamaCppCapabilityReason.BACKEND_EXPOSED,
        source="test",
    )
    return LlamaCppRuntimeCapability(
        binary_path="/usr/bin/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text=None,
        backend_capabilities=[backend_capability],
        probe_source="test",
    )


def _readiness(
    *,
    selection: RuntimeBackendSelection,
    capability: LlamaCppRuntimeCapability,
):
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

    return evaluate_execution_readiness(
        selection=selection,
        runtime_capability=capability,
    )


def _observation(
    *,
    success: bool = True,
    failure_reason: ExecutionFailureReason | None = None,
) -> ExecutionObservation:
    return ExecutionObservation(
        success=success,
        duration_seconds=1.25,
        peak_ram_bytes=1000,
        peak_vram_bytes=None,
        exit_code=0 if success else 1,
        failure_reason=failure_reason,
    )
