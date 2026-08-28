from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.comparison import (
    CompatibilityComparison,
    CompatibilityOutcome,
    MetricComparison,
    MetricComparisonAvailability,
    PredictionComparison,
)
from jaull.domain.enums import Format, RepositoryType
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
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import (
    ExperimentEnvironment,
    ExperimentIdentity,
    ExperimentPredictionInput,
    ExperimentRecord,
    ExperimentReplayabilityStatus,
    ExperimentWorkload,
)
from jaull.domain.hardware import (
    ComputeBackend,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.experiments import build_experiment_record
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.experiments.reevaluation import ExperimentReevaluationService

GIB = 1024**3
_MISSING = object()


def test_reevaluation_uses_frozen_hardware_snapshot() -> None:
    record = _record(hardware=_rtx_2060_hardware())
    seen_hardware: list[HardwareProfile] = []

    def fake_estimate(**kwargs: Any) -> MemoryEstimate:
        seen_hardware.append(kwargs["hardware"])
        return _estimate(runtime=_runtime(), total=1_200)

    result = ExperimentReevaluationService(
        estimate_memory_fn=fake_estimate,
        environment_factory=_environment,
    ).reevaluate(record)

    assert result.replayability.status is ExperimentReplayabilityStatus.REPRODUCIBLE
    assert seen_hardware == [record.hardware]
    assert seen_hardware[0].gpus[0].name == "RTX 2060"


def test_reevaluation_does_not_mutate_historical_prediction_or_observation() -> None:
    record = _record()
    before = record.model_dump(mode="json")

    ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_300),
        environment_factory=_environment,
    ).reevaluate(record)

    assert record.model_dump(mode="json") == before


def test_reevaluation_keeps_original_and_current_predictions_separate() -> None:
    record = _record(prediction=_estimate(runtime=_runtime(), total=1_000))

    result = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_400),
        environment_factory=_environment,
    ).reevaluate(record)

    assert result.original_prediction == record.prediction
    assert result.current_prediction is not None
    assert result.current_prediction.total_bytes == 1_400
    assert result.observation == record.observation


def test_comparison_rules_can_change_without_rewriting_raw_prediction_values() -> None:
    record = _record(prediction=_estimate(runtime=_runtime(), total=1_000))
    current = _estimate(runtime=_runtime(), total=1_000)

    def changed_comparison(**kwargs: Any) -> PredictionComparison:
        estimate = kwargs["estimate"]
        observation = kwargs["observation"]
        return PredictionComparison(
            ram=MetricComparison(
                predicted_bytes=estimate.weights.component.bytes
                + estimate.kv_cache.component.bytes
                + estimate.runtime_overhead.component.bytes,
                measured_bytes=observation.peak_ram_bytes,
                error_bytes=0,
                absolute_error_bytes=0,
                error_percent=0.0,
                availability=MetricComparisonAvailability.AVAILABLE,
            ),
            vram=_unavailable_metric("test evaluator does not compare VRAM"),
            compatibility=CompatibilityComparison(
                predicted_status=estimate.assessment.status,
                predicted_runnable=True,
                observed_success=observation.success,
                outcome=CompatibilityOutcome.UNKNOWN,
            ),
        )

    result = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: current,
        compare_prediction_fn=changed_comparison,
        environment_factory=_environment,
    ).reevaluate(record)

    assert result.current_prediction == current
    assert result.current_comparison is not None
    assert result.current_comparison.ram.predicted_bytes == 1_000
    assert result.current_comparison.ram.measured_bytes == 1_000
    assert result.current_comparison.compatibility.outcome is CompatibilityOutcome.UNKNOWN
    assert record.comparison.compatibility.outcome is CompatibilityOutcome.CORRECT_SUCCESS


def test_missing_prediction_input_is_not_reconstructed_silently() -> None:
    record = _record(prediction_input=None)

    result = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_200),
        environment_factory=_environment,
    ).reevaluate(record)

    assert result.replayability.status is ExperimentReplayabilityStatus.NOT_REPRODUCIBLE
    assert result.current_prediction is None
    assert "missing prediction input snapshot" in result.replayability.reasons


def test_historical_schema_without_prediction_input_loads_as_not_reproducible() -> None:
    payload = _record().model_dump(mode="json")
    payload.pop("prediction_input")
    restored = ExperimentRecord.model_validate(payload)

    result = ExperimentReevaluationService(environment_factory=_environment).reevaluate(
        restored
    )

    assert restored.prediction_input is None
    assert result.replayability.status is ExperimentReplayabilityStatus.NOT_REPRODUCIBLE
    assert "missing prediction input snapshot" in result.replayability.reasons


def test_missing_frozen_hardware_is_not_reproducible() -> None:
    record = _record().model_copy()
    incomplete = ExperimentRecord.model_construct(
        **{
            key: value
            for key, value in record.__dict__.items()
            if key != "hardware"
        }
    )

    result = ExperimentReevaluationService(environment_factory=_environment).reevaluate(
        incomplete
    )

    assert result.replayability.status is ExperimentReplayabilityStatus.NOT_REPRODUCIBLE
    assert "missing frozen hardware profile" in result.replayability.reasons


def test_missing_workload_or_artifact_data_is_reported() -> None:
    record = _record(workload=None)
    without_artifact = ExperimentRecord.model_construct(
        **{
            key: value
            for key, value in record.__dict__.items()
            if key != "artifact"
        }
    )

    result = ExperimentReevaluationService(environment_factory=_environment).reevaluate(
        without_artifact
    )

    assert result.replayability.status is ExperimentReplayabilityStatus.NOT_REPRODUCIBLE
    assert "missing workload snapshot" in result.replayability.reasons
    assert "missing artifact snapshot" in result.replayability.reasons


def test_reevaluation_is_deterministic_for_same_record_and_code() -> None:
    record = _record()
    service = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_250),
        environment_factory=_environment,
    )

    first = service.reevaluate(record)
    second = service.reevaluate(record)

    assert first == second


def test_current_host_state_is_irrelevant_to_reevaluation() -> None:
    record = _record(hardware=_rtx_2060_hardware())
    current_host = _cpu_only_hardware()
    predictions: list[int | None] = []

    def fake_estimate(**kwargs: Any) -> MemoryEstimate:
        hardware = kwargs["hardware"]
        predictions.append(hardware.gpus[0].vram_total_bytes)
        return _estimate(runtime=_runtime(), total=1_100)

    first = ExperimentReevaluationService(
        estimate_memory_fn=fake_estimate,
        environment_factory=_environment,
    ).reevaluate(record)
    second = ExperimentReevaluationService(
        estimate_memory_fn=fake_estimate,
        environment_factory=_environment,
    ).reevaluate(record)

    assert first.current_prediction == second.current_prediction
    assert current_host.gpus == []
    assert predictions == [6 * GIB, 6 * GIB]


def test_actual_estimator_can_reevaluate_from_frozen_gguf_input_without_network() -> None:
    record = _record(prediction=_estimate(runtime=_runtime(), total=1_000))

    result = ExperimentReevaluationService(environment_factory=_environment).reevaluate(
        record
    )

    assert result.replayability.status is ExperimentReplayabilityStatus.REPRODUCIBLE
    assert result.current_prediction is not None
    assert result.current_prediction.repository.repo_id == "owner/repo"
    assert result.current_comparison is not None


def test_reevaluation_blocks_network_completion_calls() -> None:
    blocked_calls: list[str] = []

    def fake_estimate(**kwargs: Any) -> MemoryEstimate:
        client = kwargs["client"]
        for name, call in (
            ("model_info", lambda: client.model_info("owner/repo")),
            (
                "download_small_file",
                lambda: client.download_small_file("owner/repo", "config.json"),
            ),
        ):
            try:
                call()
            except HuggingFaceUnavailableError:
                blocked_calls.append(name)
        return _estimate(runtime=_runtime(), total=1_100)

    result = ExperimentReevaluationService(
        estimate_memory_fn=fake_estimate,
        environment_factory=_environment,
    ).reevaluate(_record())

    assert result.current_prediction is not None
    assert blocked_calls == ["model_info", "download_small_file"]


def test_artifact_without_hash_warns_without_blocking_prediction_reproducibility() -> None:
    record = _record(artifact=_artifact(sha256=None))

    result = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_000),
        environment_factory=_environment,
    ).reevaluate(record)

    assert result.replayability.status is ExperimentReplayabilityStatus.REPRODUCIBLE
    assert result.replayability.reasons == []
    assert result.replayability.warnings == [
        "artifact hash is unavailable; prediction can be recomputed, "
        "but the physical artifact identity is not cryptographically verified"
    ]
    assert result.current_prediction is not None


def test_result_serialization_keeps_original_current_observation_and_provenance() -> None:
    record = _record()

    result = ExperimentReevaluationService(
        estimate_memory_fn=lambda **_: _estimate(runtime=_runtime(), total=1_200),
        environment_factory=_environment,
    ).reevaluate(record)
    payload = result.model_dump(mode="json")

    assert payload["mode"] == "re_evaluation"
    assert payload["original_prediction"]["total_bytes"] == 1_000
    assert payload["current_prediction"]["total_bytes"] == 1_200
    assert payload["observation"]["peak_ram_bytes"] == 1_000
    assert payload["original_comparison"]["compatibility"]["outcome"] == (
        "correct_success"
    )
    assert payload["original_prediction_engine"] == "memory-estimator-v1"
    assert payload["original_comparison_engine"] == "prediction-comparison-v1"
    assert payload["current_provenance"]["prediction_engine"] == "memory-estimator-v1"
    assert payload["current_provenance"]["comparison_engine"] == (
        "prediction-comparison-v1"
    )


def _record(
    *,
    hardware: HardwareProfile | None = None,
    artifact: ModelArtifact | None = None,
    runtime: RuntimeRecommendation | None = None,
    prediction: MemoryEstimate | None = None,
    prediction_input: ExperimentPredictionInput | object | None = _MISSING,
    workload: ExperimentWorkload | object | None = _MISSING,
) -> ExperimentRecord:
    effective_runtime = runtime or _runtime()
    effective_prediction = prediction or _estimate(runtime=effective_runtime, total=1_000)
    effective_prediction_input = (
        _prediction_input(effective_prediction)
        if prediction_input is _MISSING
        else prediction_input
    )
    effective_workload = (
        ExperimentWorkload(prompt="Explain local inference.")
        if workload is _MISSING
        else workload
    )
    capability = _runtime_capability()
    return build_experiment_record(
        hardware=hardware or _rtx_2060_hardware(),
        artifact=artifact or _artifact(),
        workload=effective_workload,
        runtime=effective_runtime,
        prediction=effective_prediction,
        prediction_input=effective_prediction_input,
        runtime_capability=capability,
        execution_readiness=_readiness(capability),
        observation=_observation(),
        identity=ExperimentIdentity(
            experiment_id="exp-test",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        environment=_environment(),
    )


def _prediction_input(prediction: MemoryEstimate) -> ExperimentPredictionInput:
    return ExperimentPredictionInput(
        analysis=_analysis(),
        inference_configuration=prediction.inference_configuration,
        resolve_base_model=False,
        recommend_runtime=False,
    )


def _analysis() -> ModelAnalysis:
    file = ModelFile(path="model.Q4_K_M.gguf", size_bytes=512)
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="owner/repo"),
        files=[file],
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=[
                GgufVariant(
                    quantization="Q4_K_M",
                    files=[file],
                    total_bytes=512,
                )
            ],
        ),
    )


def _rtx_2060_hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Ryzen 5 3600"),
        memory=MemoryInfo(total_bytes=16 * GIB, available_bytes=8 * GIB),
        gpus=[
            GpuInfo(
                name="RTX 2060",
                vram_total_bytes=6 * GIB,
                vram_available_bytes=6 * GIB,
            )
        ],
    )


def _cpu_only_hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Laptop CPU"),
        memory=MemoryInfo(total_bytes=8 * GIB, available_bytes=2 * GIB),
    )


def _artifact(**updates: object) -> ModelArtifact:
    data: dict[str, object] = {
        "repo_id": "owner/repo",
        "revision": "main",
        "filename": "model.Q4_K_M.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 512,
        "local_path": Path("/models/model.Q4_K_M.gguf"),
        "sha256": "a" * 64,
        "is_downloaded": True,
        "is_verified": True,
    }
    data.update(updates)
    return ModelArtifact(**data)


def _runtime() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value="4096",
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value="0",
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _estimate(
    *,
    runtime: RuntimeRecommendation,
    total: int,
) -> MemoryEstimate:
    weights = 500
    kv = 300
    overhead = total - weights - kv
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="owner/repo"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.CPU,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=weights,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            gguf_variant="Q4_K_M",
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=kv,
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
                bytes=overhead,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=overhead,
            weight_fraction=0.0,
            minimum_bytes=0,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=0,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=total,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMPATIBLE,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.CPU,
            effective_device=TargetDevice.CPU,
            available_ram_bytes=8 * GIB,
        ),
        runtime_recommendation=runtime,
    )


def _runtime_capability() -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path="/usr/bin/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
                source="test",
            )
        ],
    )


def _readiness(capability: LlamaCppRuntimeCapability):
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

    return evaluate_execution_readiness(
        selection=RuntimeBackendSelection(
            selected_backend=ComputeBackend.CPU,
            reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
        ),
        runtime_capability=capability,
    )


def _observation() -> ExecutionObservation:
    return ExecutionObservation(
        success=True,
        duration_seconds=1.0,
        peak_ram_bytes=1_000,
        peak_vram_bytes=None,
        exit_code=0,
    )


def _unavailable_metric(reason: str) -> MetricComparison:
    return MetricComparison(
        availability=MetricComparisonAvailability.METHODOLOGICALLY_UNAVAILABLE,
        unavailable_reason=reason,
    )


def _environment() -> ExperimentEnvironment:
    return ExperimentEnvironment(
        jaull_version="test",
        python_version="3.12.0",
        python_implementation="CPython",
        git_commit="abc123",
    )
