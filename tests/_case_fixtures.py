"""Records for experimental-case tests.

One machine, one artifact, one runtime: the shape a real case has. Every builder
takes overrides so a test can perturb exactly the one fact it is about.
"""

from __future__ import annotations

from pathlib import Path

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
    BenchmarkObservation,
    BenchmarkRecord,
    BenchmarkRequest,
    LlamaBenchBinaryStatus,
    LlamaBenchCapability,
)
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
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import (
    ComputeBackend,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelRepositoryInfo
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

GIB = 1024**3


def hardware(
    *,
    ram_available_bytes: int = 7 * GIB,
    vram_available_bytes: int = 4 * GIB,
    gpu_name: str = "NVIDIA GeForce RTX 2060",
    driver_version: str = "610.88",
    cuda_version: str = "13.3",
) -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        os_version="6.18",
        arch="x86_64",
        cpu=CpuInfo(model="AMD Ryzen 5 3600", physical_cores=6, logical_cores=12),
        memory=MemoryInfo(total_bytes=8 * GIB, available_bytes=ram_available_bytes),
        gpus=[
            GpuInfo(
                name=gpu_name,
                vram_total_bytes=6 * GIB,
                vram_available_bytes=vram_available_bytes,
                driver_version=driver_version,
                cuda_version=cuda_version,
            )
        ],
    )


def artifact(
    *,
    repo_id: str = "bartowski/Qwen2.5-7B-Instruct-GGUF",
    quantization: str = "Q4_K_M",
    sha256: str | None = "a" * 64,
) -> ModelArtifact:
    return ModelArtifact(
        repo_id=repo_id,
        revision="d1a2b3c",
        filename=f"Qwen2.5-7B-Instruct-{quantization}.gguf",
        format="gguf",
        quantization=quantization,
        size_bytes=4_683_074_240,
        local_path=Path("/models/qwen.gguf"),
        sha256=sha256,
        is_downloaded=True,
        is_verified=True,
    )


def runtime(
    *,
    name: RuntimeName = RuntimeName.LLAMA_CPP,
    context_length: int = 4096,
) -> RuntimeRecommendation:
    # ``assert_prediction_runtime_matches`` refuses a record whose --ctx-size
    # disagrees with the estimate, so the two move together here.
    return RuntimeRecommendation(
        runtime=name,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(context_length),
                source=RuntimeFlagSource.ESTIMATE,
                explanation="from estimate",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value="18",
                source=RuntimeFlagSource.HARDWARE,
                explanation="from hardware",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def runtime_capability(*, version_text: str | None = "version: 10357 (689e227db)"):
    return LlamaCppRuntimeCapability(
        binary_path="/home/ton/tools/llama.cpp/build-cuda/bin/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        version_text=version_text,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CUDA,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
                source="test",
            )
        ],
    )


def experiment_record(
    *,
    profile: HardwareProfile | None = None,
    model_artifact: ModelArtifact | None = None,
    recommendation: RuntimeRecommendation | None = None,
    version_text: str | None = "version: 10357 (689e227db)",
    context_length: int = 4096,
) -> ExperimentRecord:
    resolved_runtime = recommendation or runtime(context_length=context_length)
    capability = runtime_capability(version_text=version_text)
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

    readiness = evaluate_execution_readiness(
        selection=RuntimeBackendSelection(
            selected_backend=ComputeBackend.CUDA,
            reason=RuntimeBackendSelectionReason.NATIVE_BACKEND_AVAILABLE,
        ),
        runtime_capability=capability,
    )
    return build_experiment_record(
        hardware=profile or hardware(),
        artifact=model_artifact or artifact(),
        runtime=resolved_runtime,
        prediction=_estimate(resolved_runtime, context_length=context_length),
        runtime_capability=capability,
        execution_readiness=readiness,
        observation=ExecutionObservation(
            success=True,
            duration_seconds=5.73,
            peak_ram_bytes=4_951_216_128,
            exit_code=0,
        ),
    )


def benchmark_record(
    *,
    profile: HardwareProfile | None = None,
    model_artifact: ModelArtifact | None = None,
    recommendation: RuntimeRecommendation | None = None,
    version_text: str | None = "llama-bench version: 10357 (689e227db)",
    context_length: int | None = None,
) -> BenchmarkRecord:
    request = BenchmarkRequest(
        artifact=model_artifact or artifact(),
        runtime=recommendation or runtime(),
        backend=ComputeBackend.CPU,
        device="none",
        gpu_layers=BenchmarkGpuLayers.count_layers(0),
        context_length=context_length,
    )
    return BenchmarkRecord.create(
        hardware=profile or hardware(),
        request=request,
        observation=BenchmarkObservation(
            success=True,
            measurements=[
                BenchmarkMeasurement(
                    kind=BenchmarkMeasurementKind.GENERATION,
                    tokens=128,
                    mean_tokens_per_second=13.4,
                    stddev_tokens_per_second=0.2,
                    source_label="tg128",
                    raw_ngl="0",
                )
            ],
            repetitions=5,
            duration_seconds=42.0,
            methodology="llama_bench_v1",
            command=("llama-bench", "-m", "/models/qwen.gguf", "-ngl", "0"),
        ),
        llama_bench_capability=LlamaBenchCapability(
            binary_path="/home/ton/tools/llama.cpp/build-cuda/bin/llama-bench",
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            version_text=version_text,
        ),
    )


def _estimate(
    recommendation: RuntimeRecommendation,
    *,
    context_length: int,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=context_length,
            target_device=TargetDevice.GPU,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=4_683_074_240,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            )
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=234_881_024,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=28,
            kv_heads=4,
            head_dim=128,
            context_length=context_length,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=1_005_178_336,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=536_870_912,
            weight_fraction=0.1,
            minimum_bytes=268_435_456,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=536_870_912,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=7_106_004_964,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMPATIBLE,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.GPU,
            effective_device=TargetDevice.GPU,
            available_ram_bytes=7 * GIB,
            available_vram_bytes=4 * GIB,
        ),
        runtime_recommendation=recommendation,
    )


__all__ = [
    "GIB",
    "artifact",
    "benchmark_record",
    "experiment_record",
    "hardware",
    "runtime",
    "runtime_capability",
]
