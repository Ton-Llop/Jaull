from __future__ import annotations

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
from jaull.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelRepositoryInfo
from jaull.runtime import transformers as transformers_runtime

GIB = 1024**3


def _hw() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=32 * GIB),
        storage=[],
        gpus=[
            GpuInfo(
                name="Test",
                vram_total_bytes=24 * GIB,
                vram_available_bytes=24 * GIB,
                driver_version="1.0",
                cuda_version="12.4",
            )
        ],
        warnings=[],
    )


def _estimate(
    *,
    status: CompatibilityStatus,
    effective_device: TargetDevice,
    precision: WeightPrecision | None = WeightPrecision.FLOAT16,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="user/tf-model"),
        repository_type=RepositoryType.TRANSFORMERS,
        inference_configuration=InferenceConfiguration(
            context_length=2048, target_device=TargetDevice.AUTO
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=2 * GIB,
                source=EstimateSource.METADATA,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            precision=precision,
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=256 * 1024 * 1024,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=32,
            kv_heads=32,
            head_dim=128,
            context_length=2048,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=512 * 1024 * 1024,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=512 * 1024 * 1024,
            weight_fraction=0.1,
            minimum_bytes=256 * 1024 * 1024,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=0,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=3 * GIB,
        assessment=CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=effective_device,
            available_vram_bytes=24 * GIB,
            available_ram_bytes=32 * GIB,
            ratio=0.1,
        ),
    )


def test_gpu_recommends_cuda_device_map() -> None:
    est = _estimate(
        status=CompatibilityStatus.COMFORTABLE, effective_device=TargetDevice.GPU
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "cuda"
    assert rec.python_snippet is not None
    assert "cuda" in rec.python_snippet


def test_offloading_uses_auto_device_map_with_warning() -> None:
    est = _estimate(
        status=CompatibilityStatus.OFFLOADING_REQUIRED,
        effective_device=TargetDevice.GPU,
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "auto"
    assert any("accelerate" in w.lower() for w in rec.warnings)


def test_cpu_recommends_cpu_device_map() -> None:
    est = _estimate(
        status=CompatibilityStatus.COMPATIBLE, effective_device=TargetDevice.CPU
    )
    rec = transformers_runtime.build(est, _hw())
    device_map_flag = next(f for f in rec.flags if f.name == "device_map")
    assert device_map_flag.value == "cpu"
    assert rec.python_snippet is not None
    assert "cpu" in rec.python_snippet
