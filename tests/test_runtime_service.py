from __future__ import annotations

from local_ai_check.domain.enums import RepositoryType
from local_ai_check.domain.estimation import (
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
from local_ai_check.domain.hardware import (
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from local_ai_check.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from local_ai_check.domain.model import ModelRepositoryInfo
from local_ai_check.domain.runtime import RuntimeName
from local_ai_check.runtime import service as runtime_service

GIB = 1024**3


def _hw() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=32 * GIB),
        storage=[],
        gpus=[],
        warnings=[],
    )


def _est(
    status: CompatibilityStatus,
    repo_type: RepositoryType = RepositoryType.TRANSFORMERS,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="user/repo"),
        repository_type=repo_type,
        inference_configuration=InferenceConfiguration(
            context_length=4096, target_device=TargetDevice.AUTO
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=1 * GIB,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            precision=WeightPrecision.FLOAT16,
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=256 * 1024 * 1024,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=None,
            kv_heads=None,
            head_dim=None,
            context_length=4096,
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
        total_bytes=2 * GIB,
        assessment=CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=TargetDevice.CPU,
            available_vram_bytes=None,
            available_ram_bytes=32 * GIB,
            ratio=0.1,
        ),
    )


def test_insufficient_returns_unknown_runtime() -> None:
    rec = runtime_service.recommend(
        _est(CompatibilityStatus.INSUFFICIENT),
        _hw(),
        RepositoryType.TRANSFORMERS,
    )
    assert rec.runtime is RuntimeName.UNKNOWN
    assert any("insufficient" in w.lower() for w in rec.warnings)


def test_diffusers_type_returns_unknown_runtime() -> None:
    rec = runtime_service.recommend(
        _est(CompatibilityStatus.COMFORTABLE, repo_type=RepositoryType.DIFFUSERS),
        _hw(),
        RepositoryType.DIFFUSERS,
    )
    assert rec.runtime is RuntimeName.UNKNOWN
    assert any("out of scope" in w.lower() for w in rec.warnings)


def test_transformers_cpu_returns_transformers_runtime() -> None:
    rec = runtime_service.recommend(
        _est(CompatibilityStatus.COMPATIBLE), _hw(), RepositoryType.TRANSFORMERS
    )
    assert rec.runtime is RuntimeName.TRANSFORMERS
    assert rec.python_snippet is not None


def test_unknown_estimate_returns_unknown_runtime() -> None:
    rec = runtime_service.recommend(
        _est(CompatibilityStatus.UNKNOWN), _hw(), RepositoryType.TRANSFORMERS
    )
    assert rec.runtime is RuntimeName.UNKNOWN
