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
from jaull.runtime import vllm

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


def _estimate(architecture: str | None) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(
            repo_id="user/model", pipeline_tag="text-generation"
        ),
        repository_type=RepositoryType.TRANSFORMERS,
        inference_configuration=InferenceConfiguration(
            context_length=4096, target_device=TargetDevice.AUTO
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=2 * GIB,
                source=EstimateSource.METADATA,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            precision=WeightPrecision.BFLOAT16,
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
        total_bytes=3 * GIB,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMFORTABLE,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=24 * GIB,
            available_ram_bytes=32 * GIB,
            ratio=0.1,
        ),
        architecture=architecture,
    )


def test_vllm_is_viable_for_supported_architecture_on_gpu() -> None:
    assert vllm.is_viable(_estimate("llama"), _hw()) is True


def test_vllm_not_viable_for_unknown_architecture() -> None:
    assert vllm.is_viable(_estimate("bloom"), _hw()) is False
    assert vllm.is_viable(_estimate(None), _hw()) is False


def test_vllm_build_produces_serve_command() -> None:
    rec = vllm.build(_estimate("llama"), _hw())
    assert rec.command_preview is not None
    assert rec.command_preview.startswith("vllm serve user/model")
    assert "--dtype bfloat16" in rec.command_preview
