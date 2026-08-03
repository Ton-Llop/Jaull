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
)
from jaull.domain.model import ModelRepositoryInfo
from jaull.runtime import llama_cpp

GIB = 1024**3


def _hardware(vram_gib: int | None, ram_gib: int) -> HardwareProfile:
    gpus = (
        [
            GpuInfo(
                name="Test",
                vram_total_bytes=vram_gib * GIB,
                vram_available_bytes=vram_gib * GIB,
                driver_version="1.0",
                cuda_version="12.4",
            )
        ]
        if vram_gib is not None
        else []
    )
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=ram_gib * GIB, available_bytes=ram_gib * GIB),
        storage=[],
        gpus=gpus,
        warnings=[],
    )


def _gguf_estimate(
    *,
    status: CompatibilityStatus,
    effective_device: TargetDevice,
    weights_bytes: int,
    kv_bytes: int | None = 512 * 1024 * 1024,
    layers: int | None = 32,
    available_vram: int | None = 24 * GIB,
    available_ram: int = 32 * GIB,
) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="user/repo-GGUF"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.AUTO,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=weights_bytes,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            gguf_variant="Q4_K_M",
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=kv_bytes,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=layers,
            kv_heads=8,
            head_dim=128,
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
            bytes=512 * 1024 * 1024,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=weights_bytes + (kv_bytes or 0),
        assessment=CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.AUTO,
            effective_device=effective_device,
            available_vram_bytes=available_vram,
            available_ram_bytes=available_ram,
            ratio=0.5,
        ),
        assumptions=[],
        warnings=[],
    )


def test_full_gpu_recommends_all_layers() -> None:
    estimate = _gguf_estimate(
        status=CompatibilityStatus.COMFORTABLE,
        effective_device=TargetDevice.GPU,
        weights_bytes=int(4.5 * GIB),
    )
    rec = llama_cpp.build(estimate, _hardware(vram_gib=24, ram_gib=32))
    flag = next(f for f in rec.flags if f.name == "--n-gpu-layers")
    assert flag.value == "-1"
    assert rec.confidence is EstimationConfidence.HIGH


def test_offloading_with_block_count_splits_layers() -> None:
    estimate = _gguf_estimate(
        status=CompatibilityStatus.OFFLOADING_REQUIRED,
        effective_device=TargetDevice.GPU,
        weights_bytes=16 * GIB,
        layers=32,
        available_vram=6 * GIB,
        kv_bytes=1 * GIB,
    )
    rec = llama_cpp.build(estimate, _hardware(vram_gib=6, ram_gib=32))
    flag = next(f for f in rec.flags if f.name == "--n-gpu-layers")
    layers = int(flag.value)
    assert 0 <= layers <= 32
    # bytes_per_layer = 16 GiB / 32 = 512 MiB. Available for layers ≈ 6 - 0.5 - 0.25 - 1 = 4.25 GiB
    # => ≈ 8 layers fit. Allow a small tolerance around the heuristic.
    assert 5 <= layers <= 10
    assert rec.confidence is EstimationConfidence.MEDIUM


def test_offloading_without_block_count_uses_fallback() -> None:
    estimate = _gguf_estimate(
        status=CompatibilityStatus.OFFLOADING_REQUIRED,
        effective_device=TargetDevice.GPU,
        weights_bytes=16 * GIB,
        layers=None,  # unknown block_count
        available_vram=6 * GIB,
    )
    rec = llama_cpp.build(estimate, _hardware(vram_gib=6, ram_gib=32))
    flag = next(f for f in rec.flags if f.name == "--n-gpu-layers")
    assert int(flag.value) == 20  # LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN
    assert any("Block count unavailable" in w for w in rec.warnings)
    assert rec.confidence is EstimationConfidence.LOW


def test_cpu_device_disables_gpu_layers() -> None:
    estimate = _gguf_estimate(
        status=CompatibilityStatus.COMPATIBLE,
        effective_device=TargetDevice.CPU,
        weights_bytes=4 * GIB,
        available_vram=None,
    )
    rec = llama_cpp.build(estimate, _hardware(vram_gib=None, ram_gib=32))
    flag = next(f for f in rec.flags if f.name == "--n-gpu-layers")
    assert flag.value == "0"


def test_default_dtype_used_when_precision_missing() -> None:
    # sanity: llama.cpp builder does not depend on precision; command still forms.
    estimate = _gguf_estimate(
        status=CompatibilityStatus.COMFORTABLE,
        effective_device=TargetDevice.GPU,
        weights_bytes=4 * GIB,
    )
    rec = llama_cpp.build(estimate, _hardware(vram_gib=24, ram_gib=32))
    assert rec.command_preview is not None
    assert "llama-server" in rec.command_preview
