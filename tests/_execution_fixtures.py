"""The measured Qwen2.5-7B / RTX 2060 / ctx4096 fixture, shared across tests.

Identical numbers to tests/test_hardware_fit_analyzer.py::test_qwen_7b_rtx2060_*.
On this fixture: HardwareFitAnalyzer picks 18/28 transformer blocks and the
llama.cpp launch policy picks --n-gpu-layers 23. Both facts are load-bearing
regression anchors for the "unify the flow, not the numbers" milestone.
"""

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
from jaull.domain.hardware import CpuInfo, GpuInfo, HardwareProfile, MemoryInfo
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelRepositoryInfo
from jaull.domain.runtime import RuntimeRecommendation  # noqa: F401 (rebuild)

QWEN_WEIGHTS = 4_683_074_240
QWEN_KV = 234_881_024
QWEN_OVERHEAD = 1_005_178_336
QWEN_RESERVE = 536_870_912
QWEN_MARGIN = 646_000_452
QWEN_VRAM = 4_985_380_864
QWEN_RAM = 7_593_828_352
QWEN_BLOCKS = 28


def qwen_hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=8 * 1024**3, available_bytes=QWEN_RAM),
        storage=[],
        gpus=[
            GpuInfo(
                name="RTX 2060",
                vram_total_bytes=6 * 1024**3,
                vram_available_bytes=QWEN_VRAM,
                driver_version="1.0",
                cuda_version="12.4",
            )
        ],
        warnings=[],
    )


def qwen_ctx4096_estimate(*, with_runtime_recommendation: bool) -> MemoryEstimate:
    estimate = MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.AUTO,
            quantization="Q4_K_M",
            device_reserve_bytes=QWEN_RESERVE,
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=QWEN_WEIGHTS,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="fixture",
            ),
            gguf_variant="Q4_K_M",
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=QWEN_KV,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="fixture",
            ),
            layers=QWEN_BLOCKS,
            kv_heads=8,
            head_dim=128,
            context_length=4096,
            batch_size=1,
            dtype_bytes=2,
            formula="fixture",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=QWEN_OVERHEAD,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="fixture",
            ),
            base_bytes=536_870_912,
            weight_fraction=0.1,
            minimum_bytes=268_435_456,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=QWEN_RESERVE,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="fixture",
        ),
        safety_margin=MemoryComponent(
            name="Safety margin",
            bytes=QWEN_MARGIN,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.MEDIUM,
            explanation="fixture",
        ),
        total_bytes=QWEN_WEIGHTS + QWEN_KV + QWEN_OVERHEAD + QWEN_RESERVE + QWEN_MARGIN,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.OFFLOADING_REQUIRED,
            confidence=EstimationConfidence.MEDIUM,
            target_device=TargetDevice.AUTO,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=QWEN_VRAM,
            available_ram_bytes=QWEN_RAM,
            ratio=0.9,
        ),
        assumptions=[],
        warnings=[],
    )
    if not with_runtime_recommendation:
        return estimate

    from jaull.runtime import service as runtime_service

    rec = runtime_service.recommend(estimate, qwen_hardware(), RepositoryType.GGUF)
    return estimate.model_copy(update={"runtime_recommendation": rec})
