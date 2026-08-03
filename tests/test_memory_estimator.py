from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
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
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.estimator import service

GIB = 1024**3


@dataclass
class _FakeClient:
    summaries: dict[str, SafetensorsSummary | None] = field(default_factory=dict)

    def model_info(self, repo_id: str):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:  # pragma: no cover
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return self.summaries.get(repo_id)


def _hardware(ram: int, vram: int | None) -> HardwareProfile:
    gpus = (
        [
            GpuInfo(
                name="Test GPU",
                vram_total_bytes=vram,
                vram_available_bytes=vram,
                driver_version="1.0",
                cuda_version="12.0",
            )
        ]
        if vram is not None
        else []
    )
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=ram, available_bytes=ram),
        storage=[],
        gpus=gpus,
        warnings=[],
    )


def _gguf_analysis(variants: list[GgufVariant]) -> ModelAnalysis:
    files = [f for v in variants for f in v.files]
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="user/gguf"),
        files=files,
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=variants,
        ),
        config=None,
        relevant_files=[],
        total_size_bytes=sum(v.total_bytes for v in variants),
        warnings=[],
    )


def _transformers_analysis(config: ModelConfig) -> ModelAnalysis:
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="user/tf"),
        files=[
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(path="model.safetensors", size_bytes=int(7.6 * GIB)),
        ],
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
            gguf_variants=[],
        ),
        config=config,
        relevant_files=["config.json", "model.safetensors"],
        total_size_bytes=int(7.6 * GIB),
        warnings=[],
    )


def test_gguf_q4_km_fits_in_gpu_with_headroom() -> None:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=int(4.9 * GIB))],
        total_bytes=int(4.9 * GIB),
    )
    analysis = _gguf_analysis([variant])
    cfg = InferenceConfiguration(
        context_length=4096,
        target_device=TargetDevice.GPU,
        quantization="Q4_K_M",
        safety_margin_percent=5.0,
        device_reserve_bytes=int(0.25 * GIB),
    )
    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=24 * GIB),
        inference_cfg=cfg,
        client=_FakeClient(),
        resolve_base_model=False,
    )
    assert est.assessment.status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
    }
    assert est.weights.gguf_variant == "Q4_K_M"


def test_transformers_int8_offloading_required() -> None:
    config = ModelConfig(
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
    )
    analysis = _transformers_analysis(config)
    cfg = InferenceConfiguration(
        context_length=32768,
        target_device=TargetDevice.AUTO,
        precision=WeightPrecision.INT8,
        safety_margin_percent=10.0,
    )
    client = _FakeClient(
        summaries={
            "user/tf": SafetensorsSummary(
                total_parameters=7_615_616_512,
                parameters_by_dtype={"BF16": 7_615_616_512},
            )
        }
    )
    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=6 * GIB),
        inference_cfg=cfg,
        client=client,
    )
    assert est.assessment.status is CompatibilityStatus.OFFLOADING_REQUIRED


def test_transformers_float16_cpu_only_when_no_gpu() -> None:
    config = ModelConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
    )
    analysis = _transformers_analysis(config)
    cfg = InferenceConfiguration(
        context_length=2048,
        target_device=TargetDevice.AUTO,
        precision=WeightPrecision.FLOAT16,
    )
    client = _FakeClient(
        summaries={
            "user/tf": SafetensorsSummary(
                total_parameters=1_000_000_000,
                parameters_by_dtype={"F16": 1_000_000_000},
            )
        }
    )
    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=None),
        inference_cfg=cfg,
        client=client,
    )
    assert est.assessment.effective_device is TargetDevice.CPU
    assert est.assessment.status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
    }


def test_insufficient_when_ram_too_small() -> None:
    config = ModelConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
    )
    analysis = _transformers_analysis(config)
    cfg = InferenceConfiguration(
        context_length=2048,
        target_device=TargetDevice.CPU,
        precision=WeightPrecision.FLOAT16,
    )
    client = _FakeClient(
        summaries={
            "user/tf": SafetensorsSummary(
                total_parameters=70_000_000_000,
                parameters_by_dtype={"F16": 70_000_000_000},
            )
        }
    )
    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=4 * GIB, vram=None),
        inference_cfg=cfg,
        client=client,
    )
    assert est.assessment.status is CompatibilityStatus.INSUFFICIENT


def test_confidence_reflects_unknown_kv_cache() -> None:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.gguf", size_bytes=int(4 * GIB))],
        total_bytes=int(4 * GIB),
    )
    analysis = _gguf_analysis([variant])
    cfg = InferenceConfiguration(
        context_length=4096,
        target_device=TargetDevice.GPU,
        quantization="Q4_K_M",
    )
    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=16 * GIB),
        inference_cfg=cfg,
        client=_FakeClient(),
        resolve_base_model=False,
    )
    # No config => kv_cache is UNKNOWN, so the global confidence must fall to UNKNOWN.
    assert est.assessment.confidence is EstimationConfidence.UNKNOWN
