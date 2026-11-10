from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest

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
    summary_calls: list[str] = field(default_factory=list)

    def model_info(self, repo_id: str):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:  # pragma: no cover
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        self.summary_calls.append(repo_id)
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
    analysis = analysis.model_copy(
        update={"config": ModelConfig(
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
        )},
    )
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


MIB = 1024**2


def _coder_like_gguf(target: TargetDevice = TargetDevice.GPU, **knobs):
    """A 17.28 GiB GGUF shaped like Qwen3-Coder 30B-A3B, at 32768 tokens (D120, D123).

    The structured hardware fit is only computed for ``TargetDevice.AUTO``.
    """
    size = round(17.28 * GIB)
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=size)],
        total_bytes=size,
    )
    analysis = _gguf_analysis([variant]).model_copy(
        update={
            "config": ModelConfig(
                num_hidden_layers=48,
                num_attention_heads=32,
                num_key_value_heads=4,
                head_dim=128,
                hidden_size=2048,
            )
        }
    )
    cfg = InferenceConfiguration(
        context_length=32768,
        target_device=target,
        quantization="Q4_K_M",
        **knobs,
    )
    return service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=23613 * MIB),
        inference_cfg=cfg,
        client=_FakeClient(),
        resolve_base_model=False,
        recommend_runtime=False,
    )


def test_gguf_total_is_weights_plus_kv_plus_compute_slack_plus_one_gib() -> None:
    """D123: llama.cpp accepts a load that leaves >= 1 GiB free, so that is the whole headroom."""
    est = _coder_like_gguf(device_reserve_bytes=512 * MIB)

    assert est.runtime_overhead.component.bytes == 256 * MIB
    assert est.device_reserve.bytes == 1 * GIB
    assert est.safety_margin is None
    weights = est.weights.component.bytes
    kv = est.kv_cache.component.bytes
    assert kv == 3072 * MIB  # what llama.cpp measured at 32768 tokens
    assert est.total_bytes == weights + kv + 256 * MIB + 1 * GIB


def test_gguf_reserve_is_never_below_one_gib() -> None:
    assert _coder_like_gguf(device_reserve_bytes=0).device_reserve.bytes == 1 * GIB
    assert _coder_like_gguf(device_reserve_bytes=256 * MIB).device_reserve.bytes == 1 * GIB


def test_gguf_reserve_above_one_gib_is_kept() -> None:
    assert _coder_like_gguf(device_reserve_bytes=2 * GIB).device_reserve.bytes == 2 * GIB


def test_gguf_explicit_safety_margin_is_still_applied() -> None:
    est = _coder_like_gguf(device_reserve_bytes=512 * MIB, safety_margin_percent=5.0)

    assert est.safety_margin is not None
    subtotal = (
        est.weights.component.bytes
        + est.kv_cache.component.bytes
        + est.runtime_overhead.component.bytes
        + est.device_reserve.bytes
    )
    assert est.safety_margin.bytes == math.ceil(subtotal * 0.05)


def test_the_coder_at_32768_tokens_is_tight_and_not_insufficient_on_the_b60() -> None:
    """Measured: it loads with 2.9 GiB free. Before D123 Jaull said it did not fit (23.41 GiB)."""
    est = _coder_like_gguf(device_reserve_bytes=512 * MIB)

    assert est.total_bytes is not None
    assert est.total_bytes / (23613 * MIB) < 1.0
    assert est.assessment.status is CompatibilityStatus.TIGHT


def test_non_gguf_keeps_the_reserve_and_the_percentage_margin() -> None:
    config = ModelConfig(
        num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584
    )
    est = service.estimate_memory(
        analysis=_transformers_analysis(config),
        hardware=_hardware(ram=32 * GIB, vram=24 * GIB),
        inference_cfg=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.GPU,
            precision=WeightPrecision.FLOAT16,
            safety_margin_percent=10.0,
            device_reserve_bytes=512 * MIB,
        ),
        client=_FakeClient(),
        resolve_base_model=False,
        recommend_runtime=False,
    )

    assert est.device_reserve.bytes == 512 * MIB
    assert est.safety_margin is not None and est.safety_margin.bytes > 0
    assert est.runtime_overhead.component.bytes > 512 * MIB  # base plus 10 % of the weights


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


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_packed_quantized_transformers_uses_file_size_not_safetensors_summary(
    method: str,
) -> None:
    config = ModelConfig(
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        quantization_config={"quant_method": method, "bits": 4},
    )
    analysis = _transformers_analysis(config)
    cfg = InferenceConfiguration(
        context_length=4096,
        target_device=TargetDevice.AUTO,
        precision=WeightPrecision.INT4,
    )
    client = _FakeClient(
        summaries={
            "user/tf": SafetensorsSummary(
                total_parameters=1_960_000_000,
                parameters_by_dtype={"I4": 1_960_000_000},
            )
        }
    )

    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=12 * GIB),
        inference_cfg=cfg,
        client=client,
    )

    assert est.weights.component.bytes == int(7.6 * GIB)
    assert est.weights.num_parameters is None
    assert any("Packed quantized safetensors metadata ignored" in item for item in est.warnings)
    assert client.summary_calls == []


def test_compressed_tensors_transformers_uses_file_size_not_requested_int8() -> None:
    config = ModelConfig(
        num_hidden_layers=50,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        torch_dtype="bfloat16",
        quantization_config={
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "group_size": 128,
                        "type": "int",
                    }
                }
            },
        },
    )
    analysis = _transformers_analysis(config).model_copy(
        update={
            "repo": ModelRepositoryInfo(repo_id="speakleash/Bielik-11B-v3.0-Instruct-awq"),
            "files": [
                ModelFile(path="config.json", size_bytes=1476),
                ModelFile(path="model-00001-of-00002.safetensors", size_bytes=4_992_601_480),
                ModelFile(path="model-00002-of-00002.safetensors", size_bytes=1_200_378_608),
            ],
            "total_size_bytes": 6_197_175_414,
        }
    )
    cfg = InferenceConfiguration(
        context_length=4096,
        target_device=TargetDevice.AUTO,
        precision=WeightPrecision.INT8,
    )
    client = _FakeClient(
        summaries={
            "speakleash/Bielik-11B-v3.0-Instruct-awq": SafetensorsSummary(
                total_parameters=1_960_000_000,
                parameters_by_dtype={"I4": 1_960_000_000},
            )
        }
    )

    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=32 * GIB, vram=12 * GIB),
        inference_cfg=cfg,
        client=client,
    )

    expected_weight_bytes = 4_992_601_480 + 1_200_378_608
    assert est.weights.component.bytes == expected_weight_bytes
    assert est.weights.precision is WeightPrecision.BFLOAT16
    assert est.weights.bits_per_parameter == 16
    assert est.weights.num_parameters is None
    assert client.summary_calls == []


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


@pytest.mark.parametrize("target", list(TargetDevice))
def test_confidence_reflects_unknown_kv_cache(target: TargetDevice) -> None:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.gguf", size_bytes=int(4 * GIB))],
        total_bytes=int(4 * GIB),
    )
    analysis = _gguf_analysis([variant])
    cfg = InferenceConfiguration(
        context_length=4096,
        target_device=target,
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
    assert est.assessment.status is CompatibilityStatus.UNKNOWN
    assert est.hardware_fit is None
    assert est.total_bytes is not None  # Preserve the known lower bound.


def test_a_hybrid_gguf_goes_through_the_whole_estimate_and_the_hardware_fit() -> None:
    """Regression: the hardware fit needs the KV cache layer count to equal the weights block count.

    A hybrid model counts only its attention layers for the KV bytes, but `layers` stays the
    model block count. Qwen3.6-35B-A3B IQ3_S at 256K must fit a 24 GiB card, not need 20 GiB of KV.
    """
    size = round(12.74 * GIB)
    variant = GgufVariant(
        quantization="IQ3_S",
        files=[ModelFile(path="model.IQ3_S.gguf", size_bytes=size)],
        total_bytes=size,
    )
    hybrid = {
        "full_attention_interval": 4,
        "ssm_conv_kernel": 4,
        "ssm_inner_size": 4096,
        "ssm_state_size": 128,
        "ssm_group_count": 16,
    }
    analysis = _gguf_analysis([variant]).model_copy(
        update={
            "config": ModelConfig(
                num_hidden_layers=40,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=256,
                hidden_size=2048,
                raw_flags={"hybrid_attention": hybrid},
            )
        }
    )

    est = service.estimate_memory(
        analysis=analysis,
        hardware=_hardware(ram=28 * GIB, vram=23613 * MIB),
        inference_cfg=InferenceConfiguration(
            context_length=262_144, target_device=TargetDevice.AUTO, quantization="IQ3_S"
        ),
        client=_FakeClient(),
        resolve_base_model=False,
        recommend_runtime=False,
    )

    assert est.kv_cache.component.bytes < 5.2 * GIB  # not the 20 GiB of counting 40 layers
    assert est.kv_cache.layers == 40
    assert est.hardware_fit is not None
    assert est.total_bytes is not None and est.total_bytes < 23613 * MIB
