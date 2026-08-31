from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimateSource,
    MetadataSource,
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
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.estimator import service as estimator_service
from jaull.metadata.range_reader import RangeResponse
from tests._gguf_fixtures import build_header

GIB = 1024**3


@dataclass
class _StubModelInfo:
    card_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubHfClient:
    card_data: dict[str, Any] = field(default_factory=dict)
    base_config: dict[str, Any] | None = None
    summary: SafetensorsSummary | None = None

    def model_info(self, repo_id: str) -> _StubModelInfo:
        return _StubModelInfo(card_data=self.card_data)

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        assert self.base_config is not None
        target = Path("/tmp") / f"{repo_id.replace('/', '_')}_{filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.base_config), encoding="utf-8")
        return target

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return self.summary


@dataclass
class _StubRangeClient:
    body: bytes

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse:
        return RangeResponse(
            body=self.body[start : end + 1], honored_range=True, status_code=206
        )


def _hardware_with_gpu(vram_gib: int, ram_gib: int) -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=ram_gib * GIB, available_bytes=ram_gib * GIB),
        storage=[],
        gpus=[
            GpuInfo(
                name="Test GPU",
                vram_total_bytes=vram_gib * GIB,
                vram_available_bytes=vram_gib * GIB,
                driver_version="1.0",
                cuda_version="12.4",
            )
        ],
        warnings=[],
    )


def _gguf_analysis() -> ModelAnalysis:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=int(4.6 * GIB))],
        total_bytes=int(4.6 * GIB),
    )
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
        files=variant.files,
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=[variant],
        ),
        config=None,
        relevant_files=[],
        total_size_bytes=variant.total_bytes,
        warnings=[],
    )


def test_gguf_estimate_gets_kv_cache_after_enrichment() -> None:
    analysis = _gguf_analysis()
    client = _StubHfClient(
        card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        base_config={
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 131072,
        },
    )
    header_body = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 131072,
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    estimate = estimator_service.estimate_memory(
        analysis=analysis,
        hardware=_hardware_with_gpu(vram_gib=24, ram_gib=64),
        inference_cfg=InferenceConfiguration(
            context_length=8192,
            target_device=TargetDevice.GPU,
            quantization="Q4_K_M",
        ),
        client=client,
        range_client=_StubRangeClient(body=header_body),
    )
    assert estimate.kv_cache.component.bytes is not None
    assert estimate.kv_cache.kv_heads == 8
    assert estimate.assessment.status in {
        CompatibilityStatus.COMFORTABLE,
        CompatibilityStatus.COMPATIBLE,
    }
    assert estimate.base_model_resolution is not None
    assert estimate.base_model_resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert estimate.base_model_resolution.source is MetadataSource.MODEL_CARD_METADATA
    assert estimate.configuration_sources


def test_no_resolve_base_model_skips_enrichment() -> None:
    analysis = _gguf_analysis()
    client = _StubHfClient(
        card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        base_config={"num_hidden_layers": 32},
    )
    estimate = estimator_service.estimate_memory(
        analysis=analysis,
        hardware=_hardware_with_gpu(vram_gib=24, ram_gib=64),
        inference_cfg=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.GPU,
            quantization="Q4_K_M",
        ),
        client=client,
        resolve_base_model=False,
    )
    assert estimate.kv_cache.component.bytes is None
    assert estimate.base_model_resolution is not None
    assert estimate.base_model_resolution.source is MetadataSource.UNRESOLVED
    assert estimate.configuration_sources == {}


def test_estimate_json_includes_enrichment_fields() -> None:
    from jaull.presentation.estimation_report import (
        SCHEMA_VERSION,
        estimate_to_json_dict,
    )

    analysis = _gguf_analysis()
    client = _StubHfClient(
        card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        base_config={
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        },
    )
    header_body = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    estimate = estimator_service.estimate_memory(
        analysis=analysis,
        hardware=_hardware_with_gpu(vram_gib=24, ram_gib=64),
        inference_cfg=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.GPU,
            quantization="Q4_K_M",
        ),
        client=client,
        range_client=_StubRangeClient(body=header_body),
    )
    payload = estimate_to_json_dict(estimate)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["base_model_resolution"]["repo_id"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert payload["configuration_sources"]  # non-empty
    for value in payload["configuration_sources"].values():
        assert isinstance(value, str)


def test_gguf_header_source_is_labelled_correctly() -> None:
    # Confirms the new EstimateSource values are wired.
    assert EstimateSource.GGUF_HEADER.value == "gguf_header"
    assert EstimateSource.BASE_MODEL_CONFIG.value == "base_model_config"


def test_qwen_gguf_estimate_gets_architecture_weight_decomposition() -> None:
    total_weight_bytes = 4_683_074_240
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[
            ModelFile(
                path="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                size_bytes=total_weight_bytes,
            )
        ],
        total_bytes=total_weight_bytes,
    )
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF"),
        files=variant.files,
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=[variant],
        ),
    )
    client = _StubHfClient(
        card_data={"base_model": "Qwen/Qwen2.5-7B-Instruct"},
        base_config={
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "head_dim": 128,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "tie_word_embeddings": False,
        },
    )
    header_body = build_header(
        {
            "general.architecture": "qwen2",
            "qwen2.context_length": 32768,
            "qwen2.block_count": 28,
            "qwen2.embedding_length": 3584,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
        }
    )

    estimate = estimator_service.estimate_memory(
        analysis=analysis,
        hardware=_hardware_with_gpu(vram_gib=24, ram_gib=64),
        inference_cfg=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.AUTO,
            quantization="Q4_K_M",
        ),
        client=client,
        range_client=_StubRangeClient(body=header_body),
        recommend_runtime=False,
    )

    decomposition = estimate.weights.transformer_block_decomposition
    assert decomposition is not None
    assert decomposition.method.value == "config_parameter_decomposition"
    assert decomposition.total_weight_bytes == total_weight_bytes
    assert decomposition.estimated_transformer_block_weight_bytes == 4_012_773_975
    assert decomposition.estimated_non_block_weight_bytes == 670_300_265
    assert decomposition.estimated_bytes_per_transformer_block == 143_313_357
