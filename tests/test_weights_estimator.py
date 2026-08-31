from __future__ import annotations

import pytest

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    MemoryComponent,
    TransformerBlockWeightDecomposition,
    TransformerBlockWeightDecompositionMethod,
    WeightEstimate,
)
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.estimator import weights
from jaull.estimator.policies import bytes_per_parameter


def _weight_estimate(total_bytes: int | None) -> WeightEstimate:
    return WeightEstimate(
        component=MemoryComponent(
            name="Weights",
            bytes=total_bytes,
            source=EstimateSource.EXACT,
            confidence=EstimationConfidence.HIGH,
            explanation="test artifact",
        )
    )


def _qwen_config(*, tie_word_embeddings: bool | None = False) -> ModelConfig:
    return ModelConfig(
        model_type="qwen2",
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        head_dim=128,
        intermediate_size=18944,
        vocab_size=152064,
        tie_word_embeddings=tie_word_embeddings,
    )


def _analysis(files: list[ModelFile], primary: RepositoryType) -> ModelAnalysis:
    classification = RepositoryClassification(
        primary_type=primary,
        detected_types={primary},
        formats=(
            {Format.SAFETENSORS} if primary is RepositoryType.TRANSFORMERS else set()
        ),
        gguf_variants=[],
    )
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="user/model"),
        files=files,
        classification=classification,
        config=None,
        relevant_files=[],
        total_size_bytes=None,
        warnings=[],
    )


def test_bytes_per_parameter_table_matches_spec() -> None:
    assert bytes_per_parameter(WeightPrecision.FLOAT32) == 4.0
    assert bytes_per_parameter(WeightPrecision.FLOAT16) == 2.0
    assert bytes_per_parameter(WeightPrecision.BFLOAT16) == 2.0
    assert bytes_per_parameter(WeightPrecision.INT8) == 1.0
    assert bytes_per_parameter(WeightPrecision.INT4) == 0.5


def test_weights_from_gguf_returns_exact_size() -> None:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=4_921_000_000)],
        total_bytes=4_921_000_000,
    )
    est = weights.estimate_weights_from_gguf(variant)
    assert est.component.bytes == 4_921_000_000
    assert est.component.source.value == "exact"
    assert est.component.confidence.value == "high"
    assert est.gguf_variant == "Q4_K_M"


def test_weights_from_safetensors_uses_requested_precision() -> None:
    summary = SafetensorsSummary(
        total_parameters=1_000_000_000,
        parameters_by_dtype={"BF16": 1_000_000_000},
    )
    est = weights.estimate_weights_from_safetensors(
        summary=summary,
        requested_precision=WeightPrecision.INT8,
        config=None,
    )
    assert est.component.bytes == 1_000_000_000  # 1B * 1 byte
    assert est.precision is WeightPrecision.INT8
    assert est.num_parameters == 1_000_000_000


def test_weights_from_safetensors_falls_back_to_dominant_dtype() -> None:
    summary = SafetensorsSummary(
        total_parameters=1_000_000_000,
        parameters_by_dtype={"F16": 1_000_000_000},
    )
    est = weights.estimate_weights_from_safetensors(
        summary=summary,
        requested_precision=None,
        config=None,
    )
    assert est.precision is WeightPrecision.FLOAT16
    assert est.component.bytes == 2_000_000_000


def test_weights_from_files_sums_safetensors_and_bin() -> None:
    analysis = _analysis(
        files=[
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(path="model-00001-of-00002.safetensors", size_bytes=3_000_000_000),
            ModelFile(path="model-00002-of-00002.safetensors", size_bytes=3_000_000_000),
            ModelFile(path="tokenizer.json", size_bytes=2048),
        ],
        primary=RepositoryType.TRANSFORMERS,
    )
    est = weights.estimate_weights_from_files(
        analysis=analysis,
        requested_precision=None,
        config=None,
    )
    assert est.component.bytes == 6_000_000_000
    assert est.component.source.value == "derived"
    assert est.precision is WeightPrecision.FLOAT16


def test_weights_from_files_scales_when_precision_differs() -> None:
    analysis = _analysis(
        files=[
            ModelFile(path="model.safetensors", size_bytes=8_000_000_000),
        ],
        primary=RepositoryType.TRANSFORMERS,
    )
    config = ModelConfig(torch_dtype="bfloat16")
    est = weights.estimate_weights_from_files(
        analysis=analysis,
        requested_precision=WeightPrecision.INT4,
        config=config,
    )
    # 8B bytes at bf16 (2 bytes/param) -> 4B params -> at int4 (0.5) = 2B bytes.
    assert est.component.bytes == 2_000_000_000
    assert est.precision is WeightPrecision.INT4


def test_weights_from_files_unknown_when_no_weights_present() -> None:
    analysis = _analysis(
        files=[ModelFile(path="README.md", size_bytes=100)],
        primary=RepositoryType.TRANSFORMERS,
    )
    est = weights.estimate_weights_from_files(
        analysis=analysis, requested_precision=None, config=None
    )
    assert est.component.bytes is None
    assert est.component.source.value == "unknown"


def test_qwen_weight_decomposition_uses_config_parameter_fraction() -> None:
    total_bytes = 4_683_074_240

    estimate = weights.add_transformer_block_decomposition(
        _weight_estimate(total_bytes),
        _qwen_config(),
    )

    decomposition = estimate.transformer_block_decomposition
    assert decomposition is not None
    assert (
        decomposition.method
        is TransformerBlockWeightDecompositionMethod.CONFIG_PARAMETER_DECOMPOSITION
    )
    assert decomposition.total_weight_bytes == total_bytes
    assert decomposition.estimated_transformer_block_weight_bytes == 4_012_773_975
    assert decomposition.estimated_non_block_weight_bytes == 670_300_265
    assert decomposition.estimated_bytes_per_transformer_block == 143_313_357
    assert decomposition.total_transformer_blocks == 28
    assert (
        decomposition.estimated_transformer_block_weight_bytes
        + decomposition.estimated_non_block_weight_bytes
        == total_bytes
    )


def test_tied_embeddings_remove_the_separate_output_head_from_fraction() -> None:
    total_bytes = 4_683_074_240
    untied = weights.add_transformer_block_decomposition(
        _weight_estimate(total_bytes),
        _qwen_config(tie_word_embeddings=False),
    ).transformer_block_decomposition
    tied = weights.add_transformer_block_decomposition(
        _weight_estimate(total_bytes),
        _qwen_config(tie_word_embeddings=True),
    ).transformer_block_decomposition

    assert untied is not None
    assert tied is not None
    assert (
        tied.estimated_non_block_weight_bytes
        < untied.estimated_non_block_weight_bytes
    )
    assert (
        tied.estimated_transformer_block_weight_bytes
        > untied.estimated_transformer_block_weight_bytes
    )


@pytest.mark.parametrize(
    "config",
    [
        _qwen_config(tie_word_embeddings=None),
        _qwen_config().model_copy(update={"model_type": "unknown_architecture"}),
        _qwen_config().model_copy(update={"raw_flags": {"num_experts": 8}}),
        _qwen_config().model_copy(update={"raw_flags": {"vision_config": {}}}),
        _qwen_config().model_copy(update={"intermediate_size": None}),
    ],
)
def test_incomplete_or_unsupported_config_uses_uniform_fallback(
    config: ModelConfig,
) -> None:
    estimate = weights.add_transformer_block_decomposition(
        _weight_estimate(101),
        config,
    )

    decomposition = estimate.transformer_block_decomposition
    assert decomposition is not None
    assert (
        decomposition.method
        is TransformerBlockWeightDecompositionMethod.UNIFORM_WEIGHT_FALLBACK
    )
    assert decomposition.estimated_transformer_block_weight_bytes == 101
    assert decomposition.estimated_non_block_weight_bytes == 0
    assert decomposition.estimated_bytes_per_transformer_block == 4


def test_config_decomposition_rounding_conserves_every_artifact_byte() -> None:
    estimate = weights.add_transformer_block_decomposition(
        _weight_estimate(101),
        _qwen_config(),
    )

    decomposition = estimate.transformer_block_decomposition
    assert decomposition is not None
    assert (
        decomposition.estimated_transformer_block_weight_bytes
        + decomposition.estimated_non_block_weight_bytes
        == 101
    )
    assert decomposition.estimated_bytes_per_transformer_block == 4


@pytest.mark.parametrize("config", [None, ModelConfig(num_hidden_layers=0)])
def test_missing_or_invalid_block_count_has_no_decomposition(
    config: ModelConfig | None,
) -> None:
    estimate = weights.add_transformer_block_decomposition(
        _weight_estimate(101),
        config,
    )

    assert estimate.transformer_block_decomposition is None


def test_legacy_weight_estimate_without_decomposition_still_loads() -> None:
    estimate = WeightEstimate.model_validate(
        {
            "component": {
                "name": "Weights",
                "bytes": 123,
                "source": "exact",
                "confidence": "high",
                "explanation": "legacy payload",
            }
        }
    )

    assert estimate.transformer_block_decomposition is None


def test_weight_decomposition_rejects_non_conserving_payload() -> None:
    with pytest.raises(ValueError, match="must sum exactly"):
        TransformerBlockWeightDecomposition(
            total_weight_bytes=100,
            estimated_transformer_block_weight_bytes=80,
            estimated_non_block_weight_bytes=19,
            estimated_bytes_per_transformer_block=10,
            total_transformer_blocks=8,
            method=(
                TransformerBlockWeightDecompositionMethod.CONFIG_PARAMETER_DECOMPOSITION
            ),
        )
