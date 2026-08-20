from __future__ import annotations

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.estimation import ConfigurationSource
from jaull.domain.model import ModelConfig
from jaull.metadata import config_merger


def test_only_gguf_produces_gguf_sourced_config() -> None:
    header = GgufHeaderMetadata(
        architecture="llama",
        context_length=8192,
        embedding_length=4096,
        block_count=32,
        head_count=32,
        head_count_kv=8,
        rope_dim=128,
    )
    enriched = config_merger.merge(gguf_header=header, base_config=None)
    assert enriched is not None
    assert enriched.config.num_hidden_layers == 32
    assert enriched.config.num_key_value_heads == 8
    assert enriched.sources["num_hidden_layers"] is ConfigurationSource.GGUF_HEADER


def test_only_base_config_used_when_no_header() -> None:
    base = ModelConfig(
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        max_position_embeddings=32768,
    )
    enriched = config_merger.merge(gguf_header=None, base_config=base)
    assert enriched is not None
    assert enriched.config.num_hidden_layers == 28
    assert (
        enriched.sources["num_hidden_layers"]
        is ConfigurationSource.BASE_MODEL_CONFIG
    )


def test_quantization_config_is_preserved_from_base_config() -> None:
    base = ModelConfig(
        num_hidden_layers=28,
        quantization_config={
            "bits": 4,
            "quant_method": "awq",
            "group_size": 128,
        },
    )

    enriched = config_merger.merge(gguf_header=None, base_config=base)

    assert enriched is not None
    assert enriched.config.quantization_config == {
        "bits": 4,
        "quant_method": "awq",
        "group_size": 128,
    }
    assert (
        enriched.sources["quantization_config"]
        is ConfigurationSource.BASE_MODEL_CONFIG
    )


def test_matching_values_are_not_recorded_as_conflicts() -> None:
    header = GgufHeaderMetadata(
        architecture="llama",
        block_count=32,
        head_count=32,
    )
    base = ModelConfig(num_hidden_layers=32, num_attention_heads=32)
    enriched = config_merger.merge(gguf_header=header, base_config=base)
    assert enriched is not None
    assert enriched.conflicts == []
    assert enriched.config.num_hidden_layers == 32


def test_conflicting_values_favour_gguf_and_record_conflict() -> None:
    header = GgufHeaderMetadata(architecture="llama", context_length=8192)
    base = ModelConfig(max_position_embeddings=131072)
    enriched = config_merger.merge(gguf_header=header, base_config=base)
    assert enriched is not None
    assert enriched.config.max_position_embeddings == 8192  # GGUF wins
    assert enriched.conflicts
    assert (
        enriched.sources["max_position_embeddings"]
        is ConfigurationSource.GGUF_HEADER
    )
