"""KV cache of a model whose GGUF header lists the KV heads of every layer.

gemma-4 declares `attention.head_count_kv` once per layer (a few wide global layers among many
narrow sliding-window ones). The estimator has one KV head count, so it sizes every layer with the
largest value and the full context: an upper bound, said as such. The alternative was no estimate.
"""

from __future__ import annotations

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import PER_LAYER_KV_HEADS_FLAG, ModelConfig
from jaull.estimator import kv_cache
from jaull.metadata import config_merger
from jaull.metadata.gguf_reader import parse_header
from tests._gguf_fixtures import build_header


def _config(kv_heads: list[int]) -> ModelConfig:
    layers = len(kv_heads)
    header = parse_header(
        build_header(
            {
                "general.architecture": "gemma4",
                "gemma4.context_length": 262144,
                "gemma4.embedding_length": 3840,
                "gemma4.block_count": layers,
                "gemma4.attention.head_count": 16,
                "gemma4.attention.head_count_kv": kv_heads,
                "gemma4.attention.key_length": 512,
                "gemma4.attention.value_length": 512,
            }
        )
    )
    assert isinstance(header, GgufHeaderMetadata)
    enriched = config_merger.merge(header, None)
    assert enriched is not None
    return enriched.config


def _estimate(config: ModelConfig, context: int = 4096) -> kv_cache.KvCacheEstimate:
    return kv_cache.estimate_kv_cache(
        config=config, context_length=context, batch_size=1, kv_dtype=WeightPrecision.FLOAT16,
    )


def test_the_merged_config_keeps_the_range_of_a_per_layer_list() -> None:
    config = _config([8, 8, 8, 8, 8, 1])

    assert config.num_key_value_heads == 8
    assert config.raw_flags[PER_LAYER_KV_HEADS_FLAG] == {"min": 1, "max": 8, "layers": 6}


def test_a_per_layer_model_is_sized_with_its_largest_head_count_and_says_so() -> None:
    estimate = _estimate(_config([8, 8, 8, 8, 8, 1]))

    assert estimate.component.bytes == 2 * 6 * 8 * 512 * 4096 * 2
    assert estimate.layers == 6  # the block count: the hardware fit splits the weights over it
    assert estimate.component.confidence is EstimationConfidence.MEDIUM
    assert any("upper bound" in note and "1 to 8" in note for note in estimate.notes)


def test_an_unrolled_list_of_equal_values_is_not_an_upper_bound() -> None:
    config = _config([4, 4, 4, 4, 4, 4])

    estimate = _estimate(config)

    assert PER_LAYER_KV_HEADS_FLAG not in config.raw_flags
    assert estimate.component.bytes == 2 * 6 * 4 * 512 * 4096 * 2
    assert estimate.component.confidence is EstimationConfidence.HIGH
    assert not any("upper bound" in note for note in estimate.notes)
