from __future__ import annotations

from local_ai_check.domain.inference import WeightPrecision
from local_ai_check.domain.model import ModelConfig
from local_ai_check.estimator import kv_cache


def test_kv_cache_mha_standard() -> None:
    config = ModelConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
    )
    est = kv_cache.estimate_kv_cache(
        config=config,
        context_length=2048,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    # 2 * 32 * 32 * 128 * 2048 * 1 * 2 = 1_073_741_824
    assert est.component.bytes == 2 * 32 * 32 * 128 * 2048 * 1 * 2
    assert est.kv_heads == 32
    assert est.head_dim == 128


def test_kv_cache_gqa_reduces_bytes_and_warns() -> None:
    config = ModelConfig(
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
    )
    est = kv_cache.estimate_kv_cache(
        config=config,
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    expected = 2 * 28 * 4 * (3584 // 28) * 4096 * 1 * 2
    assert est.component.bytes == expected
    assert est.kv_heads == 4
    assert any("Grouped-Query Attention" in note for note in est.notes)


def test_kv_cache_scales_with_batch_size() -> None:
    config = ModelConfig(
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=512,
    )
    est_b1 = kv_cache.estimate_kv_cache(
        config=config, context_length=128, batch_size=1, kv_dtype=WeightPrecision.FLOAT16
    )
    est_b4 = kv_cache.estimate_kv_cache(
        config=config, context_length=128, batch_size=4, kv_dtype=WeightPrecision.FLOAT16
    )
    assert est_b4.component.bytes == est_b1.component.bytes * 4


def test_kv_cache_respects_sliding_window() -> None:
    config = ModelConfig(
        num_hidden_layers=8,
        num_attention_heads=8,
        hidden_size=1024,
        sliding_window=1024,
    )
    est = kv_cache.estimate_kv_cache(
        config=config,
        context_length=8192,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    assert est.context_length == 1024
    assert any("sliding_window" in note for note in est.notes)


def test_kv_cache_unknown_when_config_missing() -> None:
    est = kv_cache.estimate_kv_cache(
        config=None,
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    assert est.component.bytes is None
    assert est.component.source.value == "unknown"


def test_kv_cache_unknown_when_fields_missing() -> None:
    config = ModelConfig(num_hidden_layers=32)  # missing heads and hidden_size
    est = kv_cache.estimate_kv_cache(
        config=config,
        context_length=1024,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    assert est.component.bytes is None
    assert "num_attention_heads" in est.component.explanation


def test_kv_cache_flags_moe() -> None:
    config = ModelConfig(
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_size=2048,
        raw_flags={"num_experts": 8, "num_local_experts": 8},
    )
    est = kv_cache.estimate_kv_cache(
        config=config, context_length=1024, batch_size=1, kv_dtype=WeightPrecision.FLOAT16
    )
    assert est.component.bytes is None
    assert "Mixture-of-Experts" in est.component.explanation


def test_kv_cache_flags_custom_architecture() -> None:
    config = ModelConfig(
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_size=2048,
        raw_flags={"auto_map": {"AutoModel": "modeling_custom.CustomModel"}},
    )
    est = kv_cache.estimate_kv_cache(
        config=config, context_length=1024, batch_size=1, kv_dtype=WeightPrecision.FLOAT16
    )
    assert est.component.bytes is None
    assert "trust_remote_code" in est.component.explanation
