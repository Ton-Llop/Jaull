from __future__ import annotations

from jaull.domain.inference import WeightPrecision
from jaull.domain.model import ModelConfig
from jaull.estimator.kv_cache import estimate_kv_cache


def _config() -> ModelConfig:
    return ModelConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
    )


def test_concurrent_users_multiply_kv_bytes() -> None:
    single = estimate_kv_cache(
        config=_config(),
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
        concurrent_users=1,
    )
    many = estimate_kv_cache(
        config=_config(),
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
        concurrent_users=10,
    )
    assert single.component.bytes is not None
    assert many.component.bytes is not None
    assert many.component.bytes == single.component.bytes * 10


def test_concurrency_notes_are_emitted() -> None:
    result = estimate_kv_cache(
        config=_config(),
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
        concurrent_users=5,
    )
    assert any("5 concurrent" in note for note in result.notes)


def test_single_user_leaves_notes_untouched() -> None:
    result = estimate_kv_cache(
        config=_config(),
        context_length=4096,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )
    assert not any("concurrent" in note for note in result.notes)
