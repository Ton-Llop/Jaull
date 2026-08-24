from __future__ import annotations

from jaull.domain.estimation import EstimationConfidence
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import ModelConfig
from jaull.estimator import kv_cache


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


def test_kv_cache_respects_an_active_sliding_window() -> None:
    config = ModelConfig(
        num_hidden_layers=8,
        num_attention_heads=8,
        hidden_size=1024,
        sliding_window=1024,
        # Explicit: a bare `sliding_window` is no longer enough to clamp on.
        use_sliding_window=True,
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


# ---------------------------------------------------------------------------
# Sliding window: only clamp when the window is actually in force
# ---------------------------------------------------------------------------
# Measured against llama.cpp on Qwen2.5-1.5B at 131072 tokens: the config
# declares sliding_window=32768 with use_sliding_window=false, llama.cpp
# allocated the full 3584 MiB, and clamping made the estimate 4x too small.


def _windowed(
    window: int | None, use: bool | None, model_type: str | None = None
) -> ModelConfig:
    return ModelConfig(
        model_type=model_type,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_size=1536,
        head_dim=128,
        sliding_window=window,
        use_sliding_window=use,
    )


def _kv(config: ModelConfig, context_length: int = 131072):
    return kv_cache.estimate_kv_cache(
        config=config,
        context_length=context_length,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
    )


def test_declared_but_disabled_window_does_not_clamp_the_cache() -> None:
    est = _kv(_windowed(32768, False))

    assert est.context_length == 131072
    # 2 * 28 * 2 * 128 * 131072 * 1 * 2 - what llama.cpp actually allocated.
    assert est.component.bytes == 2 * 28 * 2 * 128 * 131072 * 1 * 2
    assert est.component.confidence is EstimationConfidence.HIGH
    assert any("use_sliding_window is false" in note for note in est.notes)


def test_an_enabled_window_clamps_the_cache() -> None:
    est = _kv(_windowed(32768, True))

    assert est.context_length == 32768
    assert est.component.bytes == 2 * 28 * 2 * 128 * 32768 * 1 * 2
    assert est.component.confidence is EstimationConfidence.HIGH


def test_an_unstated_flag_sizes_the_full_context_rather_than_guessing_smaller() -> None:
    """Clamping reduces the estimate, so an unknown must not clamp.

    Reporting that a model fits when it needs several times the memory is the
    failure that matters; over-estimating only costs a recommendation.
    """
    est = _kv(_windowed(32768, None))

    assert est.context_length == 131072
    assert est.component.bytes == 2 * 28 * 2 * 128 * 131072 * 1 * 2
    assert est.component.confidence is EstimationConfidence.MEDIUM
    assert any("avoid under-estimating" in note for note in est.notes)


def test_a_known_architecture_clamps_without_the_flag() -> None:
    """Mistral documents `sliding_window` alone and never exposes the flag."""
    est = _kv(_windowed(32768, None, model_type="mistral"))

    assert est.context_length == 32768
    # Decided from the architecture, not from the config, so not HIGH.
    assert est.component.confidence is EstimationConfidence.MEDIUM
    assert any("windows attention unconditionally" in note for note in est.notes)


def test_an_explicit_flag_beats_the_architecture_table() -> None:
    """A config that says so outranks what we assume about its family."""
    est = _kv(_windowed(32768, False, model_type="mistral"))

    assert est.context_length == 131072
    assert est.component.confidence is EstimationConfidence.HIGH


def test_the_flag_is_irrelevant_below_the_window() -> None:
    """No clamp applies either way, so nothing is assumed and nothing is said."""
    for use in (True, False, None):
        est = _kv(_windowed(32768, use), context_length=4096)

        assert est.context_length == 4096
        assert est.component.confidence is EstimationConfidence.HIGH
        assert not any("sliding_window" in note for note in est.notes)


def test_a_model_without_a_window_is_unaffected() -> None:
    est = _kv(_windowed(None, None))

    assert est.context_length == 131072
    assert est.component.confidence is EstimationConfidence.HIGH
