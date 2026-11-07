"""KV cache of a hybrid model: only the full-attention layers keep a KV cache.

Qwen3.5/3.6-style models alternate Gated DeltaNet layers (a constant recurrent state per
sequence) with a full-attention layer every ``full_attention_interval`` layers. Counting every
layer overstated the KV cache about four times. The numbers below come from the GGUF headers of
two real files, and the 27B one is checked against what llama.cpp measured on the B60 (D120).
"""

from __future__ import annotations

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import ModelConfig
from jaull.estimator import kv_cache
from jaull.metadata import config_merger

MIB = 1024**2
GIB = 1024**3


def _hybrid_flags(
    *, interval: int, conv: int, inner: int, state: int, groups: int
) -> dict[str, object]:
    return {
        "hybrid_attention": {
            "full_attention_interval": interval,
            "ssm_conv_kernel": conv,
            "ssm_inner_size": inner,
            "ssm_state_size": state,
            "ssm_group_count": groups,
        }
    }


def _qwen36_35b_a3b() -> ModelConfig:
    """Header of Qwen3.6-35B-A3B-UD-IQ3_S: 40 layers, an attention layer every 4th."""
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        raw_flags=_hybrid_flags(interval=4, conv=4, inner=4096, state=128, groups=16),
    )


def _qwen38_27b() -> ModelConfig:
    """Header of Qwen3.8-27B-UD-Q4_K_M: 65 layers, an attention layer every 4th."""
    return ModelConfig(
        hidden_size=5120,
        num_hidden_layers=65,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        raw_flags=_hybrid_flags(interval=4, conv=4, inner=6144, state=128, groups=16),
    )


def _estimate(config: ModelConfig, context: int, *, sessions: int = 1) -> kv_cache.KvCacheEstimate:
    return kv_cache.estimate_kv_cache(
        config=config,
        context_length=context,
        batch_size=1,
        kv_dtype=WeightPrecision.FLOAT16,
        concurrent_users=sessions,
    )


def test_a_hybrid_counts_only_its_attention_layers() -> None:
    est = _estimate(_qwen36_35b_a3b(), 262_144)

    attention_layers = 10  # 40 // 4
    kv = 2 * attention_layers * 2 * 256 * 262_144 * 2
    assert kv == 5 * GIB  # 20 MiB per thousand tokens, 5 GiB at 256K
    # `layers` stays the model's block count: the hardware fit splits the weights over it.
    assert est.layers == 40
    assert "10 of 40 layers" in " ".join(est.notes)
    # The recurrent state of the other 30 layers is constant and small next to the KV.
    state = est.component.bytes - kv
    assert 0 < state < 100 * MIB


def test_the_recurrent_state_is_per_sequence_and_does_not_grow_with_the_context() -> None:
    one = _estimate(_qwen36_35b_a3b(), 4096)
    long = _estimate(_qwen36_35b_a3b(), 262_144)
    two_sessions = _estimate(_qwen36_35b_a3b(), 4096, sessions=2)

    def state_of(est: kv_cache.KvCacheEstimate, context: int, sessions: int = 1) -> int:
        attention_kv = 2 * 10 * est.kv_heads * est.head_dim * context * 2 * sessions  # 40 // 4
        return est.component.bytes - attention_kv

    assert state_of(one, 4096) == state_of(long, 262_144)
    assert state_of(two_sessions, 4096, 2) == 2 * state_of(one, 4096)


def test_the_27b_matches_what_llama_cpp_measured_on_the_b60() -> None:
    """D120: 854 MiB of context at 4096 tokens with four slots (one recurrent state each)."""
    est = _estimate(_qwen38_27b(), 4096, sessions=1)
    attention_kv = 2 * 16 * 4 * 256 * 4096 * 2
    assert est.layers == 65
    assert "16 of 65 layers" in " ".join(est.notes)  # 65 // 4
    assert attention_kv == 256 * MIB

    state_per_sequence = est.component.bytes - attention_kv
    assert abs(state_per_sequence - 153 * MIB) < 2 * MIB

    four_slots = attention_kv + 4 * state_per_sequence
    assert abs(four_slots - 854 * MIB) / (854 * MIB) < 0.02


def test_a_hybrid_estimate_says_so() -> None:
    est = _estimate(_qwen36_35b_a3b(), 32_768)

    assert any("hybrid" in note.lower() for note in est.notes)
    assert est.component.confidence is not EstimationConfidence.HIGH


def test_a_model_without_the_hybrid_flags_is_unchanged() -> None:
    plain = ModelConfig(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
    )

    est = _estimate(plain, 32_768)

    assert est.component.bytes == 2 * 40 * 2 * 256 * 32_768 * 2
    assert est.layers == 40
    assert not any("hybrid" in note.lower() for note in est.notes)


def test_inconsistent_hybrid_flags_are_ignored() -> None:
    for broken in (
        _hybrid_flags(interval=0, conv=4, inner=4096, state=128, groups=16),
        _hybrid_flags(interval=1, conv=4, inner=4096, state=128, groups=16),
        _hybrid_flags(interval=4, conv=4, inner=4096, state=0, groups=16),
        {"hybrid_attention": {"full_attention_interval": 4}},
        {"hybrid_attention": "yes"},
    ):
        config = ModelConfig(
            hidden_size=2048,
            num_hidden_layers=40,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            raw_flags=broken,
        )
        est = _estimate(config, 4096)
        assert est.component.bytes == 2 * 40 * 2 * 256 * 4096 * 2  # the standard formula


def test_the_gguf_header_keys_reach_the_config() -> None:
    header = GgufHeaderMetadata(
        architecture="qwen35moe",
        embedding_length=2048,
        block_count=40,
        head_count=16,
        head_count_kv=2,
        raw_kv={
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.value_length": 256,
            "qwen35moe.full_attention_interval": 4,
            "qwen35moe.ssm.conv_kernel": 4,
            "qwen35moe.ssm.inner_size": 4096,
            "qwen35moe.ssm.state_size": 128,
            "qwen35moe.ssm.group_count": 16,
        },
    )

    enriched = config_merger.merge(gguf_header=header, base_config=None)

    assert enriched is not None
    assert enriched.config.raw_flags["hybrid_attention"] == {
        "full_attention_interval": 4,
        "ssm_conv_kernel": 4,
        "ssm_inner_size": 4096,
        "ssm_state_size": 128,
        "ssm_group_count": 16,
    }
    assert "10 of 40 layers" in " ".join(_estimate(enriched.config, 262_144).notes)


def test_a_dense_gguf_header_adds_no_hybrid_flag() -> None:
    header = GgufHeaderMetadata(
        architecture="qwen2", embedding_length=5120, block_count=64, head_count=40, head_count_kv=8,
        raw_kv={"qwen2.attention.key_length": 128},
    )

    enriched = config_merger.merge(gguf_header=header, base_config=None)

    assert enriched is not None
    assert "hybrid_attention" not in enriched.config.raw_flags
