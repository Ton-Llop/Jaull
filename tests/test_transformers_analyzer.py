from __future__ import annotations

import pytest

from jaull.analyzers.transformers import _model_config_from_dict


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_raw_config_preserves_quantization_config(method: str) -> None:
    raw = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "quantization_config": {
            "bits": 4,
            "quant_method": method,
            "group_size": 128,
        },
    }

    config = _model_config_from_dict(raw)

    assert config.quantization_config == {
        "bits": 4,
        "quant_method": method,
        "group_size": 128,
    }


# ---------------------------------------------------------------------------
# Sliding window: the number and the flag are separate facts
# ---------------------------------------------------------------------------
# A populated `sliding_window` does not mean attention is windowed. These are
# the three shapes real configs actually take, so the parser has to keep the
# flag's absence distinguishable from an explicit false.


def test_qwen_declares_a_window_it_does_not_use() -> None:
    """Qwen2/2.5 set the field to their full context with the flag off."""
    raw = {
        "model_type": "qwen2",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "max_position_embeddings": 32768,
        "sliding_window": 32768,
        "use_sliding_window": False,
        "max_window_layers": 21,
    }

    config = _model_config_from_dict(raw)

    assert config.sliding_window == 32768
    assert config.use_sliding_window is False


def test_mistral_style_config_carries_a_window_without_the_flag() -> None:
    """Predates `use_sliding_window`, and does window its attention."""
    raw = {
        "model_type": "mistral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "max_position_embeddings": 32768,
        "sliding_window": 4096,
    }

    config = _model_config_from_dict(raw)

    assert config.sliding_window == 4096
    assert config.use_sliding_window is None, "absent must not collapse to False"


def test_a_config_can_declare_a_window_and_switch_it_on() -> None:
    raw = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "sliding_window": 4096,
        "use_sliding_window": True,
    }

    config = _model_config_from_dict(raw)

    assert config.sliding_window == 4096
    assert config.use_sliding_window is True


def test_a_config_without_windowed_attention_reports_neither() -> None:
    raw = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32}

    config = _model_config_from_dict(raw)

    assert config.sliding_window is None
    assert config.use_sliding_window is None


def test_a_non_boolean_flag_is_not_guessed_at() -> None:
    """`_bool_or_none` must not coerce; a string is not a decision."""
    raw = {"model_type": "qwen2", "sliding_window": 4096, "use_sliding_window": "false"}

    config = _model_config_from_dict(raw)

    assert config.use_sliding_window is None
