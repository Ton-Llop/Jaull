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
