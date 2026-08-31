"""Provenance-tracked parameter counts.

The composite recommendation logic needs to know *how* a parameter count was
obtained. A 7B derived from ``safetensors`` metadata is a measurement; a 7B
derived from the repo id is a guess. Treating them the same is what leads to
reports that quietly under-report a Qwen 2.5 7B as "4.3B".
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from jaull.domain.estimation import EstimationConfidence
from jaull.domain.model import ModelConfig


class ParameterCountSource(StrEnum):
    SAFETENSORS_METADATA = "safetensors_metadata"
    MODEL_CONFIG = "model_config"
    GGUF_METADATA = "gguf_metadata"
    NAME_INFERENCE = "name_inference"
    UNKNOWN = "unknown"


class ParameterCount(BaseModel):
    """A count with the evidence that produced it."""

    model_config = ConfigDict(frozen=True)

    count: int | None
    source: ParameterCountSource
    confidence: EstimationConfidence

    @property
    def is_known(self) -> bool:
        return self.count is not None and self.count > 0


class ConfigParameterDecomposition(BaseModel):
    """Parameter split inferred from a supported dense transformer config."""

    model_config = ConfigDict(frozen=True)

    total_parameters: int
    transformer_block_parameters: int
    non_block_parameters: int
    total_transformer_blocks: int


_SUPPORTED_GATED_MLP_MODEL_TYPES = frozenset(
    {
        "gemma",
        "gemma2",
        "gemma3",
        "llama",
        "mistral",
        "phi3",
        "qwen2",
        "qwen3",
    }
)
_UNSUPPORTED_DECOMPOSITION_FLAG_NAMES = frozenset(
    {
        "audio_config",
        "auto_map",
        "kv_lora_rank",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "num_local_experts",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "text_config",
        "v_head_dim",
        "vision_config",
    }
)


def estimate_config_parameter_decomposition(
    config: ModelConfig | None,
) -> ConfigParameterDecomposition | None:
    """Return a strict config-derived block/non-block parameter split.

    The formula models dense, gated MLP transformer families whose block shape
    is represented by ``ModelConfig``. It abstains for unknown architectures,
    MoE models, incomplete dimensions and unknown embedding tying rather than
    presenting an unsupported assumption as architectural evidence.
    """

    if config is None or (config.model_type or "").lower() not in (
        _SUPPORTED_GATED_MLP_MODEL_TYPES
    ):
        return None
    if config.tie_word_embeddings is None or config.intermediate_size is None:
        return None
    if _has_unsupported_decomposition_flags(config):
        return None
    if (
        config.head_dim is None
        and config.hidden_size is not None
        and config.num_attention_heads is not None
        and config.hidden_size % config.num_attention_heads != 0
    ):
        return None
    kv_heads = config.num_key_value_heads or config.num_attention_heads
    if (
        kv_heads is not None
        and config.num_attention_heads is not None
        and kv_heads > config.num_attention_heads
    ):
        return None
    return _parameter_components(
        config,
        intermediate_size=config.intermediate_size,
        tie_word_embeddings=config.tie_word_embeddings,
    )


def estimate_total_parameters_from_config(config: ModelConfig | None) -> int | None:
    """Preserve Jaull's existing best-effort total parameter estimate.

    This compatibility calculation intentionally keeps the historical
    assumptions used by capability evidence: an absent intermediate width uses
    ``4 * hidden_size`` and unknown embedding tying is treated as untied. New
    byte decomposition uses the stricter function above.
    """

    if config is None or config.hidden_size is None:
        return None
    intermediate_size = config.intermediate_size or (4 * config.hidden_size)
    components = _parameter_components(
        config,
        intermediate_size=intermediate_size,
        tie_word_embeddings=bool(config.tie_word_embeddings),
    )
    return components.total_parameters if components is not None else None


def _parameter_components(
    config: ModelConfig,
    *,
    intermediate_size: int,
    tie_word_embeddings: bool,
) -> ConfigParameterDecomposition | None:
    layers = config.num_hidden_layers
    hidden = config.hidden_size
    heads = config.num_attention_heads
    vocab = config.vocab_size
    if not layers or not hidden or not heads or not vocab or intermediate_size <= 0:
        return None

    head_dim = config.head_dim or (hidden // heads)
    kv_heads = config.num_key_value_heads or heads
    if head_dim <= 0 or kv_heads <= 0:
        return None

    attention_per_block = (
        hidden * heads * head_dim
        + 2 * hidden * kv_heads * head_dim
        + hidden * hidden
    )
    ffn_per_block = 3 * hidden * intermediate_size
    transformer_block_parameters = layers * (
        attention_per_block + ffn_per_block
    )
    embedding_parameters = vocab * hidden
    output_head_parameters = 0 if tie_word_embeddings else vocab * hidden
    non_block_parameters = embedding_parameters + output_head_parameters
    total_parameters = transformer_block_parameters + non_block_parameters
    if total_parameters <= 0:
        return None
    return ConfigParameterDecomposition(
        total_parameters=total_parameters,
        transformer_block_parameters=transformer_block_parameters,
        non_block_parameters=non_block_parameters,
        total_transformer_blocks=layers,
    )


def _has_unsupported_decomposition_flags(config: ModelConfig) -> bool:
    for name in _UNSUPPORTED_DECOMPOSITION_FLAG_NAMES:
        value = config.raw_flags.get(name)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
        if value is not None and not isinstance(value, (bool, int, float)):
            return True
    return False


__all__ = [
    "ConfigParameterDecomposition",
    "ParameterCount",
    "ParameterCountSource",
    "estimate_config_parameter_decomposition",
    "estimate_total_parameters_from_config",
]
