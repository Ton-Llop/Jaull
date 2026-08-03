"""Combine a GGUF header with a base-model ``config.json`` under a documented policy.

Precedence rule: the artifact that is going to be executed wins. GGUF header values
override the base config for anything the header states explicitly. The base config
fills the gaps. Conflicts are reported (not raised) so the user can inspect them.
"""

from __future__ import annotations

from jaull.domain.enrichment import (
    EnrichedConfig,
    GgufHeaderMetadata,
)
from jaull.domain.estimation import ConfigurationSource
from jaull.domain.model import ModelConfig


def merge(
    gguf_header: GgufHeaderMetadata | None,
    base_config: ModelConfig | None,
) -> EnrichedConfig | None:
    """Combine ``gguf_header`` and ``base_config`` into an enriched configuration.

    Returns ``None`` when neither source carries anything useful.
    """
    if gguf_header is None and base_config is None:
        return None

    sources: dict[str, ConfigurationSource] = {}
    conflicts: list[str] = []

    def pick(
        field: str,
        header_value: object,
        base_value: object,
    ) -> object:
        if header_value is not None and base_value is not None:
            if header_value != base_value:
                conflicts.append(
                    f"{field}: gguf_header={header_value!r} vs base_config={base_value!r} "
                    f"— GGUF wins because the artifact is what will run."
                )
                sources[field] = ConfigurationSource.GGUF_HEADER
                return header_value
            sources[field] = ConfigurationSource.GGUF_HEADER
            return header_value
        if header_value is not None:
            sources[field] = ConfigurationSource.GGUF_HEADER
            return header_value
        if base_value is not None:
            sources[field] = ConfigurationSource.BASE_MODEL_CONFIG
            return base_value
        return None

    header = gguf_header
    base = base_config

    architectures: list[str] = []
    if header is not None and header.architecture:
        architectures = [_arch_to_class_name(header.architecture)]
        sources["architectures"] = ConfigurationSource.GGUF_HEADER
    elif base is not None and base.architectures:
        architectures = list(base.architectures)
        sources["architectures"] = ConfigurationSource.BASE_MODEL_CONFIG

    model_type = pick(
        "model_type",
        header.architecture if header else None,
        base.model_type if base else None,
    )

    max_pos = pick(
        "max_position_embeddings",
        header.context_length if header else None,
        base.max_position_embeddings if base else None,
    )

    hidden = pick(
        "hidden_size",
        header.embedding_length if header else None,
        base.hidden_size if base else None,
    )

    layers = pick(
        "num_hidden_layers",
        header.block_count if header else None,
        base.num_hidden_layers if base else None,
    )

    heads = pick(
        "num_attention_heads",
        header.head_count if header else None,
        base.num_attention_heads if base else None,
    )

    kv_heads = pick(
        "num_key_value_heads",
        header.head_count_kv if header else None,
        base.num_key_value_heads if base else None,
    )

    head_dim = pick(
        "head_dim",
        header.rope_dim if header else None,
        base.head_dim if base else None,
    )

    # These live only in the base config today (GGUF headers rarely carry them).
    sliding = base.sliding_window if base else None
    if sliding is not None:
        sources["sliding_window"] = ConfigurationSource.BASE_MODEL_CONFIG

    torch_dtype = base.torch_dtype if base else None
    if torch_dtype is not None:
        sources["torch_dtype"] = ConfigurationSource.BASE_MODEL_CONFIG

    intermediate = base.intermediate_size if base else None
    if intermediate is not None:
        sources["intermediate_size"] = ConfigurationSource.BASE_MODEL_CONFIG

    rope_scaling = base.rope_scaling if base else None
    if rope_scaling is not None:
        sources["rope_scaling"] = ConfigurationSource.BASE_MODEL_CONFIG

    tie_wte = base.tie_word_embeddings if base else None
    if tie_wte is not None:
        sources["tie_word_embeddings"] = ConfigurationSource.BASE_MODEL_CONFIG

    vocab = base.vocab_size if base else None
    if vocab is not None:
        sources["vocab_size"] = ConfigurationSource.BASE_MODEL_CONFIG

    raw_flags = dict(base.raw_flags) if base else {}

    config = ModelConfig(
        architectures=architectures,
        model_type=_as_str(model_type),
        torch_dtype=torch_dtype,
        max_position_embeddings=_as_int(max_pos),
        hidden_size=_as_int(hidden),
        num_hidden_layers=_as_int(layers),
        num_attention_heads=_as_int(heads),
        num_key_value_heads=_as_int(kv_heads),
        head_dim=_as_int(head_dim),
        intermediate_size=intermediate,
        sliding_window=sliding,
        rope_scaling=rope_scaling,
        tie_word_embeddings=tie_wte,
        vocab_size=vocab,
        raw_flags=raw_flags,
    )

    return EnrichedConfig(config=config, sources=sources, conflicts=conflicts)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _arch_to_class_name(architecture: str) -> str:
    """Map GGUF ``general.architecture`` values to the transformers class name.

    GGUF stores short names like ``llama``, ``qwen2``, ``mistral``. Transformers
    architecture strings look like ``LlamaForCausalLM``. This function is not
    exhaustive on purpose — the field is informational for reporting, not used
    by the estimator.
    """
    mapping = {
        "llama": "LlamaForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "mistral": "MistralForCausalLM",
        "phi3": "Phi3ForCausalLM",
        "gemma": "GemmaForCausalLM",
        "gemma2": "Gemma2ForCausalLM",
        "gemma3": "Gemma3ForCausalLM",
        "falcon": "FalconForCausalLM",
        "gptj": "GPTJForCausalLM",
        "gpt2": "GPT2LMHeadModel",
    }
    return mapping.get(architecture.lower(), f"{architecture}ForCausalLM")


__all__ = ["merge"]
