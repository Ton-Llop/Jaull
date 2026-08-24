"""Estimate the memory footprint of the transformer KV cache."""

from __future__ import annotations

from jaull.domain.estimation import (
    EstimateSource,
    EstimationConfidence,
    KvCacheEstimate,
    MemoryComponent,
)
from jaull.domain.inference import WeightPrecision
from jaull.domain.model import ModelConfig
from jaull.estimator.policies import bytes_per_parameter

FORMULA = (
    "kv_bytes = 2 * num_layers * num_kv_heads * head_dim "
    "* context * batch * concurrent_users * bytes_per_kv_element"
)

# Architectures that document ``sliding_window`` on its own and never expose
# ``use_sliding_window``: there, a populated window is in force and its absence
# from the config carries no information. Kept minimal on purpose — see
# ``_apply_sliding_window``. Every entry needs evidence and a test.
_WINDOW_ALWAYS_ACTIVE_MODEL_TYPES = frozenset({"mistral", "mixtral"})


def estimate_kv_cache(
    config: ModelConfig | None,
    context_length: int,
    batch_size: int,
    kv_dtype: WeightPrecision,
    concurrent_users: int = 1,
) -> KvCacheEstimate:
    dtype_bytes = int(bytes_per_parameter(kv_dtype))
    notes: list[str] = []

    if config is None:
        return _unknown(
            context_length=context_length,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
            reason="Model configuration is not available; KV cache cannot be estimated.",
        )

    exotic = _detect_exotic_architecture(config)
    if exotic:
        return _unknown(
            context_length=context_length,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
            reason=f"Non-standard attention architecture: {exotic}.",
        )

    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    hidden = config.hidden_size

    if layers is None or heads is None or hidden is None:
        missing = [
            name
            for name, value in (
                ("num_hidden_layers", layers),
                ("num_attention_heads", heads),
                ("hidden_size", hidden),
            )
            if value is None
        ]
        return _unknown(
            context_length=context_length,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
            reason=f"Missing configuration fields: {', '.join(missing)}.",
        )

    kv_heads: int = config.num_key_value_heads or heads
    explicit_head_dim = config.head_dim
    head_dim: int = explicit_head_dim if explicit_head_dim else hidden // heads

    effective_context, window_note, window_assumed = _apply_sliding_window(
        config, context_length
    )
    if window_note is not None:
        notes.append(window_note)

    kv_bytes = (
        2
        * layers
        * kv_heads
        * head_dim
        * effective_context
        * batch_size
        * concurrent_users
        * dtype_bytes
    )

    if config.num_key_value_heads and config.num_key_value_heads < heads:
        notes.append(
            f"Grouped-Query Attention: {kv_heads} KV heads for {heads} attention heads."
        )
    if concurrent_users > 1:
        notes.append(
            f"Sized for {concurrent_users} concurrent sessions; assumes each keeps "
            "its own KV cache."
        )

    explanation = (
        f"{layers} layers x {kv_heads} KV heads x {head_dim} head_dim x "
        f"{effective_context} tokens x batch {batch_size} x {dtype_bytes} bytes "
        f"x {concurrent_users} sessions x 2."
    )

    return KvCacheEstimate(
        component=MemoryComponent(
            name="KV cache",
            bytes=kv_bytes,
            source=EstimateSource.DERIVED,
            confidence=(
                EstimationConfidence.MEDIUM
                if window_assumed
                else EstimationConfidence.HIGH
            ),
            explanation=explanation,
        ),
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context_length=effective_context,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        formula=FORMULA,
        notes=notes,
    )


def _apply_sliding_window(
    config: ModelConfig, context_length: int
) -> tuple[int, str | None, bool]:
    """Decide whether a declared sliding window actually bounds the KV cache.

    A populated ``sliding_window`` is not by itself evidence that attention is
    windowed, and different architectures use the field differently. Qwen2 and
    Qwen2.5 ship it set to their full context and gate it behind
    ``use_sliding_window: false``; Mistral documents ``sliding_window`` alone
    and never exposes the flag at all, so there its absence does not mean the
    window is off.

    Clamping *reduces* the estimate, so guessing wrong in that direction can
    report that a model fits when it needs several times the memory — measured
    at 4x against llama.cpp on Qwen2.5-1.5B at 131072 tokens. The unknown case
    therefore resolves the other way:

    * ``true``   - the window is in force. Clamp, HIGH confidence.
    * ``false``  - declared but inactive. Do not clamp, HIGH confidence.
    * ``None``   - the config did not say:

      * on an architecture whose ``sliding_window`` semantics are known
        (:data:`_WINDOW_ALWAYS_ACTIVE_MODEL_TYPES`), clamp, MEDIUM confidence
        — the decision came from the architecture, not from the config;
      * otherwise size the full context, MEDIUM confidence. This over-estimates
        rather than under-estimates, which is the safe direction for an advisor
        that answers "does this fit".

    The known-architecture set is deliberately small. It grows when there is
    evidence and a test for the entry, not to pre-empt every family.

    Returns the context to size for, a note when there is something to say, and
    whether the decision rested on an assumption rather than on the config.
    """

    window = config.sliding_window
    if not window or context_length <= window:
        return context_length, None, False

    if config.use_sliding_window is True:
        return (
            window,
            f"Context capped at sliding_window={window} tokens "
            f"(user requested {context_length}); use_sliding_window is true.",
            False,
        )

    if config.use_sliding_window is False:
        return (
            context_length,
            f"Config declares sliding_window={window} but use_sliding_window is "
            "false, so the window does not bound the KV cache and the full "
            f"{context_length}-token context is sized.",
            False,
        )

    model_type = (config.model_type or "").lower()
    if model_type in _WINDOW_ALWAYS_ACTIVE_MODEL_TYPES:
        return (
            window,
            f"Context capped at sliding_window={window} tokens (user requested "
            f"{context_length}). {model_type} does not expose "
            "use_sliding_window and windows attention unconditionally, so the "
            "window is applied on the architecture's behaviour.",
            True,
        )

    return (
        context_length,
        f"Config declares sliding_window={window} but not use_sliding_window, "
        f"and {model_type or 'this architecture'} has no known window "
        f"semantics. Sized for the full {context_length}-token context to avoid "
        "under-estimating the KV cache.",
        True,
    )


def _unknown(
    context_length: int,
    batch_size: int,
    dtype_bytes: int,
    reason: str,
) -> KvCacheEstimate:
    return KvCacheEstimate(
        component=MemoryComponent(
            name="KV cache",
            bytes=None,
            source=EstimateSource.UNKNOWN,
            confidence=EstimationConfidence.UNKNOWN,
            explanation=reason,
        ),
        layers=None,
        kv_heads=None,
        head_dim=None,
        context_length=context_length,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        formula=FORMULA,
        notes=[reason],
    )


def _detect_exotic_architecture(config: ModelConfig) -> str | None:
    flags = config.raw_flags
    if any(k in flags for k in ("num_experts", "num_local_experts", "moe_intermediate_size")):
        return "Mixture-of-Experts (per-expert KV footprint not modelled)"
    if any(k in flags for k in ("q_lora_rank", "kv_lora_rank")):
        return "Multi-head Latent Attention (Deepseek-style)"
    if any(k in flags for k in ("text_config", "vision_config", "audio_config")):
        return "Multimodal composite configuration"
    if "auto_map" in flags:
        return "Custom architecture requiring trust_remote_code"
    return None


__all__ = ["FORMULA", "estimate_kv_cache"]
