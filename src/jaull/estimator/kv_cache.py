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

    effective_context = context_length
    if config.sliding_window and context_length > config.sliding_window:
        effective_context = config.sliding_window
        notes.append(
            f"Context capped at sliding_window={config.sliding_window} tokens "
            f"(user requested {context_length})."
        )

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
            confidence=EstimationConfidence.HIGH,
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
