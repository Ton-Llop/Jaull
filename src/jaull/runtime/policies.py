"""Documented constants for the RuntimeSelector."""

from __future__ import annotations

MiB = 1024 * 1024

# --------------------------------------------------------------------------
# Architectures that vLLM has first-class support for. Not exhaustive; the
# real support matrix in vLLM is broader. When in doubt, we fall back to
# recommending Transformers (which is architecturally portable).
# --------------------------------------------------------------------------
VLLM_SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset(
    {"llama", "qwen2", "mistral", "phi3", "gemma", "gemma2", "gemma3"}
)

# --------------------------------------------------------------------------
# llama.cpp heuristics for --n-gpu-layers.
# --------------------------------------------------------------------------
LLAMA_CPP_HEADROOM_BYTES = 256 * MiB  # kernel workspaces + KV growth
LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN = 20  # conservative fallback with warning

# Context length assumed when a launch plan is built without a memory estimate
# (e.g. `jaull run --n-gpu-layers N`, which skips estimation). Mirrors the
# defensive default in jaull.runtime.llama_cpp_runner.
LLAMA_CPP_DEFAULT_CONTEXT_SIZE = 4096
