from __future__ import annotations

import pytest

from jaull.exceptions import (
    GgufHeaderIncompleteError,
    GgufHeaderInvalidError,
)
from jaull.metadata.gguf_reader import parse_header
from tests._gguf_fixtures import build_header


def test_llama_mha_header_is_parsed() -> None:
    data = build_header(
        {
            "general.architecture": "llama",
            "general.name": "Meta-Llama-3.1-8B-Instruct",
            "llama.context_length": 131072,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 32,
            "llama.rope.dimension_count": 128,
        }
    )
    header = parse_header(data)
    assert header is not None
    assert header.architecture == "llama"
    assert header.context_length == 131072
    assert header.head_count == 32
    assert header.head_count_kv == 32
    assert header.rope_dim == 128
    assert header.block_count == 32


def test_qwen_gqa_header_is_parsed() -> None:
    data = build_header(
        {
            "general.architecture": "qwen2",
            "qwen2.context_length": 32768,
            "qwen2.embedding_length": 3584,
            "qwen2.block_count": 28,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
        }
    )
    header = parse_header(data)
    assert header is not None
    assert header.architecture == "qwen2"
    assert header.head_count == 28
    assert header.head_count_kv == 4  # GQA


def test_source_repository_is_extracted() -> None:
    data = build_header(
        {
            "general.architecture": "llama",
            "general.source.huggingface.repository": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
    )
    header = parse_header(data)
    assert header is not None
    assert header.source_repository == "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_missing_magic_returns_none() -> None:
    result = parse_header(b"NOT-GGUF-AT-ALL" + b"\x00" * 100)
    assert result is None


def test_truncated_header_raises_incomplete() -> None:
    full = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,
        }
    )
    with pytest.raises(GgufHeaderIncompleteError):
        parse_header(full[:20])


def test_unsupported_version_raises_invalid() -> None:
    # Version 99 does not exist
    fake = build_header({"general.architecture": "llama"}, version=99)
    with pytest.raises(GgufHeaderInvalidError):
        parse_header(fake)


def test_unknown_architecture_still_parses_general_only() -> None:
    data = build_header(
        {
            "general.architecture": "totally-new-arch",
            "general.name": "Weird 1B",
            # No <arch>.* keys at all
        }
    )
    header = parse_header(data)
    assert header is not None
    assert header.architecture == "totally-new-arch"
    assert header.context_length is None
    assert header.head_count is None


def test_array_of_strings_is_accepted() -> None:
    data = build_header(
        {
            "general.architecture": "llama",
            "tokenizer.ggml.tokens": ["<pad>", "<bos>", "<eos>"],
        }
    )
    header = parse_header(data)
    assert header is not None
    assert isinstance(header.raw_kv["tokenizer.ggml.tokens"], list)


def _architecture_then_tokenizer() -> tuple[bytes, int]:
    """A header written the way llama.cpp writes it: architecture keys, then a huge tokenizer."""
    architecture = build_header(
        {
            "general.architecture": "qwen3_5",
            "qwen3_5.context_length": 262144,
            "qwen3_5.embedding_length": 5120,
            "qwen3_5.block_count": 64,
            "qwen3_5.attention.head_count": 24,
            "qwen3_5.attention.head_count_kv": 4,
        }
    )
    full = build_header(
        {
            "general.architecture": "qwen3_5",
            "qwen3_5.context_length": 262144,
            "qwen3_5.embedding_length": 5120,
            "qwen3_5.block_count": 64,
            "qwen3_5.attention.head_count": 24,
            "qwen3_5.attention.head_count_kv": 4,
            "tokenizer.ggml.tokens": [f"token-{index}" for index in range(2000)],
        }
    )
    return full, len(architecture)


def test_a_header_cut_inside_the_tokenizer_still_gives_the_architecture() -> None:
    """Regression found on a B60 box: Qwen3.8-27B's tokenizer pushed the header past the 8 MiB
    download budget, so its KV cache stayed unknown and the model was never recommended, although
    every key the memory estimate needs comes before the tokenizer."""
    full, architecture_end = _architecture_then_tokenizer()

    header = parse_header(full[: architecture_end + 200])  # ends inside the token list

    assert header is not None
    assert header.architecture == "qwen3_5"
    assert header.block_count == 64
    assert header.embedding_length == 5120
    assert header.head_count == 24
    assert header.head_count_kv == 4
    assert header.is_complete is False  # keys not read yet are unknown, not absent


def test_a_complete_header_says_so() -> None:
    full, _ = _architecture_then_tokenizer()

    header = parse_header(full)

    assert header is not None and header.is_complete is True


def test_a_cut_is_not_accepted_when_a_key_the_estimate_reads_may_still_follow() -> None:
    """Found by an audit: a header written with `head_count_kv` and `context_length` after the
    tokenizer was accepted at the cut, and both were then read as absent (a full read gave 4 and
    32768). Unread keys are unknown; the reader keeps asking for bytes until it has them."""
    late = build_header(
        {
            "general.architecture": "qwen2",
            "qwen2.embedding_length": 3584,
            "qwen2.block_count": 28,
            "qwen2.attention.head_count": 28,
            "tokenizer.ggml.tokens": [f"token-{index}" for index in range(2000)],
            "qwen2.attention.head_count_kv": 4,
            "qwen2.context_length": 32768,
        }
    )
    cut = late.index(b"token-1000")

    with pytest.raises(GgufHeaderIncompleteError):
        parse_header(late[:cut])

    header = parse_header(late)
    assert header is not None and header.head_count_kv == 4 and header.context_length == 32768


def test_a_cut_outside_the_tokenizer_is_never_accepted() -> None:
    full = build_header(
        {
            "general.architecture": "qwen2",
            "qwen2.context_length": 32768,
            "qwen2.embedding_length": 3584,
            "qwen2.block_count": 28,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
            "qwen2.rope.dimension_count": 128,
        }
    )

    with pytest.raises(GgufHeaderIncompleteError):
        parse_header(full[:-3])  # inside the last architecture value


def test_a_header_cut_before_the_architecture_keys_still_asks_for_more() -> None:
    full, _ = _architecture_then_tokenizer()

    with pytest.raises(GgufHeaderIncompleteError):
        parse_header(full[:60])  # inside the architecture block


def test_the_size_label_is_exposed_and_can_be_missing() -> None:
    labelled = parse_header(build_header({
        "general.architecture": "mellum", "general.size_label": "12B-A2.5B",
    }))
    plain = parse_header(build_header({"general.architecture": "mellum"}))

    assert labelled is not None and labelled.size_label == "12B-A2.5B"
    assert plain is not None and plain.size_label is None
