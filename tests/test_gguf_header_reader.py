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
