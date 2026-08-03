from __future__ import annotations

import pytest

from jaull.exceptions import InvalidModelReferenceError
from jaull.huggingface.url_parser import normalize_repo_id


@pytest.mark.parametrize(
    "reference",
    [
        "Qwen/Qwen2.5-7B-Instruct",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json",
        "https://hf.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json",
        "https://www.huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    ],
)
def test_all_variants_produce_canonical_repo_id(reference: str) -> None:
    assert normalize_repo_id(reference) == "Qwen/Qwen2.5-7B-Instruct"


def test_repo_id_with_dot_and_dashes_is_accepted() -> None:
    assert normalize_repo_id("meta-llama/Llama-3.2-1B") == "meta-llama/Llama-3.2-1B"


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "   ",
        "not-a-repo",
        "too/many/segments",
        "https://example.com/Qwen/Qwen2.5-7B-Instruct",
        "https://github.com/foo/bar",
        "ftp://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "https://huggingface.co/",
        "https://huggingface.co/Qwen",
    ],
)
def test_invalid_references_are_rejected(reference: str) -> None:
    with pytest.raises(InvalidModelReferenceError):
        normalize_repo_id(reference)


def test_special_characters_in_segments_rejected() -> None:
    with pytest.raises(InvalidModelReferenceError):
        normalize_repo_id("bad owner/name")


def test_reserved_top_segment_rejected() -> None:
    with pytest.raises(InvalidModelReferenceError):
        normalize_repo_id("https://huggingface.co/tree/main")
