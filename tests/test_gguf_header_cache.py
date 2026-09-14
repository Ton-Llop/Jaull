from __future__ import annotations

from pathlib import Path

from jaull.adapters.cache.gguf_header_cache import GgufHeaderCache
from jaull.domain.enrichment import GgufHeaderMetadata
from tests._workflow_fixtures import gguf_analysis


def test_gguf_header_cache_round_trips_one_concrete_artifact(tmp_path: Path) -> None:
    analysis = gguf_analysis("org/model")
    variant = analysis.classification.gguf_variants[0]
    header = GgufHeaderMetadata(architecture="llama", block_count=32)

    GgufHeaderCache(root=tmp_path).put(analysis, variant, header)
    fresh = GgufHeaderCache(root=tmp_path)

    assert fresh.get(analysis, variant) == header
    assert fresh.stats.hits == 1


def test_gguf_header_cache_does_not_share_different_artifacts(tmp_path: Path) -> None:
    analysis = gguf_analysis("org/model")
    variant = analysis.classification.gguf_variants[0]
    changed_file = variant.files[0].model_copy(
        update={"path": "other-model.gguf", "size_bytes": variant.total_bytes + 1}
    )
    different_variant = variant.model_copy(update={"files": [changed_file]})
    cache = GgufHeaderCache(root=tmp_path)
    cache.put(analysis, variant, GgufHeaderMetadata(architecture="llama"))

    assert cache.get(analysis, different_variant) is None
    assert cache.stats.misses == 1


def test_gguf_header_cache_ignores_corrupt_entries(tmp_path: Path) -> None:
    analysis = gguf_analysis("org/model")
    variant = analysis.classification.gguf_variants[0]
    cache = GgufHeaderCache(root=tmp_path)
    cache.put(analysis, variant, GgufHeaderMetadata(architecture="llama"))
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")

    assert cache.get(analysis, variant) is None
    assert cache.stats.read_errors == 1


def test_gguf_header_cache_does_not_store_the_tokenizer_tables(tmp_path: Path) -> None:
    """The vocabulary is most of a GGUF header and nothing reads it back.

    A 73-entry cache came to 764 MB because ``tokenizer.ggml.merges`` and
    ``tokenizer.ggml.tokens`` were persisted verbatim; the same entries without
    them are 0.1 MB. ``raw_kv`` only exists so the reader can pull the typed
    fields out of it, so the cached copy drops those keys on purpose.
    """
    analysis = gguf_analysis("org/model")
    variant = analysis.classification.gguf_variants[0]
    header = GgufHeaderMetadata(
        architecture="llama",
        block_count=32,
        raw_kv={
            "general.architecture": "llama",
            "llama.block_count": 32,
            "tokenizer.ggml.tokens": ["tok"] * 5000,
            "tokenizer.ggml.merges": ["a b"] * 5000,
        },
    )
    cache = GgufHeaderCache(root=tmp_path)

    cache.put(analysis, variant, header)
    entry = next(tmp_path.glob("*.json"))
    restored = cache.get(analysis, variant)

    assert restored is not None
    # The fields the estimator actually consumes survive.
    assert restored.architecture == "llama"
    assert restored.block_count == 32
    assert restored.raw_kv["general.architecture"] == "llama"
    # The tokenizer tables do not.
    assert not any(key.startswith("tokenizer.") for key in restored.raw_kv)
    assert entry.stat().st_size < 4096
