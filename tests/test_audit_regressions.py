"""Offline regressions for metadata identity, KV dimensions and download reporting."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jaull.domain.inference import WeightPrecision
from jaull.domain.model import ModelConfig
from jaull.estimator.kv_cache import estimate_kv_cache
from jaull.huggingface.client import HfClient
from jaull.metadata.config_merger import merge
from jaull.metadata.gguf_reader import parse_header
from jaull.workflow.download_progress import DownloadProgressReporter
from tests._gguf_fixtures import build_header
from tests.test_artifact_service import _gguf_repo, _make_service, _stub_downloader


@pytest.mark.parametrize("explicit_lengths", [False, True])
@pytest.mark.parametrize("with_base", [False, True])
def test_partial_rope_never_shrinks_kv_heads(explicit_lengths, with_base):
    metadata = {
        "general.architecture": "llama", "llama.embedding_length": 4096,
        "llama.block_count": 32, "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 4, "llama.rope.dimension_count": 64,
    }
    if explicit_lengths:
        metadata.update({"llama.attention.key_length": 256, "llama.attention.value_length": 256})
    header = parse_header(build_header(metadata))
    base = ModelConfig(head_dim=256) if with_base else None
    enriched = merge(header, base)
    kv = estimate_kv_cache(enriched.config, 4096, 1, WeightPrecision.FLOAT16)
    assert kv.head_dim == 256
    assert kv.component.bytes == 512 * 1024**2


def test_unequal_kv_dimensions_are_not_certified_by_symmetric_formula():
    header = parse_header(build_header({
        "general.architecture": "llama", "llama.embedding_length": 4096,
        "llama.block_count": 32, "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 4,
        "llama.attention.key_length": 256, "llama.attention.value_length": 128,
    }))
    kv = estimate_kv_cache(merge(header, None).config, 4096, 1, WeightPrecision.FLOAT16)
    assert kv.component.bytes is None
    assert "key/value" in " ".join(kv.notes)


def test_explicit_gguf_key_length_overrides_base_head_dim():
    header = parse_header(build_header({
        "general.architecture": "llama", "llama.embedding_length": 4096,
        "llama.block_count": 32, "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 4, "llama.rope.dimension_count": 64,
        "llama.attention.key_length": 128, "llama.attention.value_length": 128,
    }))
    enriched = merge(header, ModelConfig(head_dim=256))
    assert enriched.config.head_dim == 128
    assert enriched.sources["head_dim"].value == "gguf_header"
    assert any("head_dim" in conflict for conflict in enriched.conflicts)


def test_revision_specific_metadata_is_used_for_download_and_verification(tmp_path: Path):
    old_bytes = b"old revision"
    old = _gguf_repo([("old-q5_k_m.gguf", len(old_bytes))]).info
    old.sha = "a" * 40
    latest = _gguf_repo([("new-q5_k_m.gguf", 100)]).info
    client = Mock()
    client.model_info.side_effect = lambda repo, **kw: old if kw.get("revision") == "v1" else latest
    service = _make_service(tmp_path, client, _stub_downloader(old_bytes))
    artifact = service.resolve("owner/repo", "Q5_K_M", revision="v1")
    client.model_info.assert_called_once_with("owner/repo", revision="v1")
    assert artifact.revision == old.sha
    assert artifact.filename == "old-q5_k_m.gguf"
    assert service.verify(service.download(artifact), full=True).is_verified


def test_hf_client_forwards_revision():
    api = Mock()
    client = HfClient(api=api)
    client.model_info("owner/repo", revision="v1")
    api.model_info.assert_called_once_with(
        repo_id="owner/repo", files_metadata=True, revision="v1",
    )


def test_resumed_download_rate_counts_only_new_bytes():
    now = [0.0]
    lines = []
    reporter = DownloadProgressReporter(lines.append, clock=lambda: now[0])
    mib = 1024**2
    reporter(9000 * mib, 10000 * mib)
    now[0] = 1.0
    reporter(9001 * mib, 10000 * mib)
    assert "1.0 MiB/s" in lines[-1]
    assert "ETA 16:39" in lines[-1]
    assert "90 %" in lines[-1]
    # A restarted transfer must establish a new baseline, not a negative speed.
    now[0] = 2.0
    reporter(0, 10000 * mib)
    now[0] = 3.0
    reporter(mib, 10000 * mib)
    assert "1.0 MiB/s" in lines[-1]


def test_metadata_cache_separates_revisions():
    from jaull.workflow.orchestrator import _RunMetadataClient

    client = Mock()
    client.model_info.side_effect = lambda repo, **kw: SimpleNamespace(sha=kw.get("revision"))
    cached = _RunMetadataClient(client, Mock())
    assert cached.model_info("owner/repo", revision="a").sha == "a"
    assert cached.model_info("owner/repo", revision="b").sha == "b"
    assert cached.model_info("owner/repo", revision="a").sha == "a"
    assert client.model_info.call_count == 2
