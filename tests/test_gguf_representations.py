from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaull.adapters.cache.model_analysis_cache import ModelAnalysisCache
from jaull.domain.candidates import ModelCandidate
from jaull.domain.estimation import EstimateSource, EstimationConfidence
from jaull.domain.gguf_variants import preferred_gguf_variants
from jaull.domain.model import GgufVariant, ModelFile
from jaull.estimator.gguf_selection import select_variant
from jaull.estimator.weights import estimate_weights_from_gguf
from jaull.exceptions import GgufVariantSelectionError
from jaull.huggingface.artifact_resolver import HuggingFaceArtifactResolver
from jaull.huggingface.classifiers import classify_repository
from tests._workflow_fixtures import gguf_analysis
from tests.test_artifact_service import _FakeHfClient, _FakeInfo, _Sibling


def _variants(files: list[ModelFile]) -> list[GgufVariant]:
    return classify_repository(files).gguf_variants


@pytest.mark.parametrize("folder", ["", "split/", "Q4_K_M/"])
def test_whole_and_parts_are_alternatives_and_resolver_matches_estimator(folder: str) -> None:
    files = [
        ModelFile(path="m-Q4_K_M.gguf", size_bytes=5000),
        ModelFile(path=f"{folder}m-Q4_K_M-00001-of-00002.gguf", size_bytes=3100),
        ModelFile(path=f"{folder}m-Q4_K_M-00002-of-00002.gguf", size_bytes=2100),
    ]
    variants = _variants(files)
    assert len(variants) == 2
    choice = select_variant(variants, "Q4_K_M").variant
    assert choice.total_bytes == 5000
    assert [file.path for file in choice.files] == ["m-Q4_K_M.gguf"]
    assert preferred_gguf_variants(variants) == [choice]
    assert select_variant(list(reversed(variants)), "Q4_K_M").variant == choice

    client = _FakeHfClient(
        _FakeInfo(siblings=[_Sibling(rfilename=file.path, size=file.size_bytes) for file in files])
    )
    artifact = HuggingFaceArtifactResolver(client).resolve("org/model", quantization="Q4_K_M")
    assert artifact.filename == choice.files[0].path
    assert artifact.size_bytes == estimate_weights_from_gguf(choice).component.bytes


@pytest.mark.parametrize("size", [None, 0])
def test_missing_whole_size_is_unknown_even_when_split_sizes_are_known(size: int | None) -> None:
    variants = _variants(
        [
            ModelFile(path="m-Q4_K_M.gguf", size_bytes=size),
            ModelFile(path="m-Q4_K_M-00001-of-00002.gguf", size_bytes=3000),
            ModelFile(path="m-Q4_K_M-00002-of-00002.gguf", size_bytes=2000),
        ]
    )
    whole = select_variant(variants, "Q4_K_M").variant
    estimate = estimate_weights_from_gguf(whole).component
    assert whole.total_bytes is None
    assert estimate.bytes is None
    assert estimate.source is EstimateSource.UNKNOWN
    assert estimate.confidence is EstimationConfidence.UNKNOWN


def test_unrelated_whole_files_are_not_summed_or_chosen_arbitrarily() -> None:
    files = [
        ModelFile(path="a-Q4_K_M.gguf", size_bytes=3000),
        ModelFile(path="b-Q4_K_M.gguf", size_bytes=4000),
    ]
    variants = _variants(files)
    assert sorted(v.total_bytes for v in variants) == [3000, 4000]
    assert preferred_gguf_variants(variants) == []
    with pytest.raises(GgufVariantSelectionError, match="Multiple artifacts"):
        select_variant(variants, "Q4_K_M")
    client = _FakeHfClient(
        _FakeInfo(siblings=[_Sibling(rfilename=file.path, size=file.size_bytes) for file in files])
    )
    with pytest.raises(GgufVariantSelectionError, match="Multiple artifacts"):
        HuggingFaceArtifactResolver(client).resolve("org/model", quantization="Q4_K_M")


@pytest.mark.parametrize(
    "paths",
    [
        ["m-Q4_K_M-00001-of-00002.gguf"],
        ["m-Q4_K_M-00000-of-00002.gguf", "m-Q4_K_M-00001-of-00002.gguf"],
        ["m-Q4_K_M-00001-of-00000.gguf"],
        ["a/m-Q4_K_M-00001-of-00002.gguf", "b/m-Q4_K_M-00002-of-00002.gguf"],
        ["m-Q4_K_M-00001-of-00002.gguf", "m-Q4_K_M-00002-of-00003.gguf"],
    ],
)
def test_incomplete_or_mixed_shard_sets_do_not_become_complete_artifacts(paths: list[str]) -> None:
    variants = _variants([ModelFile(path=path, size_bytes=1000) for path in paths])
    assert all(not variant.is_complete and variant.total_bytes is None for variant in variants)
    with pytest.raises(GgufVariantSelectionError, match="Incomplete"):
        select_variant(variants, "Q4_K_M")


def test_two_complete_shard_sets_are_independent_and_ambiguous() -> None:
    variants = _variants(
        [
            ModelFile(path=f"{folder}/m-Q4_K_M-{index:05d}-of-00002.gguf", size_bytes=1000)
            for folder in ("a", "b")
            for index in (1, 2)
        ]
    )
    assert len(variants) == 2
    assert all(variant.is_complete and variant.total_bytes == 2000 for variant in variants)
    with pytest.raises(GgufVariantSelectionError, match="Multiple artifacts"):
        select_variant(variants, "Q4_K_M")


def test_ambiguous_quantization_does_not_hide_another_unambiguous_option() -> None:
    variants = _variants(
        [
            ModelFile(path="a-Q4_K_M.gguf", size_bytes=3000),
            ModelFile(path="b-Q4_K_M.gguf", size_bytes=4000),
            ModelFile(path="a-Q8_0.gguf", size_bytes=6000),
        ]
    )
    preferred = preferred_gguf_variants(variants)
    assert [variant.quantization for variant in preferred] == ["Q8_0"]
    assert select_variant(variants, "Q8_0").variant == preferred[0]


def test_partial_size_does_not_become_an_exact_shard_set_size() -> None:
    variants = _variants(
        [
            ModelFile(path="m-Q4_K_M-00001-of-00002.gguf", size_bytes=1000),
            ModelFile(path="m-Q4_K_M-00002-of-00002.gguf", size_bytes=None),
        ]
    )
    assert variants[0].is_complete
    assert (
        estimate_weights_from_gguf(select_variant(variants, "Q4_K_M").variant).component.bytes
        is None
    )


def test_previous_analysis_cache_schema_is_invalidated(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    candidate = ModelCandidate(repo_id="org/model", revision_hint="abc")
    cache.put(candidate, gguf_analysis("org/model"))
    path = next(tmp_path.glob("*.json"))
    payload = json.loads(path.read_text())
    payload["schema_version"] = 4
    path.write_text(json.dumps(payload))
    assert cache.get(candidate) is None
    assert cache.stats.unsupported_schema == 1
