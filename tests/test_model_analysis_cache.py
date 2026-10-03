from __future__ import annotations

from pathlib import Path

from jaull.adapters.cache.model_analysis_cache import (
    ANALYSIS_CACHE_SCHEMA_VERSION,
    ModelAnalysisCache,
)
from jaull.advisor.service import AdvisorService
from jaull.domain.candidates import ModelCandidate
from jaull.domain.model import ModelAnalysis
from tests._workflow_fixtures import FakeHfClient, gguf_analysis, hardware, transformers_analysis


def test_model_analysis_cache_round_trips_exact_analysis(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    candidate = ModelCandidate(repo_id="org/model", revision_hint="abc123")
    analysis = transformers_analysis("org/model")

    cache.put(candidate, analysis)

    assert cache.get(candidate) == analysis
    assert cache.stats.hits == 1


def test_changed_revision_is_a_cache_miss(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    cache.put(
        ModelCandidate(repo_id="org/model", revision_hint="old"),
        transformers_analysis("org/model"),
    )

    assert cache.get(ModelCandidate(repo_id="org/model", revision_hint="new")) is None
    assert cache.stats.misses == 1


def test_cache_survives_new_instance(tmp_path: Path) -> None:
    candidate = ModelCandidate(repo_id="org/model", revision_hint="abc123")
    analysis = gguf_analysis("org/model")
    ModelAnalysisCache(root=tmp_path).put(candidate, analysis)

    fresh = ModelAnalysisCache(root=tmp_path)

    assert fresh.get(candidate) == analysis


def test_corrupt_json_is_ignored(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    candidate = ModelCandidate(repo_id="org/model", revision_hint="abc123")
    cache.put(candidate, transformers_analysis("org/model"))
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")

    assert cache.get(candidate) is None
    assert cache.stats.read_errors == 1


def test_unsupported_schema_is_ignored(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    candidate = ModelCandidate(repo_id="org/model", revision_hint="abc123")
    cache.put(candidate, transformers_analysis("org/model"))
    path = next(tmp_path.glob("*.json"))
    path.write_text(
        (
            '{"schema_version": '
            f"{ANALYSIS_CACHE_SCHEMA_VERSION + 1}, "
            '"repo_id": "org/model"}'
        ),
        encoding="utf-8",
    )

    assert cache.get(candidate) is None
    assert cache.stats.unsupported_schema == 1


def test_write_failure_does_not_raise(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    cache = ModelAnalysisCache(root=tmp_path)

    def fail_write(path: Path, payload: dict[str, object]) -> None:
        del path, payload
        raise OSError("nope")

    monkeypatch.setattr(cache, "_write_atomic", fail_write)

    cache.put(
        ModelCandidate(repo_id="org/model", revision_hint="abc123"),
        transformers_analysis("org/model"),
    )

    assert cache.stats.write_errors == 1


def test_repo_id_cannot_path_traverse(tmp_path: Path) -> None:
    cache = ModelAnalysisCache(root=tmp_path / "cache")
    candidate = ModelCandidate(repo_id="../../evil/model", revision_hint="abc123")

    cache.put(candidate, transformers_analysis("../../evil/model"))

    assert not (tmp_path / "evil").exists()
    assert list((tmp_path / "cache").glob("*.json"))


def test_explicit_inspection_refresh_preserves_default_cache_and_updates_missing_digest(
    tmp_path: Path,
) -> None:
    cache = ModelAnalysisCache(root=tmp_path)
    candidate = ModelCandidate(repo_id="org/model-GGUF")
    old = gguf_analysis(candidate.repo_id)
    cache.put(candidate, old)
    fresh = old.model_copy(deep=True)
    variant = fresh.classification.gguf_variants[0]
    variant.files[0] = variant.files[0].model_copy(update={"sha256": "a" * 64})
    calls: list[str] = []

    def inspect(repo_id: str, **kwargs: object) -> ModelAnalysis:
        calls.append(repo_id)
        return fresh

    advisor = AdvisorService.build(
        hf_client=FakeHfClient(),
        detect_hardware=hardware,
        inspect_model=inspect,
        estimate_memory=lambda **kwargs: None,
        model_analysis_cache=cache,
    )
    assert advisor.inspect_model(candidate.repo_id) == old
    assert calls == []
    assert advisor.inspect_model(candidate.repo_id, refresh=True) == fresh
    assert calls == [candidate.repo_id]
    assert advisor.inspect_model(candidate.repo_id) == fresh
    assert calls == [candidate.repo_id]
