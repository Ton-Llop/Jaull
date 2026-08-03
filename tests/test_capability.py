from __future__ import annotations

import pytest

from jaull.discovery.models import ModelCandidate
from jaull.domain.estimation import EstimationConfidence
from jaull.recommendation.capability import (
    DEFAULT_CAPABILITY_SCORE,
    MetadataCapabilityAnalyzer,
    capability_score,
    detect_family,
    parameter_count,
)
from tests._workflow_fixtures import transformers_analysis


def _candidate(
    repo_id: str,
    tags: list[str] | None = None,
    downloads: int = 0,
    likes: int = 0,
    confidence: EstimationConfidence = EstimationConfidence.MEDIUM,
) -> ModelCandidate:
    return ModelCandidate(
        repo_id=repo_id,
        tags=tags or [],
        downloads=downloads,
        likes=likes,
        metadata_confidence=confidence,
        pipeline_tag="text-generation",
    )


def test_larger_beats_smaller_with_similar_metadata() -> None:
    analyzer = MetadataCapabilityAnalyzer()
    small = analyzer.analyze(_candidate("Qwen/Qwen2.5-0.5B-Instruct"), None)
    big = analyzer.analyze(_candidate("Qwen/Qwen2.5-7B-Instruct"), None)
    assert big.score > small.score
    assert big.parameter_count == 7_000_000_000


def test_unknown_family_uses_metadata_without_table_penalty() -> None:
    candidate = _candidate(
        "random-user/random-model-7B",
        tags=["safetensors"],
        downloads=10_000,
        likes=100,
        confidence=EstimationConfidence.HIGH,
    )
    signal = MetadataCapabilityAnalyzer().analyze(candidate, None)
    assert signal.family == "random-model"
    assert signal.score >= DEFAULT_CAPABILITY_SCORE


def test_thin_metadata_without_size_stays_moderate() -> None:
    candidate = _candidate(
        "random-user/random-model",
        confidence=EstimationConfidence.LOW,
    )
    signal = MetadataCapabilityAnalyzer().analyze(candidate, None)
    assert signal.family == "unknown"
    assert 0.35 <= signal.score <= 0.55


def test_popularity_is_secondary_to_parameter_count() -> None:
    analyzer = MetadataCapabilityAnalyzer()
    tiny_popular = analyzer.analyze(
        _candidate("org/Tiny-0.5B", downloads=1_000_000, likes=20_000),
        None,
    )
    large_quiet = analyzer.analyze(_candidate("org/Large-7B"), None)
    assert large_quiet.score > tiny_popular.score


def test_parameter_count_reads_repo_id_suffix() -> None:
    assert parameter_count(_candidate("org/Model-7B"), None) == 7_000_000_000
    assert parameter_count(_candidate("org/Model-0.5B"), None) == 500_000_000
    assert parameter_count(_candidate("org/Model-unknown"), None) is None


def test_parameter_count_prefers_inspected_config() -> None:
    analysis = transformers_analysis(repo_id="org/Model-7B")
    assert parameter_count(_candidate("org/Model-7B"), analysis) == (
        12 * 32 * 4096 * 4096
    )


@pytest.mark.parametrize(
    ("repo_id", "expected"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama-3.1"),
        ("random-user/random-model", "unknown"),
    ],
)
def test_detect_family_derives_stable_series_keys(repo_id: str, expected: str) -> None:
    assert detect_family(_candidate(repo_id)) == expected


def test_compatibility_wrapper_returns_analyzer_score() -> None:
    candidate = _candidate("org/Model-7B")
    assert capability_score(candidate, None) == pytest.approx(
        MetadataCapabilityAnalyzer().analyze(candidate, None).score
    )
