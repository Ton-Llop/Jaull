from __future__ import annotations

import pytest

from jaull.analyzers.transformers import _model_config_from_dict
from jaull.domain.candidates import ModelCandidate
from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import EstimationConfidence
from jaull.domain.families import resolve_parameter_count
from jaull.domain.model import (
    ModelAnalysis,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.domain.parameters import ParameterCountSource
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


def test_unknown_metadata_confidence_does_not_break_capability_analysis() -> None:
    candidate = _candidate(
        "random-user/random-model-7B",
        confidence=EstimationConfidence.UNKNOWN,
    )
    signal = MetadataCapabilityAnalyzer().analyze(candidate, None)
    assert signal.confidence is EstimationConfidence.UNKNOWN
    assert 0.0 <= signal.score <= 1.0


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
    """When the config carries every dimension we need, use it — otherwise
    fall through to the repo-id suffix, which is exactly what happens with
    the shared fixture (no ``vocab_size`` declared)."""
    analysis = transformers_analysis(repo_id="org/Model-7B")
    # Fixture lacks vocab_size so the config path abstains and the name-
    # suffix supplies the count. The important property: it does not silently
    # return the old 12·L·H² undercount.
    assert parameter_count(_candidate("org/Model-7B"), analysis) == 7_000_000_000


def test_packed_awq_safetensors_count_is_not_logical_parameter_count() -> None:
    config = _model_config_from_dict(
        {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "quantization_config": {
                "bits": 4,
                "quant_method": "awq",
                "group_size": 128,
            },
        }
    )
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ"),
        files=[],
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
        ),
        config=config,
        safetensors_summary=SafetensorsSummary(
            total_parameters=1_960_000_000,
            parameters_by_dtype={"I4": 1_960_000_000},
        ),
    )

    resolved = resolve_parameter_count(
        _candidate("Qwen/Qwen2.5-7B-Instruct-AWQ"),
        analysis,
    )

    assert resolved.source is not ParameterCountSource.SAFETENSORS_METADATA
    assert resolved.count is not None
    assert resolved.count > 7_000_000_000


def test_compressed_tensors_safetensors_count_is_not_logical_parameter_count() -> None:
    config = _model_config_from_dict(
        {
            "model_type": "llama",
            "num_hidden_layers": 50,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32128,
            "quantization_config": {
                "format": "pack-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "group_size": 128,
                        }
                    }
                },
            },
        }
    )
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="speakleash/Bielik-11B-v3.0-Instruct-awq"),
        files=[],
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
        ),
        config=config,
        safetensors_summary=SafetensorsSummary(
            total_parameters=1_960_000_000,
            parameters_by_dtype={"I4": 1_960_000_000},
        ),
    )

    resolved = resolve_parameter_count(
        _candidate("speakleash/Bielik-11B-v3.0-Instruct-awq"),
        analysis,
    )

    assert resolved.source is not ParameterCountSource.SAFETENSORS_METADATA
    assert resolved.count is not None
    assert resolved.count > 10_000_000_000


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
