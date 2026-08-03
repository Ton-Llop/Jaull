from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jaull.domain.enrichment import GgufHeaderMetadata
from jaull.domain.estimation import EstimationConfidence, MetadataSource
from jaull.metadata import base_model_resolver


@dataclass
class _FakeModelInfo:
    card_data: dict[str, Any] = field(default_factory=dict)


def test_string_base_model_resolves_high_confidence() -> None:
    info = _FakeModelInfo(card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"})
    resolution = base_model_resolver.resolve_base_model(
        model_info=info, gguf_header=None, repo_id="user/anything-GGUF"
    )
    assert resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert resolution.source is MetadataSource.MODEL_CARD_METADATA
    assert resolution.confidence is EstimationConfidence.HIGH


def test_list_of_length_one_resolves() -> None:
    info = _FakeModelInfo(card_data={"base_model": ["meta-llama/Meta-Llama-3.1-8B-Instruct"]})
    resolution = base_model_resolver.resolve_base_model(
        model_info=info, gguf_header=None, repo_id="user/x-GGUF"
    )
    assert resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_ambiguous_list_returns_unresolved_with_candidates() -> None:
    info = _FakeModelInfo(
        card_data={"base_model": ["meta-llama/A", "meta-llama/B"]}
    )
    resolution = base_model_resolver.resolve_base_model(
        model_info=info, gguf_header=None, repo_id="user/x-GGUF"
    )
    assert resolution.repo_id is None
    assert resolution.source is MetadataSource.UNRESOLVED
    assert set(resolution.candidates) == {"meta-llama/A", "meta-llama/B"}
    assert resolution.warnings


def test_dict_with_relations_picks_single_reference() -> None:
    info = _FakeModelInfo(
        card_data={"base_model": {"finetune": "meta-llama/Meta-Llama-3.1-8B-Instruct"}}
    )
    resolution = base_model_resolver.resolve_base_model(
        model_info=info, gguf_header=None, repo_id="user/x-GGUF"
    )
    assert resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_gguf_source_repository_wins_when_no_model_card() -> None:
    header = GgufHeaderMetadata(
        source_repository="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    resolution = base_model_resolver.resolve_base_model(
        model_info=None, gguf_header=header, repo_id="user/x-GGUF"
    )
    assert resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert resolution.source is MetadataSource.GGUF_METADATA
    assert resolution.confidence is EstimationConfidence.HIGH


def test_no_information_returns_unresolved_and_never_guesses() -> None:
    # Naming hint (-GGUF) MUST NOT be treated as a resolution.
    resolution = base_model_resolver.resolve_base_model(
        model_info=None,
        gguf_header=None,
        repo_id="user/Some-Model-GGUF",
    )
    assert resolution.repo_id is None
    assert resolution.source is MetadataSource.UNRESOLVED
    # The hint is surfaced as evidence but never as the answer.
    assert any("hint" in e.lower() for e in resolution.evidence)


def test_url_reference_medium_confidence() -> None:
    info = _FakeModelInfo(
        card_data={"source": "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"}
    )
    resolution = base_model_resolver.resolve_base_model(
        model_info=info, gguf_header=None, repo_id="user/x-GGUF"
    )
    assert resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert resolution.source is MetadataSource.EXPLICIT_REPOSITORY_REFERENCE
    assert resolution.confidence is EstimationConfidence.MEDIUM
