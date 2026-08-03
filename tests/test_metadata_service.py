from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import EstimationConfidence, MetadataSource
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.exceptions import ModelAccessDeniedError
from jaull.metadata import service
from jaull.metadata.range_reader import RangeResponse
from tests._gguf_fixtures import build_header


@dataclass
class _StubModelInfo:
    card_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubClient:
    card_data: dict[str, Any] = field(default_factory=dict)
    base_config_json: dict[str, Any] | None = None
    raises_on_base_download: Exception | None = None

    def model_info(self, repo_id: str) -> _StubModelInfo:
        return _StubModelInfo(card_data=self.card_data)

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        if self.raises_on_base_download:
            raise self.raises_on_base_download
        assert self.base_config_json is not None
        tmp = Path("/tmp") / f"{repo_id.replace('/', '_')}_{filename}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.base_config_json), encoding="utf-8")
        return tmp

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


@dataclass
class _StubRangeClient:
    body: bytes

    def fetch_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> RangeResponse:
        return RangeResponse(
            body=self.body[start : end + 1], honored_range=True, status_code=206
        )


def _gguf_analysis() -> ModelAnalysis:
    variant = GgufVariant(
        quantization="Q4_K_M",
        files=[ModelFile(path="model.Q4_K_M.gguf", size_bytes=4_500_000_000)],
        total_bytes=4_500_000_000,
    )
    return ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
        files=variant.files,
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=[variant],
        ),
        config=None,
        relevant_files=[],
        total_size_bytes=variant.total_bytes,
        warnings=[],
    )


def test_enrich_with_resolvable_base_and_gguf_header() -> None:
    analysis = _gguf_analysis()
    variant = analysis.classification.gguf_variants[0]
    client = _StubClient(
        card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        base_config_json={
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 131072,
        },
    )
    header_body = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,  # note: mismatch to trigger conflict
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    range_client = _StubRangeClient(body=header_body)

    result = service.enrich(
        analysis=analysis,
        variant=variant,
        client=client,
        range_client=range_client,
    )
    assert result.base_model_resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert result.gguf_header is not None
    assert result.enriched_config is not None
    assert result.enriched_config.config.num_hidden_layers == 32
    assert result.enriched_config.config.max_position_embeddings == 8192  # GGUF wins
    assert any("GGUF wins" in w for w in result.warnings)


def test_enrich_without_base_model_declared_still_uses_gguf_header() -> None:
    analysis = _gguf_analysis()
    variant = analysis.classification.gguf_variants[0]
    client = _StubClient()  # no card metadata at all
    header_body = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    result = service.enrich(
        analysis=analysis,
        variant=variant,
        client=client,
        range_client=_StubRangeClient(body=header_body),
    )
    assert result.base_model_resolution.source is MetadataSource.UNRESOLVED
    assert result.enriched_config is not None
    assert result.enriched_config.config.num_hidden_layers == 32


def test_enrich_when_base_is_gated_degrades_cleanly() -> None:
    analysis = _gguf_analysis()
    variant = analysis.classification.gguf_variants[0]
    client = _StubClient(
        card_data={"base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
        raises_on_base_download=ModelAccessDeniedError("gated"),
    )
    header_body = build_header(
        {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
    )
    result = service.enrich(
        analysis=analysis,
        variant=variant,
        client=client,
        range_client=_StubRangeClient(body=header_body),
    )
    assert result.base_model_resolution.repo_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert any("gated" in w.lower() for w in result.warnings)
    # Still returns an enriched config from the GGUF header alone.
    assert result.enriched_config is not None
    assert result.enriched_config.config.num_hidden_layers == 32


def test_unresolved_result_helper() -> None:
    result = service.unresolved_result(warnings=["disabled"])
    assert result.enriched_config is None
    assert result.base_model_resolution.source is MetadataSource.UNRESOLVED
    assert result.base_model_resolution.confidence is EstimationConfidence.UNKNOWN
    assert "disabled" in result.warnings
