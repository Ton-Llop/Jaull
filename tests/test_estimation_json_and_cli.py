from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from jaull.advisor.service import AdvisorService
from jaull.cli.app import app
from jaull.domain.enums import Format, RepositoryType
from jaull.domain.inference import TargetDevice
from jaull.domain.model import (
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.estimator import service as estimator_service
from jaull.hardware.detector import detect_hardware
from jaull.presentation.estimation_report import (
    SCHEMA_VERSION,
    estimate_to_json_dict,
)


@dataclass
class _StubClient:
    summary: SafetensorsSummary | None = None
    files: list[ModelFile] = field(default_factory=list)
    config: ModelConfig | None = None
    primary: RepositoryType = RepositoryType.TRANSFORMERS

    def model_info(self, repo_id: str):
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return self.summary


def _stub_analysis(client: _StubClient) -> ModelAnalysis:
    return ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id="user/model",
            author="user",
            pipeline_tag="text-generation",
            library_name="transformers",
            license="apache-2.0",
        ),
        files=client.files
        or [
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(path="model.safetensors", size_bytes=1_000_000_000),
        ],
        classification=RepositoryClassification(
            primary_type=client.primary,
            detected_types={client.primary},
            formats={Format.SAFETENSORS},
            gguf_variants=[],
        ),
        config=client.config,
        relevant_files=[],
        total_size_bytes=1_000_000_000,
        warnings=[],
    )


def _install_fake_advisor(
    monkeypatch, *, analysis: ModelAnalysis, client: _StubClient
) -> None:
    """Replace ``AdvisorService.default`` with a hand-wired advisor.

    The fake reuses the real estimator + hardware detection (the tests want to
    exercise the real memory maths) but shortcircuits ``inspect_model`` to
    return the canned analysis and hands the stub HfClient to the estimator
    so its ``safetensors_summary`` probe returns pre-canned data.
    """
    advisor = AdvisorService.build(
        hf_client=client,  # type: ignore[arg-type]
        detect_hardware=detect_hardware,
        inspect_model=lambda repo_id, client=None: analysis,  # type: ignore[misc]
        estimate_memory=estimator_service.estimate_memory,
    )
    monkeypatch.setattr(AdvisorService, "default", classmethod(lambda cls: advisor))


def _run_and_estimate(*, monkeypatch) -> dict[str, Any]:
    client = _StubClient(
        summary=SafetensorsSummary(
            total_parameters=500_000_000,
            parameters_by_dtype={"F16": 500_000_000},
        ),
        config=ModelConfig(
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            max_position_embeddings=2048,
        ),
    )
    analysis = _stub_analysis(client)
    _install_fake_advisor(monkeypatch, analysis=analysis, client=client)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["estimate", "user/model", "--context", "1024", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    return payload


def test_json_output_has_stable_schema(monkeypatch) -> None:
    payload = _run_and_estimate(monkeypatch=monkeypatch)
    assert payload["schema_version"] == SCHEMA_VERSION
    for key in ("model", "inference_configuration", "memory", "hardware", "assessment"):
        assert key in payload
    assert payload["model"]["repo_id"] == "user/model"
    assert payload["memory"]["weights_bytes"] == 500_000_000 * 2
    assert payload["kv_cache"]["layers"] == 8
    decomposition = payload["weights"]["transformer_block_decomposition"]
    assert decomposition["method"] == "uniform_weight_fallback"
    assert decomposition["total_weight_bytes"] == 500_000_000 * 2
    assert decomposition["estimated_transformer_block_weight_bytes"] == (
        500_000_000 * 2
    )
    assert decomposition["estimated_non_block_weight_bytes"] == 0
    assert payload["assessment"]["status"] in {
        "comfortable",
        "compatible",
        "tight",
        "offloading_required",
        "insufficient",
        "unknown",
    }
    # Fase 4 additive fields must be present and JSON-serialisable.
    assert "runtime_recommendation" in payload
    assert "architecture" in payload
    if payload["runtime_recommendation"] is not None:
        assert payload["runtime_recommendation"]["runtime"] in {
            "llama.cpp",
            "transformers",
            "vllm",
            "unknown",
        }


def test_estimate_to_json_dict_is_json_serialisable() -> None:
    from jaull.domain.estimation import (
        CompatibilityAssessment,
        CompatibilityStatus,
        EstimateSource,
        EstimationConfidence,
        KvCacheEstimate,
        MemoryComponent,
        MemoryEstimate,
        RuntimeOverheadEstimate,
        WeightEstimate,
    )
    from jaull.domain.inference import InferenceConfiguration

    weights_component = MemoryComponent(
        name="Weights",
        bytes=1_000_000,
        source=EstimateSource.EXACT,
        confidence=EstimationConfidence.HIGH,
        explanation="test",
    )
    estimate = MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="foo/bar"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=1024, target_device=TargetDevice.CPU
        ),
        weights=WeightEstimate(component=weights_component),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=None,
                source=EstimateSource.UNKNOWN,
                confidence=EstimationConfidence.UNKNOWN,
                explanation="none",
            ),
            layers=None,
            kv_heads=None,
            head_dim=None,
            context_length=1024,
            batch_size=1,
            dtype_bytes=2,
            formula="x",
            notes=[],
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=512,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=256,
            weight_fraction=0.1,
            minimum_bytes=256,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=0,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="none",
        ),
        safety_margin=None,
        total_bytes=1_000_512,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMFORTABLE,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.CPU,
            effective_device=TargetDevice.CPU,
            available_ram_bytes=100_000_000,
            ratio=0.01,
        ),
    )

    payload = estimate_to_json_dict(estimate)
    dumped = json.dumps(payload)
    assert json.loads(dumped) == payload
    assert payload["memory"]["weights_bytes"] == 1_000_000


def test_cli_estimate_rich_output(monkeypatch) -> None:
    """Rich rendering path does not crash and prints the expected sections."""

    client = _StubClient(
        summary=SafetensorsSummary(
            total_parameters=100_000_000,
            parameters_by_dtype={"F16": 100_000_000},
        ),
        config=ModelConfig(
            num_hidden_layers=4,
            num_attention_heads=4,
            hidden_size=256,
            max_position_embeddings=2048,
        ),
    )
    analysis = _stub_analysis(client)
    _install_fake_advisor(monkeypatch, analysis=analysis, client=client)

    runner = CliRunner()
    result = runner.invoke(app, ["estimate", "user/model"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Model estimation" in result.stdout
    assert "Memory breakdown" in result.stdout
    assert "Assessment" in result.stdout


def test_service_end_to_end_shape_is_used(monkeypatch) -> None:
    """Sanity check the pass-through: monkeypatched service is actually invoked."""

    called = {}

    def fake_estimate(**kwargs: Any) -> None:
        called["hit"] = True
        raise RuntimeError("boom")

    monkeypatch.setattr(estimator_service, "estimate_memory", fake_estimate)

    client = _StubClient()
    analysis = _stub_analysis(client)
    # AdvisorService.build captures ``estimator_service.estimate_memory`` at
    # call time, so patching the module attribute above and then constructing
    # the fake advisor here picks up the ``fake_estimate`` sentinel.
    _install_fake_advisor(monkeypatch, analysis=analysis, client=client)

    runner = CliRunner()
    result = runner.invoke(
        app, ["estimate", "user/model"], catch_exceptions=True
    )
    assert called.get("hit") is True
    # Runtime error surfaces (not one of our JaullError types → not caught).
    assert result.exit_code != 0
