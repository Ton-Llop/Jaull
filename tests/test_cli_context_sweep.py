"""``jaull estimate --context-sweep``: the memory a model needs at each context."""

from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from jaull.advisor.service import AdvisorService
from jaull.cli.app import app
from jaull.domain.estimation import CompatibilityStatus
from jaull.domain.model import ModelConfig, SafetensorsSummary
from jaull.estimator import policies
from jaull.estimator import service as estimator_service
from tests._workflow_fixtures import hardware
from tests.test_estimation_json_and_cli import _stub_analysis, _StubClient

_RESIDENT = {
    CompatibilityStatus.COMFORTABLE.value,
    CompatibilityStatus.COMPATIBLE.value,
    CompatibilityStatus.TIGHT.value,
}
_NATIVE_CONTEXT = 65536


def _install_advisor(monkeypatch: Any, *, vram_gib: int = 12) -> None:
    client = _StubClient(
        summary=SafetensorsSummary(
            total_parameters=500_000_000, parameters_by_dtype={"F16": 500_000_000}
        ),
        config=ModelConfig(
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            max_position_embeddings=_NATIVE_CONTEXT,
        ),
    )
    analysis = _stub_analysis(client)
    advisor = AdvisorService.build(
        hf_client=client,  # type: ignore[arg-type]
        detect_hardware=lambda: hardware(vram_gib=vram_gib),
        inspect_model=lambda repo_id, client=None: analysis,  # type: ignore[misc]
        estimate_memory=estimator_service.estimate_memory,
    )
    monkeypatch.setattr(AdvisorService, "default", classmethod(lambda cls: advisor))


def _sweep_json(monkeypatch: Any, *, vram_gib: int = 12) -> dict[str, Any]:
    _install_advisor(monkeypatch, vram_gib=vram_gib)
    result = CliRunner().invoke(
        app,
        ["estimate", "user/model", "--context-sweep", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.stdout)
    return payload


def test_json_has_its_own_schema_and_names_the_model(monkeypatch: Any) -> None:
    payload = _sweep_json(monkeypatch)
    assert payload["schema_version"] == 2
    assert payload["provenance"]["hardware"]
    assert payload["provenance"]["configuration"]["kv_cache_dtype"] == "float16"
    assert payload["provenance"]["estimate_snapshot"]
    assert payload["model"]["repo_id"] == "user/model"
    assert "context_sweep" in payload
    # It is a separate document: the estimate JSON is not touched.
    assert "memory" not in payload


def test_the_sweep_covers_the_ladder_up_to_the_native_context(monkeypatch: Any) -> None:
    sweep = _sweep_json(monkeypatch)["context_sweep"]
    contexts = [point["context_length"] for point in sweep["points"]]
    expected = [c for c in policies.CONTEXT_LADDER if c <= _NATIVE_CONTEXT]
    assert contexts == expected
    assert sweep["native_limit"] == _NATIVE_CONTEXT


def test_the_kv_cache_grows_linearly_with_the_context(monkeypatch: Any) -> None:
    points = _sweep_json(monkeypatch)["context_sweep"]["points"]
    first, last = points[0], points[-1]
    ratio = last["context_length"] // first["context_length"]
    assert last["kv_cache_bytes"] == first["kv_cache_bytes"] * ratio
    totals = [point["total_bytes"] for point in points]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_a_model_that_fits_everywhere_reports_the_native_context(monkeypatch: Any) -> None:
    sweep = _sweep_json(monkeypatch, vram_gib=12)["context_sweep"]
    assert sweep["largest_fitting"] == _NATIVE_CONTEXT
    assert all(point["status"] in _RESIDENT for point in sweep["points"])


def test_a_small_gpu_stops_the_sweep_before_the_native_context(monkeypatch: Any) -> None:
    sweep = _sweep_json(monkeypatch, vram_gib=3)["context_sweep"]
    largest = sweep["largest_fitting"]
    assert largest is not None
    assert largest < _NATIVE_CONTEXT
    last = sweep["points"][-1]
    assert last["status"] not in _RESIDENT


def test_the_report_names_the_largest_context_that_fits(monkeypatch: Any) -> None:
    _install_advisor(monkeypatch)
    result = CliRunner().invoke(
        app, ["estimate", "user/model", "--context-sweep"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "Largest context that fits" in result.stdout
    assert str(_NATIVE_CONTEXT) in result.stdout


def test_confidence_is_said_once_not_repeated_on_every_row(monkeypatch: Any) -> None:
    _install_advisor(monkeypatch)
    result = CliRunner().invoke(
        app, ["estimate", "user/model", "--context-sweep"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("onfidence") == 1
    assert "Confidence of the estimate" in result.stdout


def test_the_json_keeps_the_confidence_of_each_point(monkeypatch: Any) -> None:
    points = _sweep_json(monkeypatch)["context_sweep"]["points"]
    assert all("confidence" in point for point in points)


def test_a_fixed_context_makes_no_sense_with_a_sweep(monkeypatch: Any) -> None:
    _install_advisor(monkeypatch)
    result = CliRunner().invoke(
        app,
        ["estimate", "user/model", "--context-sweep", "--context", "4096"],
    )
    assert result.exit_code == 2
    assert "--context" in result.output
