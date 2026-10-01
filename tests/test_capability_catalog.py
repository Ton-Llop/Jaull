"""Synthetic fixtures; shipped-catalog tests separately pin sourced publisher claims."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from jaull.application.recommendation.service import recommend
from jaull.domain.candidates import ModelCandidate
from jaull.domain.capability_evidence import (
    CapabilityCatalog,
    CapabilityEvaluation,
    CapabilitySubject,
    EvaluationProtocol,
    comparison_blockers,
)
from jaull.domain.model import ModelAnalysis
from jaull.domain.recommendation import ExternalEvaluationEvidence
from jaull.domain.requirements import RecommendationPriority
from jaull.recommendation.capability import CapabilitySignal, MetadataCapabilityAnalyzer
from jaull.recommendation.capability_catalog import (
    attach_capability_evidence,
    load_capability_catalog,
)
from tests._workflow_fixtures import hardware
from tests.test_recommendation_ranking import _evaluated, _req


def _evaluation(**changes: object) -> CapabilityEvaluation:
    data: dict[str, object] = {
        "evaluation_id": "synthetic-knowledge",
        "repo_id": "synthetic/Example-32B-Instruct",
        "variant": "instruct",
        "revision": "1" * 40,
        "precision": "float16",
        "task": "knowledge",
        "benchmark": "SyntheticBenchmark",
        "benchmark_version": "test-v1",
        "metric": "accuracy",
        "unit": "percent",
        "direction": "higher_is_better",
        "value": 75.0,
        "source": "https://publisher.invalid/synthetic-evaluation",
        "source_kind": "publisher_reported",
        "date": "2026-10-01",
        "protocol": {
            "evaluation_method": "synthetic-harness-v1",
            "few_shot": 5,
            "reasoning_mode": "disabled",
            "max_reasoning_tokens": 0,
            "max_generation_tokens": 128,
            "tools": [],
            "temperature": 0.0,
        },
    }
    return CapabilityEvaluation.model_validate(data | changes)


def _write_catalog(tmp_path: Path, *entries: CapabilityEvaluation) -> Path:
    path = tmp_path / "catalog.json"
    catalog = CapabilityCatalog(catalog_version="synthetic-v1", evaluations=entries)
    path.write_text(catalog.model_dump_json(), encoding="utf-8")
    return path


def test_catalog_roundtrip_keeps_provenance_and_explicit_unknowns(tmp_path: Path) -> None:
    entry = _evaluation(protocol={}, benchmark_version=None)
    independent = _evaluation(evaluation_id="independent", source_kind="independent")
    path = _write_catalog(tmp_path, entry, independent)
    result = load_capability_catalog(path)

    assert result.status == "loaded"
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.catalog is not None
    assert result.catalog.evaluations == (entry, independent)
    assert entry.model_dump(mode="json")["protocol"] == dict.fromkeys(
        EvaluationProtocol.model_fields
    )
    assert entry.benchmark_version is None
    assert entry.source_kind == "publisher_reported"
    assert independent.source_kind == "independent"
    with pytest.raises(ValidationError, match="frozen"):
        entry.value = 99.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"value": True},
        {"value": "75"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 101.0},
        {"value": -1.0},
        {"unit": "fraction", "value": 1.1},
        {"source_kind": "official"},
        {"protocol": {"few_shot": True}},
        {"protocol": {"max_generation_tokens": 0}},
        {"protocol": {"max_reasoning_tokens": -1}},
        {"protocol": {"temperature": float("nan")}},
        {"repo_id": "not-a-repository"},
        {"revision": "main"},
        {"date": "not-a-date"},
        {"source": "not-a-url"},
        {"invented_quality_bonus": 0.1},
    ],
)
def test_invalid_evaluations_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _evaluation(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_id": " "},
        {"variant": "\t"},
        {"precision": "\n"},
        {"benchmark": " "},
        {"benchmark_version": "\u00a0"},
        {"metric": " "},
        {"protocol": {"evaluation_method": " "}},
        {"protocol": {"reasoning_mode": "\t"}},
        {"protocol": {"tools": [""]}},
        {"protocol": {"tools": [" \t"]}},
        {"protocol": {"tools": ["search", " "]}},
    ],
)
def test_blank_catalog_labels_are_invalid_not_known_conditions(
    tmp_path: Path, changes: dict[str, object],
) -> None:
    entry = _evaluation()
    data = entry.model_dump(mode="json") | changes
    protocol_changes = changes.get("protocol")
    if isinstance(protocol_changes, dict):
        data["protocol"] = entry.protocol.model_dump(mode="json") | protocol_changes
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "catalog_version": "synthetic-v1",
        "evaluations": [data],
    }), encoding="utf-8")
    result = load_capability_catalog(path)

    assert result.status == "invalid"
    assert result.catalog is None
    assert next(iter(changes)) in (result.diagnostic or "")
    baseline = CapabilitySignal(score=0.73, reasons=("Unchanged scale prior.",))
    enriched = attach_capability_evidence(baseline, entry.subject, result)
    assert enriched.evaluation_evidence == ()
    assert enriched.evidence_diagnostics == (result.diagnostic,)
    assert enriched.score == baseline.score
    assert enriched.reasons == baseline.reasons


@pytest.mark.parametrize("field", ["variant", "precision"])
def test_blank_lookup_subject_labels_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        CapabilitySubject.model_validate(_evaluation().subject.model_dump() | {field: " "})


def test_blank_catalog_version_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text('{"catalog_version":" "}', encoding="utf-8")
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert "catalog_version" in (result.diagnostic or "")


def test_nonblank_labels_stay_exact_and_null_conditions_stay_unknown(tmp_path: Path) -> None:
    entry = _evaluation(
        variant=" instruct ", precision=" float16 ", benchmark_version=None,
        protocol={"evaluation_method": " named harness ", "tools": [" python "]},
    )
    result = load_capability_catalog(_write_catalog(tmp_path, entry))
    assert result.status == "loaded"
    assert result.catalog is not None
    assert result.catalog.evaluations == (entry,)
    assert entry.subject.variant == " instruct "
    assert entry.subject.precision == " float16 "
    assert entry.protocol.evaluation_method == " named harness "
    assert entry.protocol.tools == (" python ",)
    assert entry.protocol.reasoning_mode is None
    assert "benchmark_version is unknown" in comparison_blockers(entry, entry)
    assert "protocol reasoning_mode is unknown" in comparison_blockers(entry, entry)
    signal = CapabilitySignal(score=0.5)
    assert attach_capability_evidence(signal, entry.subject, result).evaluation_evidence == (entry,)
    assert not attach_capability_evidence(signal, _evaluation().subject, result).evaluation_evidence


def test_duplicate_ids_and_unknown_schema_are_rejected(tmp_path: Path) -> None:
    entry = _evaluation()
    with pytest.raises(ValidationError, match="unique"):
        CapabilityCatalog(catalog_version="v1", evaluations=(entry, entry))
    path = tmp_path / "catalog.json"
    path.write_text('{"schema_version":2,"catalog_version":"future"}', encoding="utf-8")
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert "schema_version" in (result.diagnostic or "")


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_schema_version_requires_an_integer(tmp_path: Path, version: object) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema_version": version, "catalog_version": "synthetic-v1"}))
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert "schema_version" in (result.diagnostic or "")


@pytest.mark.parametrize("duplicate", [
    '"catalog_version":"ambiguous",',
    '"repo_id":"synthetic/Other-32B",',
    '"repo\\u005fid":"synthetic/Other-32B",',
    '"value":99.0,',
    '"temperature":1.0,',
])
def test_duplicate_json_keys_cannot_supply_trusted_evidence(
    tmp_path: Path, duplicate: str,
) -> None:
    entry = _evaluation()
    path = _write_catalog(tmp_path, entry)
    raw = path.read_text(encoding="utf-8")
    key = duplicate.split(":", 1)[0].replace("\\u005f", "_")
    raw = raw.replace(key + ":", duplicate + key + ":", 1)
    path.write_text(raw, encoding="utf-8")

    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "duplicate" in (result.diagnostic or "").lower()
    signal = CapabilitySignal(score=0.73, reasons=("Unchanged scale prior.",))
    enriched = attach_capability_evidence(signal, entry.subject, result)
    assert not enriched.evaluation_evidence
    assert enriched.evidence_diagnostics == (result.diagnostic,)
    assert enriched.score == signal.score
    assert enriched.reasons == signal.reasons


def test_invalid_utf8_catalog_is_a_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_bytes(b'{"catalog_version":"\xff"}')
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert result.diagnostic


def test_excessively_nested_json_catalog_is_a_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert result.diagnostic


def test_invalid_catalog_diagnostic_does_not_echo_url_credentials(tmp_path: Path) -> None:
    entry = _evaluation().model_dump(mode="json")
    entry["source"] = "https://private-user:private-password@example.invalid/evaluation"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"catalog_version": "v1", "evaluations": [entry]}))
    result = load_capability_catalog(path)
    assert result.status == "invalid"
    assert result.catalog is None
    assert "source" in (result.diagnostic or "")
    assert "private-user" not in (result.diagnostic or "")
    assert "private-password" not in (result.diagnostic or "")


@pytest.mark.parametrize("content", [None, "{broken", '{"catalog_version":"v1","bad":true}'])
def test_missing_or_invalid_catalog_keeps_legacy_signal(
    tmp_path: Path, content: str | None,
) -> None:
    path = tmp_path / "catalog.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    result = load_capability_catalog(path)
    signal = CapabilitySignal(score=0.73, reasons=("Existing scale prior.",))
    enriched = attach_capability_evidence(signal, _evaluation().subject, result)
    assert result.status == ("missing" if content is None else "invalid")
    assert enriched.score == signal.score
    assert enriched.confidence == signal.confidence
    assert enriched.reasons == signal.reasons
    assert not enriched.evaluation_evidence
    assert enriched.evidence_diagnostics == (result.diagnostic,)


def test_unreadable_catalog_is_a_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(path: Path) -> bytes:
        raise PermissionError("synthetic denial")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    result = load_capability_catalog(tmp_path / "catalog.json")
    assert result.status == "invalid"
    assert "PermissionError" in (result.diagnostic or "")


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_id": "finetuner/Example-32B-Instruct"},
        {"repo_id": "synthetic/Example-32B-Instruct-GGUF"},
        {"repo_id": "synthetic/Example-32B-Instruct-AWQ"},
        {"repo_id": "synthetic/Example-32B-A3B-Instruct"},
        {"variant": "base"},
        {"variant": None},
        {"revision": "2" * 40},
        {"revision": None},
        {"precision": "int4"},
        {"precision": None},
    ],
)
def test_matching_is_exact_without_family_or_variant_inheritance(
    tmp_path: Path, changes: dict[str, object],
) -> None:
    entry = _evaluation()
    result = load_capability_catalog(_write_catalog(tmp_path, entry))
    signal = CapabilitySignal(score=0.5, family="example")
    assert attach_capability_evidence(signal, entry.subject, result).evaluation_evidence == (entry,)
    other = CapabilitySubject.model_validate(entry.subject.model_dump() | changes)
    unmatched = attach_capability_evidence(signal, other, result)
    assert unmatched.evaluation_evidence == ()
    assert "No exact" in unmatched.evidence_diagnostics[0]
    assert unmatched.score == signal.score


def test_unknown_revision_is_explicit_not_a_wildcard(tmp_path: Path) -> None:
    entry = _evaluation(revision=None)
    result = load_capability_catalog(_write_catalog(tmp_path, entry))
    signal = attach_capability_evidence(CapabilitySignal(score=0.5), entry.subject, result)
    assert signal.evaluation_evidence == (entry,)
    assert "not revision-confirmed" in signal.evidence_diagnostics[0]
    assert "evaluated model revision is unknown" in comparison_blockers(entry, entry)
    pinned = _evaluation().subject
    assert not attach_capability_evidence(signal, pinned, result).evaluation_evidence


def test_comparison_checks_conditions_not_scores_or_source_authority() -> None:
    first = _evaluation()
    second = _evaluation(
        evaluation_id="second",
        repo_id="synthetic/Other-32B-Instruct", revision="2" * 40,
        value=90.0,
        source_kind="independent",
    )
    assert comparison_blockers(first, second) == ()
    assert first.source_kind != second.source_kind
    assert first.value != second.value
    assert "additional protocol notes differ" in comparison_blockers(
        first, _evaluation(notes=["synthetic alternative prompt template"])
    )


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"benchmark_version": None}, "benchmark_version is unknown"),
        ({"benchmark_version": "v2"}, "benchmark_version differs"),
        ({"metric": "pass@1"}, "metric differs"),
        ({"task": "coding"}, "task differs"),
        ({"direction": "lower_is_better"}, "direction differs"),
        ({"protocol": {}}, "protocol evaluation_method is unknown"),
    ],
)
def test_incompatible_or_unknown_protocol_is_not_comparable(
    changes: dict[str, object], expected: str,
) -> None:
    assert expected in comparison_blockers(_evaluation(), _evaluation(**changes))


@pytest.mark.parametrize(
    "field,value",
    [("max_reasoning_tokens", 100), ("max_generation_tokens", 256), ("tools", ["search"]),
     ("reasoning_mode", "enabled"), ("few_shot", 0), ("temperature", 1.0)],
)
def test_reasoning_generation_and_tool_budgets_are_not_interchangeable(
    field: str, value: object,
) -> None:
    first = _evaluation()
    second = _evaluation(protocol=first.protocol.model_dump() | {field: value})
    assert f"protocol {field} differs" in comparison_blockers(first, second)


def test_existing_signal_construction_needs_no_evidence() -> None:
    signal = CapabilitySignal(score=0.5)
    assert signal.evaluation_evidence == ()
    assert signal.evidence_catalog_version is None
    assert signal.evidence_catalog_sha256 is None
    assert signal.evidence_diagnostics == ()


def test_catalog_reuses_external_evidence_without_breaking_legacy_records() -> None:
    legacy = ExternalEvaluationEvidence(
        benchmark="SyntheticLegacy", task="coding", value="unknown", metric="pass@1",
        source="synthetic_legacy_source", revision="legacy-tag",
    )
    assert ExternalEvaluationEvidence.model_validate_json(legacy.model_dump_json()) == legacy
    entry = _evaluation()
    assert isinstance(entry, ExternalEvaluationEvidence)
    assert entry.verified is None
    assert entry.subject.revision == entry.revision
    assert entry.model_dump(mode="json")["value"] == 75.0
    # Strict catalog validation must not change the historical record parser.
    blank_legacy = ExternalEvaluationEvidence(
        benchmark=" ", task="coding", value="unknown", metric=" ", source=" ",
    )
    assert ExternalEvaluationEvidence.model_validate_json(
        blank_legacy.model_dump_json()
    ) == blank_legacy


@pytest.mark.parametrize("priority", list(RecommendationPriority))
@pytest.mark.parametrize("with_hardware", [False, True])
@pytest.mark.parametrize("catalog_state", ["loaded", "missing", "invalid"])
def test_evidence_never_changes_scores_breakdowns_or_order(
    tmp_path: Path, priority: RecommendationPriority, with_hardware: bool, catalog_state: str,
) -> None:
    entry = _evaluation()
    path = _write_catalog(tmp_path, entry)
    if catalog_state == "missing":
        path.unlink()
    elif catalog_state == "invalid":
        path.write_text("{broken", encoding="utf-8")
    catalog = load_capability_catalog(path)
    baseline = MetadataCapabilityAnalyzer()

    class DiagnosticAnalyzer:
        def analyze(
            self, candidate: ModelCandidate, analysis: ModelAnalysis | None,
        ) -> CapabilitySignal:
            signal = baseline.analyze(candidate, analysis)
            enriched = attach_capability_evidence(
                signal, entry.subject.model_copy(update={"repo_id": candidate.repo_id}), catalog,
            )
            for field, value in asdict(signal).items():
                if not field.startswith("evidence_") and field != "evaluation_evidence":
                    assert getattr(enriched, field) == value
            if catalog_state == "loaded" and candidate.repo_id == entry.subject.repo_id:
                assert enriched.evaluation_evidence == (entry,)
            return enriched

    candidates = [_evaluated(entry.subject.repo_id), _evaluated("synthetic/Different-3B-Instruct")]
    req = _req(priority=priority)
    hw = hardware() if with_hardware else None
    original = recommend(candidates, req, hardware=hw)  # type: ignore[arg-type]
    diagnostic = recommend(
        candidates, req, hardware=hw, capability_analyzer=DiagnosticAnalyzer(),  # type: ignore[arg-type]
    )
    assert original
    assert [item.model_dump() for item in diagnostic] == [item.model_dump() for item in original]


def _shipped_catalog() -> CapabilityCatalog:
    path = Path(__file__).parents[1] / "src/jaull/recommendation/capability_catalog.json"
    result = load_capability_catalog(path)
    assert result.status == "loaded"
    assert result.catalog is not None
    return result.catalog


def test_shipped_catalog_keeps_publisher_claims_and_protocol_gaps_explicit() -> None:
    catalog = _shipped_catalog()
    expected = {
        ("Qwen/Qwen2.5-32B-Instruct", "MMLU-Redux", None): 83.9,
        ("Qwen/Qwen2.5-32B-Instruct", "GPQA-Diamond", None): 49.5,
        ("Qwen/Qwen2.5-32B-Instruct", "MATH-500", None): 84.6,
        ("Qwen/Qwen2.5-32B-Instruct", "LiveCodeBench", "v5 (2024.10-2025.02)"): 26.4,
        ("Qwen/Qwen2.5-32B-Instruct", "IFEval", None): 79.5,
        ("Qwen/Qwen3-32B", "MMLU-Redux", None): 85.7,
        ("Qwen/Qwen3-32B", "GPQA-Diamond", None): 54.6,
        ("Qwen/Qwen3-32B", "MATH-500", None): 88.6,
        ("Qwen/Qwen3-32B", "LiveCodeBench", "v5 (2024.10-2025.02)"): 31.3,
        ("Qwen/Qwen3-32B", "IFEval", None): 83.2,
        ("Qwen/Qwen2.5-32B-Instruct", "LiveCodeBench", "2305-2409"): 51.2,
    }
    assert catalog.catalog_version == "0.2.0"
    assert len(catalog.evaluations) == len(expected)
    assert {
        (entry.repo_id, entry.benchmark, entry.benchmark_version): entry.value
        for entry in catalog.evaluations
    } == expected
    for entry in catalog.evaluations:
        assert entry.source_kind == "publisher_reported"
        assert entry.verified is None
        assert entry.revision is None
        assert entry.precision is None
        assert entry.date == "2026-10-01"
        assert entry.protocol.evaluation_method is None
        assert entry.protocol.few_shot is None
        assert entry.protocol.max_reasoning_tokens is None
        assert entry.protocol.tools is None
        assert entry.unit == "percent"
        assert entry.direction == "higher_is_better"
        assert "assessment date is source review" in " ".join(entry.notes)
        if entry.repo_id == "Qwen/Qwen3-32B":
            assert entry.variant == "post-trained/non-thinking"
            assert entry.protocol.reasoning_mode == "non-thinking"
            assert entry.protocol.max_generation_tokens == 32768
            assert entry.protocol.temperature == 0.7
            assert entry.source == "https://arxiv.org/html/2505.09388v1#S4.T14"
        else:
            assert entry.variant == "instruct"
            assert entry.protocol.reasoning_mode is None
            assert entry.protocol.max_generation_tokens is None
            assert entry.protocol.temperature is None


def test_shipped_model_profiles_are_not_protocol_confirmed() -> None:
    entries = _shipped_catalog().evaluations
    for task in ("knowledge", "reasoning", "maths", "coding", "instruction_following"):
        pair = [
            entry for entry in entries
            if entry.task == task and "qwen3-report-v1" in entry.evaluation_id
        ]
        assert len(pair) == 2
        blockers = comparison_blockers(*pair)
        assert "evaluated model revision is unknown" in blockers
        assert "subject precision is unknown" in blockers
        assert "protocol evaluation_method is unknown" in blockers


def test_different_real_coding_windows_are_retained_not_merged() -> None:
    coding = [
        entry for entry in _shipped_catalog().evaluations
        if entry.task == "coding" and entry.repo_id == "Qwen/Qwen2.5-32B-Instruct"
    ]
    assert len(coding) == 2
    assert {entry.value for entry in coding} == {26.4, 51.2}
    assert "benchmark_version differs" in comparison_blockers(*coding)
    assert coding[0].source != coding[1].source


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_id": "Qwen/Qwen3-32B-FP8"},
        {"repo_id": "converter/Qwen3-32B-GGUF"},
        {"repo_id": "finetuner/Qwen3-32B"},
        {"repo_id": "Qwen/Qwen3-30B-A3B"},
        {"variant": "post-trained/thinking"},
        {"revision": "1" * 40},
        {"precision": "bfloat16"},
    ],
)
def test_real_unpinned_evidence_does_not_transfer_to_other_representations(
    changes: dict[str, object],
) -> None:
    path = Path(__file__).parents[1] / "src/jaull/recommendation/capability_catalog.json"
    result = load_capability_catalog(path)
    subject = CapabilitySubject(repo_id="Qwen/Qwen3-32B", variant="post-trained/non-thinking")
    baseline = CapabilitySignal(score=0.71, reasons=("Unchanged scale prior.",))
    matched = attach_capability_evidence(baseline, subject, result)
    assert len(matched.evaluation_evidence) == 5
    assert matched.score == baseline.score
    assert matched.confidence == baseline.confidence
    assert matched.reasons == baseline.reasons
    assert "not revision-confirmed" in matched.evidence_diagnostics[0]
    unsupported = CapabilitySubject.model_validate(subject.model_dump() | changes)
    assert not attach_capability_evidence(baseline, unsupported, result).evaluation_evidence
