"""Saved-pool quality refresh with synthetic evidence; no Hub, Docker or GPU."""

import hashlib
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from jaull.advisor.service import AdvisorService
from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import HardwareFitMode, HardwareFitResult, HardwareMemoryTopology
from jaull.domain.hardware import ComputeBackend
from jaull.domain.recommendation import AssessmentLevel
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.evaluation.quality_records import digest
from jaull.evaluation.quality_storage import QualityEvidenceStore
from jaull.recommendation import shadow
from jaull.reporting.recommendation import report_to_dict
from jaull.runtime import quality_eval_runner
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationCancelled,
    QualityEvaluationError,
    QualityProfile,
)
from jaull.workflow.orchestrator import run_workflow
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import answers, gguf_analysis, hardware
from tests.test_cli_quality import _request
from tests.test_execution_plans import _selection
from tests.test_ifeval_applicability import _full_record
from tests.test_workflow_orchestrator import _container, _search_with


def _state(*, use_case=UseCase.GENERAL_CHAT, priority=RecommendationPriority.QUALITY):
    names = ("org/A-7B-Instruct-GGUF", "org/B-7B-Instruct-GGUF")
    analyses = {}
    for name, sha in zip(names, ("a" * 64, "9" * 64), strict=True):
        analysis = gguf_analysis(name, quantizations=("Q5_K_M",))
        variant = analysis.classification.gguf_variants[0]
        variant.files[0] = variant.files[0].model_copy(update={"sha256": sha})
        analyses[name] = analysis
    services = _container(_search_with(*names), analyses=analyses)
    state = run_workflow(answers(use_case, priority, languages=["English"]), hardware(), services)
    assert state.completed and state.ranked_plans and state.shadow is not None
    assert {item.plan.artifact.sha256 for item in state.ranked_plans} == {"a" * 64, "9" * 64}
    return state, services


def _forbidden(*args, **kwargs):
    pytest.fail("Recompare must not discover, inspect, estimate, execute or rebuild plans")


def _advisor(services, store, monkeypatch):
    monkeypatch.setattr(services.search_client, "search", _forbidden)
    services = replace(services, detect_hardware=_forbidden, inspect_model=_forbidden,
                       estimate_memory=_forbidden)
    monkeypatch.setattr("jaull.recommendation.engine_v2.generate_execution_plans", _forbidden)
    for method in ("_plan_ranking_context", "_artifacts", "prepare_quality_evaluation_setup",
                   "run_quality_evaluation", "run_quality_evaluation_for_plan"):
        monkeypatch.setattr(AdvisorService, method, _forbidden)
    return AdvisorService(services=services, quality_store=store)


def test_search_then_new_records_then_recompare_preserves_the_original(tmp_path, monkeypatch):
    state, services = _state()
    before = state.model_dump_json()
    exported = report_to_dict(state)
    exported.pop("timestamp")
    store = QualityEvidenceStore(tmp_path)
    advisor = _advisor(services, store, monkeypatch)
    initial = advisor.recompare_quality(state)
    assert initial == state.shadow
    store.save(_full_record())
    store.save(_full_record(stronger=True))
    updated = advisor.recompare_quality(state)
    assert updated.base_order == initial.base_order
    assert updated.base_top5 == initial.base_top5
    assert updated.shadow_order == initial.base_order[::-1]
    assert updated.moves and updated.groups
    assert updated != initial and advisor.recompare_quality(state) == updated
    assert state.model_dump_json() == before
    after_export = report_to_dict(state)
    after_export.pop("timestamp")  # Export time is not part of the saved search.
    assert after_export == exported
    # New evidence is read, not rewritten or followed by a fresh experiment.
    assert len(store.list_ids()) == 2


@pytest.mark.parametrize("priority", [RecommendationPriority.SPEED,
                                     RecommendationPriority.BALANCED,
                                     RecommendationPriority.MEMORY])
def test_quality_alone_cannot_unlock_other_policies(tmp_path, monkeypatch, priority):
    state, services = _state(priority=priority)
    store = QualityEvidenceStore(tmp_path)
    store.save(_full_record())
    store.save(_full_record(stronger=True))
    updated = _advisor(services, store, monkeypatch).recompare_quality(state)
    assert updated.shadow_order == updated.base_order
    assert not updated.moves


def test_single_success_and_bad_file_do_not_promote_or_lose_candidates(tmp_path, monkeypatch):
    state, services = _state()
    before = state.model_dump_json()
    store = QualityEvidenceStore(tmp_path)
    store.save(_full_record(stronger=True))
    store.path_for("c" * 64).write_text("{}", encoding="utf-8")
    updated = _advisor(services, store, monkeypatch).recompare_quality(state)
    assert updated.shadow_order == updated.base_order
    assert set(updated.base_order) == set(state.shadow.base_order)
    assert not updated.moves
    assert state.model_dump_json() == before


def test_recompare_cannot_transfer_quality_to_another_artifact(tmp_path, monkeypatch):
    state, services = _state()
    store = QualityEvidenceStore(tmp_path)
    store.save(_full_record())
    changed = _full_record(stronger=True)
    changed["identity"]["artifact_sha256"] = "f" * 64
    changed["identity_sha256"] = digest(changed["identity"])
    store.save(changed)
    updated = _advisor(services, store, monkeypatch).recompare_quality(state)
    assert not updated.groups and not updated.moves
    assert updated.shadow_order == state.shadow.base_order


def test_coding_recompare_does_not_apply_ifeval(tmp_path, monkeypatch):
    state, services = _state(use_case=UseCase.CODING)
    store = QualityEvidenceStore(tmp_path)
    store.save(_full_record())
    store.save(_full_record(stronger=True))
    updated = _advisor(services, store, monkeypatch).recompare_quality(state)
    assert not updated.groups and not updated.moves


def test_recompare_keeps_search_time_constraints_and_speed(tmp_path, monkeypatch):
    state, services = _state()
    weaker, stronger = state.ranked_plans
    stronger = replace(stronger, assessment=stronger.assessment.model_copy(update={
        "suitability": AssessmentLevel.WEAK,
    }))
    state = state.model_copy(update={"ranked_plans": (weaker, stronger)})
    before = state.model_dump_json()
    store = QualityEvidenceStore(tmp_path)
    store.save(_full_record())
    store.save(_full_record(stronger=True))
    build = shadow.build_shadow_report

    def check_snapshot(refreshed, requirements, *, limit):
        for old, new in zip(state.ranked_plans, refreshed, strict=True):
            assert new.plan is old.plan and new.evaluated is old.evaluated
            assert new.assessment.model_dump(exclude={"quality"}) == (
                old.assessment.model_dump(exclude={"quality"})
            )
            assert new.assessment.quality.applicable is not None
        return build(refreshed, requirements, limit=limit)

    monkeypatch.setattr(shadow, "build_shadow_report", check_snapshot)
    updated = _advisor(services, store, monkeypatch).recompare_quality(state)
    assert updated.shadow_order == updated.base_order
    assert not updated.groups and not updated.moves
    assert state.model_dump_json() == before


def test_old_state_without_pool_loads_but_requires_a_new_search(tmp_path):
    state, services = _state()
    data = state.model_dump(mode="json")
    data.pop("ranked_plans")
    old = RecommendationWorkflowState.model_validate(data)
    assert old.ranked_plans == ()
    advisor = AdvisorService(services=services, quality_store=QualityEvidenceStore(tmp_path))
    with pytest.raises(ValueError, match="saved plan pool"):
        advisor.recompare_quality(old)
    restored = RecommendationWorkflowState.model_validate_json(state.model_dump_json())
    assert restored.ranked_plans == state.ranked_plans


def test_incomplete_search_cannot_be_recompared(tmp_path: Path):
    advisor = AdvisorService(services=_container(_search_with()),
                             quality_store=QualityEvidenceStore(tmp_path))
    with pytest.raises(ValueError, match="completed search"):
        advisor.recompare_quality(RecommendationWorkflowState())


@pytest.mark.parametrize("cancel_second", [False, True])
def test_search_select_evaluate_persist_recompare_offline(tmp_path, monkeypatch, cancel_second):
    """Real product flow with synthetic bytes/outcomes, not a hardware measurement."""
    names = ("org/A-7B-Instruct-GGUF", "org/B-7B-Instruct-GGUF")
    data = {name: f"synthetic GGUF {name}".encode() for name in names}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()}
    analyses = {}
    for name in names:
        analysis = gguf_analysis(name, quantizations=("Q5_K_M",))
        variant = analysis.classification.gguf_variants[0]
        file = variant.files[0].model_copy(update={
            "sha256": hashes[name], "size_bytes": len(data[name]),
        })
        analyses[name] = analysis.model_copy(update={
            "classification": analysis.classification.model_copy(update={
                "gguf_variants": [variant.model_copy(update={
                    "total_bytes": len(data[name]), "files": [file],
                })],
            }),
        })
    services = _container(_search_with(*names), analyses=analyses)
    estimate = services.estimate_memory

    def full_device_estimate(**kwargs):
        prediction = estimate(**kwargs)
        return prediction.model_copy(update={"hardware_fit": HardwareFitResult(
            mode=HardwareFitMode.GPU_RESIDENT,
            memory_topology=HardwareMemoryTopology.DISCRETE_MEMORY,
            weights_bytes=kwargs["analysis"].classification.gguf_variants[0].total_bytes,
            kv_cache_bytes=0, overhead_bytes=0,
            reason="Synthetic confirmed fit, not measured",
        )})

    services = replace(services, estimate_memory=full_device_estimate)
    calls = []

    class Resolver:
        def resolve(self, repo_id, **kwargs):
            variant = analyses[repo_id].classification.gguf_variants[0]
            return ModelArtifact(
                repo_id=repo_id, revision="f" * 40, filename=variant.files[0].path,
                format="gguf", quantization=variant.quantization,
                size_bytes=len(data[repo_id]), sha256=hashes[repo_id],
            )

    def download(**kwargs):
        calls.append(("download", kwargs["repo_id"]))
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(data[kwargs["repo_id"]])
        return str(path)

    artifacts = ArtifactService(Resolver(), ArtifactStorage(tmp_path / "models"), download)
    store = QualityEvidenceStore(tmp_path / "quality")
    advisor = AdvisorService(services=services, artifacts=artifacts, quality_store=store)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    monkeypatch.setattr(AdvisorService, "select_runtime_backend",
                        lambda self, hw: _selection(ComputeBackend.CUDA))
    monkeypatch.setattr(AdvisorService, "_plan_runtime_readiness", lambda self, **kw: None)
    for method in ("_stored_benchmark_records", "_stored_experiment_records",
                   "_published_evaluations"):
        monkeypatch.setattr(AdvisorService, method, lambda self: [])
    monkeypatch.setattr(quality_eval_runner, "prepare_quality_setup", lambda *a, **kw: None)
    cancelled = Event()

    def evaluate(request, **kwargs):
        assert request.profile is QualityProfile.IFEVAL
        artifact = ModelArtifact.model_validate_json(request.artifact_json.read_text())
        assert artifact.is_verified and artifact.local_path.read_bytes() == data[artifact.repo_id]
        assert artifact.sha256 == hashes[artifact.repo_id]
        calls.append(("evaluate", artifact.repo_id))
        if artifact.repo_id == names[1]:
            assert len(store.list_ids()) == 1  # First result persisted before the next run.
        if cancel_second and artifact.repo_id == names[1]:
            cancelled.set()
            raise QualityEvaluationCancelled("Synthetic cancellation")
        record = _full_record(stronger=artifact.repo_id == names[1])
        record["identity"]["artifact_sha256"] = artifact.sha256
        record["identity_sha256"] = digest(record["identity"])
        return record

    monkeypatch.setattr(quality_eval_runner, "run_quality_evaluation", evaluate)
    state = advisor.recommend(answers(UseCase.GENERAL_CHAT, RecommendationPriority.QUALITY,
                                     languages=["English"]), hardware=hardware())
    assert state.completed and len(state.ranked_plans) == 2
    before = state.model_dump_json()
    exported = report_to_dict(state)
    exported.pop("timestamp")
    selection = advisor.prepare_quality_candidates(state, profile=QualityProfile.IFEVAL)
    assert len(selection.candidates) == 2 and not selection.notices
    assert not calls and not store.list_ids()  # Search and selection never execute.
    assert [c.plan.plan_id for c in selection.candidates] == [
        rec.plan.plan_id for rec in state.recommendations
    ]
    assert [c.plan.artifact.sha256 for c in selection.candidates] == list(hashes.values())
    request = replace(_request(tmp_path / "inputs"), profile=QualityProfile.IFEVAL)
    # Real consent guard: no downloader or producer may run without permission.
    with pytest.raises(QualityEvaluationError, match="download permission"):
        advisor.run_quality_evaluation_for_plan(selection.candidates[0].plan, request)
    assert not calls
    outcomes = advisor.run_quality_candidates(
        selection.candidates, request, allow_download=True, is_cancelled=cancelled.is_set,
    )
    assert len(outcomes) == 2
    assert sum(outcome.path is not None for outcome in outcomes) == (1 if cancel_second else 2)
    assert calls == [(action, name) for name in names for action in ("download", "evaluate")]
    assert len(store.list_ids()) == (1 if cancel_second else 2)
    saved_bytes = {path: path.read_bytes() for path in (o.path for o in outcomes) if path}
    for path in saved_bytes:
        record = advisor.load_quality_record(path.stem)
        assert store.lookup(record["identity"]) == record["result"]
    refreshed = advisor.prepare_quality_candidates(state, profile=QualityProfile.IFEVAL)
    assert [c.settled for c in refreshed.candidates] == [True, not cancel_second]
    assert len(calls) == 4  # Reading stored results does not launch another evaluation.
    # Recompare must read only the saved pool/store, not rediscover or estimate.
    monkeypatch.setattr(services.search_client, "search", _forbidden)
    monkeypatch.setattr(AdvisorService, "inspect_model", _forbidden)
    monkeypatch.setattr(AdvisorService, "estimate_model", _forbidden)
    monkeypatch.setattr(AdvisorService, "_plan_ranking_context", _forbidden)
    report = advisor.recompare_quality(state)
    assert report.shadow_order == (report.base_order if cancel_second else report.base_order[::-1])
    assert bool(report.moves) is not cancel_second
    assert state.model_dump_json() == before
    after_export = report_to_dict(state)
    after_export.pop("timestamp")
    assert after_export == exported
    assert all(path.read_bytes() == content for path, content in saved_bytes.items())
    # A stored result for different bytes cannot settle the missing candidate.
    other_quantization = _full_record(stronger=True)
    other_quantization["identity"]["artifact_sha256"] = "e" * 64
    other_quantization["identity_sha256"] = digest(other_quantization["identity"])
    store.save(other_quantization)
    assert advisor.recompare_quality(state) == report


@pytest.mark.parametrize("kind", ["conflict", "off-contract"])
def test_recompare_diagnostics_keep_record_rejection_reasons(tmp_path, monkeypatch, kind):
    state, services = _state()
    store = QualityEvidenceStore(tmp_path)
    first = _full_record()
    if kind == "conflict":
        store.save(first)
        sample = first["result"]["samples"][first["identity"]["suite"]["name"]][0]
        sample["resps"] = [["Synthetic different reply."]]
        sample["filtered_resps"] = ["Synthetic different reply."]
        expected = "different results; no rule selects one"
    else:
        first["identity"]["runtime"]["server_sha256"] = "0" * 64
        first["identity_sha256"] = digest(first["identity"])
        expected = "matching the audited pins"
    store.save(first)
    report = _advisor(services, store, monkeypatch).recompare_quality(state)
    fallback = next(f for f in report.fallback if f.artifact_sha256 == "a" * 64)
    assert expected in " ".join(fallback.reasons)
    assert "Quality references for this SHA256" in " ".join(fallback.reasons)
    assert not report.groups and not report.moves
