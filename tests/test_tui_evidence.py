"""How much evidence stands behind an execution plan.

These cover the resolver that replaced the substring scan of
``ExecutionPlan.evidence``. That scan looked for ``"validated"`` and
``"benchmarked"``, which ``_plan_evidence`` never emits, so both states were
permanently unreachable and the two filters were permanently empty.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TabbedContent

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.tui.app import JaullApp
from jaull.tui.evidence import EvidenceIndex, EvidenceState
from jaull.tui.screens import estimate as estimate_module
from jaull.tui.screens.estimate import EstimateScreen
from jaull.tui.screens.execution_paths import (
    ExecutionPathsScreen,
    _ExecutionPathsLoaded,
    _PathOption,
)
from jaull.tui.screens.recommendation_results import _RecommendationRow
from jaull.tui.widgets.technical_details import TechnicalDetails
from tests._workflow_fixtures import hardware
from tests.test_benchmarks import _record as _benchmark_record
from tests.test_execution_plans import _gguf_recommendation
from tests.test_experiment_store import _sample_record
from tests.test_quality_eval_records import synthetic_full_record, synthetic_limited_record
from tests.test_tui_recommendation_execution import _wait_until


class _FakeStoreAdvisor:
    """Only the record-store methods the resolver actually calls."""

    def __init__(
        self,
        experiments: dict[str, Any] | None = None,
        benchmarks: dict[str, Any] | None = None,
        *,
        quality: dict[str, Any] | None = None,
        broken: bool = False,
    ) -> None:
        self._experiments = experiments or {}
        self._benchmarks = benchmarks or {}
        self._broken = broken
        self._quality = quality or {}

    def list_experiment_ids(self) -> list[str]:
        if self._broken:
            raise OSError("data directory unreadable")
        return list(self._experiments)

    def load_experiment_record(self, experiment_id: str) -> Any:
        return self._experiments[experiment_id]

    def list_benchmark_ids(self) -> list[str]:
        if self._broken:
            raise OSError("data directory unreadable")
        return list(self._benchmarks)

    def load_benchmark_record(self, benchmark_id: str) -> Any:
        return self._benchmarks[benchmark_id]

    def list_quality_ids(self) -> list[str]:
        if self._broken:
            raise OSError("quality directory unreadable")
        return list(self._quality)

    def load_quality_record(self, identity_sha256: str) -> Any:
        return self._quality[identity_sha256]


def _plan(
    *,
    repo_id: str = "owner/repo",
    quantization: str | None = "Q4_K_M",
    runtime: RuntimeName = RuntimeName.LLAMA_CPP,
    ready: bool = True,
    sha256: str | None = "1" * 64,
    file_count: int = 1,
    revision: str = "main",
    backend: ComputeBackend = ComputeBackend.CPU,
    context_length: int = 4096,
) -> ExecutionPlan:
    identity = ModelIdentity(model_name=repo_id.split("/")[-1])
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id=repo_id,
        revision=revision,
        format=ArtifactVariantFormat.GGUF,
        filename="model.gguf",
        sha256=sha256,
        file_count=file_count,
        quantization=quantization,
        compatible_runtimes=[runtime],
    )
    # A real readiness object rather than a hand-built one: it carries a
    # selection and a runtime capability, and the resolver only reads `.status`.
    sample = _sample_record()
    recommendation = (
        sample.runtime if runtime is RuntimeName.LLAMA_CPP else RuntimeRecommendation(
            runtime=runtime,
            confidence=EstimationConfidence.HIGH,
        )
    )
    readiness = sample.preflight.execution_readiness if ready else None
    prediction = sample.prediction
    if context_length != 4096:
        prediction = prediction.model_copy(update={
            "inference_configuration": prediction.inference_configuration.model_copy(
                update={"context_length": context_length}
            ),
        })
    selection = sample.preflight.execution_readiness.selection.model_copy(
        update={"selected_backend": backend}
    )
    return ExecutionPlan(
        plan_id=f"{repo_id}:{quantization}",
        model_identity=identity,
        artifact=variant,
        runtime=recommendation,
        runtime_family=runtime,
        execution_readiness=readiness,
        backend_selection=selection,
        memory_prediction=prediction,
        hardware=sample.hardware,
    )


def _experiment(
    *,
    repo_id: str = "owner/repo",
    quantization: str = "Q4_K_M",
    success: bool = True,
) -> Any:
    record = _sample_record()
    artifact = record.artifact.model_copy(
        update={"repo_id": repo_id, "quantization": quantization}
    )
    observation = record.observation.model_copy(update={"success": success})
    return record.model_copy(update={"artifact": artifact, "observation": observation})


def test_plan_with_no_records_is_ready_not_validated() -> None:
    index = EvidenceIndex.load(_FakeStoreAdvisor(), ModelIdentity(model_name="repo"))

    evidence = index.for_plan(_plan())

    assert evidence.state is EvidenceState.READY
    assert evidence.validated is False
    assert evidence.benchmarked is False


def test_plan_without_readiness_is_only_estimated() -> None:
    index = EvidenceIndex.empty()

    evidence = index.for_plan(_plan(ready=False))

    assert evidence.state is EvidenceState.ESTIMATED
    assert evidence.summary() == "Estimated only"


def test_a_stored_experiment_makes_the_plan_validated() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))
    evidence = index.for_plan(_plan())

    assert evidence.state is EvidenceState.VALIDATED
    assert evidence.experiment_ids == ("exp-1",)
    assert "Validated" in evidence.summary()


def test_a_failed_experiment_does_not_count_as_validated() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(success=False)})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).state is EvidenceState.READY


def test_evidence_does_not_cross_quantizations() -> None:
    """A Q4_K_M run says nothing about Q5_K_M: different artifact, different run."""
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(quantization="Q4_K_M")})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(quantization="Q4_K_M")).validated is True
    assert index.for_plan(_plan(quantization="Q5_K_M")).validated is False


def test_evidence_does_not_cross_runtimes() -> None:
    """A llama.cpp GGUF run is not evidence for a Transformers plan."""
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(runtime=RuntimeName.LLAMA_CPP)).validated is True
    assert index.for_plan(_plan(runtime=RuntimeName.TRANSFORMERS)).validated is False


def test_evidence_does_not_cross_repositories() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment(repo_id="owner/repo")})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(repo_id="other/repo")).validated is False


def test_evidence_does_not_cross_artifact_hashes_or_mutable_unpinned_revisions() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})
    index = EvidenceIndex.load(advisor)

    assert not index.for_plan(_plan(sha256="2" * 64)).validated
    assert not index.for_plan(_plan(sha256=None)).validated


def test_pinned_revision_and_filename_can_identify_artifact_without_sha() -> None:
    revision = "a" * 40
    experiment = _experiment()
    experiment = experiment.model_copy(update={"artifact": experiment.artifact.model_copy(
        update={"revision": revision, "sha256": None}
    )})
    index = EvidenceIndex.load(_FakeStoreAdvisor(experiments={"exp-1": experiment}))

    assert index.for_plan(_plan(sha256=None, revision=revision)).validated
    assert not index.for_plan(_plan(sha256=None, revision="b" * 40)).validated


def test_evidence_does_not_cross_backend_context_or_machine() -> None:
    advisor = _FakeStoreAdvisor(experiments={"exp-1": _experiment()})
    index = EvidenceIndex.load(advisor)

    assert not index.for_plan(_plan(backend=ComputeBackend.VULKAN)).validated
    assert not index.for_plan(_plan(context_length=8192)).validated
    different_machine = _plan().model_copy(update={"hardware": hardware()})
    assert not index.for_plan(different_machine).validated
    changed_flags = _plan().runtime.model_copy(update={"flags": []})
    assert not index.for_plan(_plan().model_copy(update={"runtime": changed_flags})).validated


def test_benchmark_evidence_is_specific_to_artifact_backend_context_and_machine(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark_record(tmp_path)
    request = benchmark.request.model_copy(update={"context_length": 4096})
    benchmark = benchmark.model_copy(update={"request": request})
    index = EvidenceIndex.load(_FakeStoreAdvisor(benchmarks={
        benchmark.identity.benchmark_id: benchmark,
    }))

    assert index.for_plan(_plan()).benchmarked
    assert not index.for_plan(_plan(sha256="2" * 64)).benchmarked
    assert not index.for_plan(_plan(backend=ComputeBackend.VULKAN)).benchmarked
    assert not index.for_plan(_plan(context_length=8192)).benchmarked
    different_machine = _plan().model_copy(update={"hardware": hardware()})
    assert not index.for_plan(different_machine).benchmarked


def test_loading_without_an_identity_indexes_every_model() -> None:
    """The results screen scans the stores once for several models at a time."""
    advisor = _FakeStoreAdvisor(
        experiments={
            "exp-1": _experiment(repo_id="owner/repo"),
            "exp-2": _experiment(repo_id="other/second"),
        }
    )

    index = EvidenceIndex.load(advisor)

    assert index.for_plan(_plan(repo_id="owner/repo")).validated is True
    assert index.for_plan(_plan(repo_id="other/second")).validated is True


def test_an_unreadable_store_degrades_to_no_evidence() -> None:
    """Evidence enriches a screen; a plan must not vanish because a log failed."""
    index = EvidenceIndex.load(_FakeStoreAdvisor(broken=True), ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).state is EvidenceState.READY


def test_a_corrupt_record_is_skipped_rather_than_fatal() -> None:
    class _Exploding(_FakeStoreAdvisor):
        def load_experiment_record(self, experiment_id: str) -> Any:
            raise ValueError("truncated json")

    advisor = _Exploding(experiments={"exp-1": None})

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan()).validated is False


def test_artifact_without_quantization_falls_back_to_the_filename() -> None:
    record = _sample_record()
    artifact: ModelArtifact = record.artifact.model_copy(
        update={"quantization": None, "filename": "model.gguf"}
    )
    advisor = _FakeStoreAdvisor(
        experiments={"exp-1": record.model_copy(update={"artifact": artifact})}
    )

    index = EvidenceIndex.load(advisor, ModelIdentity(model_name="repo"))

    assert index.for_plan(_plan(quantization=None)).validated is True


def test_sample_record_artifact_path_is_untouched() -> None:
    """Guards the fixture assumption the tests above are built on."""
    assert _sample_record().artifact.local_path == Path("/models/model.gguf")


def test_quality_is_associated_by_exact_bytes_without_becoming_execution_evidence() -> None:
    record = synthetic_limited_record()
    advisor = _FakeStoreAdvisor(quality={record["identity_sha256"]: record})
    index = EvidenceIndex.load(advisor)
    plan = _plan(sha256=record["identity"]["artifact_sha256"], ready=False)
    before = deepcopy(plan)
    found = index.for_plan(plan)
    assert found.state is EvidenceState.ESTIMATED
    assert not found.validated and not found.benchmarked
    assert len(found.quality_records) == 1
    assert found.quality_records[0].reusable is False
    text = found.quality_summary()
    assert "limited" in text and "100/10042" in text and "ctx 2048" in text
    assert "current execution/protocol not verified" in text
    assert record["identity_sha256"] in found.quality_details()
    assert plan == before
    # Repository labels do not replace exact content identity.
    mirror = _plan(repo_id="mirror/same-bytes", sha256=plan.artifact.sha256)
    assert index.for_plan(mirror).quality_records


def test_quality_never_inherits_to_unknown_different_or_multipart_artifacts() -> None:
    record = synthetic_full_record()
    index = EvidenceIndex.load(_FakeStoreAdvisor(quality={record["identity_sha256"]: record}))
    sha256 = record["identity"]["artifact_sha256"]
    for plan in (
        _plan(), _plan(sha256="7" * 64), _plan(sha256=sha256, file_count=2),
        _plan(sha256=sha256, runtime=RuntimeName.TRANSFORMERS),
    ):
        assert not index.for_plan(plan).quality_records
        assert index.for_plan(plan).quality_summary() == ""


def test_corrupt_quality_does_not_hide_valid_execution_or_quality_records() -> None:
    good = synthetic_full_record()
    experiment = _experiment()
    experiment = experiment.model_copy(update={"artifact": experiment.artifact.model_copy(
        update={"sha256": good["identity"]["artifact_sha256"]}
    )})
    index = EvidenceIndex.load(_FakeStoreAdvisor(
        experiments={"exp-1": experiment},
        quality={"bad": {}, good["identity_sha256"]: good},
    ))
    found = index.for_plan(_plan(sha256=good["identity"]["artifact_sha256"]))
    assert found.validated
    assert len(found.quality_records) == 1


def test_unreadable_quality_store_does_not_hide_execution_evidence() -> None:
    class Unreadable(_FakeStoreAdvisor):
        def list_quality_ids(self) -> list[str]:
            raise OSError("quality directory unreadable")

    index = EvidenceIndex.load(Unreadable(experiments={"exp-1": _experiment()}))
    found = index.for_plan(_plan())
    assert found.validated
    assert not found.quality_records


def test_paths_clears_historical_evaluation_when_recheck_removes_all_plans() -> None:
    record = synthetic_limited_record()
    plan = _plan(sha256=record["identity"]["artifact_sha256"])

    class PathsAdvisor(_FakeStoreAdvisor):
        services = None

        def execution_plans_for_recommendation(self, recommendation: Any) -> list[ExecutionPlan]:
            return [plan]

    async def scenario() -> None:
        advisor = PathsAdvisor(quality={record["identity_sha256"]: record})
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = ExecutionPathsScreen(_gguf_recommendation())
            app.push_screen(screen)
            deadline = asyncio.get_running_loop().time() + 10
            while asyncio.get_running_loop().time() < deadline:
                await pilot.pause(0)
                if screen.query("#paths-quality-provenance") and screen.query_one(
                    "#paths-quality-provenance"
                ).display:
                    break
                await asyncio.sleep(0.01)
            assert screen._plans
            assert screen.query_one("#paths-quality-provenance").display
            await screen._paths_loaded(_ExecutionPathsLoaded([], EvidenceIndex.empty()))
            assert not screen.query_one("#paths-quality-provenance").display
            assert not screen.query_one("#paths-quality").display
            assert not str(screen.query_one("#paths-quality-details", Static).content)
            for name in ("run", "validate", "benchmark"):
                assert screen.query_one(f"#paths-{name}", Button).disabled

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (110, 32)])
def test_quality_display_is_selected_only_and_clears_when_the_artifact_changes(
    size: tuple[int, int],
) -> None:
    record = synthetic_limited_record()
    index = EvidenceIndex.load(_FakeStoreAdvisor(quality={record["identity_sha256"]: record}))
    plan = _plan(sha256=record["identity"]["artifact_sha256"])
    rec = _gguf_recommendation().model_copy(update={"plan": plan})

    class View(App[None]):
        CSS_PATH = str(Path(__file__).parents[1] / "src/jaull/tui/styles.tcss")

        def compose(self) -> ComposeResult:
            row = _RecommendationRow(0, rec, selected=True)
            yield row
            yield row.detail
            yield _PathOption(plan, index.for_plan(plan), selected=True, id="option")

    async def scenario() -> None:
        app = View()
        async with app.run_test(size=size) as pilot:
            row = app.query_one(_RecommendationRow)
            detail = row.detail
            action_states = [button.disabled for button in detail.query(Button)]
            row.show_evidence(index.for_plan(plan))
            detail.query_one(TabbedContent).active = "tab-3"
            await pilot.pause(0)
            quality = detail.query_one("#rec-quality-0", Static)
            assert quality.display
            assert "limited" in str(quality.content)
            assert "current execution/protocol not verified" in str(quality.content)
            assert record["identity_sha256"] in str(quality.tooltip)
            assert quality.region.right <= size[0]
            actions = detail.query_one(".rec-actions")
            # Actions stay beside the model header, above the scrolling tab.
            assert actions.region.bottom <= quality.parent.content_region.y
            assert [button.disabled for button in detail.query(Button)] == action_states
            row.set_selected(False)
            assert not detail.display
            row.set_selected(True)
            assert detail.display and quality.display

            other = _plan(sha256="7" * 64, quantization="Q5_K_M")
            option = app.query_one(_PathOption)
            option.rebind(other, index.for_plan(other))
            assert not option.query_one(".-quality", Static).display
            row.show_evidence(index.for_plan(other))
            assert not quality.display and not str(quality.content)
            assert [button.disabled for button in detail.query(Button)] == action_states

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (110, 32)])
def test_manual_estimate_displays_only_quality_for_the_selected_exact_gguf(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = synthetic_limited_record()
    rec = _gguf_recommendation()
    analysis = rec.evaluated.analysis
    assert analysis is not None
    variant = next(
        item for item in analysis.classification.gguf_variants if item.quantization == "Q5_K_M"
    )
    variant.files[0] = variant.files[0].model_copy(update={
        "sha256": record["identity"]["artifact_sha256"],
    })
    requests: list[InferenceConfiguration] = []

    class ManualAdvisor(_FakeStoreAdvisor):
        def inspect_model(self, repo_id: str, *, refresh: bool = False) -> Any:
            assert repo_id == rec.repo_id
            assert refresh is True
            return analysis

        def scan_hardware(self) -> Any:
            return hardware()

        def estimate_model(
            self, *, analysis: Any, hardware: Any, inference_cfg: InferenceConfiguration,
        ) -> MemoryEstimate:
            requests.append(inference_cfg)
            estimate = rec.evaluated.memory_estimate
            assert estimate is not None
            return estimate.model_copy(update={"inference_configuration": inference_cfg})

    advisor = ManualAdvisor(quality={record["identity_sha256"]: record})
    monkeypatch.setattr(estimate_module, "_advisor", lambda screen: advisor)

    class ManualView(App[None]):
        CSS_PATH = JaullApp.CSS_PATH

        def on_mount(self) -> None:
            self.push_screen(EstimateScreen())

    async def scenario() -> None:
        app = ManualView()
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0)
            screen = app.screen
            assert isinstance(screen, EstimateScreen)

            async def wait_for(selector: str) -> None:
                await _wait_until(pilot, lambda: bool(screen.query(selector)))

            screen.query_one("#est-input", Input).value = rec.repo_id
            screen._start_detect()
            await wait_for("#est-variant")
            picker = screen.query_one("#est-variant", Select)
            picker.value = "Q5_K_M"
            await _wait_until(pilot, lambda: screen.state.quantization == "Q5_K_M")
            screen._start_estimate()
            screen._start_estimate()
            await wait_for("#est-quality")
            quality = screen.query_one("#est-quality", Vertical)
            text = "\n".join(str(widget.content) for widget in quality.query(Static))
            assert "Measured evaluation" in text
            assert "Limited evaluation" in text
            assert "100 of 10042" in text
            assert "2048 tokens" in text
            assert "current execution and protocol have not been verified" in text
            assert "not a general capability assessment" in text
            assert "Accuracy" in text and "Length-normalized accuracy" in text
            metrics = record["result"]["results"][record["identity"]["suite"]["name"]]
            for name in ("acc,none", "acc_norm,none"):
                assert f"{metrics[name]:.1%}" in text
            assert quality.region.right <= size[0]
            details = quality.query_one(TechnicalDetails)
            assert details.collapsed
            details_text = "\n".join(str(widget.content) for widget in details.query(Static))
            assert record["identity_sha256"] in details_text
            assert "Placement: unknown" in details_text
            assert record["identity"]["suite"]["name"] in details_text
            body = screen.query_one("#est-body", VerticalScroll)
            assert body.scroll_y == 0
            details.collapsed = False
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            assert not details.collapsed
            result_text = "\n".join(
                str(widget.content) for widget in screen.query_one("#est-result").query(Static)
            )
            assert "Estimation context" in result_text
            assert f"{requests[0].context_length} tokens" in result_text
            assert not screen.query_one("#est-parameters").display
            assert screen.query_one("#est-output").display
            body.scroll_to_widget(screen.query_one("#est-edit"), animate=False, immediate=True)
            await pilot.pause()
            await pilot.click("#est-edit")
            await _wait_until(pilot, lambda: screen.query_one("#est-parameters").display)
            assert screen.query_one("#est-parameters").display
            assert not screen.query_one("#est-output").display
            assert picker.value == "Q5_K_M"
            assert screen.query_one("#est-input", Input).value == rec.repo_id
            screen._start_estimate()
            await _wait_until(
                pilot,
                lambda: len(requests) == 2 and screen.query_one("#est-output").display,
            )
            assert screen.query_one("#est-quality", Vertical).display
            assert not screen.query_one("#est-parameters").display
            screen.query_one("#est-edit", Button).press()
            await _wait_until(pilot, lambda: screen.query_one("#est-parameters").display)
            picker.value = "Q4_K_M"
            await _wait_until(pilot, lambda: screen.state.quantization == "Q4_K_M")
            screen._start_estimate()
            await _wait_until(
                pilot,
                lambda: len(requests) == 3
                and screen.query_one("#est-output").display
                and not screen.query("#est-quality"),
            )
            assert not screen.query("#est-quality")
            assert [cfg.quantization for cfg in requests] == ["Q5_K_M", "Q5_K_M", "Q4_K_M"]

    asyncio.run(scenario())
