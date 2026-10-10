"""IFEval and Recompare in the candidate screen. Synthetic advisor; no Docker/GPU/network."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TabbedContent

from jaull.domain.requirements import RecommendationPriority, UseCase, UserRequirements
from jaull.recommendation.shadow import (
    Measured,
    PlanEvidence,
    ShadowFallback,
    ShadowMove,
    ShadowReport,
    ShadowTop5Change,
    build_shadow_report,
)
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationCancelled,
    QualityProfile,
    default_dataset,
)
from jaull.tui.app import JaullApp
from jaull.tui.screens.quality_evaluation import (
    QualityEvaluationScreen,
    _QualityCandidatesReady,
    recompare_readout,
)
from jaull.tui.widgets.technical_details import TechnicalDetails
from jaull.workflow.state import RecommendationWorkflowState
from tests.test_shadow_policies import _pool
from tests.test_tui_quality import CandidateAdvisor, _settled, _wait_until


class ProfileAdvisor(CandidateAdvisor):
    def __init__(self) -> None:
        super().__init__()
        self.profiles: list[QualityProfile] = []
        self.recompared: list[RecommendationWorkflowState] = []

    def prepare_quality_candidates(self, state: Any, **kwargs: Any) -> Any:
        self.profiles.append(kwargs["profile"])
        return super().prepare_quality_candidates(state, **kwargs)

    def recompare_quality(self, state: RecommendationWorkflowState) -> ShadowReport:
        self.recompared.append(state)
        return ShadowReport(
            priority=RecommendationPriority.QUALITY, applied=True,
            base_order=("plan-a", "plan-b"), shadow_order=("plan-b", "plan-a"),
            moves=(ShadowMove(
                plan_id="plan-b", repo_id="org/B", base_index=1, shadow_index=0,
                group="run0", rule="jaull-ifeval-chat-v1 prompt_level_strict_acc 1",
            ),),
            top5_changes=(ShadowTop5Change(plan_id="plan-b", change="entered", moved=True),),
        )


def _state(use_case: UseCase = UseCase.GENERAL_CHAT) -> RecommendationWorkflowState:
    return RecommendationWorkflowState(requirements=UserRequirements(
        use_case=use_case, priority=RecommendationPriority.QUALITY, languages=["en"],
        desired_context=4096, pipeline_tag="text-generation",
    ))


def test_english_chat_defaults_to_ifeval_and_selects_for_it() -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            assert advisor.profiles == [QualityProfile.IFEVAL]
            assert screen.query_one("#quality-profile", Select).value == "ifeval"
            assert screen.query_one("#quality-dataset", Input).value == str(
                default_dataset(QualityProfile.IFEVAL),
            )
            protocol = str(screen.query_one("#quality-protocol", Static).render())
            assert "IFEval" in protocol and "chat template" in protocol
            assert not advisor.calls  # Opening the screen never evaluates.
    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_candidates_do_not_require_nested_controls_to_mount_immediately(size, monkeypatch) -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            container = screen.query_one("#quality-candidates", Vertical)
            mount = container.mount
            pending = []

            async def delayed_mount(*widgets, **kwargs):
                # Awaiting a parent mount need not make its descendants queryable yet.
                for widget in widgets:
                    if (widget.id or "").startswith("quality-row-"):
                        pending.append(widget)
                    else:
                        await mount(widget, **kwargs)

            monkeypatch.setattr(container, "mount", delayed_mount)
            selection = advisor.prepare_quality_candidates(_state(), profile=QualityProfile.IFEVAL)
            await screen._candidates_ready(_QualityCandidatesReady(selection))
            assert len(pending) == len(selection.candidates)
            assert not screen.query("#quality-candidates Checkbox")
            monkeypatch.setattr(container, "mount", mount)
            await mount(*pending)
            await pilot.pause()
            for index, candidate in enumerate(selection.candidates):
                checkbox = screen.query_one(f"#quality-candidate-{index}", Checkbox)
                assert checkbox.tooltip == candidate.plan.model_identity.model_name
                assert checkbox.value
            assert screen._checked() == selection.candidates
            assert not screen.query_one("#quality-start", Button).disabled
            assert not advisor.calls
    asyncio.run(scenario())


def test_ifeval_is_not_offered_where_it_could_never_apply() -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state(UseCase.CODING))
            await app.push_screen(screen)
            await _settled(pilot, screen)
            options = [value for _, value in screen.query_one("#quality-profile", Select)._options]
            assert "ifeval" not in options
            assert advisor.profiles == [QualityProfile.SMOKE]
    asyncio.run(scenario())


def test_changing_profile_reselects_and_keeps_a_typed_dataset(tmp_path: Path) -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            screen.query_one("#quality-profile", Select).value = "smoke"
            await _settled(pilot, screen)
            assert advisor.profiles == [QualityProfile.IFEVAL, QualityProfile.SMOKE]
            assert "HellaSwag" in str(screen.query_one("#quality-protocol", Static).render())
            # A path the user typed survives a profile switch.
            typed = str(tmp_path / "my-dataset.jsonl")
            screen.query_one("#quality-dataset", Input).value = typed
            screen.query_one("#quality-profile", Select).value = "ifeval"
            await _settled(pilot, screen)
            assert screen.query_one("#quality-dataset", Input).value == typed
            assert advisor.profiles[-1] is QualityProfile.IFEVAL
    asyncio.run(scenario())


def test_recompare_shows_a_proposal_and_leaves_the_search_untouched() -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        state = _state()
        before = state.model_dump_json()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=state)
            await app.push_screen(screen)
            await _settled(pilot, screen)
            screen.query_one("#quality-tabs", TabbedContent).active = "quality-results-tab"
            button = screen.query_one("#quality-recompare", Button)
            button.scroll_visible(animate=False)
            await pilot.pause()
            assert button.region.right <= 80
            await pilot.click("#quality-recompare")
            await _settled(pilot, screen)
            text = str(screen.query_one("#quality-result", Static).render())
            assert "Proposal only; the search result stays as it was" in text
            assert "#2 -> #1  org/B" in text
            assert advisor.recompared == [state] and not advisor.calls
            assert state.model_dump_json() == before
            for name in ("start", "prepare", "cancel", "back"):
                assert screen.query_one(f"#quality-{name}", Button).region.right <= 80
    asyncio.run(scenario())


@pytest.mark.parametrize("applied,moves,expected", [
    (False, (), "keeps the active order"),
    (True, (), "No moves"),
])
def test_recompare_readout_says_why_nothing_moved(applied, moves, expected) -> None:
    report = ShadowReport(priority=RecommendationPriority.MEMORY, applied=applied,
                          base_order=("p",), shadow_order=("p",), moves=moves)
    text = recompare_readout(report, {})
    assert expected in text and "Proposal only" in text


def test_balanced_without_speed_says_so_instead_of_blaming_quality() -> None:
    # Two plans with applicable quality: Balanced still forms no group without speed.
    requirements, items = _pool("A", "B")
    requirements = requirements.model_copy(update={"priority": RecommendationPriority.BALANCED})
    report = build_shadow_report(items, requirements, limit=5, evidence={
        item.plan.plan_id: PlanEvidence(quality=Measured("same-suite", 0.5 + index / 10, "m"))
        for index, item in enumerate(items)
    })
    assert report.applied and not report.groups
    text = recompare_readout(report, {})
    assert "no speed measurement applies yet" in text and "Quality priority" in text
    assert "no two plans" not in text


@pytest.mark.parametrize("first_value", [0.5, 0.8])
def test_no_moves_with_comparable_evidence_is_not_reported_as_no_evidence(first_value) -> None:
    requirements, items = _pool("A", "B")
    report = build_shadow_report(items, requirements, limit=5, evidence={
        item.plan.plan_id: PlanEvidence(quality=Measured(
            "same-suite", first_value if index == 0 else 0.5, "same metric",
        )) for index, item in enumerate(items)
    })
    assert report.groups and not report.moves
    text = recompare_readout(report, {})
    assert "comparable evidence preserves the base order" in text
    assert "no two plans" not in text


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_recompare_does_not_propose_quality_moves_already_shown(size) -> None:
    from jaull.application.recommendation.service import recommend
    from jaull.recommendation.engine_v2 import PlanRankingContext
    from jaull.recommendation.shadow import recompare_quality
    from tests._workflow_fixtures import hardware
    from tests.test_ifeval_applicability import _full_record, _gguf
    from tests.test_recommendation_engine_v2 import _requirements

    records = [_full_record(), _full_record(stronger=True)]
    requirements = _requirements(RecommendationPriority.QUALITY, use_case=UseCase.GENERAL_CHAT)
    pools: list = []
    recommendations = recommend(
        [_gguf("org/A-7B-GGUF", "a" * 64), _gguf("org/B-7B-GGUF", "9" * 64)],
        requirements, hardware=hardware(),
        plan_context=PlanRankingContext(hardware=hardware(), quality_records=records),
        on_ranked_plans=pools.append,
    )
    state = RecommendationWorkflowState(requirements=requirements, recommendations=recommendations,
                                        ranked_plans=tuple(pools[0]))
    assert [rec.repo_id for rec in recommendations] == ["org/B-7B-GGUF", "org/A-7B-GGUF"]

    class StoredAdvisor(ProfileAdvisor):
        def recompare_quality(self, saved):
            return recompare_quality(saved.ranked_plans, saved.requirements, records, limit=5)

    async def scenario() -> None:
        app = JaullApp(advisor=StoredAdvisor())  # type: ignore[arg-type]
        before = state.model_dump_json()
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen(search_state=state)
            await app.push_screen(screen)
            await _settled(pilot, screen)
            screen.query_one("#quality-recompare", Button).press()
            await _settled(pilot, screen)
            text = str(screen.query_one("#quality-result", Static).content)
            assert "No changes to the shown recommendations" in text
            assert "#2 -> #1" not in text and "Top 5 entered" not in text
            assert "base order" in text
            assert state.model_dump_json() == before
            assert screen.query_one("#quality-back", Button).region.right <= size[0]
    asyncio.run(scenario())


@pytest.mark.parametrize("shown,expected,absent", [
    (("b", "a"), "No changes to the shown recommendations", "#2 -> #1"),
    (("a", "b"), "#2 -> #1  B", "No changes to the shown recommendations"),
    (("b", "c"), "Top 5 entered: A (#2)", "#2 -> #1"),
])
def test_recompare_uses_visible_positions_not_base_pool_positions(shown, expected, absent) -> None:
    report = ShadowReport(
        priority=RecommendationPriority.QUALITY, applied=True,
        base_order=("a", "b", "c"), shadow_order=("b", "a", "c"),
        base_top5=("a", "b"), shadow_top5=("b", "a"),
        moves=(ShadowMove(plan_id="b", repo_id="B", base_index=1, shadow_index=0,
                          group="same", rule="measured"),),
    )
    text = recompare_readout(report, {"a": "A", "b": "B", "c": "C"}, shown_order=shown)
    assert expected in text and absent not in text
    assert "base order" in text
    if "c" in shown:
        assert "Top 5 left: C" in text


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_recompare_lists_each_plan_and_keeps_actions_visible(size) -> None:
    class DiagnosticAdvisor(ProfileAdvisor):
        def recompare_quality(self, state):
            ids = tuple(f"plan-{i}" for i in range(12))
            return ShadowReport(
                priority=RecommendationPriority.QUALITY, applied=True,
                base_order=ids, shadow_order=ids,
                fallback=tuple(ShadowFallback(
                    plan_id=plan_id, repo_id=f"synthetic/Model-{i}",
                    artifact_label="gguf Q6_K", artifact_sha256=f"{i:064x}",
                    reasons=("Quality: no evidence for this artifact.",
                             "Speed: no benchmark of this exact artifact."),
                ) for i, plan_id in enumerate(ids)),
            )

    async def scenario():
        advisor = DiagnosticAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            screen.query_one("#quality-recompare", Button).press()
            await _wait_until(pilot, lambda: (
                not screen._busy and bool(screen.query("#quality-plan-diagnostics Static"))
            ))
            text = str(screen.query_one("#quality-result", Static).content)
            assert "plans kept their fallback positions" in text
            assert "SHA256" not in text
            diagnostics = screen.query_one("#quality-plan-diagnostics")
            details = list(diagnostics.query(TechnicalDetails))
            assert len(details) == 12 and all(d.collapsed for d in details)
            await pilot.pause()
            first_model = next(iter(diagnostics.query(Static)))
            body = screen.query_one("#quality-results-body")
            assert first_model.region.height > 0
            assert first_model.region.y >= body.content_region.y
            assert first_model.region.bottom <= body.content_region.bottom
            detail_text = "\n".join(str(w.content) for w in diagnostics.query(Static))
            for i in range(12):
                assert f"#{i + 1} synthetic/Model-{i} / gguf Q6_K" in detail_text
                assert f"{i:064x}" in detail_text
            assert str(details[0].title) == "Quality: no evidence for this artifact."
            assert "no benchmark of this exact artifact" in detail_text
            details[0].collapsed = False
            await pilot.pause()
            assert not details[0].collapsed
            assert "the search result stays as it was" in text
            for name in ("start", "prepare", "cancel", "back"):
                button = screen.query_one(f"#quality-{name}", Button)
                assert button.region.right <= size[0]
                assert button.region.bottom <= size[1] - 1
            assert not advisor.calls
            screen.query_one("#quality-profile", Select).value = "smoke"
            await _wait_until(pilot, lambda: (
                screen._profile is QualityProfile.SMOKE
                and not screen.query("#quality-plan-diagnostics Static")
            ))
            assert not screen.query("#quality-plan-diagnostics TechnicalDetails")
    asyncio.run(scenario())


def test_profile_selection_after_cancel_does_not_inherit_cancellation() -> None:
    class CancelAdvisor(ProfileAdvisor):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()
            self.cancel_states: list[bool] = []

        def prepare_quality_candidates(self, state: Any, **kwargs: Any) -> Any:
            self.cancel_states.append(kwargs["is_cancelled"]())
            if len(self.cancel_states) == 1:
                self.started.set()
                assert self.release.wait(10), "test did not release selection worker"
            if kwargs["is_cancelled"]():
                raise QualityEvaluationCancelled("Synthetic cancellation")
            return super().prepare_quality_candidates(state, **kwargs)

    async def scenario() -> None:
        advisor = CancelAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            try:
                async with asyncio.timeout(10):
                    while not advisor.started.is_set():
                        await pilot.pause(0.01)
                screen.query_one("#quality-cancel", Button).press()
                await pilot.pause()
            finally:
                advisor.release.set()
            await _settled(pilot, screen)
            assert not screen.candidate_plans
            screen.query_one("#quality-profile", Select).value = "smoke"
            await pilot.pause()
            await _settled(pilot, screen)
            assert advisor.cancel_states == [False, False]
            assert screen.candidate_plans
            assert not screen.query_one("#quality-start", Button).disabled
    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_evaluation_workspace_separates_candidates_setup_and_results(size, tmp_path) -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            tabs = screen.query_one("#quality-tabs", TabbedContent)
            assert tabs.active == "quality-selection-tab"
            assert not screen.query_one("#quality-setup-tab").display
            assert not screen.query_one("#quality-results-tab").display
            for name in ("start", "prepare", "cancel", "back"):
                button = screen.query_one(f"#quality-{name}", Button)
                assert button.region.right <= size[0]
                assert button.region.bottom < size[1]
            checkbox = screen.query_one("#quality-candidate-0", Checkbox)
            assert checkbox.region.height > 0
            assert checkbox.region.y > screen.query_one("#quality-profile").region.bottom
            assert checkbox.region.bottom < screen.query_one("#quality-actions").region.y
            assert screen.query_one("#quality-row-0").has_class("-selected")
            checkbox.value = False
            await pilot.pause()
            assert not screen.query_one("#quality-row-0").has_class("-selected")
            app.save_screenshot(f"quality-candidates-{size[0]}.svg", path=str(tmp_path))
            tabs.active = "quality-setup-tab"
            await pilot.pause()
            assert screen.query_one("#quality-setup-tab").display
            advanced = screen.query_one("#quality-setup", Collapsible)
            assert advanced.collapsed
            pane = screen.query_one("#quality-setup-tab")
            actions = screen.query_one("#quality-actions")
            assert advanced.region.bottom <= min(pane.region.bottom, actions.region.y)
            for name in ("dataset", "dataset-download", "download"):
                control = screen.query_one(f"#quality-{name}")
                assert pane.region.y <= control.region.y
                assert control.region.bottom <= min(pane.region.bottom, actions.region.y)
                assert control.region.right <= size[0]
                assert control not in advanced.query("Input, Checkbox")
            for name in ("dataset-download", "download"):
                permission = screen.query_one(f"#quality-{name}", Checkbox)
                assert not permission.value
                await pilot.click(f"#quality-{name}")
                assert permission.value
                await pilot.click(f"#quality-{name}")
                assert not permission.value
            app.save_screenshot(f"quality-protocol-{size[0]}.svg", path=str(tmp_path))
            assert not advisor.calls and not advisor.preparations
    asyncio.run(scenario())


def test_missing_server_opens_advanced_without_hiding_dataset(tmp_path: Path) -> None:
    async def scenario() -> None:
        advisor = ProfileAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = QualityEvaluationScreen(search_state=_state())
            await app.push_screen(screen)
            await _settled(pilot, screen)
            typed = str(tmp_path / "typed-dataset.jsonl")
            screen.query_one("#quality-dataset", Input).value = typed
            screen.query_one("#quality-prepare", Button).press()
            await _wait_until(
                pilot, lambda: app.focused is screen.query_one("#quality-server", Input),
            )
            assert screen.query_one("#quality-tabs", TabbedContent).active == "quality-setup-tab"
            advanced = screen.query_one("#quality-setup", Collapsible)
            assert not advanced.collapsed
            advanced.collapsed = True
            assert screen.query_one("#quality-dataset", Input).value == typed
            assert screen.query_one("#quality-dataset", Input) not in advanced.query(Input)
            assert not advisor.calls and not advisor.preparations
    asyncio.run(scenario())
