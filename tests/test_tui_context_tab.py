"""The Context tab: how much memory a model needs at each context, inside the Results screen."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Static, TabbedContent

from jaull.domain.estimation import CompatibilityStatus, EstimationConfidence
from jaull.estimator.context_fit import ContextPoint, ContextSweep
from jaull.exceptions import EstimationError
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_results import (
    RecommendationResultsScreen,
    _RecommendationInspector,
    _RecommendationRow,
)
from jaull.tui.widgets.context_sweep_view import ContextSweepView
from jaull.workflow.state import RecommendationWorkflowState
from tests._workflow_fixtures import GIB, hardware
from tests.test_execution_plans import _gguf_recommendation
from tests.test_recommendation_engine_v2 import _requirements
from tests.test_tui_evidence import _FakeStoreAdvisor

STYLES = str(Path(__file__).parents[1] / "src/jaull/tui/styles.tcss")
_COMFORTABLE = CompatibilityStatus.COMFORTABLE
_WALL = CompatibilityStatus.INSUFFICIENT


def _point(context: int, total_gib: float, status: CompatibilityStatus) -> ContextPoint:
    return ContextPoint(
        context_length=context,
        total_bytes=int(total_gib * GIB),
        kv_cache_bytes=int(0.25 * GIB * context / 4096),
        status=status,
        confidence=EstimationConfidence.LOW,
    )


def _sweep(*, largest: int | None = 32768, notes: list[str] | None = None) -> ContextSweep:
    points = (
        _point(4096, 2.0, _COMFORTABLE),
        _point(32768, 4.0, _COMFORTABLE),
        _point(131072, 9.0, _WALL),
    )
    return ContextSweep(points=points, largest_fitting=largest, notes=notes or [])


def _texts(view: ContextSweepView) -> str:
    return " | ".join(str(widget.content) for widget in view.query(Static))


class _ViewApp(App[None]):
    CSS_PATH = STYLES

    def compose(self) -> ComposeResult:
        yield ContextSweepView()


async def _view_after(action: Any) -> tuple[ContextSweepView, str]:
    app = _ViewApp()
    async with app.run_test(size=(100, 40)) as pilot:
        view = app.query_one(ContextSweepView)
        action(view)
        await pilot.pause()
        await pilot.pause()
        return view, _texts(view)


def test_the_view_names_the_largest_context_that_fits() -> None:
    _, text = asyncio.run(_view_after(lambda v: v.show_sweep(_sweep())))
    assert "Largest context that fits: 32,768 tokens" in text


def test_there_is_one_row_per_context_and_the_largest_is_marked() -> None:
    async def scenario() -> None:
        app = _ViewApp()
        async with app.run_test(size=(100, 40)) as pilot:
            view = app.query_one(ContextSweepView)
            view.show_sweep(_sweep())
            await pilot.pause()
            await pilot.pause()
            rows = list(view.query(".context-point"))
            assert len(rows) == 3
            assert [row.has_class("-largest") for row in rows] == [False, True, False]

    asyncio.run(scenario())


def test_each_row_says_its_verdict_in_words_and_not_only_in_colour() -> None:
    _, text = asyncio.run(_view_after(lambda v: v.show_sweep(_sweep())))
    assert text.count("Comfortable") == 2
    assert "Insufficient" in text


def test_the_total_and_the_context_are_on_the_row_and_the_kv_is_in_its_tooltip() -> None:
    async def scenario() -> None:
        app = _ViewApp()
        async with app.run_test(size=(100, 40)) as pilot:
            view = app.query_one(ContextSweepView)
            view.show_sweep(_sweep())
            await pilot.pause()
            await pilot.pause()
            row = list(view.query(".context-point"))[1]
            row_text = " ".join(str(s.content) for s in row.query(Static))
            assert "32,768" in row_text
            assert "4.00 GiB" in row_text
            assert "KV cache" in str(row.tooltip)
            assert "low" in str(row.tooltip)

    asyncio.run(scenario())


def test_nothing_fitting_is_said_as_such_and_marked_bad() -> None:
    async def scenario() -> None:
        app = _ViewApp()
        async with app.run_test(size=(100, 40)) as pilot:
            view = app.query_one(ContextSweepView)
            view.show_sweep(_sweep(largest=None))
            await pilot.pause()
            await pilot.pause()
            headline = view.query_one(".context-headline", Static)
            assert "none" in str(headline.content).lower()
            assert headline.has_class("fit-bad")

    asyncio.run(scenario())


def test_confidence_is_shown_once() -> None:
    _, text = asyncio.run(_view_after(lambda v: v.show_sweep(_sweep())))
    assert text.count("Confidence of the estimate") == 1


def test_notes_are_shown() -> None:
    sweep = _sweep(notes=["The sweep stops at 262144 tokens and it all fits."])
    _, text = asyncio.run(_view_after(lambda v: v.show_sweep(sweep)))
    assert "may fit more" in text or "all fits" in text


def test_loading_and_unavailable_states_replace_the_previous_content() -> None:
    _, loading = asyncio.run(_view_after(lambda v: v.show_loading()))
    assert "Computing" in loading
    _, unavailable = asyncio.run(
        _view_after(lambda v: v.show_unavailable("No model analysis is available."))
    )
    assert "No model analysis is available." in unavailable
    assert "Computing" not in unavailable


def test_showing_a_sweep_after_loading_leaves_no_loading_text() -> None:
    def action(view: ContextSweepView) -> None:
        view.show_loading()
        view.show_sweep(_sweep())

    _, text = asyncio.run(_view_after(action))
    assert "Computing" not in text
    assert "Largest context that fits" in text


# --- the tab inside the inspector -------------------------------------------------------------


class _InspectorApp(App[None]):
    CSS_PATH = STYLES

    def __init__(self) -> None:
        super().__init__()
        self.requested: list[int] = []

    def compose(self) -> ComposeResult:
        row = _RecommendationRow(0, _gguf_recommendation(), selected=True)
        yield row
        yield row.detail

    @on(_RecommendationInspector.ContextRequested)
    def _requested(self, message: _RecommendationInspector.ContextRequested) -> None:
        self.requested.append(message.index)


def test_opening_the_context_tab_asks_for_the_sweep_exactly_once() -> None:
    async def scenario() -> None:
        app = _InspectorApp()
        async with app.run_test(size=(110, 40)) as pilot:
            tabs = app.query_one(_RecommendationRow).detail.query_one(TabbedContent)
            await pilot.pause()
            assert app.requested == []
            tabs.active = "rec-context-tab-0"
            await pilot.pause()
            assert app.requested == [0]
            tabs.active = "tab-1"
            await pilot.pause()
            tabs.active = "rec-context-tab-0"
            await pilot.pause()
            assert app.requested == [0]

    asyncio.run(scenario())


def test_the_existing_tabs_keep_their_ids() -> None:
    async def scenario() -> None:
        app = _InspectorApp()
        async with app.run_test(size=(110, 40)) as pilot:
            tabs = app.query_one(_RecommendationRow).detail.query_one(TabbedContent)
            await pilot.pause()
            ids = [pane.id for pane in tabs.query("TabPane")]
            assert ids[:3] == ["tab-1", "tab-2", "tab-3"]
            assert ids[3] == "rec-context-tab-0"

    asyncio.run(scenario())


# --- the whole Results screen -------------------------------------------------------------------


class _SweepAdvisor(_FakeStoreAdvisor):
    services = None

    def __init__(self, *, sweep: ContextSweep | None = None, fail: bool = False) -> None:
        super().__init__()
        self._sweep = sweep
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    def sweep_context_for_model(
        self,
        analysis: Any,
        hardware: Any,
        inference_cfg: Any,
        *,
        native_limit: int | None = None,
        resolve_base_model: bool = True,
    ) -> ContextSweep:
        self.calls.append({"analysis": analysis, "config": inference_cfg, "hardware": hardware})
        if self._fail:
            raise EstimationError("could not read the header")
        assert self._sweep is not None
        return self._sweep


def _state(*, analysis: bool = True) -> tuple[RecommendationWorkflowState, Any]:
    rec = _gguf_recommendation()
    if not analysis:
        rec = rec.model_copy(
            update={"evaluated": rec.evaluated.model_copy(update={"analysis": None})}
        )
    state = RecommendationWorkflowState(
        requirements=_requirements(), hardware=hardware(), recommendations=[rec]
    )
    return state, rec


async def _open_context(advisor: _SweepAdvisor, state: RecommendationWorkflowState) -> str:
    app = JaullApp(advisor=advisor)
    async with app.run_test(size=(120, 45)) as pilot:
        screen = RecommendationResultsScreen(state)
        await app.push_screen(screen)
        await pilot.pause()
        detail = screen.query_one(_RecommendationRow).detail
        detail.query_one(TabbedContent).active = "rec-context-tab-0"
        view = detail.query_one(ContextSweepView)
        async with asyncio.timeout(10):
            while "Computing" in _texts(view) or not _texts(view).strip():
                await pilot.pause(0.02)
        await pilot.pause()
        return _texts(view)


def test_the_screen_sweeps_with_the_recommendations_own_analysis_and_configuration() -> None:
    state, rec = _state()
    advisor = _SweepAdvisor(sweep=_sweep())
    text = asyncio.run(_open_context(advisor, state))
    assert "Largest context that fits: 32,768 tokens" in text
    assert len(advisor.calls) == 1
    assert advisor.calls[0]["analysis"] is rec.evaluated.analysis
    assert advisor.calls[0]["config"] is rec.evaluated.selected_configuration
    assert advisor.calls[0]["hardware"] is state.hardware


def test_a_recommendation_without_analysis_says_so_and_never_calls_the_advisor() -> None:
    state, _ = _state(analysis=False)
    advisor = _SweepAdvisor(sweep=_sweep())
    text = asyncio.run(_open_context(advisor, state))
    assert "not available" in text.lower()
    assert advisor.calls == []


def test_an_error_is_shown_and_does_not_leave_the_tab_computing_forever() -> None:
    state, _ = _state()
    advisor = _SweepAdvisor(fail=True)
    text = asyncio.run(_open_context(advisor, state))
    assert "could not read the header" in text
    assert "Computing" not in text
