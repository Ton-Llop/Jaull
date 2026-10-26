"""Results inspectors accept evidence before nested tab panes have mounted."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widgets import Button, Static, TabbedContent, TabPane

from jaull.evaluation.quality_records import describe_record
from jaull.presentation.speed import SpeedReadout
from jaull.tui.evidence import EvidenceState, PlanEvidence
from jaull.tui.screens.recommendation_results import _RecommendationInspector
from tests.test_quality_eval_records import synthetic_limited_record
from tests.test_tui_recommendation_execution import _recommendation, _visible_text


@pytest.mark.parametrize("width", [80, 140])
@pytest.mark.parametrize("blocked", [False, True])
def test_early_evidence_survives_tab_mount_and_repeated_inspector_lifecycle(width, blocked):
    async def scenario():
        app = App()
        rec = _recommendation()
        if blocked:
            rec = rec.model_copy(
                update={
                    "evaluated": rec.evaluated.model_copy(
                        update={"memory_estimate": None},
                    )
                }
            )
        evidence = PlanEvidence(
            state=EvidenceState.BENCHMARKED,
            benchmark_ids=("bench-before-mount",),
            quality_records=(describe_record(synthetic_limited_record()),),
            published_summary="References received before mounting.",
            published_provenance="Published provenance received before mounting.",
        )
        speed = SpeedReadout("Measured 42.0 tok/s", "Saved speed provenance.", measured=True)
        async with app.run_test(size=(width, 50)) as pilot:
            for _ in range(3):
                detail = _RecommendationInspector(0, rec)
                assert not detail.is_mounted
                # Deterministically exercise the same missing-DOM window as the
                # asynchronous evidence worker, without scheduler-dependent sleeps.
                detail.show_evidence(evidence)
                detail.show_speed(speed)
                detail.show_alternative_quality([])
                await app.screen.mount(detail)
                await pilot.pause(0)
                assert detail.query_one("#rec-actions-reason-0").display is blocked
                assert detail.query_one("#res-run-0", Button).disabled is blocked
                assert detail.query_one("#rec-quality-0").display
                assert detail.query_one("#rec-quality-details-0").display
                assert detail.query_one("#rec-quality-provenance-0").display
                assert detail.query_one("#rec-published-provenance-0").display
                assert not detail.query_one("#rec-quality-empty-0").display
                tabs = detail.query_one(TabbedContent)
                for pane in tabs.query(TabPane):
                    assert pane.id is not None
                    tabs.active = pane.id
                    await pilot.pause(0)
                text = _visible_text(detail)
                assert evidence.published_summary in text
                assert "Measured 42.0 tok/s" in text
                assert "Saved speed provenance." in text
                # A later refresh still updates the same widgets after mounting.
                detail.show_evidence(PlanEvidence(state=EvidenceState.READY))
                assert not detail.query_one("#rec-quality-0").display
                assert not detail.query_one("#rec-quality-provenance-0").display
                assert detail.query_one("#rec-quality-empty-0").display
                assert len(detail.query("#rec-actions-reason-0")) == 1
                await detail.remove()
                await pilot.pause(0)
                assert not app.screen.query(_RecommendationInspector)

    asyncio.run(scenario())


def test_empty_inspector_initializes_tab_fields_without_a_mount_query():
    async def scenario():
        app = App()
        async with app.run_test() as pilot:
            detail = _RecommendationInspector(0, _recommendation())
            await app.screen.mount(detail)
            await pilot.pause(0)
            for suffix in (
                "quality",
                "quality-details",
                "quality-provenance",
                "quality-alternatives",
                "published-provenance",
            ):
                assert not detail.query_one(f"#rec-{suffix}-0").display
            assert detail.query_one("#rec-quality-empty-0", Static).display

    asyncio.run(scenario())
