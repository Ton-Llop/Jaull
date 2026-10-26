"""Paths refreshes local history without rediscovery or upgrading metadata into proof."""

from __future__ import annotations

import asyncio
from threading import Event

from textual.screen import Screen
from textual.widgets import Button, Static

from jaull.tui.app import JaullApp
from jaull.tui.evidence import EvidenceIndex
from jaull.tui.screens.execution_paths import ExecutionPathsScreen, _PathOption
from tests.test_benchmarks import _record
from tests.test_execution_plans import _gguf_recommendation
from tests.test_results_speed import _transformers_case
from tests.test_tui_evidence import _FakeStoreAdvisor, _plan
from tests.test_tui_recommendation_execution import _wait_until


def test_metadata_history_does_not_award_a_confirmed_benchmark_badge():
    plan, record = _transformers_case()
    index = EvidenceIndex.load(_FakeStoreAdvisor(benchmarks={record.identity.benchmark_id: record}))
    evidence = index.for_plan(plan)
    assert not evidence.benchmarked
    assert evidence.has_benchmark_history
    assert evidence.historical_benchmark_ids == (record.identity.benchmark_id,)
    assert "current plan not verified" in evidence.summary()
    other = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={
        "sha256": "b" * 64,
    })})
    conflict = record.model_copy(update={"artifact": record.artifact.model_copy(update={
        "sha256": "a" * 64,
    })})
    conflict_index = EvidenceIndex.load(_FakeStoreAdvisor(benchmarks={"conflict": conflict}))
    assert not conflict_index.for_plan(other).has_benchmark_history


def test_paths_refreshes_saved_benchmarks_on_return_and_clears_an_empty_inspector(tmp_path):
    primary = _plan()
    other = _plan(quantization="Q5_K_M", sha256="2" * 64)
    record = _record(tmp_path)

    class Advisor(_FakeStoreAdvisor):
        services = None
        plan_calls = 0

        def execution_plans_for_recommendation(self, recommendation):
            self.plan_calls += 1
            return [primary, other]

        def prepare_execution_plan(self, *args, **kwargs):
            raise AssertionError("History refresh must not prepare or download a model")

        def run_benchmark(self, *args, **kwargs):
            raise AssertionError("History refresh must not run benchmarks")

    async def scenario():
        advisor = Advisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 50)) as pilot:
            screen = ExecutionPathsScreen(_gguf_recommendation())
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: len(screen.query(_PathOption)) == 2)
            screen.query_one("#paths-filter-benchmarked", Button).press()
            await _wait_until(pilot, lambda: not screen.query(_PathOption))
            assert not screen.query_one("#paths-callout").display
            assert not screen.query_one("#paths-facts").display
            assert not screen.query_one("#paths-selected").display
            for action in ("run", "validate", "benchmark"):
                assert screen.query_one(f"#paths-{action}", Button).disabled

            await app.push_screen(Screen())
            advisor._benchmarks[record.identity.benchmark_id] = record
            app.pop_screen()
            await _wait_until(pilot, lambda: bool(screen.query("#paths-plan-0")))
            assert screen._evidence.for_plan(primary).benchmarked
            assert not screen.query_one("#paths-benchmark", Button).disabled
            assert "✓ Benchmarked" in str(screen.query_one("#paths-fact-evidence", Static).content)
            assert not screen.query("#paths-plan-1")
            assert advisor.plan_calls == 1
            for _ in range(2):
                previous = screen._evidence
                await app.push_screen(Screen())
                app.pop_screen()
                await _wait_until(pilot, lambda previous=previous: screen._evidence is not previous)
                await pilot.pause(0)
                assert len(screen.query("#paths-plan-0")) == 1
                assert advisor.plan_calls == 1

    asyncio.run(scenario())


def test_leaving_paths_during_history_refresh_does_not_wait_for_the_store():
    entered, release = Event(), Event()

    class Advisor(_FakeStoreAdvisor):
        services = None
        block = False

        def execution_plans_for_recommendation(self, recommendation):
            return [_plan()]

        def list_benchmark_ids(self):
            if self.block:
                entered.set()
                release.wait(5)
            return []

    async def scenario():
        advisor = Advisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 50)) as pilot:
            screen = ExecutionPathsScreen(_gguf_recommendation())
            await app.push_screen(screen)
            await _wait_until(pilot, lambda: bool(screen.query(_PathOption)))
            await app.push_screen(Screen())
            advisor.block = True
            app.pop_screen()
            try:
                await _wait_until(pilot, entered.is_set)
                app.pop_screen()
                await _wait_until(pilot, screen._paths_closing.is_set)
                assert screen._evidence_future is None
                assert not release.is_set()
            finally:
                release.set()
            await pilot.pause(0)

    asyncio.run(scenario())
