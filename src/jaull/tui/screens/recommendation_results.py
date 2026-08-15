from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Input, Static

from jaull.domain.benchmarks import BenchmarkMeasurementKind
from jaull.domain.execution_plans import ArtifactVariantFormat, ExecutionPlan
from jaull.evaluation.benchmark_comparison import (
    BenchmarkComparison,
    BenchmarkPlanSummary,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.reporting.writer import (
    default_report_name,
    default_reports_dir,
    write_recommendation_report,
)
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.memory_usage_bar import MemoryUsageBar
from jaull.tui.widgets.score_bar import ScoreBar
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowStep
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class RecommendationResultsScreen(Screen[None]):
    """Step 4: the best recommendation, alternatives, and secondary actions."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Results",
            "Best fit first. Technical detail is available when you need it.",
        )
        with VerticalScroll(id="results-body"):
            if not self._state.recommendations:
                yield from self._compose_no_results()
            else:
                yield from self._compose_results()
            yield from _results_actions(bool(self._state.recommendations))
        yield Footer()

    def _compose_no_results(self) -> ComposeResult:
        yield Static("No compatible models found", classes="section-title")
        yield WarningsPanel(self._state.no_results_reason or ["No candidates could be evaluated."])

    def _compose_results(self) -> ComposeResult:
        primary = self._state.recommendations[0]
        yield _PrimaryRecommendationPanel(primary)
        if len(self._state.recommendations) > 1:
            yield Static("ALTERNATIVES", classes="results-section-title")
            for index, rec in enumerate(self._state.recommendations[1:], start=1):
                yield _AlternativeRecommendationRow(index, rec)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self._app()
        button_id = event.button.id or ""
        if button_id.startswith("res-run-"):
            index = int(button_id.removeprefix("res-run-"))
            if 0 <= index < len(self._state.recommendations):
                app.run_recommendation(self._state.recommendations[index])
            return
        if button_id.startswith("res-validate-"):
            index = int(button_id.removeprefix("res-validate-"))
            if 0 <= index < len(self._state.recommendations):
                app.validate_recommendation(self._state.recommendations[index])
            return
        if button_id.startswith("res-benchmark-"):
            index = int(button_id.removeprefix("res-benchmark-"))
            if 0 <= index < len(self._state.recommendations):
                app.benchmark_recommendation(self._state.recommendations[index])
            return
        if button_id.startswith("res-paths-"):
            index = int(button_id.removeprefix("res-paths-"))
            if 0 <= index < len(self._state.recommendations):
                self.app.push_screen(
                    ExecutionPathsScreen(self._state.recommendations[index])
                )
            return

        match button_id:
            case "res-compare":
                self.app.push_screen(RecommendationCompareScreen(self._state))
            case "res-compare-paths":
                if self._state.recommendations:
                    self.app.push_screen(
                        ExecutionPathBenchmarkCompareScreen(
                            self._state.recommendations[0]
                        )
                    )
            case "res-details":
                self.app.push_screen(RecommendationDetailsScreen(self._state))
            case "res-export":
                self.app.push_screen(ExportReportModal(self._state))
            case "res-restart":
                app.restart_workflow()
            case "res-advanced":
                app.goto_advanced_tools()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class RecommendationCompareScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Compare",
            "A compact table of the ranked recommendations.",
        )
        with VerticalScroll(id="compare-body"):
            if not self._state.recommendations:
                yield Static("Nothing to compare.", classes="text-muted")
            else:
                yield _comparison_table(self._state.recommendations)
            yield Button("Back", id="compare-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare-back":
            self.app.pop_screen()


class _BenchmarkComparisonLoaded(Message):
    def __init__(self, comparison: BenchmarkComparison) -> None:
        super().__init__()
        self.comparison = comparison


class _BenchmarkComparisonFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class ExecutionPathBenchmarkCompareScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._comparison_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-benchmark-compare",
        )
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Compare execution paths",
            "Uses saved benchmark records. No benchmark is run from this screen.",
        )
        with VerticalScroll(id="path-compare-body"):
            yield Static(
                "Loading saved benchmark records...",
                id="path-compare-status",
                classes="status-line",
            )
            yield Static("", id="path-compare-error", classes="warning-line")
            with Vertical(id="path-compare-content"):
                yield Static("", id="path-compare-empty", classes="text-muted")
            yield Button("Back", id="path-compare-back")
        yield Footer()

    def on_mount(self) -> None:
        self._comparison_closing.clear()
        self._future = self._executor.submit(self._load_worker, self._app().advisor)

    def on_unmount(self) -> None:
        self._comparison_closing.set()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _load_worker(self, advisor: AdvisorService) -> None:
        try:
            comparison = advisor.compare_saved_benchmarks_for_recommendation(
                self._recommendation
            )
        except Exception as exc:
            if not self._comparison_closing.is_set():
                self.post_message(_BenchmarkComparisonFailed(str(exc)))
            return
        if not self._comparison_closing.is_set():
            self.post_message(_BenchmarkComparisonLoaded(comparison))

    @on(_BenchmarkComparisonLoaded)
    async def _comparison_loaded(self, message: _BenchmarkComparisonLoaded) -> None:
        if self._comparison_closing.is_set():
            return
        self.query_one("#path-compare-status", Static).display = False
        await self._render_comparison(message.comparison)

    @on(_BenchmarkComparisonFailed)
    def _comparison_failed(self, message: _BenchmarkComparisonFailed) -> None:
        if self._comparison_closing.is_set():
            return
        self.query_one("#path-compare-status", Static).display = False
        error = self.query_one("#path-compare-error", Static)
        error.update(message.message)
        error.display = True

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "path-compare-back":
            self.app.pop_screen()

    async def _render_comparison(self, comparison: BenchmarkComparison) -> None:
        container = self.query_one("#path-compare-content", Vertical)
        await container.remove_children()
        if len(comparison.plans) < 2:
            await container.mount(
                Static(
                    "At least two saved benchmark records are needed for this model.",
                    classes="text-muted",
                )
            )
            return

        await container.mount(
            Static(
                f"{comparison.model_identity.model_name} · This machine",
                classes="section-title",
            )
        )
        await container.mount(_benchmark_plan_table(comparison))
        if comparison.metrics:
            await container.mount(Static("Measured differences", classes="section-title"))
            await container.mount(_benchmark_difference_table(comparison))
        if comparison.warnings:
            await container.mount(
                SummaryCard(
                    "Methodology warnings",
                    [
                        (f"Warning {index}", warning)
                        for index, warning in enumerate(comparison.warnings, start=1)
                    ],
                    plain=True,
                )
            )

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class RecommendationDetailsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Technical details",
            "Scoring, memory, assumptions and the equivalent CLI command.",
        )
        with VerticalScroll(id="details-body"):
            if not self._state.recommendations:
                yield Static("No technical details.", classes="text-muted")
            else:
                yield from self._compose_details()
            yield Button("Back", id="details-back")
        yield Footer()

    def _compose_details(self) -> ComposeResult:
        primary = self._state.recommendations[0]
        yield ScoreBar(primary.score)

        estimate = primary.evaluated.memory_estimate
        if estimate is not None:
            yield SummaryCard("Memory breakdown", _breakdown_rows(estimate), plain=True)
            if estimate.total_bytes is not None:
                yield MemoryUsageBar(
                    "Available VRAM",
                    estimate.total_bytes,
                    estimate.assessment.available_vram_bytes,
                    plain=True,
                )
                yield MemoryUsageBar(
                    "Available RAM",
                    estimate.total_bytes,
                    estimate.assessment.available_ram_bytes,
                    plain=True,
                )

        rows: list[tuple[str, str]] = [
            ("Configuration chosen", primary.evaluated.configuration_reason or "-"),
            ("Variants considered", ", ".join(primary.evaluated.alternatives_considered) or "-"),
            ("Search queries", str(len(self._state.search_queries))),
            ("Candidates found", str(len(self._state.candidates))),
            ("Candidates evaluated", str(len(self._state.evaluated_candidates))),
        ]
        if estimate is not None:
            rows.append(("Repository type", estimate.repository_type.value))
            if estimate.architecture:
                rows.append(("Architecture", estimate.architecture))
            base = estimate.base_model_resolution
            if base is not None and base.repo_id:
                rows.append(("Base model", f"{base.repo_id} ({base.source.value})"))
        yield SummaryCard("Pipeline", rows, plain=True)
        yield SummaryCard(
            "Execution paths",
            _execution_path_detail_rows(primary),
            plain=True,
        )

        if self._state.requirements is not None:
            yield SummaryCard(
                "Assumptions",
                [
                    (str(index), assumption)
                    for index, assumption in enumerate(
                        self._state.requirements.assumptions,
                        start=1,
                    )
                ]
                or [("-", "none recorded")],
                plain=True,
            )
        yield CliEquivalent(_equivalent_cli_for(primary))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "details-back":
            self.app.pop_screen()


class _ExecutionPathsLoaded(Message):
    def __init__(self, plans: list[ExecutionPlan]) -> None:
        super().__init__()
        self.plans = plans


class _ExecutionPathsFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class ExecutionPathsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._plans: list[ExecutionPlan] = []
        self._selected_plan_id: str | None = None
        self._show_advanced_quantizations = False
        self._paths_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-execution-paths",
        )
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Execution paths",
            "Select one concrete plan before running, validating or benchmarking.",
        )
        with VerticalScroll(id="paths-body"):
            yield Static(self._recommendation.repo_id, classes="section-title")
            yield Static("Loading execution paths...", id="paths-status", classes="status-line")
            yield Static("", id="paths-error", classes="warning-line")
            with Vertical(id="paths-list"):
                yield Static("", id="paths-empty", classes="text-muted")
            with Horizontal(id="paths-actions"):
                yield Button("Run", id="paths-run", classes="-primary", disabled=True)
                yield Button("Validate", id="paths-validate", disabled=True)
                yield Button("Benchmark", id="paths-benchmark", disabled=True)
                yield Button("Back", id="paths-back")
        yield Footer()

    def on_mount(self) -> None:
        self._paths_closing.clear()
        self._future = self._executor.submit(self._load_worker, self._app().advisor)

    def on_unmount(self) -> None:
        self._paths_closing.set()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _load_worker(self, advisor: AdvisorService) -> None:
        try:
            plans = advisor.execution_plans_for_recommendation(self._recommendation)
        except Exception as exc:
            if not self._paths_closing.is_set():
                self.post_message(_ExecutionPathsFailed(str(exc)))
            return
        if not self._paths_closing.is_set():
            self.post_message(_ExecutionPathsLoaded(plans))

    @on(_ExecutionPathsLoaded)
    async def _paths_loaded(self, message: _ExecutionPathsLoaded) -> None:
        if self._paths_closing.is_set():
            return
        self._plans = message.plans
        self._selected_plan_id = self._plans[0].plan_id if self._plans else None
        self.query_one("#paths-status", Static).update("")
        self.query_one("#paths-status", Static).display = False
        await self._render_plans()

    @on(_ExecutionPathsFailed)
    def _paths_failed(self, message: _ExecutionPathsFailed) -> None:
        if self._paths_closing.is_set():
            return
        self.query_one("#paths-status", Static).display = False
        error = self.query_one("#paths-error", Static)
        error.update(message.message)
        error.display = True

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("paths-plan-"):
            index = int(button_id.removeprefix("paths-plan-"))
            plans = self._primary_path_plans()
            if 0 <= index < len(plans):
                self._selected_plan_id = plans[index].plan_id
                await self._render_plans()
            return
        if button_id == "paths-gguf":
            gguf_plan = self._selected_gguf_plan() or next(iter(self._gguf_plans()), None)
            if gguf_plan is not None:
                self._selected_plan_id = gguf_plan.plan_id
                await self._render_plans()
            return
        if button_id.startswith("paths-quant-"):
            index = int(button_id.removeprefix("paths-quant-"))
            gguf_plans = self._gguf_plans()
            if 0 <= index < len(gguf_plans):
                self._selected_plan_id = gguf_plans[index].plan_id
                await self._render_plans()
            return
        if button_id == "paths-toggle-advanced":
            self._show_advanced_quantizations = not self._show_advanced_quantizations
            await self._render_plans()
            return
        if button_id == "paths-back":
            self.app.pop_screen()
            return
        plan = self._selected_plan()
        if plan is None:
            return
        app = self._app()
        if button_id == "paths-run":
            app.run_recommendation(self._recommendation, plan)
        elif button_id == "paths-validate":
            app.validate_recommendation(self._recommendation, plan)
        elif button_id == "paths-benchmark":
            app.benchmark_recommendation(self._recommendation, plan)

    async def _render_plans(self) -> None:
        container = self.query_one("#paths-list", Vertical)
        await container.remove_children()
        if not self._plans:
            await container.mount(
                Static("No execution paths were found.", classes="text-muted")
            )
            self._set_actions(False)
            return
        await container.mount(Static("Runtime", classes="results-section-title"))
        for index, plan in enumerate(self._primary_path_plans()):
            marker = "●" if plan.plan_id == self._selected_plan_id else "○"
            await container.mount(
                Button(
                    f"{marker} {_path_label(plan)}",
                    id=f"paths-plan-{index}",
                    classes="-compact",
                )
            )
            await container.mount(Static(_plan_summary(plan), classes="text-muted"))

        gguf_plan = self._selected_gguf_plan() or next(iter(self._gguf_plans()), None)
        if gguf_plan is not None:
            marker = "●" if gguf_plan.plan_id == self._selected_plan_id else "○"
            await container.mount(
                Button(
                    f"{marker} llama.cpp · {_backend_hint(gguf_plan.runtime)} · GGUF",
                    id="paths-gguf",
                    classes="-compact",
                )
            )
            await container.mount(
                Static(
                    f"Quantization: {gguf_plan.artifact.quantization or 'unknown'}",
                    classes="text-muted",
                )
            )
            await self._render_quantization_picker(container, gguf_plan)
        self._set_actions(True)

    async def _render_quantization_picker(
        self,
        container: Vertical,
        selected_plan: ExecutionPlan,
    ) -> None:
        gguf_plans = self._gguf_plans()
        suggested = [
            plan
            for plan in gguf_plans
            if (plan.artifact.quantization or "") in _SUGGESTED_QUANTIZATIONS
        ]
        advanced = [plan for plan in gguf_plans if plan not in suggested]
        selected_advanced = [
            plan for plan in advanced if plan.plan_id == selected_plan.plan_id
        ]
        visible_advanced = (
            advanced if self._show_advanced_quantizations else selected_advanced
        )

        if suggested:
            await container.mount(Static("Suggested quantizations", classes="text-muted"))
            for plan in suggested:
                await container.mount(self._quantization_button(plan, gguf_plans))

        if visible_advanced:
            await container.mount(Static("Advanced quantizations", classes="text-muted"))
            for plan in visible_advanced:
                await container.mount(self._quantization_button(plan, gguf_plans))

        hidden = len(advanced) - len(visible_advanced)
        if hidden > 0:
            label = (
                f"Show {hidden} advanced quantizations"
                if not self._show_advanced_quantizations
                else "Hide advanced quantizations"
            )
            await container.mount(
                Button(label, id="paths-toggle-advanced", classes="-compact -quiet")
            )

    def _quantization_button(
        self,
        plan: ExecutionPlan,
        gguf_plans: list[ExecutionPlan],
    ) -> Widget:
        marker = "●" if plan.plan_id == self._selected_plan_id else "○"
        index = gguf_plans.index(plan)
        quantization = plan.artifact.quantization or plan.artifact.label
        return Button(
            f"{marker} {quantization}",
            id=f"paths-quant-{index}",
            classes="-compact",
        )

    def _set_actions(self, enabled: bool) -> None:
        self.query_one("#paths-run", Button).disabled = not enabled
        self.query_one("#paths-validate", Button).disabled = not enabled
        self.query_one("#paths-benchmark", Button).disabled = not enabled

    def _selected_plan(self) -> ExecutionPlan | None:
        if self._selected_plan_id is None:
            return None
        return next(
            (plan for plan in self._plans if plan.plan_id == self._selected_plan_id),
            None,
        )

    def _primary_path_plans(self) -> list[ExecutionPlan]:
        return [plan for plan in self._plans if not _is_gguf_plan(plan)]

    def _gguf_plans(self) -> list[ExecutionPlan]:
        return sorted(
            (plan for plan in self._plans if _is_gguf_plan(plan)),
            key=_gguf_sort_key,
        )

    def _selected_gguf_plan(self) -> ExecutionPlan | None:
        selected = self._selected_plan()
        if selected is not None and _is_gguf_plan(selected):
            return selected
        return None

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class ExportReportModal(ModalScreen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Cancel")]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="export-modal"):
            yield Static("Export report", classes="card-title")
            yield Static(
                f"Writes JSON and Markdown under {default_reports_dir()} unless "
                "you enter an absolute or explicitly-relative path. No tokens or "
                "credentials are included.",
                classes="text-muted",
            )
            yield Input(value=default_report_name(), id="res-export-path")
            yield Static("", id="res-export-error", classes="warning-line")
            yield Static("", id="res-export-status", classes="status-ok")
            with Horizontal(id="res-export-actions"):
                yield Button("Export", id="res-export-confirm", classes="-primary")
                yield Button("Cancel", id="res-export-cancel")
                yield Button("Close", id="res-export-close")

    def on_mount(self) -> None:
        self.query_one("#res-export-path", Input).focus()
        self._set_error("")
        self._set_status("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "res-export-cancel" | "res-export-close":
                self.app.pop_screen()
            case "res-export-confirm":
                self._write_report()

    def _write_report(self) -> None:
        raw = self.query_one("#res-export-path", Input).value.strip()
        target = Path(raw) if raw else None
        self._set_error("")
        self._set_status("Writing report...")
        try:
            written = write_recommendation_report(self._state, target)
        except OSError as exc:
            self._set_status("")
            self._set_error(f"Could not write the report: {exc}")
            self.query_one("#res-export-path", Input).focus()
            return
        files = ", ".join(str(path) for path in written)
        self._set_status(f"Report written: {files}")

    def _set_error(self, message: str) -> None:
        widget = self.query_one("#res-export-error", Static)
        widget.update(message)
        widget.display = bool(message)

    def _set_status(self, message: str) -> None:
        widget = self.query_one("#res-export-status", Static)
        widget.update(message)
        widget.display = bool(message)


def _results_actions(has_results: bool) -> ComposeResult:
    """Three tiers: run (in the panel), inspect, and leave.

    Compare/Details/Export are ordinary buttons; starting over and dropping
    into the advanced tools are quiet, because neither is what the user came
    here to do.
    """
    if has_results:
        with Horizontal(id="results-actions"):
            yield Button("Compare", id="res-compare")
            yield Button("Compare paths", id="res-compare-paths")
            yield Button("Details", id="res-details")
            yield Button("Export", id="res-export")
    with Horizontal(id="results-actions-secondary"):
        yield Button("Start again", id="res-restart", classes="-quiet")
        yield Button("Advanced tools", id="res-advanced", classes="-quiet")


class _PrimaryRecommendationPanel(Vertical):
    """The one block on this screen allowed to wear a border."""

    DEFAULT_CLASSES = "results-primary"

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._rec = recommendation

    def compose(self) -> ComposeResult:
        rec = self._rec
        yield Static("BEST MATCH", classes="results-kicker")
        yield Static(rec.repo_id, classes="results-model-name")
        with Horizontal():
            yield Static(
                f"{rec.score.out_of_100} / 100",
                classes="results-score-line",
            )
            yield Static(
                _status_label(rec),
                classes=f"results-fit {_fit_class(rec)}",
            )
        meta = " · ".join(_recommendation_metadata(rec))
        if meta:
            yield Static(meta, classes="results-meta-line")
        for reason in rec.reasons[:3]:
            yield Static(f"✓ {reason}", classes="reason-line")
        path_lines = _execution_path_lines(rec)
        if path_lines:
            yield Static("Execution paths", classes="results-section-title")
            for line in path_lines:
                yield Static(line, classes="reason-line")
        with Horizontal(classes="actions-right"):
            yield Button("Paths", id="res-paths-0")
            yield Button("Run model", id="res-run-0", classes="-primary")
            yield Button(
                "Validate",
                id="res-validate-0",
                disabled=not _can_validate(rec),
            )
            yield Button(
                "Benchmark",
                id="res-benchmark-0",
                disabled=not _can_benchmark(rec),
            )


class _AlternativeRecommendationRow(Horizontal):
    """One line per alternative: rank, name, score, fit, and a way to run it."""

    DEFAULT_CLASSES = "results-alternative"

    def __init__(self, index: int, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._index = index
        self._rec = recommendation

    def compose(self) -> ComposeResult:
        rec = self._rec
        yield Static(f"#{self._index + 1}", classes="results-alt-rank")
        yield Static(rec.repo_id, classes="results-alt-name")
        yield Static(str(rec.score.out_of_100), classes="results-alt-score")
        yield Static(
            _status_label(rec),
            classes=f"results-alt-fit {_fit_class(rec)}",
        )
        yield Button("Run", id=f"res-run-{self._index}", classes="-compact")
        yield Button("Paths", id=f"res-paths-{self._index}", classes="-compact")
        yield Button(
            "Validate",
            id=f"res-validate-{self._index}",
            classes="-compact",
            disabled=not _can_validate(rec),
        )
        yield Button(
            "Benchmark",
            id=f"res-benchmark-{self._index}",
            classes="-compact",
            disabled=not _can_benchmark(rec),
        )


def _comparison_table(recommendations: list[ModelRecommendation]) -> DataTable[str]:
    table: DataTable[str] = DataTable()
    table.add_columns(
        "Model",
        "Configuration",
        "Memory",
        "Compatibility",
        "Score",
        "Confidence",
        "License",
        "Main reason",
    )
    for rec in recommendations:
        table.add_row(
            rec.repo_id,
            _configuration_label(rec),
            _memory_label(rec),
            rec.status.value,
            f"{rec.score.out_of_100}/100",
            rec.confidence.value,
            rec.evaluated.candidate.license or "not declared",
            rec.reasons[0] if rec.reasons else "-",
        )
    return table


def _benchmark_plan_table(comparison: BenchmarkComparison) -> DataTable[str]:
    table: DataTable[str] = DataTable()
    labels = [_benchmark_plan_label(plan) for plan in comparison.plans]
    table.add_columns("Metric", *labels)
    table.add_row(
        "Artifact",
        *[
            " ".join(
                item
                for item in (plan.artifact_format, plan.quantization_or_precision)
                if item
            )
            for plan in comparison.plans
        ],
    )
    table.add_row("Backend", *[plan.backend.upper() for plan in comparison.plans])
    table.add_row(
        "Methodology",
        *[plan.methodology or "unknown" for plan in comparison.plans],
    )
    table.add_row(
        "Compatible runs",
        *[str(plan.compatible_run_count) for plan in comparison.plans],
    )
    table.add_row("Peak RAM", *[_bytes(plan.peak_ram_bytes) for plan in comparison.plans])
    table.add_row(
        "Peak VRAM",
        *[_bytes(plan.peak_vram_bytes) for plan in comparison.plans],
    )
    table.add_row(
        "Model load",
        *[_seconds(plan.model_load_seconds) for plan in comparison.plans],
    )
    table.add_row(
        "TTFT",
        *[_seconds(plan.time_to_first_token_seconds) for plan in comparison.plans],
    )
    return table


def _benchmark_difference_table(comparison: BenchmarkComparison) -> DataTable[str]:
    table: DataTable[str] = DataTable()
    table.add_columns("Metric", "Baseline", "Candidate", "Difference")
    for metric in comparison.metrics:
        table.add_row(
            _metric_label(metric.kind, metric.tokens),
            f"{metric.baseline_tokens_per_second:.1f} t/s",
            f"{metric.candidate_tokens_per_second:.1f} t/s",
            _throughput_ratio(metric.relative_throughput),
        )
    memory_difference = _memory_difference(comparison.plans)
    if memory_difference is not None:
        baseline, candidate, ratio = memory_difference
        table.add_row("Peak RAM", _bytes(baseline), _bytes(candidate), ratio)
    return table


def _benchmark_plan_label(plan: BenchmarkPlanSummary) -> str:
    if plan.runtime == "transformers":
        return "PyTorch"
    if plan.runtime == "llama.cpp":
        return "llama.cpp"
    return plan.runtime


def _throughput_ratio(relative: float) -> str:
    if relative <= 0:
        return "unavailable"
    if relative >= 1:
        return f"{relative:.1f}x"
    return f"{1 / relative:.1f}x lower"


def _memory_difference(
    plans: list[BenchmarkPlanSummary],
) -> tuple[int, int, str] | None:
    if len(plans) < 2:
        return None
    baseline = plans[0].peak_ram_bytes
    candidate = plans[1].peak_ram_bytes
    if baseline is None or candidate is None or baseline <= 0 or candidate <= 0:
        return None
    relative = candidate / baseline
    label = f"{relative:.1f}x higher" if relative >= 1 else f"{1 / relative:.1f}x lower"
    return baseline, candidate, label


def export_report(state: RecommendationWorkflowState, target: Path) -> list[Path]:
    """Backwards-compatible wrapper around :func:`write_recommendation_report`."""
    return write_recommendation_report(state, target)


def _recommendation_metadata(rec: ModelRecommendation) -> list[str]:
    config = rec.evaluated.selected_configuration
    estimate = rec.evaluated.memory_estimate
    runtime = estimate.runtime_recommendation if estimate is not None else None

    values: list[str] = []
    if config is not None and config.quantization:
        values.append(config.quantization)
    if runtime is not None:
        values.append(runtime.runtime.value)
    if config is not None and config.context_length:
        values.append(f"ctx {config.context_length}")
    if estimate is not None and estimate.total_bytes is not None:
        values.append(_gib(estimate.total_bytes))
    return values


def _execution_path_lines(rec: ModelRecommendation) -> list[str]:
    rows = _execution_path_detail_rows(rec)
    lines = [f"✓ {label} · {value}" for label, value in rows[:1]]
    for related in rec.related_repositories[:2]:
        lines.append(f"○ Related artifact repository · {related}")
    return lines


def _execution_path_detail_rows(rec: ModelRecommendation) -> list[tuple[str, str]]:
    from jaull.execution_plans import execution_plan_for_recommendation

    try:
        plan = execution_plan_for_recommendation(rec)
    except ValueError:
        return [("Current artifact", "no executable runtime recommendation")]

    runtime = plan.runtime.runtime.value
    backend = _backend_hint(plan.runtime)
    readiness = (
        plan.execution_readiness.status.value
        if plan.execution_readiness is not None
        else "preflight not checked"
    )
    rows = [
        (f"{runtime} · {backend}", f"{plan.artifact.label} · {readiness}"),
        ("Model identity", plan.model_identity.model_name),
        ("Identity confidence", plan.model_identity.confidence.value),
        ("Identity match", plan.artifact.identity_match.value),
        ("Repository", plan.artifact.repo_id),
    ]
    if plan.artifact.filename:
        rows.append(("Filename", plan.artifact.filename))
    if plan.artifact.quantization:
        rows.append(("Quantization", plan.artifact.quantization))
    if plan.artifact.precision:
        rows.append(("Precision", plan.artifact.precision))
    if rec.related_repositories:
        rows.append(("Related repositories", ", ".join(rec.related_repositories)))
    return rows


def _backend_hint(runtime: object) -> str:
    from jaull.domain.runtime import RuntimeRecommendation

    assert isinstance(runtime, RuntimeRecommendation)
    if runtime.runtime.value == "transformers":
        return next(
            (flag.value for flag in runtime.flags if flag.name == "device_map"),
            "cpu",
        )
    gpu_layers = next(
        (flag.value for flag in runtime.flags if flag.name == "--n-gpu-layers"),
        None,
    )
    if gpu_layers in {None, "0"}:
        return "CPU"
    return "accelerated"


def _plan_label(plan: ExecutionPlan) -> str:
    backend = (
        plan.backend_selection.selected_backend.value
        if plan.backend_selection is not None
        else _backend_hint(plan.runtime)
    )
    readiness = (
        plan.execution_readiness.status.value
        if plan.execution_readiness is not None
        else "prepare on demand"
    )
    return (
        f"{plan.runtime.runtime.value} · {backend} · "
        f"{plan.artifact.label} · {readiness}"
    )


_SUGGESTED_QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q6_K")


def _path_label(plan: ExecutionPlan) -> str:
    return f"{plan.runtime.runtime.value} · {_backend_hint(plan.runtime)}"


def _plan_summary(plan: ExecutionPlan) -> str:
    readiness = (
        plan.execution_readiness.status.value
        if plan.execution_readiness is not None
        else "prepare on action"
    )
    return f"{plan.artifact.label} · {readiness}"


def _is_gguf_plan(plan: ExecutionPlan) -> bool:
    return plan.artifact.format == ArtifactVariantFormat.GGUF


def _gguf_sort_key(plan: ExecutionPlan) -> tuple[int, str]:
    quantization = plan.artifact.quantization or ""
    try:
        priority = _SUGGESTED_QUANTIZATIONS.index(quantization)
    except ValueError:
        priority = len(_SUGGESTED_QUANTIZATIONS)
    return priority, quantization or plan.artifact.label


def _configuration_label(rec: ModelRecommendation) -> str:
    config = rec.evaluated.selected_configuration
    if config is None:
        return "unknown"
    return config.quantization or (config.precision.value if config.precision else "default")


def _memory_label(rec: ModelRecommendation) -> str:
    estimate = rec.evaluated.memory_estimate
    if estimate is None:
        return "unknown"
    return _gib(estimate.total_bytes)


def _status_label(rec: ModelRecommendation) -> str:
    return rec.status.value.replace("_", " ").capitalize()


def _can_validate(rec: ModelRecommendation) -> bool:
    from jaull.domain.runtime import RuntimeName

    estimate = rec.evaluated.memory_estimate
    if estimate is None or estimate.runtime_recommendation is None:
        return False
    return estimate.runtime_recommendation.runtime in {
        RuntimeName.LLAMA_CPP,
        RuntimeName.TRANSFORMERS,
    }


def _can_benchmark(rec: ModelRecommendation) -> bool:
    from jaull.domain.runtime import RuntimeName

    estimate = rec.evaluated.memory_estimate
    if estimate is None or estimate.runtime_recommendation is None:
        return False
    return estimate.runtime_recommendation.runtime in {
        RuntimeName.LLAMA_CPP,
        RuntimeName.TRANSFORMERS,
    }


# Green comfortable, amber tight, red insufficient — the same reading the rest
# of the app gives these words.
_FIT_CLASSES = {
    "comfortable": "fit-good",
    "compatible": "fit-good",
    "tight": "fit-warn",
    "offloading_required": "fit-warn",
    "insufficient": "fit-bad",
}


def _fit_class(rec: ModelRecommendation) -> str:
    return _FIT_CLASSES.get(rec.status.value, "fit-unknown")


def _breakdown_rows(estimate: object) -> list[tuple[str, str]]:
    from jaull.domain.estimation import MemoryEstimate

    assert isinstance(estimate, MemoryEstimate)
    rows = [
        ("Weights", _gib(estimate.weights.component.bytes)),
        ("KV cache", _gib(estimate.kv_cache.component.bytes)),
        ("Runtime overhead", _gib(estimate.runtime_overhead.component.bytes)),
        ("Device reserve", _gib(estimate.device_reserve.bytes)),
    ]
    if estimate.safety_margin is not None:
        rows.append(("Safety margin", _gib(estimate.safety_margin.bytes)))
    rows.append(("Estimated total", _gib(estimate.total_bytes)))
    rows.append(("Available VRAM", _gib(estimate.assessment.available_vram_bytes)))
    rows.append(("Available RAM", _gib(estimate.assessment.available_ram_bytes)))
    return rows


def _equivalent_cli_for(primary: ModelRecommendation) -> str:
    from jaull.domain.runtime import RuntimeName

    estimate = primary.evaluated.memory_estimate
    if estimate is not None and estimate.runtime_recommendation is not None:
        runtime = estimate.runtime_recommendation
        if runtime.runtime is not RuntimeName.UNKNOWN and runtime.command_preview:
            return runtime.command_preview

    config = primary.evaluated.selected_configuration
    if config is None:
        return f"jaull inspect {primary.repo_id}"
    parts = ["jaull estimate", primary.repo_id]
    if config.quantization:
        parts.append(f"--quantization {config.quantization}")
    if config.precision:
        parts.append(f"--dtype {config.precision.value}")
    parts.append(f"--context {config.context_length}")
    return " ".join(parts)


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**3:.2f} GiB"


def _bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    gib = value / 1024**3
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / 1024**2:.1f} MiB"


def _seconds(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f} s"


def _metric_label(kind: BenchmarkMeasurementKind, tokens: int) -> str:
    prefix = "Prompt" if kind is BenchmarkMeasurementKind.PREFILL else "Generation"
    return f"{prefix} {tokens}"


__all__ = [
    "ExecutionPathBenchmarkCompareScreen",
    "ExecutionPathsScreen",
    "ExportReportModal",
    "RecommendationCompareScreen",
    "RecommendationDetailsScreen",
    "RecommendationResultsScreen",
    "export_report",
]
