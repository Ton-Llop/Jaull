from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Input, Static

from jaull.domain.benchmarks import BenchmarkMeasurementKind
from jaull.domain.execution_plans import ModelIdentity
from jaull.evaluation.benchmark_comparison import (
    BenchmarkComparison,
    BenchmarkPlanMetricComparison,
    BenchmarkPlanSummary,
)
from jaull.presentation.plan_labels import (
    backend_hint,
    format_gib,
    model_display_name,
    plan_summary_line,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.recommendation.tier import TIER_HEADINGS, RecommendationTier
from jaull.reporting.writer import (
    default_report_name,
    default_reports_dir,
    write_recommendation_report,
)
from jaull.tui.evidence import EvidenceIndex, state_class
from jaull.tui.screens.execution_paths import ExecutionPathsScreen
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.bars import ratio_bar
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.memory_usage_bar import MemoryUsageBar
from jaull.tui.widgets.metric_list import MetricRow
from jaull.tui.widgets.score_bar import ScoreBar
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.technical_details import TechnicalDetails
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowStep
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _RecommendationRow(Vertical):
    """One recommendation, in the same shape whether or not it is selected.

    The screen used to render #1 as a bordered card and the rest as a single
    dense line each, which made the list impossible to scan as a list: two
    layouts, two typographies, and four buttons repeated on every row. Here
    every row reads the same and the selected one expands in place.

    The expanded parts stay mounted and are hidden with ``display``. Mounting
    and removing them per selection would churn widget ids — the thing three
    separate tests exist to prevent.
    """

    DEFAULT_CLASSES = "rec-row selectable"

    can_focus = True

    BINDINGS = [("enter", "choose", "Select")]

    class Chosen(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(
        self,
        index: int,
        recommendation: ModelRecommendation,
        *,
        selected: bool,
    ) -> None:
        super().__init__(id=f"rec-row-{index}")
        self._index = index
        self._rec = recommendation
        self._selected = selected
        # Filled in once the evidence scan lands; until then the row simply
        # does not claim anything about prior runs.
        self._evidence_summary = ""
        self.set_class(selected, "-selected")

    def compose(self) -> ComposeResult:
        rec = self._rec
        with Horizontal(classes="rec-headline"):
            yield Static(f"{self._index + 1}", classes="rec-rank")
            yield Static(_model_title(rec), classes="rec-name")
            yield Static(
                f"{_status_label(rec)} · {_memory_label(rec)}",
                classes=f"rec-fit {_fit_class(rec)}",
            )
        yield Static(_plan_line(rec), classes="rec-plan")
        yield Static(
            "",
            id=f"rec-evidence-{self._index}",
            classes="rec-evidence status-estimated",
        )
        if rec.reasons:
            yield Static(rec.reasons[0], classes="rec-reason -expanded")
        with Horizontal(classes="rec-actions -expanded"):
            yield Button("Run", id=f"res-run-{self._index}", classes="-primary -compact")
            yield ActionButton(
                "Validate",
                id=f"res-validate-{self._index}",
                disabled=not _can_validate(rec),
            )
            yield ActionButton(
                "Benchmark",
                id=f"res-benchmark-{self._index}",
                disabled=not _can_benchmark(rec),
            )
            yield ActionButton("Paths", id=f"res-paths-{self._index}")

    def on_mount(self) -> None:
        self._apply()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self._apply()

    def show_evidence(self, summary: str, state_class: str) -> None:
        self._evidence_summary = summary
        widget = self.query_one(f"#rec-evidence-{self._index}", Static)
        widget.update(summary)
        widget.set_classes(f"rec-evidence {state_class}")
        widget.display = bool(summary) and self._selected

    def _apply(self) -> None:
        for widget in self.query(".-expanded"):
            widget.display = self._selected
        evidence = self.query_one(f"#rec-evidence-{self._index}", Static)
        evidence.display = self._selected and bool(self._evidence_summary)

    def on_click(self) -> None:
        self.post_message(self.Chosen(self._index))

    def action_choose(self) -> None:
        self.post_message(self.Chosen(self._index))


class _RecommendationEvidenceLoaded(Message):
    def __init__(self, evidence: EvidenceIndex) -> None:
        super().__init__()
        self.evidence = evidence


class RecommendationResultsScreen(Screen[None]):
    """Step 4: the ranked recommendations, and what can be done with one."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "quit", "Quit"),
        Binding("down,j", "move(1)", "Next", key_display="↓/j"),
        Binding("up,k", "move(-1)", "Previous", key_display="↑/k"),
    ]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state
        self._selected = 0
        self._evidence_closing = Event()
        self._evidence_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-results-evidence",
        )
        self._evidence_future: Future[None] | None = None

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
        recommendations = self._state.recommendations
        with Horizontal(classes="rec-list-head"):
            yield Static("Recommendations", classes="section-title")
            yield Static(
                _list_aside(recommendations),
                classes=f"rec-list-aside {_tier_class(recommendations[0])}",
            )
        for index, rec in enumerate(recommendations):
            yield _RecommendationRow(index, rec, selected=index == 0)

    def on_mount(self) -> None:
        if not self._state.recommendations:
            return
        self._evidence_closing.clear()
        self._evidence_future = self._evidence_executor.submit(
            self._evidence_worker,
            self._app().advisor,
        )

    def on_unmount(self) -> None:
        self._evidence_closing.set()
        if self._evidence_future is not None:
            self._evidence_future.cancel()
            self._evidence_future = None
        self._evidence_executor.shutdown(wait=False, cancel_futures=True)

    def _evidence_worker(self, advisor: AdvisorService) -> None:
        """One scan of both record stores, shared by every row."""
        try:
            evidence = EvidenceIndex.load(advisor)
        except Exception:
            return
        if not self._evidence_closing.is_set():
            self.post_message(_RecommendationEvidenceLoaded(evidence))

    @on(_RecommendationEvidenceLoaded)
    def _evidence_loaded(self, message: _RecommendationEvidenceLoaded) -> None:
        if self._evidence_closing.is_set():
            return
        for row in self.query(_RecommendationRow):
            summary, state_class = _evidence_for_recommendation(
                message.evidence,
                self._state.recommendations[row._index],
            )
            row.show_evidence(summary, state_class)

    @on(_RecommendationRow.Chosen)
    def _row_chosen(self, message: _RecommendationRow.Chosen) -> None:
        self._select(message.index)

    def action_move(self, delta: int) -> None:
        if not self._state.recommendations:
            return
        count = len(self._state.recommendations)
        self._select((self._selected + delta) % count)
        rows = list(self.query(_RecommendationRow))
        if 0 <= self._selected < len(rows):
            rows[self._selected].focus()

    def _select(self, index: int) -> None:
        self._selected = index
        for row in self.query(_RecommendationRow):
            row.set_selected(row._index == index)

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
                # The selected row, not always the top one: comparing a model
                # the user is not looking at was simply a bug.
                if self._state.recommendations:
                    self.app.push_screen(
                        ExecutionPathBenchmarkCompareScreen(
                            self._state.recommendations[self._selected]
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
            yield ActionButton("Back", id="compare-back")
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

    @property
    def recommendation(self) -> ModelRecommendation:
        """Which model is being compared, as the other action screens expose it."""
        return self._recommendation

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
            yield ActionButton("Back", id="path-compare-back")
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
        # An empty Static still occupies its row.
        self.query_one("#path-compare-empty", Static).display = False
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

        # `same_hardware` was computed and then ignored: the screen said "This
        # machine" even when the records came from different machines, which is
        # the one thing that would invalidate the whole comparison.
        machine = (
            "same machine" if comparison.same_hardware else "machines differ"
        )
        await container.mount(
            Static(
                f"{model_display_name(comparison.model_identity)} · {machine}",
                classes="section-title",
            )
        )
        await container.mount(_benchmark_plan_table(comparison))
        if comparison.metrics:
            await container.mount(Static("Measured differences", classes="section-title"))
            for widget in _difference_blocks(comparison):
                await container.mount(widget)
        if comparison.warnings:
            await container.mount(_comparison_notes(comparison.warnings))
        await container.mount(
            TechnicalDetails(_comparison_technical_rows(comparison))
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
            yield ActionButton("Back", id="details-back")
        yield Footer()

    def _compose_details(self) -> ComposeResult:
        primary = self._state.recommendations[0]
        yield ScoreBar(primary.score)

        estimate = primary.evaluated.memory_estimate
        if estimate is not None:
            yield SummaryCard("Memory breakdown", _breakdown_rows(estimate))
            if estimate.total_bytes is not None:
                yield MemoryUsageBar(
                    "Available VRAM",
                    estimate.total_bytes,
                    estimate.assessment.available_vram_bytes,
                )
                yield MemoryUsageBar(
                    "Available RAM",
                    estimate.total_bytes,
                    estimate.assessment.available_ram_bytes,
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
        yield SummaryCard("Pipeline", rows)
        yield SummaryCard(
            "Execution paths",
            _execution_path_detail_rows(primary),
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
            )
        yield CliEquivalent(_equivalent_cli_for(primary))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "details-back":
            self.app.pop_screen()


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
                yield ActionButton("Cancel", id="res-export-cancel")
                yield ActionButton("Close", id="res-export-close")

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
            yield ActionButton("Compare", id="res-compare")
            yield ActionButton("Compare paths", id="res-compare-paths")
            yield ActionButton("Details", id="res-details")
            yield ActionButton("Export", id="res-export")
    with Horizontal(id="results-actions-secondary"):
        yield Button("Start again", id="res-restart", classes="-quiet")
        yield Button("Advanced tools", id="res-advanced", classes="-quiet")


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
    table.add_columns("", *labels)
    table.add_row(
        "Artifact",
        *[
            plan.quantization_or_precision or plan.artifact_format or "unknown"
            for plan in comparison.plans
        ],
    )
    metric_map = _benchmark_metric_values(comparison)
    table.add_row(
        "Prompt 128",
        *[
            _tokens_per_second(metric_map.get((plan.label, BenchmarkMeasurementKind.PREFILL, 128)))
            for plan in comparison.plans
        ],
    )
    table.add_row(
        "Prompt 512",
        *[
            _tokens_per_second(metric_map.get((plan.label, BenchmarkMeasurementKind.PREFILL, 512)))
            for plan in comparison.plans
        ],
    )
    table.add_row(
        "Generation",
        *[
            _tokens_per_second(
                _first_generation_metric(metric_map, plan.label),
            )
            for plan in comparison.plans
        ],
    )
    table.add_row("Peak RAM", *[_bytes(plan.peak_ram_bytes) for plan in comparison.plans])
    return table


_COMPARE_BAR_WIDTH = 16


def _difference_blocks(comparison: BenchmarkComparison) -> list[Widget]:
    """One block per metric, each plan on its own bar.

    Two columns of numbers make the reader do the division; bars on a shared
    scale make the size of the difference the first thing you see. The ratio
    beside each block is `BenchmarkComparison.relative_throughput` — the domain
    already computes it, so nothing here invents a comparison.

    There is deliberately no winner, no overall score and no "better": the
    plans differ in artifact and often in methodology, which the notes below
    say plainly.
    """
    values = _benchmark_metric_values(comparison)
    names = {plan.label: _benchmark_plan_label(plan) for plan in comparison.plans}
    blocks: list[Widget] = []
    seen: list[tuple[BenchmarkMeasurementKind, int]] = []
    for metric in comparison.metrics:
        key = (metric.kind, metric.tokens)
        if key in seen:
            continue
        seen.append(key)
        rows: list[tuple[str, float | None]] = [
            (
                _benchmark_plan_label(plan),
                values.get((plan.label, metric.kind, metric.tokens)),
            )
            for plan in comparison.plans
        ]
        blocks.append(
            _bar_group(
                f"{_metric_label(metric.kind, metric.tokens)}"
                f"  ·  {_relative_phrase(metric, names)}",
                rows,
                lambda value: f"{value:.1f} tok/s",
            )
        )

    memory_rows: list[tuple[str, float | None]] = [
        (
            _benchmark_plan_label(plan),
            float(plan.peak_ram_bytes) if plan.peak_ram_bytes is not None else None,
        )
        for plan in comparison.plans
    ]
    if any(value is not None for _, value in memory_rows):
        difference = _memory_phrase(comparison.plans)
        heading = "Peak RAM" + (f"  ·  {difference}" if difference else "")
        blocks.append(_bar_group(heading, memory_rows, lambda v: _bytes(int(v))))
    return blocks


def _relative_phrase(
    metric: BenchmarkPlanMetricComparison,
    names: dict[str, str],
) -> str:
    """Name both sides of the ratio.

    "5.7x lower" leaves the reader to work out lower than what. Saying
    "llama.cpp 5.7x PyTorch" states the measured relationship and still avoids
    declaring a winner — which of the two is preferable depends on constraints
    this screen does not know.
    """
    relative = metric.relative_throughput
    baseline = names.get(metric.baseline_label, metric.baseline_label)
    candidate = names.get(metric.candidate_label, metric.candidate_label)
    if relative <= 0:
        return "not comparable"
    if relative >= 1:
        return f"{candidate} {relative:.1f}x {baseline}"
    return f"{baseline} {1 / relative:.1f}x {candidate}"


def _memory_phrase(plans: list[BenchmarkPlanSummary]) -> str | None:
    if len(plans) < 2:
        return None
    first, second = plans[0], plans[1]
    if not first.peak_ram_bytes or not second.peak_ram_bytes:
        return None
    first_name = _benchmark_plan_label(first)
    second_name = _benchmark_plan_label(second)
    if second.peak_ram_bytes >= first.peak_ram_bytes:
        factor = second.peak_ram_bytes / first.peak_ram_bytes
        return f"{second_name} {factor:.1f}x {first_name}"
    factor = first.peak_ram_bytes / second.peak_ram_bytes
    return f"{first_name} {factor:.1f}x {second_name}"


def _bar_group(
    heading: str,
    rows: list[tuple[str, float | None]],
    render: Callable[[float], str],
) -> Widget:
    """A labelled group of bars sharing one scale, each with its exact value."""
    measured = [value for _, value in rows if value is not None]
    peak = max(measured) if measured else 0.0
    children: list[Widget] = [Static(heading, classes="metric-group")]
    for label, value in rows:
        children.append(
            MetricRow(
                label,
                render(value) if value is not None else "not measured",
                bar=ratio_bar(value, peak, _COMPARE_BAR_WIDTH),
                emphasis="measured" if value is not None else None,
            ).add_class("-indented")
        )
    return Vertical(*children, classes="section")


def _comparison_notes(warnings: list[str]) -> Widget:
    """Visible, but not dominant.

    These are caveats on how to read the numbers above, not failures, so they
    get a marker and muted text rather than the amber warning panel — which
    would shout louder than the measurements it qualifies.
    """
    children: list[Widget] = [Static("Comparison notes", classes="section-title")]
    children.extend(
        Static(f"⚠ {warning}", classes="comparison-note") for warning in warnings
    )
    children.append(
        Static(
            "Results compare complete execution plans, not runtime engines in "
            "isolation.",
            classes="text-muted-tight",
        )
    )
    return Vertical(*children, classes="section")


def _comparison_technical_rows(
    comparison: BenchmarkComparison,
) -> list[tuple[str, str]]:
    """Methodology and provenance, which the screen used to drop entirely."""
    rows: list[tuple[str, str]] = []
    for plan in comparison.plans:
        label = _benchmark_plan_label(plan)
        rows.append((f"{label} benchmark id", plan.benchmark_id))
        rows.append((f"{label} methodology", plan.methodology or "unknown"))
        rows.append((f"{label} runs compared", str(plan.compatible_run_count)))
        if plan.peak_vram_bytes is not None:
            rows.append((f"{label} peak VRAM", _bytes(plan.peak_vram_bytes)))
        if plan.model_load_seconds is not None:
            rows.append((f"{label} model load", f"{plan.model_load_seconds:.2f} s"))
        if plan.time_to_first_token_seconds is not None:
            rows.append(
                (f"{label} TTFT", f"{plan.time_to_first_token_seconds:.2f} s")
            )
    return rows


def _benchmark_plan_label(plan: BenchmarkPlanSummary) -> str:
    if plan.runtime == "transformers":
        return "PyTorch"
    if plan.runtime == "llama.cpp":
        return "llama.cpp"
    return plan.runtime


def _benchmark_metric_values(
    comparison: BenchmarkComparison,
) -> dict[tuple[str, BenchmarkMeasurementKind, int], float]:
    values: dict[tuple[str, BenchmarkMeasurementKind, int], float] = {}
    for metric in comparison.metrics:
        values[
            (metric.baseline_label, metric.kind, metric.tokens)
        ] = metric.baseline_tokens_per_second
        values[
            (metric.candidate_label, metric.kind, metric.tokens)
        ] = metric.candidate_tokens_per_second
    return values


def _first_generation_metric(
    values: dict[tuple[str, BenchmarkMeasurementKind, int], float],
    label: str,
) -> float | None:
    generation = [
        (tokens, value)
        for (plan_label, kind, tokens), value in values.items()
        if plan_label == label and kind is BenchmarkMeasurementKind.GENERATION
    ]
    if not generation:
        return None
    return sorted(generation)[0][1]


def _tokens_per_second(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} tok/s"


def export_report(state: RecommendationWorkflowState, target: Path) -> list[Path]:
    """Backwards-compatible wrapper around :func:`write_recommendation_report`."""
    return write_recommendation_report(state, target)


def _model_title(rec: ModelRecommendation) -> str:
    """The model's name, not its repository path.

    ``Qwen/Qwen2.5-7B-Instruct-GGUF`` is a storage location; the artifact
    format is shown next to it already, and the full path stays in technical
    details.
    """
    identity = ModelIdentity(model_name=rec.repo_id.split("/")[-1])
    return model_display_name(identity, fallback=rec.repo_id)


def _plan_line(rec: ModelRecommendation) -> str:
    """Runtime, artifact and backend — how this would actually execute."""
    from jaull.execution_plans import execution_plan_for_recommendation

    try:
        plan = execution_plan_for_recommendation(rec)
    except ValueError:
        return " · ".join(_recommendation_metadata(rec))
    return plan_summary_line(plan)


def _list_aside(recommendations: list[ModelRecommendation]) -> str:
    """The honest heading for the ranking, plus how many results there are.

    ``choose_tier`` already downgrades "best match" when the evidence is thin;
    the screen used to hard-code the strongest wording regardless.
    """
    count = len(recommendations)
    noun = "model" if count == 1 else "models"
    heading = _tier_heading(recommendations[0])
    return f"{heading} · {count} {noun}" if heading else f"{count} {noun}"


def _tier_heading(rec: ModelRecommendation) -> str:
    try:
        return TIER_HEADINGS[RecommendationTier(rec.tier)].title()
    except ValueError:
        return ""


def _tier_class(rec: ModelRecommendation) -> str:
    try:
        return f"tier-{RecommendationTier(rec.tier).value.replace('_', '-')}"
    except ValueError:
        return "text-muted-tight"


def _evidence_for_recommendation(
    evidence: EvidenceIndex,
    rec: ModelRecommendation,
) -> tuple[str, str]:
    """The evidence line for a recommendation's default execution plan."""
    from jaull.execution_plans import execution_plan_for_recommendation

    try:
        plan = execution_plan_for_recommendation(rec)
    except ValueError:
        return "", "rec-evidence"
    found = evidence.for_plan(plan)
    return found.summary(), state_class(found.state)


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
        values.append(format_gib(estimate.total_bytes))
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
    backend = backend_hint(plan.runtime)
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


def _configuration_label(rec: ModelRecommendation) -> str:
    config = rec.evaluated.selected_configuration
    if config is None:
        return "unknown"
    return config.quantization or (config.precision.value if config.precision else "default")


def _memory_label(rec: ModelRecommendation) -> str:
    estimate = rec.evaluated.memory_estimate
    if estimate is None:
        return "unknown"
    return format_gib(estimate.total_bytes)


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
        ("Weights", format_gib(estimate.weights.component.bytes)),
        ("KV cache", format_gib(estimate.kv_cache.component.bytes)),
        ("Runtime overhead", format_gib(estimate.runtime_overhead.component.bytes)),
        ("Device reserve", format_gib(estimate.device_reserve.bytes)),
    ]
    if estimate.safety_margin is not None:
        rows.append(("Safety margin", format_gib(estimate.safety_margin.bytes)))
    rows.append(("Estimated total", format_gib(estimate.total_bytes)))
    rows.append(("Available VRAM", format_gib(estimate.assessment.available_vram_bytes)))
    rows.append(("Available RAM", format_gib(estimate.assessment.available_ram_bytes)))
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


def _bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    gib = value / 1024**3
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / 1024**2:.1f} MiB"


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
