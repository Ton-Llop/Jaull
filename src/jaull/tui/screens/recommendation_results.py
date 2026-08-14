from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Input, Static

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

        match button_id:
            case "res-compare":
                self.app.push_screen(RecommendationCompareScreen(self._state))
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
        with Horizontal(classes="actions-right"):
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


__all__ = [
    "ExportReportModal",
    "RecommendationCompareScreen",
    "RecommendationDetailsScreen",
    "RecommendationResultsScreen",
    "export_report",
]
