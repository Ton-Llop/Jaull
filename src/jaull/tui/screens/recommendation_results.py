from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from jaull.recommendation import policies, report
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.memory_usage_bar import MemoryUsageBar
from jaull.tui.widgets.recommendation_card import RecommendationCard
from jaull.tui.widgets.score_bar import ScoreBar
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowStep
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp


class RecommendationResultsScreen(Screen[None]):
    """Step 4: the primary recommendation, its alternatives, and what to do next."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, state: RecommendationWorkflowState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Recommendations",
            "Ranked by how well each model fits this machine and your answers.",
        )
        with VerticalScroll(id="results-body"):
            if not self._state.recommendations:
                yield from self._compose_no_results()
            else:
                yield from self._compose_results()
            with Horizontal(id="results-actions"):
                yield Button("Compare", id="res-compare")
                yield Button("Technical details", id="res-details")
                yield Button("Export report", id="res-export")
                yield Button("Start again", id="res-restart")
                yield Button("Advanced tools", id="res-advanced")
            yield Vertical(id="results-extra")
        yield Footer()

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def _compose_no_results(self) -> ComposeResult:
        yield Static("No compatible models found", classes="card-title")
        reasons = self._state.no_results_reason or [
            "No candidates could be evaluated."
        ]
        yield WarningsPanel(reasons)

    def _compose_results(self) -> ComposeResult:
        primary = self._state.recommendations[0]
        yield RecommendationCard(primary)

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

        if len(self._state.recommendations) > 1:
            yield Static("Alternatives", classes="card-title")
            for alternative in self._state.recommendations[1:]:
                yield RecommendationCard(alternative)

        yield Static(policies.LEGAL_DISCLAIMER, classes="text-muted")

        yield CliEquivalent(_equivalent_cli_for(primary))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self._app()
        match event.button.id:
            case "res-compare":
                self._show_comparison()
            case "res-details":
                self._show_details()
            case "res-export":
                self._prompt_export()
            case "res-restart":
                app.restart_workflow()
            case "res-advanced":
                app.goto_advanced_tools()
            case "res-export-confirm":
                self._do_export()

    def _extra(self) -> Vertical:
        container = self.query_one("#results-extra", Vertical)
        container.remove_children()
        return container

    def _show_comparison(self) -> None:
        container = self._extra()
        if not self._state.recommendations:
            container.mount(Static("Nothing to compare.", classes="text-muted"))
            return
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
        for rec in self._state.recommendations:
            config = rec.evaluated.selected_configuration
            configuration = "unknown"
            if config is not None:
                configuration = config.quantization or (
                    config.precision.value if config.precision else "default"
                )
            estimate = rec.evaluated.memory_estimate
            memory = (
                _gib(estimate.total_bytes)
                if estimate is not None and estimate.total_bytes is not None
                else "unknown"
            )
            table.add_row(
                rec.repo_id,
                configuration,
                memory,
                rec.status.value,
                f"{rec.score.out_of_100}/100",
                rec.confidence.value,
                rec.evaluated.candidate.license or "not declared",
                rec.reasons[0] if rec.reasons else "-",
            )
        container.mount(Static("Comparison", classes="card-title"))
        container.mount(table)

    def _show_details(self) -> None:
        container = self._extra()
        if not self._state.recommendations:
            container.mount(Static("No technical details.", classes="text-muted"))
            return
        primary = self._state.recommendations[0]
        container.mount(Static("Technical details", classes="card-title"))
        container.mount(ScoreBar(primary.score))

        rows: list[tuple[str, str]] = [
            ("Configuration chosen", primary.evaluated.configuration_reason or "-"),
            (
                "Variants considered",
                ", ".join(primary.evaluated.alternatives_considered) or "-",
            ),
            ("Search queries", str(len(self._state.search_queries))),
            ("Candidates found", str(len(self._state.candidates))),
            ("Candidates evaluated", str(len(self._state.evaluated_candidates))),
        ]
        estimate = primary.evaluated.memory_estimate
        if estimate is not None:
            rows.append(("Repository type", estimate.repository_type.value))
            if estimate.architecture:
                rows.append(("Architecture", estimate.architecture))
            base = estimate.base_model_resolution
            if base is not None and base.repo_id:
                rows.append(("Base model", f"{base.repo_id} ({base.source.value})"))
        container.mount(SummaryCard("Pipeline", rows))

        if self._state.requirements is not None:
            container.mount(
                SummaryCard(
                    "Assumptions",
                    [
                        (f"{index}", assumption)
                        for index, assumption in enumerate(
                            self._state.requirements.assumptions, start=1
                        )
                    ]
                    or [("-", "none recorded")],
                )
            )

    def _prompt_export(self) -> None:
        container = self._extra()
        container.mount(Static("Export report", classes="card-title"))
        container.mount(
            Static(
                "Writes JSON, plus a Markdown summary alongside it. "
                "No tokens or credentials are included.",
                classes="text-muted",
            )
        )
        container.mount(
            Input(value="jaull-report.json", id="res-export-path")
        )
        container.mount(Button("Write file", id="res-export-confirm", classes="-primary"))

    def _do_export(self) -> None:
        target = Path(self.query_one("#res-export-path", Input).value.strip())
        container = self._extra()
        try:
            written = export_report(self._state, target)
        except OSError as exc:
            container.mount(WarningsPanel([f"Could not write the report: {exc}"]))
            return
        container.mount(Static("Report written", classes="card-title"))
        container.mount(
            SummaryCard("Files", [(path.suffix.lstrip("."), str(path)) for path in written])
        )

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


def export_report(
    state: RecommendationWorkflowState, target: Path
) -> list[Path]:
    """Write the JSON report and a Markdown twin. Returns the paths written."""
    json_path = target if target.suffix == ".json" else target.with_suffix(".json")
    markdown_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.report_to_json(state), encoding="utf-8")
    markdown_path.write_text(report.report_to_markdown(state), encoding="utf-8")
    return [json_path, markdown_path]


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


def _equivalent_cli_for(primary) -> str:  # type: ignore[no-untyped-def]
    """Prefer the actual runtime command; fall back to `jaull estimate`."""
    from jaull.domain.runtime import RuntimeName
    from jaull.recommendation.models import ModelRecommendation

    assert isinstance(primary, ModelRecommendation)
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


__all__ = ["RecommendationResultsScreen", "export_report"]
