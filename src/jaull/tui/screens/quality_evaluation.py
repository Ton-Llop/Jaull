"""Explicit, bounded quality pilot; no work starts on screen entry."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from uuid import uuid4

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Footer,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from jaull.advisor.quality import QualityReadiness
from jaull.advisor.quality_candidates import QualityCandidateSelection
from jaull.domain.requirements import RecommendationPriority
from jaull.evaluation.quality_records import describe_record
from jaull.paths import user_data_dir
from jaull.recommendation.quality import requested_profile
from jaull.runtime.quality_eval_runner import (
    DATASET_BYTES,
    PROFILE_SUITE,
    QualityEvaluationCancelled,
    QualityEvaluationError,
    QualityProfile,
    QualityRunRequest,
    default_dataset,
)
from jaull.tui import palette
from jaull.tui.evidence import EvidenceState, PlanEvidence
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar
from jaull.tui.widgets.technical_details import TechnicalDetails

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jaull.advisor.quality_candidates import QualityCandidate
    from jaull.advisor.service import AdvisorService
    from jaull.domain.execution_plans import ExecutionPlan
    from jaull.evaluation.quality_records import QualityEvidence
    from jaull.recommendation.quality import StoredQuality
    from jaull.recommendation.shadow import ShadowReport
    from jaull.tui.app import JaullApp
    from jaull.workflow.state import RecommendationWorkflowState

_log = logging.getLogger(__name__)

_HELLASWAG = "HellaSwag / raw prompts / zero-shot / ctx 2048 / CUDA0 / full offload"
_PROTOCOL = {
    QualityProfile.SMOKE: _HELLASWAG,
    QualityProfile.HELLASWAG100: _HELLASWAG,
    QualityProfile.IFEVAL: (
        "IFEval / 541 prompts / each GGUF's own chat template / reasoning off / ctx 4096 / "
        "CUDA0 / full offload"
    ),
}
_PROFILE_LABELS = {
    QualityProfile.SMOKE: "HellaSwag - 3 samples / smoke",
    QualityProfile.HELLASWAG100: "HellaSwag - 100 samples / limited",
    QualityProfile.IFEVAL: "IFEval - 541 prompts / full split",
}


def recompare_readout(
    report: ShadowReport, names: Mapping[str, str], *, shown_order: Sequence[str] | None = None,
) -> str:
    """Say what the proposal would change, and that the shown result does not."""
    lines = [
        f"{report.priority.value.capitalize()} / {report.policy_version}. "
        "The shown recommendations are unchanged."
    ]
    no_swaps = "No pool swaps" if shown_order is not None else "No moves"
    if shown_order is not None:
        if tuple(shown_order) == report.shadow_top5:
            lines.append("No changes to the shown recommendations.")
        else:
            lines.append("Proposed changes to the shown recommendations:")
            positions = {plan_id: i + 1 for i, plan_id in enumerate(shown_order)}
            for new_position, plan_id in enumerate(report.shadow_top5, start=1):
                name = names.get(plan_id, plan_id)
                old_position = positions.get(plan_id)
                if old_position is None:
                    lines.append(f"  Top 5 entered: {name} (#{new_position})")
                elif old_position != new_position:
                    lines.append(f"  #{old_position} -> #{new_position}  {name}")
            lines.extend(
                f"  Top 5 left: {names.get(plan_id, plan_id)}"
                for plan_id in shown_order if plan_id not in report.shadow_top5
            )
    if not report.applied:
        lines.append("This priority keeps the active order; there is nothing to propose.")
    elif report.moves:
        if shown_order is None:
            lines.append("Pool moves from the base order (before Quality/diversity):")
            lines.extend(
                f"  #{move.base_index + 1} -> #{move.shadow_index + 1}  {move.repo_id}: {move.rule}"
                for move in report.moves
            )
        else:
            lines.append(f"{len(report.moves)} pool swaps from the base order (before diversity).")
    elif report.groups:
        lines.append(f"{no_swaps}: comparable evidence preserves the base order.")
    elif report.priority is RecommendationPriority.BALANCED:
        # shadow-v1 groups Balanced plans only on both axes, and speed never applies yet.
        lines.append(
            f"{no_swaps}: no speed measurement applies yet. "
            "Use the Quality priority for stored quality alone."
        )
    else:
        lines.append(
            f"{no_swaps}: no comparable pair in the same eligibility stratum."
        )
    if shown_order is None:
        for change in report.top5_changes:
            name = names.get(change.plan_id, change.plan_id)
            how = "moved" if change.moved else "diversity, not moved itself"
            lines.append(f"  Base top 5 {change.change}: {name} ({how})")
    lines.append(f"{len(report.fallback)} plans kept their fallback positions.")
    return "\n".join(lines)


_READY_STYLE = {
    "ok": ("✓", f"bold {palette.OK}"),
    "warn": ("!", f"bold {palette.WARN}"),
    "bad": ("✗", f"bold {palette.BAD}"),
}


def readiness_markup(rows: tuple[tuple[str, str, str], ...]) -> str:
    """One line per check: glyph and colour carry the state, the words say what to do."""
    lines = []
    for state, name, detail in rows:
        glyph, style = _READY_STYLE[state]
        lines.append(
            f"[{style}]{glyph}[/] [bold]{escape(name)}[/]  "
            f"[{palette.INK_2}]{escape(detail)}[/]"
        )
    return "\n".join(lines)


METRIC_LABELS = {
    "acc": "Accuracy", "acc_norm": "Length-normalized accuracy",
    "prompt_level_strict_acc": "Prompt accuracy / strict",
    "prompt_level_loose_acc": "Prompt accuracy / loose",
    "inst_level_strict_acc": "Instruction accuracy / strict",
    "inst_level_loose_acc": "Instruction accuracy / loose",
}
_BAR = 8
#: Table labels: short enough that two models fit beside them at 80 columns.
_SHORT_LABELS = {
    "prompt_level_strict_acc": "Prompt strict", "prompt_level_loose_acc": "Prompt loose",
    "inst_level_strict_acc": "Instr. strict", "inst_level_loose_acc": "Instr. loose",
    "acc_norm": "Acc. normalized",
}


def _bar(value: float) -> str:
    filled = round(max(0.0, min(1.0, value)) * _BAR)
    return "█" * filled + "░" * (_BAR - filled)


def not_comparable(evidence: list[QualityEvidence]) -> str | None:
    """Why these runs cannot be ranked against each other, or None when they can.

    The fields are the comparator's own (``quality_comparison.comparison_fields``),
    so the screen marks a winner exactly when ``jaull quality compare`` would compare.
    """
    if len(evidence) < 2:
        return None
    fields = [dict(record.comparison) for record in evidence]
    if not all(fields):
        return "Not comparable: a run lacks its comparison fields; no winner is marked."
    differing = [name for name in fields[0] if len({row.get(name) for row in fields}) > 1]
    if not differing:
        return None
    return (
        f"Not comparable: {', '.join(differing).replace('_', ' ')} "
        f"{'differs' if len(differing) == 1 else 'differ'}; no winner is marked."
    )


def comparison_table(records: tuple[tuple[str, QualityEvidence], ...]) -> DataTable[Text]:
    """Models side by side, one column each; the best value of each metric in green.

    A winner is marked only when every run shares the comparator's protocol; a
    metric a model did not measure shows a dash instead of borrowing a number.
    """
    table: DataTable[Text] = DataTable(
        id="quality-comparison", cursor_type="none", zebra_stripes=True,
    )
    names = [name for name, _ in records]
    table.add_column(Text(""), key="metric")
    for index, name in enumerate(names):
        label = name if names.count(name) == 1 else f"{name} #{index + 1}"
        table.add_column(Text(label, style="bold"), key=f"model-{index}")
    evidence = [record for _, record in records]
    dim = palette.INK_2
    # Always each result's own suite: the picker above may already show another one.
    table.add_row(Text("Suite", style=dim), *(Text(r.suite) for r in evidence))
    comparable = not_comparable(evidence) is None
    table.add_row(Text("Grade", style=dim), *(
        Text(r.classification.capitalize()) for r in evidence
    ))
    table.add_row(Text("Samples", style=dim), *(
        Text(f"{r.samples_used}/{r.samples_available}") for r in evidence
    ))
    metrics = [{m.name: m for m in r.metrics} for r in evidence]
    order = list(dict.fromkeys(name for row in metrics for name in row))
    for metric in order:
        values = [row[metric].value for row in metrics if metric in row]
        best = max(values) if comparable and len(set(values)) > 1 else None
        cells = []
        for row in metrics:
            if metric not in row:
                cells.append(Text("—", style=dim))
                continue
            item = row[metric]
            style = f"bold {palette.OK}" if item.value == best else ""
            cell = Text(_bar(item.value) + " ", style=palette.OK if style else dim)
            cell.append(f"{item.value:.1%}", style=style)
            cells.append(cell)
        table.add_row(Text(_SHORT_LABELS.get(metric, METRIC_LABELS.get(metric, metric))), *cells)
    return table


def stored_records(
    candidates: tuple[QualityCandidate, ...], ran: frozenset[str] = frozenset(),
) -> tuple[tuple[str, QualityEvidence], ...]:
    """The applicable stored result of each candidate not run now; never a conflict."""
    return tuple(
        (f"{candidate.plan.model_identity.model_name} (stored)", candidate.stored.applicable)
        for candidate in candidates
        if candidate.stored.applicable is not None and candidate.plan.plan_id not in ran
    )


def stored_badge(stored: StoredQuality) -> tuple[str, str] | None:
    """(text, class) for a candidate whose rerun could not change what applies."""
    if stored.applicable is not None:
        repeats = (f"; {stored.complete} runs, identical answers"
                   if stored.complete > 1 else "")
        return (
            f"Already measured ({stored.applicable.summary}{repeats}); unchecked, its "
            "stored result is compared in Results.", "quality-measured",
        )
    if stored.conflict:
        return (
            f"{stored.complete} complete runs disagree ({stored.distinct} results); none "
            "applies and a rerun cannot settle it. Unchecked.", "status-fail",
        )
    return None


def _size(bytes_: int) -> str:
    return f"{bytes_ / 1024**3:.2f} GiB" if bytes_ >= 1024**3 // 10 else (
        f"{bytes_ / 1024**2:.1f} MB"
    )


class _QuietCollapsible(Collapsible):
    """A Collapsible that does not scroll itself into view when toggled.

    Textual queues ``scroll_visible`` after every change of ``collapsed``. The run
    report is the last widget of Results, so folding it after a run dragged the
    pane to its bottom, racing the scroll home that shows the measurements first.
    Which one won depended on how many refreshes happened in between. Same body as
    Textual's watcher (textual/widgets/_collapsible.py) minus that scroll.
    """

    def _watch_collapsed(self, collapsed: bool) -> None:
        self._update_collapsed(collapsed)
        self.post_message(self.Collapsed(self) if self.collapsed else self.Expanded(self))


class _Readiness(Message):
    def __init__(self, readiness: QualityReadiness) -> None:
        super().__init__()
        self.readiness = readiness


class _QualityFinished(Message):
    def __init__(
        self, path: Path | None, readout: str, *,
        records: tuple[tuple[str, QualityEvidence], ...] = (), errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.path = path
        self.readout = readout
        self.records = records
        self.errors = errors


class _QualityProgress(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class _QualitySetupFinished(Message):
    def __init__(self, ready: bool, text: str) -> None:
        super().__init__()
        self.ready = ready
        self.text = text


class _Recompared(Message):
    def __init__(self, text: str, report: ShadowReport | None = None) -> None:
        super().__init__()
        self.text = text
        self.report = report


class _QualityCandidatesReady(Message):
    def __init__(self, selection: QualityCandidateSelection) -> None:
        super().__init__()
        self.selection = selection


class QualityEvaluationScreen(Screen[Path | None]):
    BINDINGS = [("escape", "back", "Back"), ("q", "quit", "Quit")]

    DEFAULT_CSS = """
    QualityEvaluationScreen #quality-header { height: auto; padding: 0 2; }
    QualityEvaluationScreen #quality-header .section-title { padding-bottom: 0; }
    QualityEvaluationScreen #quality-profile { margin: 0; }
    QualityEvaluationScreen .quality-permission { height: 1; margin: 0; }
    QualityEvaluationScreen #quality-protocol { margin-top: 1; }
    QualityEvaluationScreen #quality-tabs { height: 1fr; }
    QualityEvaluationScreen ContentSwitcher { height: 1fr; }
    QualityEvaluationScreen TabPane { height: 1fr; padding: 0; }
    QualityEvaluationScreen VerticalScroll { padding: 1 2; }
    QualityEvaluationScreen #quality-results-body { padding: 0 2; }
    QualityEvaluationScreen #quality-report > Contents { padding: 0 0 1 0; }
    QualityEvaluationScreen Input, QualityEvaluationScreen Select { margin-bottom: 1; }
    QualityEvaluationScreen #quality-actions { height: 3; padding: 0 2; }
    QualityEvaluationScreen #quality-status { margin: 0 2; max-height: 4; }
    QualityEvaluationScreen #quality-result { height: auto; }
    QualityEvaluationScreen #quality-plan-diagnostics { height: auto; }
    QualityEvaluationScreen #quality-plan-diagnostics .technical-details { margin: 0 0 1 0; }
    QualityEvaluationScreen #quality-candidates { height: auto; }
    QualityEvaluationScreen .quality-row {
        height: auto; padding: 0 1; margin-bottom: 1;
    }
    QualityEvaluationScreen .quality-row Horizontal { height: auto; }
    QualityEvaluationScreen .quality-row Checkbox { width: 1fr; height: 1; }
    QualityEvaluationScreen .quality-storage {
        width: 24; text-align: right;
    }
    QualityEvaluationScreen .quality-row .technical-details { margin: 0; }
    QualityEvaluationScreen .quality-row .text-secondary { padding-left: 2; }
    """

    def __init__(
        self, plan: ExecutionPlan | None = None, *,
        search_state: RecommendationWorkflowState | None = None,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._search_state = search_state
        self._selection = QualityCandidateSelection()
        self._selecting = search_state is not None
        self._cancel = Event()
        self._quality_closing = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jaull-quality")
        self._busy = False
        self._leave = False
        self._quit = False
        self._stored: Path | None = None
        requirements = search_state.requirements if search_state is not None else None
        # A search offers IFEval only when its assessment would accept the result.
        self._ifeval_offered = search_state is None or (
            requirements is not None
            and requested_profile(requirements) == "ifeval-instructions-en-v1"
        )
        self._profile = (
            QualityProfile.IFEVAL if search_state is not None and self._ifeval_offered
            else QualityProfile.SMOKE
        )
        self._hellaswag_dataset = ""
        self._auto_dataset: str | None = None
        # Filled by the automatic check; Start waits for it and refuses on a red row.
        self._checking = False
        self._blocked = False
        self._block_reason = ""
        self._dataset_missing = False

    @property
    def candidate_plans(self) -> tuple[ExecutionPlan, ...]:
        return tuple(candidate.plan for candidate in self._selection.candidates)

    def compose(self) -> ComposeResult:
        yield ContextBar(
            "Evaluate search candidates" if self._search_state else "Quality evaluation",
            aside="Quality / diagnostic",
        )
        with Vertical(id="quality-header"):
            if self._plan is not None:
                yield Static(
                    f"{self._plan.model_identity.model_name} / {self._plan.artifact.label}",
                    classes="section-title", markup=False,
                )
            yield Static("Evaluation suite", classes="field-label")
            yield Select(
                [(_PROFILE_LABELS[profile], profile.value) for profile in QualityProfile
                 if profile is not QualityProfile.IFEVAL or self._ifeval_offered],
                value=self._profile.value, allow_blank=False, id="quality-profile",
            )
        with TabbedContent(id="quality-tabs", initial="quality-selection-tab"):
            with TabPane(
                "Candidates" if self._search_state else "Artifact", id="quality-selection-tab",
            ), VerticalScroll(id="quality-body"):
                yield Static("Readiness", classes="field-label")
                yield Static("Checking the evaluator...", id="quality-readiness")
                yield Static("", id="quality-downloads")
                if self._search_state is not None:
                    yield Vertical(id="quality-candidates")
                if self._plan is None and self._search_state is None:
                    yield Static("Verified artifact manifest (JSON)", classes="field-label")
                    yield Input(
                        placeholder="Verified artifact manifest (JSON)", id="quality-artifact",
                    )
                if self._plan is not None or self._search_state is not None:
                    size = self._plan.artifact.size_bytes if self._plan else None
                    if size is not None:
                        yield Static(
                            f"GGUF size: {size / 1024**3:.2f} GiB", classes="text-secondary",
                        )
                    if self._plan is not None:
                        yield TechnicalDetails(rows=(
                            ("Repository", self._plan.artifact.repo_id),
                            ("File", self._plan.artifact.filename or "unknown"),
                            ("SHA256", self._plan.artifact.sha256 or "unknown"),
                        ), title="Exact artifact")
                yield Static(
                    "Sequential execution / exact artifacts / recommendations unchanged",
                    classes="text-secondary",
                )
            with TabPane("Settings", id="quality-setup-tab"), VerticalScroll(
                id="quality-settings-body",
            ):
                yield Static("Dataset file", classes="field-label")
                yield Input(placeholder="Evaluation dataset", id="quality-dataset")
                yield Static("Download permissions", classes="field-label")
                yield Checkbox(
                    "Download dataset if missing", id="quality-dataset-download", value=False,
                    classes="quality-permission",
                )
                if self._plan is not None or self._search_state is not None:
                    yield Checkbox(
                        "Download models if missing", id="quality-download", value=False,
                        classes="quality-permission",
                    )
                with Collapsible(
                    title="Advanced / server, Docker and output", collapsed=True,
                    id="quality-setup",
                ):
                    yield Static(
                        "Filled in automatically; change a field only to override it.",
                        classes="text-secondary",
                    )
                    for key, label, value in (
                        ("server", "Pinned llama-server executable", ""),
                        ("root", "Trusted pilot checkout", ""),
                        ("image", "Local evaluator image", "jaull-quality-eval:gguf-v1"),
                        ("output", "New output directory",
                         str(user_data_dir("quality-runs") / uuid4().hex)),
                    ):
                        yield Static(label, classes="field-label")
                        yield Input(value=value, placeholder=label, id=f"quality-{key}")
                yield Static(
                    self._protocol_text(), classes="text-secondary", id="quality-protocol",
                )
                yield Static(
                    "Fit and effective protocol are checked before evaluation. "
                    "Runtime duration is not inference speed.", classes="text-secondary",
                )
            with TabPane("Results", id="quality-results-tab"), VerticalScroll(
                id="quality-results-body",
            ):
                if self._search_state is not None:
                    yield ActionButton(
                        "Recompare this search with stored quality", id="quality-recompare",
                    )
                yield Vertical(id="quality-measurements")
                with _QuietCollapsible(
                    title="Run report", id="quality-report", collapsed=False,
                ):
                    yield Static(
                        "No evaluation results in this session.", id="quality-result",
                        markup=False,
                    )
                yield Vertical(id="quality-plan-diagnostics")
                # Last on purpose: cleaning up comes after reading the results.
                yield ActionButton("Free disk space", id="quality-storage")
        yield Static("", id="quality-status", classes="status-line", markup=False)
        with Horizontal(id="quality-actions"):
            yield ActionButton("Prepare evaluator", id="quality-prepare")
            yield Button("Start evaluation", id="quality-start", classes="-primary")
            yield ActionButton("Cancel", id="quality-cancel", disabled=True)
            yield ActionButton("Back", id="quality-back")
        yield Footer()

    def on_mount(self) -> None:
        self._set_busy(False)
        self._set_status("")
        try:
            setup = self._app().advisor.quality_evaluation_setup()
            for key, value in setup.items():
                self.query_one(f"#quality-{key}", Input).value = value
        except (OSError, ValueError, QualityEvaluationError) as exc:
            _log.warning("Could not restore quality setup", exc_info=True)
            self._set_status(f"Could not restore evaluator setup: {exc}")
        self._hellaswag_dataset = self.query_one("#quality-dataset", Input).value
        self._apply_profile_dataset()
        self._check_readiness()
        if self._search_state is not None:
            self._start_selection()

    def _protocol_text(self) -> str:
        return (
            f"{_PROTOCOL[self._profile]}\n"
            "Evaluation protocol is separate from the selected execution plan."
        )

    def _apply_profile_dataset(self) -> None:
        """Point at this profile's pinned dataset, never over a path the user typed."""
        dataset = self.query_one("#quality-dataset", Input)
        if self._auto_dataset is not None and dataset.value != self._auto_dataset:
            return
        value = (
            str(default_dataset(QualityProfile.IFEVAL))
            if self._profile is QualityProfile.IFEVAL else self._hellaswag_dataset
        )
        dataset.value = self._auto_dataset = value

    def _check_readiness(self) -> None:
        """Resolve and verify the evaluator locally; never download, pull or build."""
        values = {
            key: self.query_one(f"#quality-{key}", Input).value.strip()
            for key in ("dataset", "server", "root", "image", "output")
        }
        self._checking = True
        self._set_busy(self._busy)
        self.query_one("#quality-readiness", Static).update(
            f"[{palette.INK_2}]Checking the evaluator for this suite...[/]",
        )
        self._executor.submit(
            self._readiness_worker, self._app().advisor, self._profile, values,
        )

    def _readiness_worker(
        self, advisor: AdvisorService, profile: QualityProfile, values: dict[str, str],
    ) -> None:
        try:
            readiness = advisor.quality_readiness(
                profile, values, is_cancelled=self._cancel.is_set,
            )
        except QualityEvaluationCancelled:
            return
        except Exception as exc:
            _log.exception("Quality readiness check failed")
            readiness = QualityReadiness((("bad", "Readiness", str(exc)),))
        if not self._quality_closing.is_set():
            self.post_message(_Readiness(readiness))

    @on(_Readiness)
    def _readiness_ready(self, message: _Readiness) -> None:
        readiness = message.readiness
        if readiness.image:
            self.query_one("#quality-image", Input).value = readiness.image
        self._checking = False
        self._blocked = readiness.blocked
        self._dataset_missing = readiness.dataset_missing
        self.query_one("#quality-readiness", Static).update(readiness_markup(readiness.rows))
        self._update_downloads()
        self._set_busy(self._busy)
        # First sentence only: the red row above already carries the full fix.
        self._block_reason = next(
            (f"Not ready: {detail.split('. ')[0].rstrip('.')}. See the red row."
             for state, _, detail in readiness.rows if state == "bad"), "",
        )
        if self._blocked and not self._busy:
            self._set_status(self._block_reason)

    def _update_downloads(self) -> None:
        parts = []
        if self._dataset_missing:
            parts.append(f"dataset {_size(DATASET_BYTES[PROFILE_SUITE[self._profile]])}")
        pending = [
            candidate.plan.artifact.size_bytes or 0
            for index, candidate in enumerate(self._selection.candidates)
            if not candidate.downloaded and self.query(f"#quality-candidate-{index}")
            and self.query_one(f"#quality-candidate-{index}", Checkbox).value
        ]
        if pending:
            parts.append(f"{len(pending)} model{'s' if len(pending) > 1 else ''} "
                         f"{_size(sum(pending))}")
        downloads = self.query_one("#quality-downloads", Static)
        if not parts:
            downloads.update(f"[{palette.OK}]✓[/] [{palette.INK_2}]Nothing to download.[/]")
            return
        def permitted(name: str) -> bool:
            boxes = self.query(f"#quality-{name}")
            return bool(boxes) and self.query_one(f"#quality-{name}", Checkbox).value

        # Each permission matters only for what it would actually download.
        allowed = (not self._dataset_missing or permitted("dataset-download")) and (
            not pending or permitted("download")
        )
        style = palette.OK if allowed else palette.WARN
        permission = (
            "permitted" if allowed else "needs permission in Settings > Download permissions"
        )
        downloads.update(
            f"[bold {style}]{'✓' if allowed else '!'}[/] To download: "
            f"{escape(' / '.join(parts))}  [{style}]{permission}[/]"
        )

    def _start_selection(self) -> None:
        self._cancel.clear()
        self._selecting = True
        self._busy = True
        self._set_busy(True)
        self._set_status("Inspecting candidate metadata; no models downloaded")
        self._executor.submit(self._select_candidates, self._app().advisor, self._profile)

    @on(Select.Changed, "#quality-profile")
    async def _profile_changed(self, event: Select.Changed) -> None:
        profile = QualityProfile(str(event.value))
        if profile is self._profile:
            return
        self._profile = profile
        self.query_one("#quality-protocol", Static).update(self._protocol_text())
        self._apply_profile_dataset()
        self._check_readiness()
        # Results belong to the suite they ran; never leave them under another one.
        await self._show_measurements(())
        self.query_one("#quality-result", Static).update("No evaluation results in this session.")
        if self._search_state is not None and not self._busy:
            # Candidates depend on the protocol: its context, and for IFEval the
            # exact shown plans. Never run a selection made for another profile.
            self._selection = QualityCandidateSelection()
            await self.query_one("#quality-candidates", Vertical).remove_children()
            self._start_selection()

    def _select_candidates(self, advisor: AdvisorService, profile: QualityProfile) -> None:
        assert self._search_state is not None
        def progress(text: str) -> None:
            if not self._quality_closing.is_set():
                self.post_message(_QualityProgress(text))
        try:
            selection = advisor.prepare_quality_candidates(
                self._search_state, profile=profile, is_cancelled=self._cancel.is_set,
                on_progress=progress,
            )
        except Exception as exc:
            _log.exception("Quality candidate selection failed")
            selection = QualityCandidateSelection(notices=(f"Selection failed: {exc}",))
        if not self._quality_closing.is_set():
            self.post_message(_QualityCandidatesReady(selection))

    @on(_QualityCandidatesReady)
    async def _candidates_ready(self, message: _QualityCandidatesReady) -> None:
        self._selection = message.selection
        self._selecting = False
        container = self.query_one("#quality-candidates", Vertical)
        await container.remove_children()
        await container.mount(Static(
            f"{len(message.selection.candidates)} candidates from "
            f"{message.selection.considered} eligible inspected models",
            classes="section-title", markup=False,
        ))
        for index, candidate in enumerate(message.selection.candidates):
            plan = candidate.plan
            size = (
                f"{plan.artifact.size_bytes / 1024**3:.2f} GiB"
                if plan.artifact.size_bytes is not None else "size unknown"
            )
            await container.mount(Vertical(
                Horizontal(
                    Checkbox(
                        f"{index + 1}. {plan.model_identity.model_name}",
                        value=not candidate.settled, id=f"quality-candidate-{index}",
                    ),
                    Static(
                        f"{'Local' if candidate.downloaded else 'Download'} / {size}",
                        classes="quality-storage" + (" -local" if candidate.downloaded else ""),
                        markup=False,
                    ),
                ),
                Static(plan.artifact.label, markup=False, classes="text-secondary"),
                *([Static(
                    "GGUF path of the shown model, not the shown plan",
                    classes="quality-alternative", markup=False,
                )] if candidate.reason.startswith("GGUF path of the shown model") else []),
                *([Static(badge[0], classes=badge[1], markup=False)]
                  if (badge := stored_badge(candidate.stored)) else []),
                TechnicalDetails(
                    title="Artifact and selection details",
                    extra=[Static(
                        f"{plan.artifact.repo_id} / {plan.artifact.filename}\n"
                        f"{candidate.reason}\n"
                        f"Historical records: {candidate.historical_records} (not reused).",
                        markup=False,
                    )],
                ),
                classes="quality-row" + ("" if candidate.settled else " -selected"),
                id=f"quality-row-{index}",
            ))
            self.query_one(f"#quality-candidate-{index}", Checkbox).tooltip = (
                plan.model_identity.model_name
            )
        # Replaced even when empty: an earlier suite's table must not linger.
        await self._show_measurements(stored_records(message.selection.candidates))
        if message.selection.notices:
            with_details = Collapsible(
                Static("\n".join(message.selection.notices), markup=False),
                title="Selection details", collapsed=bool(message.selection.candidates),
            )
            await container.mount(with_details)
        self._busy = False
        self._update_downloads()
        self._set_busy(False)
        # A readiness blocker outranks the selection summary; never hide why Start is off.
        self._set_status(
            self._block_reason if self._blocked
            else "Review candidates before starting; execution is sequential."
            if message.selection.candidates else "No models available for this evaluation protocol."
        )
        if self._quit:
            self.app.exit()
        elif self._leave:
            self.dismiss(self._stored)

    def on_unmount(self) -> None:
        self._quality_closing.set()
        self._cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @on(Checkbox.Changed)
    def _candidate_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id or ""
        if checkbox_id.startswith("quality-candidate-"):
            row = self.query_one("#" + checkbox_id.replace("candidate", "row"), Vertical)
            row.set_class(event.value, "-selected")
        if checkbox_id.startswith("quality-candidate-") or checkbox_id in {
            "quality-dataset-download", "quality-download",
        }:
            self._update_downloads()

    def _checked(self) -> tuple[QualityCandidate, ...]:
        return tuple(
            candidate for index, candidate in enumerate(self._selection.candidates)
            if self.query_one(f"#quality-candidate-{index}", Checkbox).value
        )

    def _request(self, *, require_artifact: bool = True) -> QualityRunRequest:
        values = {
            key: self.query_one(f"#quality-{key}", Input).value.strip()
            for key in ("artifact", "dataset", "server", "root", "image", "output")
            if key != "artifact" or (
                self._plan is None and self._search_state is None and require_artifact
            )
        }
        missing = [
            self.query_one(f"#quality-{key}", Input).placeholder
            for key, value in values.items() if not value
        ]
        if missing:
            raise ValueError("Missing: " + "; ".join(missing) + ".")
        return QualityRunRequest(
            artifact_json=Path(values["artifact"]) if "artifact" in values else None,
            dataset_file=Path(values["dataset"]),
            llama_server=Path(values["server"]), pilot_root=Path(values["root"]),
            image=values["image"], output=Path(values["output"]),
            profile=QualityProfile(str(self.query_one("#quality-profile", Select).value)),
            allow_dataset_download=self.query_one("#quality-dataset-download", Checkbox).value,
        )

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quality-back":
            self.action_back()
        elif event.button.id == "quality-storage":
            if self._busy:
                # A running evaluation may be reading the very file one would delete.
                self._set_status("Wait for the evaluation to finish before freeing space.")
                return
            from jaull.tui.screens.storage import StorageScreen

            def returned(_: None) -> None:
                # Local/Download marks may be stale after a deletion.
                if self._search_state is not None and not self._busy:
                    self._start_selection()

            self.app.push_screen(StorageScreen(), returned)
        elif event.button.id == "quality-recompare" and not self._busy:
            self._busy = True
            self._set_busy(True)
            self._set_status("Recomparing the saved search pool with stored quality")
            self._executor.submit(self._recompare, self._app().advisor)
        elif event.button.id == "quality-cancel":
            self._request_cancel()
        elif event.button.id == "quality-start" and not self._busy and self._checking:
            # The check takes a moment; a red row disables Start with its reason instead.
            self._set_status("Still checking the evaluator; start again in a moment.")
        elif (event.button.id == "quality-start" and not self._busy
              and self._search_state is not None and not self._checked()):
            # Before the path checks: with everything measured there is nothing to run.
            self._set_status(
                "Every candidate is already measured or in conflict; see each badge."
                if all(c.settled for c in self._selection.candidates)
                else "Select at least one candidate."
            )
        elif event.button.id in ("quality-start", "quality-prepare") and not self._busy:
            preparing = event.button.id == "quality-prepare"
            try:
                request = self._request(require_artifact=not preparing)
            except ValueError as exc:
                self._set_status(str(exc))
                for widget in self.query(Input):
                    if preparing and widget.id == "quality-artifact":
                        continue
                    if not widget.value.strip():
                        if widget.id in (
                            "quality-server", "quality-root", "quality-image", "quality-output",
                        ):
                            self.query_one("#quality-setup", Collapsible).collapsed = False
                        self.query_one("#quality-tabs", TabbedContent).active = (
                            "quality-selection-tab" if widget.id == "quality-artifact"
                            else "quality-setup-tab"
                        )
                        self.call_after_refresh(widget.focus)
                        break
                return
            self._cancel.clear()
            self._busy = True
            self._set_busy(True)
            if preparing:
                self._set_status("Preparing evaluator infrastructure")
                self._executor.submit(self._prepare, self._app().advisor, request)
                return
            candidates = self._checked() if self._search_state is not None else ()
            self.query_one("#quality-measurements", Vertical).display = False
            self.query_one("#quality-report", Collapsible).collapsed = False
            self.query_one("#quality-result", Static).update("Evaluation in progress.")
            self._set_status(
                f"Preparing {request.profile.value}; output: {request.output}"
            )
            allow_download = (
                self.query_one("#quality-download", Checkbox).value
                if self._plan is not None or self._search_state is not None else False
            )
            if self._search_state is not None:
                self._executor.submit(
                    self._run_candidates, self._app().advisor, request, allow_download, candidates,
                )
            else:
                self._executor.submit(self._run, self._app().advisor, request, allow_download)

    def _run_candidates(
        self, advisor: AdvisorService, request: QualityRunRequest, allow_download: bool,
        candidates: tuple[QualityCandidate, ...],
    ) -> None:
        stored = None
        records: list[tuple[str, QualityEvidence]] = []
        errors: list[str] = []
        def progress(text: str) -> None:
            if not self._quality_closing.is_set():
                self.post_message(_QualityProgress(text))
        try:
            outcomes = advisor.run_quality_candidates(
                candidates, request, allow_download=allow_download,
                is_cancelled=self._cancel.is_set,
                on_progress=progress,
            )
            readouts: list[str] = []
            for outcome in outcomes:
                readout = outcome.plan.model_identity.model_name
                if outcome.path is not None:
                    stored = outcome.path
                    evidence = describe_record(advisor.load_quality_record(outcome.path.stem))
                    records.append((outcome.plan.model_identity.model_name, evidence))
                    readout += "\n" + PlanEvidence(
                        state=EvidenceState.ESTIMATED, quality_records=(evidence,),
                    ).quality_readout()
                    readout += f"\nSaved: {outcome.path}"
                else:
                    readout += f"\nFailed: {outcome.error}"
                    errors.append(readout)
                readouts.append(readout)
            if self._cancel.is_set():
                readouts.append("Cancelled. Earlier completed records remain saved.")
            elif outcomes:
                try:
                    advisor.remember_quality_evaluation_setup(request)
                except (OSError, ValueError) as exc:
                    _log.warning("Could not remember quality setup", exc_info=True)
                    readouts.append(f"Evaluator setup was not saved: {exc}")
            # Unchecked because already measured: compared from the store, not rerun.
            records.extend(stored_records(
                self._selection.candidates,
                frozenset(candidate.plan.plan_id for candidate in candidates),
            ))
            completed = sum(outcome.path is not None for outcome in outcomes)
            readouts.insert(0, (
                f"{'Cancelled. ' if self._cancel.is_set() else ''}"
                f"Completed {completed}/{len(candidates)}; "
                f"failed {len(outcomes) - completed}; "
                f"not started {len(candidates) - len(outcomes)}."
            ))
            readouts.append("Per-artifact diagnostics only; recommendation order unchanged.")
            message = _QualityFinished(
                stored, "\n\n".join(readouts), records=tuple(records), errors=tuple(errors),
            )
        except Exception as exc:
            _log.exception("Candidate evaluations failed")
            message = _QualityFinished(
                stored, f"Evaluation failed: {exc}\nOutput: {request.output}",
            )
        if not self._quality_closing.is_set():
            self.post_message(message)

    def _prepare(self, advisor: AdvisorService, request: QualityRunRequest) -> None:
        def progress(text: str) -> None:
            if not self._quality_closing.is_set():
                self.post_message(_QualityProgress(text))

        try:
            advisor.prepare_quality_evaluation_setup(
                request, is_cancelled=self._cancel.is_set,
                on_progress=progress,
            )
            if self._cancel.is_set():
                raise QualityEvaluationCancelled("Setup cancelled; no settings saved.")
            advisor.remember_quality_evaluation_setup(request)
            message = _QualitySetupFinished(
                True, "Evaluator infrastructure ready. Model fit is checked at evaluation start.",
            )
        except Exception as exc:
            _log.exception("Quality setup failed")
            message = _QualitySetupFinished(False, str(exc))
        if not self._quality_closing.is_set():
            self.post_message(message)

    @on(_QualitySetupFinished)
    def _setup_finished(self, message: _QualitySetupFinished) -> None:
        self._busy = False
        self._set_busy(False)
        self._set_status(message.text)
        self.query_one("#quality-setup", Collapsible).collapsed = message.ready
        if message.ready:
            # A blocker from before the setup must not outlive it: check again.
            self._check_readiness()
        if self._quit:
            self.app.exit()
        elif self._leave:
            self.dismiss(self._stored)

    def _run(
        self, advisor: AdvisorService, request: QualityRunRequest, allow_download: bool,
    ) -> None:
        stored = None

        def progress(text: str) -> None:
            if not self._quality_closing.is_set():
                self.post_message(_QualityProgress(text))

        try:
            if self._plan is not None:
                stored = advisor.run_quality_evaluation_for_plan(
                    self._plan, request, allow_download=allow_download,
                    is_cancelled=self._cancel.is_set,
                    on_progress=progress,
                )
            else:
                stored = advisor.run_quality_evaluation(request, is_cancelled=self._cancel.is_set)
            evidence = describe_record(advisor.load_quality_record(stored.stem))
            readout = PlanEvidence(
                state=EvidenceState.ESTIMATED, quality_records=(evidence,),
            ).quality_readout()
            try:
                advisor.remember_quality_evaluation_setup(request)
            except (OSError, ValueError) as exc:
                _log.warning("Could not remember quality setup", exc_info=True)
                readout += f"\nEvaluator setup was not saved: {exc}"
            message = _QualityFinished(
                stored, readout,
                records=((self._plan.model_identity.model_name if self._plan else "Exact GGUF",
                          evidence),),
            )
        except QualityEvaluationCancelled as exc:
            message = _QualityFinished(None, str(exc))
        except Exception as exc:
            # Worker boundary: preserve unexpected failures, never import a success.
            _log.exception("Quality evaluation failed")
            stage = "Result display failed" if stored is not None else "Evaluation failed"
            message = _QualityFinished(stored, f"{stage}: {exc}\nOutput: {request.output}")
        if not self._quality_closing.is_set():
            self.post_message(message)

    def _recompare(self, advisor: AdvisorService) -> None:
        assert self._search_state is not None
        report = None
        try:
            report = advisor.recompare_quality(self._search_state)
            names = {
                ranked.plan.plan_id: ranked.evaluated.repo_id
                for ranked in self._search_state.ranked_plans
            }
            shown = tuple(rec.plan.plan_id for rec in self._search_state.recommendations
                          if rec.plan is not None)
            text = recompare_readout(report, names, shown_order=shown or None)
        except Exception as exc:
            _log.exception("Recompare failed")
            text = f"Recompare failed: {exc}"
        if not self._quality_closing.is_set():
            self.post_message(_Recompared(text, report))

    @on(_Recompared)
    async def _recompared(self, message: _Recompared) -> None:
        # The proposal itself is in the result; repeating its first line here is noise.
        self._set_status(
            "Recompare failed; see below." if message.text.startswith("Recompare failed")
            else "Proposal ready below; the shown recommendations are unchanged."
        )
        result = self.query_one("#quality-result", Static)
        await self.query_one("#quality-measurements", Vertical).remove_children()
        diagnostics = self.query_one("#quality-plan-diagnostics", Vertical)
        await diagnostics.remove_children()
        if message.report is not None:
            positions = {p: i + 1 for i, p in enumerate(message.report.base_order)}
            for fallback in message.report.fallback:
                name = fallback.repo_id or fallback.plan_id
                await diagnostics.mount(
                    Static(
                        f"#{positions[fallback.plan_id]} {name}"
                        + (f" / {fallback.artifact_label}" if fallback.artifact_label else ""),
                        classes="text-secondary", markup=False,
                    ),
                    TechnicalDetails(
                        title=fallback.reasons[0] if fallback.reasons else "Evidence details",
                        extra=[Static(
                            f"SHA256:\n{fallback.artifact_sha256 or 'unknown'}\n\n"
                            + "\n".join(fallback.reasons), markup=False,
                        )],
                    ),
                )
        report_widget = self.query_one("#quality-report", Collapsible)
        report_widget.title = "Ranking proposal"
        report_widget.collapsed = False
        result.update(message.text)
        self._busy = False
        self._set_busy(False)
        self.query_one("#quality-tabs", TabbedContent).active = "quality-results-tab"
        self.call_after_refresh(
            self.call_after_refresh,
            self.query_one("#quality-results-body", VerticalScroll).scroll_home,
            animate=False, immediate=True,
        )
        if self._quit:
            self.app.exit()
        elif self._leave:
            self.dismiss(self._stored)

    @on(_QualityProgress)
    def _progress(self, message: _QualityProgress) -> None:
        if not self._cancel.is_set():
            self._set_status(message.text)

    @on(_QualityFinished)
    async def _finished(self, message: _QualityFinished) -> None:
        await self._show_measurements(message.records, message.errors)
        self.query_one("#quality-report", Collapsible).collapsed = bool(message.records)
        self._busy = False
        self._stored = message.path
        self._set_busy(False)
        self._set_status(
            f"Saved: {message.path}" if message.path and self._search_state is None
            else message.readout.splitlines()[0]
        )
        result = self.query_one("#quality-result", Static)
        result.update(message.readout)
        self.query_one("#quality-tabs", TabbedContent).active = "quality-results-tab"
        # Collapsible queues its own scroll on the first refresh.
        self.call_after_refresh(
            self.call_after_refresh,
            self.query_one("#quality-results-body", VerticalScroll).scroll_home,
            animate=False, immediate=True,
        )
        if self._quit:
            self.app.exit()
        elif self._leave:
            self.dismiss(self._stored)

    async def _show_measurements(
        self, records: tuple[tuple[str, QualityEvidence], ...], errors: tuple[str, ...] = (),
    ) -> None:
        await self.query_one("#quality-plan-diagnostics", Vertical).remove_children()
        self.query_one("#quality-report", Collapsible).title = "Run report"
        measurements = self.query_one("#quality-measurements", Vertical)
        measurements.display = True
        await measurements.remove_children()
        if records:
            reason = not_comparable([record for _, record in records])
            await measurements.mount(
                Static("Side by side", classes="section-title"),
                comparison_table(records),
                *([Static(reason, classes="status-fail", markup=False)] if reason else []),
                Static(
                    "Historical artifact results / current execution and protocol not verified. "
                    "Not a general capability assessment.", classes="comparison-note",
                ),
            )
        for name, record in records:
            await measurements.mount(TechnicalDetails(
                title=f"{name}: provenance and limitations",
                extra=[Static(
                    "\n".join((*record.limitations, PlanEvidence(
                        state=EvidenceState.ESTIMATED, quality_records=(record,),
                    ).quality_details(), *(
                        f"{METRIC_LABELS.get(m.name, m.name)}: {m.correct}/{m.samples}"
                        for m in record.metrics
                    ), f"Context: {record.context_length or 'unknown'} tokens")),
                    markup=False,
                )],
            ))
        for error in errors:
            await measurements.mount(Static(error, classes="status-fail", markup=False))

    def _set_busy(self, busy: bool) -> None:
        for widget in self.query(Input):
            widget.disabled = busy
        for checkbox in self.query(Checkbox):
            checkbox.disabled = busy
        self.query_one("#quality-profile", Select).disabled = busy
        self.query_one("#quality-start", Button).disabled = busy or self._blocked or (
            self._search_state is not None and not self._selection.candidates
        )
        self.query_one("#quality-prepare", Button).disabled = busy
        if self._search_state is not None:
            self.query_one("#quality-recompare", Button).disabled = busy
        if busy:
            start_reason = (
                "Inspecting candidate metadata." if self._selecting else "Evaluation in progress."
            )
        elif self._checking:
            start_reason = "Checking the evaluator."
        elif self._blocked:
            start_reason = "A readiness check failed; it says what to do."
        elif self._search_state is not None and not self._selection.candidates:
            start_reason = "No evaluation candidates available. Selection details show the reasons."
        else:
            start_reason = None
        self.query_one("#quality-start", Button).tooltip = start_reason
        self.query_one("#quality-cancel", Button).disabled = not busy
        self.query_one("#quality-cancel", Button).tooltip = (
            None if busy else "No active evaluation."
        )

    def _request_cancel(self) -> None:
        self._cancel.set()
        self._set_status(
            "Cancelling; waiting for process cleanup..."
        )
        self.query_one("#quality-cancel", Button).disabled = True
        self.query_one("#quality-cancel", Button).tooltip = "Cancellation already requested."

    def _set_status(self, text: str) -> None:
        status = self.query_one("#quality-status", Static)
        status.update(text)
        status.display = bool(text)

    def action_back(self) -> None:
        if self._busy or self._selecting:
            self._leave = True
            self._request_cancel()
        else:
            self.dismiss(self._stored)

    def action_quit(self) -> None:
        if self._busy or self._selecting:
            self._quit = True
            self._request_cancel()
        else:
            self.app.exit()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app
