"""Choose how a model runs on this machine.

`PRODUCT.md`: *group variants by decision — quantizations belong under a
llama.cpp execution option, not as dozens of peer paths*. A popular GGUF
repository yields ten to twenty plans, and listing them as peers asks the user
to choose a runtime and a quantization in the same breath, as if those were the
same kind of decision. Here the runtime is the choice and the quantization is a
setting on it.

The list is one line per path, grouped by runtime option: the recommended group
leads and is open, the others start closed, and paths that cannot work on this
machine wait behind a toggle with their reason. Everything longer lives in the
detail pane of the selected path.

Two things changed structurally from the original version:

* Selection is a state on the option widget, not a `●`/`○` glyph baked into a
  button label. The list is no longer torn down and rebuilt on every click, so
  keyboard focus survives selecting something.
* Validated/Benchmarked come from the persisted records via
  :mod:`jaull.tui.evidence` rather than from a substring scan of
  ``plan.evidence``, which could never match.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Static, TabbedContent, TabPane

from jaull.advisor.quality import quality_evaluation_block_reason
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.runtime import RuntimeName
from jaull.presentation.plan_labels import (
    SUGGESTED_QUANTIZATIONS,
    artifact_display,
    execution_option_label,
    fits_here,
    format_gib,
    is_gguf_plan,
    is_ready_plan,
    model_display_name,
    readiness_detail,
    readiness_label,
    runtime_block_reason,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.evidence import EvidenceIndex, PlanEvidence
from jaull.tui.screens.quality_evaluation import QualityEvaluationScreen
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar
from jaull.tui.widgets.selection_workspace import SelectionWorkspace
from jaull.tui.widgets.technical_details import TechnicalDetails

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp

_log = logging.getLogger(__name__)

_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("ready", "Ready"),
    ("validated", "Validated"),
    ("benchmarked", "Benchmarks"),
)


class _ExecutionPathsLoaded(Message):
    def __init__(self, plans: list[ExecutionPlan], evidence: EvidenceIndex) -> None:
        super().__init__()
        self.plans = plans
        self.evidence = evidence


class _ExecutionPathsFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class _ExecutionPathsEvidenceLoaded(Message):
    def __init__(self, evidence: EvidenceIndex) -> None:
        super().__init__()
        self.evidence = evidence


class _ListMove(Message):
    """An arrow key on a list entry: the screen decides where focus goes next."""

    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta


_LIST_KEYS: list[BindingType] = [
    Binding("down,j", "list_move(1)", "Next", show=False),
    Binding("up,k", "list_move(-1)", "Previous", show=False),
]


class _ListToggle(ActionButton):
    """A quiet toggle inside the list, reachable with the same arrow keys."""

    BINDINGS = _LIST_KEYS
    group_key: str | None = None

    def action_list_move(self, delta: int) -> None:
        self.post_message(_ListMove(delta))


class _PathGroupHeader(Static):
    """One runtime option; Enter or a click opens and closes its paths."""

    DEFAULT_CLASSES = "path-group-head"

    can_focus = True

    BINDINGS = [Binding("enter,space", "open_close", "Open/close"), *_LIST_KEYS]

    class Toggled(Message):
        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(
        self,
        key: str,
        label: str,
        count: int,
        *,
        recommended: bool,
        expanded: bool,
        summary: str,
        id: str | None = None,
    ) -> None:
        self._key = key
        self._summary = summary
        self._label = label
        self._count = count
        self._recommended = recommended
        self._expanded = expanded
        super().__init__(self._text(), id=id, markup=False)

    def _text(self) -> str:
        glyph = "▾" if self._expanded else "▸"
        plural = "" if self._count == 1 else "s"
        parts = [self._label]
        if self._recommended:
            parts.append("★ recommended")
        parts += [self._summary, f"{self._count} path{plural}"]
        return f"{glyph} " + " · ".join(parts)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.update(self._text())

    def on_click(self) -> None:
        self.action_open_close()

    def action_open_close(self) -> None:
        self.post_message(self.Toggled(self._key))

    def action_list_move(self, delta: int) -> None:
        self.post_message(_ListMove(delta))


class _PathOption(Vertical):
    """One path on one line: variant, state, memory and saved evidence.

    Selection is a state on the widget, not a glyph baked into a label, so the
    list is not torn down on every click and keyboard focus survives selecting
    something. Anything longer than a line belongs to the detail pane.
    """

    DEFAULT_CLASSES = "path-option selectable"

    can_focus = True

    BINDINGS = [("enter", "choose", "Select"), *_LIST_KEYS]

    class Chosen(Message):
        def __init__(self, plan_id: str) -> None:
            super().__init__()
            self.plan_id = plan_id

    def __init__(
        self,
        plan: ExecutionPlan,
        evidence: PlanEvidence,
        *,
        selected: bool,
        recommended: bool = False,
        name: str | None = None,
        reason: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._plan = plan
        self._evidence = evidence
        self._selected = selected
        self._recommended = recommended
        self._variant = name or _variant_name(plan)
        self._reason = reason
        self.set_class(selected, "-selected")

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def compose(self) -> ComposeResult:
        state, state_class = _short_state(self._plan)
        with Horizontal(classes="po-line"):
            yield Static(self._mark(), classes="po-mark")
            yield Static(self._variant, classes="po-name", markup=False)
            yield Static(state, classes=f"po-state {state_class}", markup=False)
            yield Static(_memory_short(self._plan), classes="po-mem", markup=False)
            yield Static(
                _evidence_short(self._evidence),
                classes=f"po-evid {_evidence_class(self._evidence)}",
                markup=False,
            )
        yield Static(self._reason or "", classes="po-reason", markup=False)
        yield Static(
            self._evidence.quality_summary(),
            classes="path-option-meta -quality",
            markup=False,
        )

    def on_mount(self) -> None:
        reason = self.query_one(".po-reason", Static)
        reason.display = bool(self._reason)
        reason.tooltip = self._reason
        quality = self.query_one(".-quality", Static)
        quality.display = bool(self._evidence.quality_records)
        quality.tooltip = self._evidence.quality_details() or None

    def _mark(self) -> str:
        return ("●" if self._selected else " ") + ("★" if self._recommended else " ")

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self.query_one(".po-mark", Static).update(self._mark())

    def on_click(self) -> None:
        self.post_message(self.Chosen(self._plan.plan_id))

    def action_choose(self) -> None:
        self.post_message(self.Chosen(self._plan.plan_id))

    def action_list_move(self, delta: int) -> None:
        self.post_message(_ListMove(delta))


class ExecutionPathsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(
        self, recommendation: ModelRecommendation, *, evaluation: bool = False,
    ) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._open_evaluation = evaluation
        self._plans: list[ExecutionPlan] = []
        self._evidence = EvidenceIndex.empty()
        self._selected_plan_id: str | None = None
        self._advanced_groups: set[str] = set()
        self._show_unfit = False
        self._path_filter = "all"
        # Runtime groups the user (or the default) has open. Closing a group only
        # hides its rows; it never touches the selection.
        self._expanded: set[str] = set()
        self._paths_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-execution-paths",
        )
        self._future: Future[None] | None = None
        self._evidence_future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield ContextBar(
            model_display_name_for(self._recommendation),
            self._recommendation.repo_id,
            aside="Execution",
        )
        yield SelectionWorkspace(
            self._compose_choices,
            self._compose_inspector,
            list_label="Paths",
            master_id="paths-body",
            detail_id="paths-inspector",
        )
        yield Footer()

    def _compose_choices(self) -> ComposeResult:
        yield Static("Execution paths", classes="section-title")
        with Horizontal(id="paths-filters"):
            for key, label in _FILTERS:
                chip = Button(
                    label,
                    id=f"paths-filter-{key}",
                    classes="chip -active" if key == "all" else "chip",
                )
                if key == "benchmarked":
                    chip.tooltip = "Saved benchmarks, including history not verified for this plan."
                yield chip
        with Vertical(id="paths-list"):
            yield Static("", id="paths-empty", classes="text-muted")

    def _compose_inspector(self) -> ComposeResult:
        yield Static("Loading execution paths...", id="paths-status", classes="status-line")
        yield Static("", id="paths-error", classes="warning-line")
        yield Static("", id="paths-selected", classes="section-title", markup=False)
        yield Static("", id="paths-selected-meta", classes="text-muted", markup=False)
        with Vertical(id="paths-callout", classes="path-callout"):
            yield Static("", id="paths-state-word", markup=False)
            yield Static("", id="paths-state-text", markup=False)
        with Horizontal(id="paths-actions"):
            yield Button("Run", id="paths-run", classes="-primary", disabled=True)
            yield ActionButton("Validate", id="paths-validate", disabled=True)
            yield ActionButton("Benchmark", id="paths-benchmark", disabled=True)
        with TabbedContent(initial="paths-evaluation" if self._open_evaluation else ""):
            with TabPane("Plan"), VerticalScroll():
                yield Static("", id="paths-warnings", classes="warning-line")
                with Vertical(id="paths-facts"):
                    for key, label in (
                        ("memory", "Memory"), ("artifact", "Artifact"), ("evidence", "Evidence"),
                    ):
                        with Horizontal(classes="fact-row"):
                            yield Static(label, classes="fact-label", markup=False)
                            yield Static(
                                "", id=f"paths-fact-{key}", classes="fact-value", markup=False,
                            )
                yield Static("", id="paths-action-reason", classes="warning-line")
            with TabPane("Evaluation", id="paths-evaluation"), VerticalScroll():
                yield Static("Published model references", classes="section-title")
                yield Static("", id="paths-published", classes="text-secondary", markup=False)
                yield TechnicalDetails(
                    title="Published reference provenance",
                    extra=[Static("", id="paths-published-details", markup=False)],
                    id="paths-published-provenance",
                )
                yield Static("Measured on this artifact", classes="section-title")
                yield ActionButton("Evaluate quality", id="paths-evaluate", disabled=True)
                yield Static("", id="paths-evaluate-reason", classes="warning-line", markup=False)
                yield Static(
                    "No measured evaluation for this exact artifact.",
                    id="paths-quality-empty",
                    classes="text-secondary",
                )
                yield Static("", id="paths-quality", classes="text-secondary", markup=False)
                yield TechnicalDetails(
                    title="Artifact measurement provenance",
                    extra=[Static("", id="paths-quality-details", markup=False)],
                    id="paths-quality-provenance",
                )
        with Horizontal(classes="actions-row"):
            yield ActionButton("Recheck runtime", id="paths-recheck", disabled=True)
            yield ActionButton("Back", id="paths-back")

    def on_mount(self) -> None:
        # An empty Static still occupies its row; the error and warning lines
        # only exist once they have something to say.
        for widget_id in (
            "#paths-error", "#paths-warnings", "#paths-selected", "#paths-quality",
            "#paths-quality-provenance", "#paths-callout",
        ):
            self.query_one(widget_id).display = False
        self._paths_closing.clear()
        self._future = self._executor.submit(self._load_worker, self._app().advisor)

    @property
    def execution_plans(self) -> tuple[ExecutionPlan, ...]:
        """Already inspected choices, available to the parent without another Hub scan."""
        return tuple(self._plans)

    def on_unmount(self) -> None:
        self._paths_closing.set()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._evidence_future is not None:
            self._evidence_future.cancel()
            self._evidence_future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def on_screen_resume(self) -> None:
        if self._paths_closing.is_set() or not self._plans:
            return
        if self._evidence_future is not None and not self._evidence_future.done():
            return
        self._evidence_future = self._executor.submit(self._reload_evidence, self._app().advisor)

    def _reload_evidence(self, advisor: AdvisorService) -> None:
        """Returning from Run/Validate/Benchmark reads stores, never rediscovers plans."""
        try:
            evidence = EvidenceIndex.load(advisor)
        except Exception:
            _log.debug("evidence refresh failed", exc_info=True)
            return
        if not self._paths_closing.is_set():
            self.post_message(_ExecutionPathsEvidenceLoaded(evidence))

    @on(_ExecutionPathsEvidenceLoaded)
    async def _evidence_loaded(self, message: _ExecutionPathsEvidenceLoaded) -> None:
        if self._paths_closing.is_set():
            return
        self._evidence = message.evidence
        await self._render_plans()

    def _load_worker(self, advisor: AdvisorService, *, refresh: bool = False) -> None:
        try:
            if refresh:
                advisor.inspect_pytorch_runtime(refresh=True)
            plans = advisor.execution_plans_for_recommendation(self._recommendation)
        except Exception as exc:
            if not self._paths_closing.is_set():
                self.post_message(_ExecutionPathsFailed(str(exc)))
            return
        # Reading both record stores is a full directory scan, which is exactly
        # why it happens here on the worker rather than during render. It is
        # also enrichment: if it fails, the paths still list, just without their
        # Validated/Benchmarked history.
        evidence = EvidenceIndex.empty()
        if plans and not self._paths_closing.is_set():
            try:
                evidence = EvidenceIndex.load(advisor)
            except Exception:
                _log.debug("evidence lookup failed", exc_info=True)
        if not self._paths_closing.is_set():
            self.post_message(_ExecutionPathsLoaded(plans, evidence))

    @on(_ExecutionPathsLoaded)
    async def _paths_loaded(self, message: _ExecutionPathsLoaded) -> None:
        if self._paths_closing.is_set():
            return
        self._plans = message.plans
        self._evidence = message.evidence
        if self._selected_plan_id not in {plan.plan_id for plan in self._plans}:
            self._selected_plan_id = self._default_plan_id()
        selected_plan = self._selected_plan()
        if selected_plan is not None and not self._expanded:
            self._expanded = {execution_option_label(selected_plan)}
        recheck = self.query_one("#paths-recheck", Button)
        recheck.display = any(
            plan.runtime_family is RuntimeName.TRANSFORMERS for plan in self._plans
        )
        recheck.disabled = not recheck.display
        status = self.query_one("#paths-status", Static)
        status.update("")
        status.display = False
        # `_render_plans` only reaches `_apply_selection` after mounting every
        # option, and each mount yields. The selection is already known here, so
        # settle the actions first: otherwise the screen looks ready while
        # `#paths-run` is still disabled, and Textual swallows the press.
        selected = self._selected_plan()
        reason = self._action_block_reason(selected)
        self._set_actions(
            enabled=reason is None,
            reason=reason,
        )
        await self._render_plans()

    @on(_ExecutionPathsFailed)
    def _paths_failed(self, message: _ExecutionPathsFailed) -> None:
        if self._paths_closing.is_set():
            return
        self.query_one("#paths-status", Static).display = False
        error = self.query_one("#paths-error", Static)
        error.update(message.message)
        error.display = True
        recheck = self.query_one("#paths-recheck", Button)
        recheck.disabled = not recheck.display

    @on(_PathOption.Chosen)
    def _option_chosen(self, message: _PathOption.Chosen) -> None:
        self._selected_plan_id = message.plan_id
        self._apply_selection()
        self.query_one(SelectionWorkspace).show_detail()

    @on(_PathGroupHeader.Toggled)
    def _group_toggled(self, message: _PathGroupHeader.Toggled) -> None:
        if message.key in self._expanded:
            self._expanded.discard(message.key)
        else:
            self._expanded.add(message.key)
        for header in self.query(_PathGroupHeader):
            if header._key == message.key:
                header.set_expanded(message.key in self._expanded)
        for body in self.query(".path-group-body"):
            if body.name == message.key:
                body.display = message.key in self._expanded

    @on(_ListMove)
    def _list_move(self, message: _ListMove) -> None:
        message.stop()
        stops = [
            widget
            for widget in self.query("#paths-list *")
            if widget.can_focus
            and not getattr(widget, "disabled", False)
            and all(node.display for node in (widget, *widget.ancestors))
        ]
        if not stops:
            return
        focused = self.focused
        if focused in stops:
            target = stops.index(focused) + message.delta
        else:
            target = 0 if message.delta > 0 else len(stops) - 1
        stops[max(0, min(target, len(stops) - 1))].focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("paths-filter-"):
            self._path_filter = button_id.removeprefix("paths-filter-")
            self._apply_filter_chips()
            await self._render_plans()
            return
        if button_id.startswith("paths-toggle-advanced"):
            key = event.button.group_key if isinstance(event.button, _ListToggle) else None
            if key is None:
                return
            self._advanced_groups.symmetric_difference_update({key})
            await self._render_plans()
            return
        if button_id == "paths-toggle-unfit":
            self._show_unfit = not self._show_unfit
            await self._render_plans()
            return
        if button_id == "paths-back":
            self.app.pop_screen()
            return
        if button_id == "paths-recheck":
            self.query_one("#paths-recheck", Button).disabled = True
            status = self.query_one("#paths-status", Static)
            status.update("Checking runtime readiness...")
            status.display = True
            self.query_one("#paths-error", Static).display = False
            self._set_actions(enabled=False, reason="Checking runtime readiness.")
            self._future = self._executor.submit(
                self._load_worker, self._app().advisor, refresh=True
            )
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
        elif button_id == "paths-evaluate" and is_gguf_plan(plan):
            self.app.push_screen(QualityEvaluationScreen(plan), self._quality_finished)

    def _quality_finished(self, stored: Path | None) -> None:
        if stored is not None and not self._paths_closing.is_set():
            self._future = self._executor.submit(self._load_worker, self._app().advisor)

    # ------------------------------------------------------------------
    # Rendering
    #
    # `_render_plans` rebuilds the option list and runs only when the visible
    # set changes — a filter, or the advanced toggle. Selecting something goes
    # through `_apply_selection`, which mutates classes and text in place so
    # the focused widget stays focused.
    # ------------------------------------------------------------------

    async def _render_plans(self) -> None:
        container = self.query_one("#paths-list", Vertical)
        # Removing the focused control would hand focus to whatever Textual finds
        # last, which is a filter chip; put it back on the same control afterwards.
        focused = self.focused
        restore = focused.id if focused is not None and container in focused.ancestors else None
        await container.remove_children()
        await self._build_list(container)
        if restore:
            for widget in self.query(f"#{restore}"):
                widget.focus()

    async def _build_list(self, container: Vertical) -> None:
        if not self._plans:
            await container.mount(Static("No execution paths were found.", classes="text-muted"))
            self._apply_selection()
            return

        selected = self._selected_plan()
        if selected is None or not self._plan_matches_filter(selected):
            first = next((plan for plan in self._plans if self._plan_matches_filter(plan)), None)
            if first is not None:
                self._selected_plan_id = first.plan_id
                self._expanded.add(execution_option_label(first))

        fitting, unfit = self._split_by_fit()
        recommended_id = self._recommended_plan_id()
        indexes = {plan.plan_id: index for index, plan in enumerate(self._plans)}
        shown = False
        for number, (key, group) in enumerate(self._ordered_groups(fitting, recommended_id)):
            visible = [plan for plan in group if self._plan_matches_filter(plan)]
            if not visible:
                continue
            shown = True
            expanded = key in self._expanded
            body = Vertical(classes="path-group-body", id=f"paths-group-{number}", name=key)
            body.display = expanded
            await container.mount(
                _PathGroupHeader(
                    key,
                    key,
                    len(visible),
                    recommended=any(plan.plan_id == recommended_id for plan in visible),
                    expanded=expanded,
                    summary=_group_summary(visible),
                ),
                body,
            )
            show_advanced = key in self._advanced_groups
            options, hidden = self._group_rows(visible, recommended_id, show_advanced=show_advanced)
            await body.mount(*(
                _PathOption(
                    plan,
                    self._evidence.for_plan(plan),
                    selected=plan.plan_id == self._selected_plan_id,
                    recommended=plan.plan_id == recommended_id,
                    id=f"paths-plan-{indexes[plan.plan_id]}",
                )
                for plan in options
            ))
            if hidden or (show_advanced and _has_advanced(visible)):
                plural = "" if hidden == 1 else "s"
                label = (
                    "Hide advanced quantizations"
                    if show_advanced
                    else f"Show {hidden} advanced quantization{plural}"
                )
                toggle = _ListToggle(
                    label, id="paths-toggle-advanced" + (f"-{number}" if number else ""),
                    classes="-quiet",
                )
                toggle.group_key = key
                await body.mount(toggle)

        visible_unfit = [plan for plan in unfit if self._plan_matches_filter(plan)]
        if visible_unfit:
            shown = True
            await self._mount_unfit(container, visible_unfit, indexes)
        if not shown:
            await container.mount(
                Static(_empty_filter_label(self._path_filter), classes="text-muted")
            )
        self._apply_selection()

    async def _mount_unfit(
        self,
        container: Vertical,
        plans: list[ExecutionPlan],
        indexes: dict[str, int],
    ) -> None:
        """Paths this machine cannot run, one line each with the reason."""
        count = len(plans)
        label = (
            f"Hide {count} that don't fit"
            if self._show_unfit
            else f"Show {count} that don't fit"
        )
        await container.mount(_ListToggle(label, id="paths-toggle-unfit", classes="-quiet"))
        if not self._show_unfit:
            return
        await container.mount(*(
            _PathOption(
                plan,
                self._evidence.for_plan(plan),
                selected=plan.plan_id == self._selected_plan_id,
                name=f"{_runtime_name(plan)} · {_variant_name(plan)}",
                reason=_why_not(plan),
                id=f"paths-plan-{indexes[plan.plan_id]}",
            )
            for plan in plans
        ))

    def _group_rows(
        self, plans: list[ExecutionPlan], recommended_id: str | None, *, show_advanced: bool,
    ) -> tuple[list[ExecutionPlan], int]:
        """The rows a group shows, and how many advanced quantizations stay behind.

        Suggested quantizations lead; the rest wait behind the group's toggle
        unless one of them is the path the user has chosen or Jaull recommends.
        """
        if not all(is_gguf_plan(plan) for plan in plans):
            return plans, 0
        advanced = [plan for plan in plans if _is_advanced(plan)]
        if show_advanced:
            return plans, 0
        keep = {self._selected_plan_id, recommended_id}
        shown = [plan for plan in plans if plan not in advanced or plan.plan_id in keep]
        return shown, len(plans) - len(shown)

    def _split_by_fit(self) -> tuple[list[ExecutionPlan], list[ExecutionPlan]]:
        """Paths that can work here, and those that cannot; never an empty list.

        When nothing fits, everything is shown as a normal path rather than
        leaving the user in front of a screen with only a toggle.
        """
        fitting = [plan for plan in self._plans if fits_here(plan)]
        if not fitting:
            return list(self._plans), []
        return fitting, [plan for plan in self._plans if not fits_here(plan)]

    def _ordered_groups(
        self, plans: list[ExecutionPlan], recommended_id: str | None,
    ) -> list[tuple[str, list[ExecutionPlan]]]:
        groups: dict[str, list[ExecutionPlan]] = {}
        for plan in plans:
            groups.setdefault(execution_option_label(plan), []).append(plan)

        def ordered(group: list[ExecutionPlan]) -> list[ExecutionPlan]:
            if all(is_gguf_plan(plan) for plan in group):
                group = sorted(group, key=_gguf_sort_key)
            return sorted(group, key=lambda plan: plan.plan_id != recommended_id)

        ranked = sorted(
            groups.items(),
            key=lambda item: all(plan.plan_id != recommended_id for plan in item[1]),
        )
        return [(key, ordered(group)) for key, group in ranked]

    def _recommended_plan_id(self) -> str | None:
        plan = self._recommendation.plan
        if plan is not None and any(item.plan_id == plan.plan_id for item in self._plans):
            return plan.plan_id
        return self._plans[0].plan_id if self._plans else None

    def _default_plan_id(self) -> str | None:
        """The recommended path when it fits here, otherwise the first one that does."""
        if not self._plans:
            return None
        fitting, _ = self._split_by_fit()
        recommended = self._recommended_plan_id()
        if any(plan.plan_id == recommended for plan in fitting):
            return recommended
        groups = self._ordered_groups(fitting, recommended)
        return groups[0][1][0].plan_id

    def _apply_selection(self) -> None:
        """Reflect the current plan everywhere, without rebuilding the list."""
        selected = self._selected_plan()
        if selected is not None and not self._plan_matches_filter(selected):
            selected = None
        evaluate = self.query_one("#paths-evaluate", Button)
        evaluation_reason = (
            quality_evaluation_block_reason(selected) if selected is not None
            else "Select a GGUF path for the quality pilot."
        )
        evaluate.disabled = evaluation_reason is not None
        evaluate.tooltip = (
            evaluation_reason or "Open evaluation setup; preflight is required before running."
        )
        reason = self.query_one("#paths-evaluate-reason", Static)
        reason.update(evaluation_reason or "")
        reason.display = evaluation_reason is not None
        for option in self.query(_PathOption):
            option.set_selected(option.plan_id == self._selected_plan_id)

        title = self.query_one("#paths-selected", Static)
        meta = self.query_one("#paths-selected-meta", Static)
        warnings = self.query_one("#paths-warnings", Static)
        if selected is None:
            self.query_one("#paths-published", Static).update("No execution path selected.")
            self.query_one("#paths-published-provenance").display = False
            title.display = False
            meta.display = False
            self.query_one("#paths-callout").display = False
            warnings.display = False
            self.query_one("#paths-quality", Static).display = False
            self.query_one("#paths-quality-empty").display = False
            self.query_one("#paths-quality-provenance").display = False
            self.query_one("#paths-quality-details", Static).update("")
            for key in ("memory", "artifact", "evidence"):
                self.query_one(f"#paths-fact-{key}", Static).update("")
            self.query_one("#paths-facts").display = False
            self.query_one("#paths-validate", ActionButton).set_action_label("Validate")
            self.query_one("#paths-benchmark", ActionButton).set_action_label("Benchmark")
            self._set_actions(
                enabled=False,
                reason="No execution path is selected.",
            )
            return

        evidence = self._evidence.for_plan(selected)
        self.query_one("#paths-facts").display = True
        self.query_one("#paths-published", Static).update(evidence.published_summary)
        self.query_one("#paths-published-details", Static).update(evidence.published_provenance)
        self.query_one("#paths-published-provenance").display = bool(evidence.published_provenance)
        badge = "   ★ recommended" if selected.plan_id == self._recommended_plan_id() else ""
        title.update(f"{_variant_name(selected)}{badge}")
        title.display = True
        meta.update(f"{execution_option_label(selected)} · {_format_name(selected)}")
        meta.display = True
        state, state_class = _short_state(selected)
        callout = self.query_one("#paths-callout")
        callout.set_classes(f"path-callout {_CALLOUT_CLASSES[state_class]}")
        callout.display = True
        word = self.query_one("#paths-state-word", Static)
        word.update(f"● {state}")
        word.set_classes(state_class)
        self.query_one("#paths-state-text", Static).update(_state_message(selected))
        quality = self.query_one("#paths-quality", Static)
        quality.update(evidence.quality_readout())
        quality.display = bool(evidence.quality_records)
        quality.tooltip = evidence.quality_details() or None
        self.query_one("#paths-quality-empty").display = not bool(evidence.quality_records)
        details = self.query_one("#paths-quality-details", Static)
        details.update(evidence.quality_details())
        details.display = bool(evidence.quality_records)
        self.query_one("#paths-quality-provenance").display = bool(evidence.quality_records)
        self.query_one("#paths-fact-memory", Static).update(_memory(selected))
        self.query_one("#paths-fact-artifact", Static).update(artifact_display(selected))
        self.query_one("#paths-fact-evidence", Static).update(evidence.summary())

        # Plan warnings are computed today and were never rendered.
        text = " ".join(selected.warnings)
        warnings.update(text)
        warnings.display = bool(text)

        self._set_actions(
            enabled=self._action_block_reason(selected) is None,
            reason=self._action_block_reason(selected),
        )
        self._label_actions(evidence)

    def _apply_filter_chips(self) -> None:
        for key, _ in _FILTERS:
            chip = self.query_one(f"#paths-filter-{key}", Button)
            chip.set_class(key == self._path_filter, "-active")

    def _set_actions(self, *, enabled: bool, reason: str | None = None) -> None:
        for button_id in ("#paths-run", "#paths-validate", "#paths-benchmark"):
            self.query_one(button_id, Button).disabled = not enabled
            self.query_one(button_id, Button).tooltip = reason
        reason_widget = self.query_one("#paths-action-reason", Static)
        reason_widget.update(f"Actions unavailable: {reason}" if reason else "")
        reason_widget.display = reason is not None

    def _action_block_reason(self, plan: ExecutionPlan | None) -> str | None:
        if plan is None:
            return "No execution path is selected."
        if not self._plan_matches_filter(plan):
            return "The selected execution path is hidden by the current filter."
        return runtime_block_reason(plan)

    def _label_actions(self, evidence: PlanEvidence) -> None:
        """A tick on an action that already has evidence behind it.

        Use the shared action-label API, preserving the button's presentation.
        """
        self.query_one("#paths-validate", ActionButton).set_action_label(
            "Validate ✓" if evidence.validated else "Validate"
        )
        self.query_one("#paths-benchmark", ActionButton).set_action_label(
            "Benchmark ✓" if evidence.benchmarked else
            "Benchmark history" if evidence.has_benchmark_history else "Benchmark"
        )

    # ------------------------------------------------------------------
    # Plan selection helpers
    # ------------------------------------------------------------------

    def _selected_plan(self) -> ExecutionPlan | None:
        if self._selected_plan_id is None:
            return None
        return next(
            (plan for plan in self._plans if plan.plan_id == self._selected_plan_id),
            None,
        )

    def _plan_matches_filter(self, plan: ExecutionPlan) -> bool:
        if self._path_filter == "ready":
            return is_ready_plan(plan)
        if self._path_filter == "validated":
            return self._evidence.for_plan(plan).validated
        if self._path_filter == "benchmarked":
            return self._evidence.for_plan(plan).has_benchmark_history
        return True

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


def model_display_name_for(recommendation: ModelRecommendation) -> str:
    """Title for a recommendation, without resolving a full ModelIdentity."""
    from jaull.domain.execution_plans import ModelIdentity

    identity = ModelIdentity(model_name=recommendation.repo_id.split("/")[-1])
    return model_display_name(identity, fallback=recommendation.repo_id)


def _variant_name(plan: ExecutionPlan) -> str:
    """What tells two paths of the same runtime apart: a quantization or a precision."""
    if is_gguf_plan(plan):
        return plan.artifact.quantization or plan.artifact.label
    name = artifact_display(plan)
    precision = plan.artifact.precision
    if precision and precision not in name:
        return f"{name} · {precision}"
    return name


def _is_advanced(plan: ExecutionPlan) -> bool:
    return (plan.artifact.quantization or "") not in SUGGESTED_QUANTIZATIONS


def _has_advanced(plans: list[ExecutionPlan]) -> bool:
    return all(is_gguf_plan(plan) for plan in plans) and any(_is_advanced(p) for p in plans)


# Short row labels for the readiness words `readiness_label` produces, with the
# colour class that carries the same meaning for anyone who can see it.
_STATE_STYLES: dict[str, tuple[str, str]] = {
    "Ready": ("Ready", "status-ok"),
    "Preparable": ("Preparable", "status-ok"),
    "Blocked": ("Blocked", "status-warn"),
    "Insufficient memory": ("No memory", "status-fail"),
    "Unsupported configuration": ("Unsupported", "status-fail"),
}


def _short_state(plan: ExecutionPlan) -> tuple[str, str]:
    label = readiness_label(plan)
    return _STATE_STYLES.get(label, (label, "status-estimated"))


_CALLOUT_CLASSES = {
    "status-ok": "-ok",
    "status-warn": "-warn",
    "status-fail": "-fail",
    "status-estimated": "-muted",
}


def _state_message(plan: ExecutionPlan) -> str:
    """The long readiness text, for the one path the user is looking at."""
    if not fits_here(plan):
        return _why_not(plan)
    readiness = plan.execution_readiness
    if readiness is not None and readiness.message:
        return readiness.message
    return readiness_detail(plan)


def _format_name(plan: ExecutionPlan) -> str:
    return "GGUF" if is_gguf_plan(plan) else plan.artifact.format.value


def _runtime_name(plan: ExecutionPlan) -> str:
    """Just the runtime of an option label, for rows that sit outside a group."""
    return execution_option_label(plan).split(" · ")[0]


def _group_summary(plans: list[ExecutionPlan]) -> str:
    """What a closed group still has to say: the best state among its paths."""
    best = next((plan for plan in plans if is_ready_plan(plan)), plans[0])
    return _short_state(best)[0].lower()


def _memory_short(plan: ExecutionPlan) -> str:
    if plan.memory_prediction is None:
        return "—"
    return format_gib(plan.memory_prediction.total_bytes)


def _evidence_short(evidence: PlanEvidence) -> str:
    parts = []
    if evidence.validated:
        parts.append("✓ validated")
    if evidence.benchmarked:
        parts.append("✓ benchmarked")
    elif evidence.has_benchmark_history:
        parts.append("benchmark history")
    return " · ".join(parts)


def _why_not(plan: ExecutionPlan) -> str:
    """The reason in one clause: the part of the readiness detail after the label."""
    return readiness_detail(plan).split(" — ", 1)[-1]


def _memory(plan: ExecutionPlan) -> str:
    """Memory as a figure, not as a second claim about evidence.

    The evidence summary already says whether anything was measured, so
    repeating "Estimated" here only produced "Estimated only · Estimated 4 GiB".
    """
    if plan.memory_prediction is None:
        return "memory unknown"
    return f"{format_gib(plan.memory_prediction.total_bytes)} predicted"


def _evidence_class(evidence: PlanEvidence) -> str:
    return f"status-{evidence.state.value}"


def _empty_filter_label(path_filter: str) -> str:
    if path_filter == "ready":
        return "No ready execution options yet."
    if path_filter == "validated":
        return "No validated execution options yet."
    if path_filter == "benchmarked":
        return "No saved benchmarks for these execution paths yet."
    return "No execution options match this filter."


def _gguf_sort_key(plan: ExecutionPlan) -> tuple[int, str]:
    quantization = plan.artifact.quantization or ""
    try:
        priority = SUGGESTED_QUANTIZATIONS.index(quantization)
    except ValueError:
        priority = len(SUGGESTED_QUANTIZATIONS)
    return priority, quantization or plan.artifact.label


__all__ = ["ExecutionPathsScreen"]
