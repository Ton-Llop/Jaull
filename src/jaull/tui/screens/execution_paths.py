"""Choose how a model runs on this machine.

`PRODUCT.md`: *group variants by decision — quantizations belong under a
llama.cpp execution option, not as dozens of peer paths*. A popular GGUF
repository yields ten to twenty plans, and listing them as peers asks the user
to choose a runtime and a quantization in the same breath, as if those were the
same kind of decision. Here the runtime is the choice and the quantization is a
setting on it.

Two things changed structurally from the previous version:

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
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Select, Static

from jaull.domain.execution_plans import ExecutionPlan
from jaull.presentation.plan_labels import (
    SUGGESTED_QUANTIZATIONS,
    artifact_display,
    execution_option_label,
    format_gib,
    is_gguf_plan,
    is_ready_plan,
    model_display_name,
    readiness_label,
    selected_plan_label,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.evidence import EvidenceIndex, PlanEvidence
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp

_log = logging.getLogger(__name__)

_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("ready", "Ready"),
    ("validated", "Validated"),
    ("benchmarked", "Benchmarked"),
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


class _PathOption(Vertical):
    """One runtime option: focusable, selectable, and its own click target.

    A `Button` would have been less code, but a button label is the wrong place
    for three lines of state, and toggling a glyph inside it meant rebuilding
    the widget — and losing focus — to show a selection change.
    """

    DEFAULT_CLASSES = "path-option selectable"

    can_focus = True

    BINDINGS = [("enter", "choose", "Select")]

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
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._plan = plan
        self._evidence = evidence
        self.set_class(selected, "-selected")

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def compose(self) -> ComposeResult:
        yield Static(execution_option_label(self._plan), classes="path-option-title")
        yield Static(artifact_display(self._plan), classes="path-option-meta -artifact")
        yield Static(
            self._evidence.summary(),
            classes=f"path-option-meta -evidence {_evidence_class(self._evidence)}",
        )

    def rebind(self, plan: ExecutionPlan, evidence: PlanEvidence) -> None:
        """Point this option at a different plan behind the same choice.

        The llama.cpp option stands for whichever quantization is currently
        picked, so changing the quantization has to move the artifact and
        evidence lines with it — otherwise the option keeps advertising the
        artifact the user just navigated away from.
        """
        self._plan = plan
        self._evidence = evidence
        for widget in self.query(".-artifact").results(Static):
            widget.update(artifact_display(plan))
        for widget in self.query(".-evidence").results(Static):
            widget.update(evidence.summary())
            widget.set_classes(f"path-option-meta -evidence {_evidence_class(evidence)}")

    def on_click(self) -> None:
        self.post_message(self.Chosen(self._plan.plan_id))

    def action_choose(self) -> None:
        self.post_message(self.Chosen(self._plan.plan_id))


class ExecutionPathsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._plans: list[ExecutionPlan] = []
        self._evidence = EvidenceIndex.empty()
        self._selected_plan_id: str | None = None
        self._show_advanced_quantizations = False
        self._path_filter = "all"
        # Set while a Select is being repopulated, so the Changed event it
        # emits for its own new value is not read back as a user choice.
        self._suppress_quantization_event = False
        self._paths_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-execution-paths",
        )
        self._future: Future[None] | None = None

    def compose(self) -> ComposeResult:
        yield ContextBar(
            model_display_name_for(self._recommendation),
            self._recommendation.repo_id,
            aside="Execution",
        )
        with VerticalScroll(id="paths-body"):
            yield Static("Loading execution paths...", id="paths-status", classes="status-line")
            yield Static("", id="paths-error", classes="warning-line")
            yield Static("", id="paths-selected", classes="section-title")
            yield Static("", id="paths-selected-meta", classes="text-muted")
            with Horizontal(id="paths-filters"):
                for key, label in _FILTERS:
                    yield Button(
                        label,
                        id=f"paths-filter-{key}",
                        classes="chip -active" if key == "all" else "chip",
                    )
            with Vertical(id="paths-list"):
                yield Static("", id="paths-empty", classes="text-muted")
            yield Static("", id="paths-warnings", classes="warning-line")
            with Horizontal(id="paths-actions"):
                yield Button("Run", id="paths-run", classes="-primary", disabled=True)
                yield ActionButton("Validate", id="paths-validate", disabled=True)
                yield ActionButton("Benchmark", id="paths-benchmark", disabled=True)
                yield ActionButton("Back", id="paths-back")
        yield Footer()

    def on_mount(self) -> None:
        # An empty Static still occupies its row; the error and warning lines
        # only exist once they have something to say.
        for widget_id in ("#paths-error", "#paths-warnings", "#paths-selected"):
            self.query_one(widget_id, Static).display = False
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
        # Reading both record stores is a full directory scan, which is exactly
        # why it happens here on the worker rather than during render. It is
        # also enrichment: if it fails, the paths still list, just without their
        # Validated/Benchmarked history.
        evidence = EvidenceIndex.empty()
        if plans and not self._paths_closing.is_set():
            try:
                evidence = EvidenceIndex.load(advisor, plans[0].model_identity)
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
        self._selected_plan_id = self._plans[0].plan_id if self._plans else None
        status = self.query_one("#paths-status", Static)
        status.update("")
        status.display = False
        # `_render_plans` only reaches `_apply_selection` after mounting every
        # option, and each mount yields. The selection is already known here, so
        # settle the actions first: otherwise the screen looks ready while
        # `#paths-run` is still disabled, and Textual swallows the press.
        selected = self._selected_plan()
        self._set_actions(
            enabled=selected is not None and self._plan_matches_filter(selected)
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

    @on(_PathOption.Chosen)
    def _option_chosen(self, message: _PathOption.Chosen) -> None:
        if message.plan_id == self._selected_plan_id:
            return
        self._selected_plan_id = message.plan_id
        self._apply_selection()

    @on(Select.Changed, "#paths-quant")
    def _quantization_changed(self, event: Select.Changed) -> None:
        if self._suppress_quantization_event or event.value is Select.BLANK:
            return
        plan_id = str(event.value)
        if plan_id == self._selected_plan_id:
            return
        self._selected_plan_id = plan_id
        self._apply_selection()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("paths-filter-"):
            self._path_filter = button_id.removeprefix("paths-filter-")
            self._apply_filter_chips()
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
        await container.remove_children()
        if not self._plans:
            await container.mount(
                Static("No execution paths were found.", classes="text-muted")
            )
            self._apply_selection()
            return

        visible_primary = self._primary_path_plans(visible_only=True)
        for index, plan in enumerate(visible_primary):
            await container.mount(
                _PathOption(
                    plan,
                    self._evidence.for_plan(plan),
                    selected=plan.plan_id == self._selected_plan_id,
                    id=f"paths-plan-{index}",
                )
            )

        gguf_plan = self._visible_gguf_plan()
        if gguf_plan is not None:
            await container.mount(
                _PathOption(
                    gguf_plan,
                    self._evidence.for_plan(gguf_plan),
                    selected=self._selected_is_gguf(),
                    id="paths-gguf",
                )
            )
            await self._mount_quantization_picker(container, gguf_plan)

        if not visible_primary and gguf_plan is None:
            await container.mount(
                Static(_empty_filter_label(self._path_filter), classes="text-muted")
            )
        self._apply_selection()

    async def _mount_quantization_picker(
        self,
        container: Vertical,
        gguf_plan: ExecutionPlan,
    ) -> None:
        """A quantization is a setting on the llama.cpp option, not a peer path.

        The suggested quantizations lead and are labelled as such; the rest
        stay behind the toggle so the dropdown opens on a shortlist rather than
        on twenty near-identical entries.
        """
        gguf_plans = self._gguf_plans(visible_only=True)
        if len(gguf_plans) <= 1:
            return
        suggested = [
            plan
            for plan in gguf_plans
            if (plan.artifact.quantization or "") in SUGGESTED_QUANTIZATIONS
        ]
        advanced = [plan for plan in gguf_plans if plan not in suggested]
        visible_advanced = advanced if self._show_advanced_quantizations else [
            plan for plan in advanced if plan.plan_id == self._selected_plan_id
        ]

        options = [
            (f"{_quantization_name(plan)} · recommended", plan.plan_id)
            for plan in suggested
        ] + [(_quantization_name(plan), plan.plan_id) for plan in visible_advanced]
        if not options:
            return

        picker = Vertical(classes="path-quant")
        await container.mount(picker)
        await picker.mount(Static("Quantization", classes="path-quant-label"))
        self._suppress_quantization_event = True
        try:
            await picker.mount(
                Select(
                    options,
                    id="paths-quant",
                    value=gguf_plan.plan_id,
                    allow_blank=False,
                )
            )
        finally:
            self._suppress_quantization_event = False

        hidden = len(advanced) - len(visible_advanced)
        if hidden > 0 or self._show_advanced_quantizations:
            plural = "" if hidden == 1 else "s"
            label = (
                "Hide advanced quantizations"
                if self._show_advanced_quantizations
                else f"Show {hidden} advanced quantization{plural}"
            )
            await picker.mount(
                ActionButton(label, id="paths-toggle-advanced", classes="-quiet")
            )

    def _apply_selection(self) -> None:
        """Reflect the current plan everywhere, without rebuilding the list."""
        selected = self._selected_plan()
        gguf_plan = self._visible_gguf_plan()
        for option in self.query(_PathOption):
            if option.id == "paths-gguf":
                if gguf_plan is not None:
                    option.rebind(gguf_plan, self._evidence.for_plan(gguf_plan))
                option.set_class(self._selected_is_gguf(), "-selected")
            else:
                option.set_class(option.plan_id == self._selected_plan_id, "-selected")

        title = self.query_one("#paths-selected", Static)
        meta = self.query_one("#paths-selected-meta", Static)
        warnings = self.query_one("#paths-warnings", Static)
        if selected is None:
            title.display = False
            meta.display = False
            warnings.display = False
            self._set_actions(enabled=False)
            return

        evidence = self._evidence.for_plan(selected)
        title.update(f"Selected · {selected_plan_label(selected)}")
        title.display = True
        meta.update(
            f"{readiness_label(selected)} · {evidence.summary()} · {_memory(selected)}"
        )
        meta.display = True

        # Plan warnings are computed today and were never rendered.
        text = " ".join(selected.warnings)
        warnings.update(text)
        warnings.display = bool(text)

        self._set_actions(enabled=self._plan_matches_filter(selected))
        self._label_actions(evidence)

    def _apply_filter_chips(self) -> None:
        for key, _ in _FILTERS:
            chip = self.query_one(f"#paths-filter-{key}", Button)
            chip.set_class(key == self._path_filter, "-active")

    def _set_actions(self, *, enabled: bool) -> None:
        for button_id in ("#paths-run", "#paths-validate", "#paths-benchmark"):
            self.query_one(button_id, Button).disabled = not enabled

    def _label_actions(self, evidence: PlanEvidence) -> None:
        """A tick on an action that already has evidence behind it.

        Through `set_action_label` so the brackets survive: assigning `.label`
        directly would replace the whole rendered label, escaping included.
        """
        self.query_one("#paths-validate", ActionButton).set_action_label(
            "Validate ✓" if evidence.validated else "Validate"
        )
        self.query_one("#paths-benchmark", ActionButton).set_action_label(
            "Benchmark ✓" if evidence.benchmarked else "Benchmark"
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

    def _selected_is_gguf(self) -> bool:
        selected = self._selected_plan()
        return selected is not None and is_gguf_plan(selected)

    def _primary_path_plans(self, *, visible_only: bool = False) -> list[ExecutionPlan]:
        plans = [plan for plan in self._plans if not is_gguf_plan(plan)]
        if visible_only:
            return [plan for plan in plans if self._plan_matches_filter(plan)]
        return plans

    def _gguf_plans(self, *, visible_only: bool = False) -> list[ExecutionPlan]:
        plans = sorted(
            (plan for plan in self._plans if is_gguf_plan(plan)),
            key=_gguf_sort_key,
        )
        if visible_only:
            return [plan for plan in plans if self._plan_matches_filter(plan)]
        return plans

    def _visible_gguf_plan(self) -> ExecutionPlan | None:
        """The GGUF plan the single llama.cpp option currently stands for."""
        visible = self._gguf_plans(visible_only=True)
        selected = self._selected_plan()
        if selected is not None and selected in visible:
            return selected
        return next(iter(visible), None)

    def _plan_matches_filter(self, plan: ExecutionPlan) -> bool:
        if self._path_filter == "ready":
            return is_ready_plan(plan)
        if self._path_filter == "validated":
            return self._evidence.for_plan(plan).validated
        if self._path_filter == "benchmarked":
            return self._evidence.for_plan(plan).benchmarked
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


def _quantization_name(plan: ExecutionPlan) -> str:
    return plan.artifact.quantization or plan.artifact.label


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
        return "No benchmarked execution options yet."
    return "No execution options match this filter."


def _gguf_sort_key(plan: ExecutionPlan) -> tuple[int, str]:
    quantization = plan.artifact.quantization or ""
    try:
        priority = SUGGESTED_QUANTIZATIONS.index(quantization)
    except ValueError:
        priority = len(SUGGESTED_QUANTIZATIONS)
    return priority, quantization or plan.artifact.label


__all__ = ["ExecutionPathsScreen"]
