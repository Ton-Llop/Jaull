"""Explicit, bounded quality pilot; no work starts on screen entry."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from uuid import uuid4

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Collapsible, Footer, Input, Select, Static

from jaull.evaluation.quality_records import describe_record
from jaull.paths import user_data_dir
from jaull.runtime.quality_eval_runner import (
    QualityEvaluationCancelled,
    QualityEvaluationError,
    QualityProfile,
    QualityRunRequest,
)
from jaull.tui.evidence import EvidenceState, PlanEvidence
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.domain.execution_plans import ExecutionPlan
    from jaull.tui.app import JaullApp

_log = logging.getLogger(__name__)


class _QualityFinished(Message):
    def __init__(self, path: Path | None, readout: str) -> None:
        super().__init__()
        self.path = path
        self.readout = readout


class _QualityProgress(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class _QualitySetupFinished(Message):
    def __init__(self, ready: bool, text: str) -> None:
        super().__init__()
        self.ready = ready
        self.text = text


class QualityEvaluationScreen(Screen[Path | None]):
    BINDINGS = [("escape", "back", "Back"), ("q", "quit", "Quit")]

    DEFAULT_CSS = """
    QualityEvaluationScreen #quality-body { padding: 1 2; }
    QualityEvaluationScreen Input, QualityEvaluationScreen Select { margin-bottom: 1; }
    QualityEvaluationScreen #quality-actions { height: 3; }
    QualityEvaluationScreen #quality-status { margin: 0 2 1 2; }
    QualityEvaluationScreen #quality-result { height: auto; margin-top: 1; }
    """

    def __init__(self, plan: ExecutionPlan | None = None) -> None:
        super().__init__()
        self._plan = plan
        self._cancel = Event()
        self._quality_closing = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jaull-quality")
        self._busy = False
        self._leave = False
        self._quit = False
        self._stored: Path | None = None

    def compose(self) -> ComposeResult:
        yield ContextBar("Quality evaluation", aside="Diagnostic pilot")
        with VerticalScroll(id="quality-body"):
            yield Static(
                (f"{self._plan.artifact.repo_id}\n"
                 f"{self._plan.artifact.filename or self._plan.artifact.label}")
                if self._plan else "Exact local GGUF",
                classes="section-title",
                markup=False,
            )
            yield Static(
                "HellaSwag / raw prompts / zero-shot / ctx 2048 / CUDA0 / full offload\n"
                "Evaluation protocol is separate from the selected execution plan.",
                classes="text-secondary",
            )
            if self._plan is None:
                yield Static("Verified artifact manifest (JSON)", classes="field-label")
                yield Input(placeholder="Verified artifact manifest (JSON)", id="quality-artifact")
            yield Static("Evaluation samples", classes="field-label")
            yield Select(
                [("Smoke - 3 examples", "smoke"), ("Limited - 100 examples", "hellaswag100")],
                value="smoke", allow_blank=False, id="quality-profile",
            )
            if self._plan is not None:
                size = self._plan.artifact.size_bytes
                yield Static(
                    f"GGUF download size: {size / 1024**3:.2f} GiB" if size is not None
                    else "GGUF download size: unknown", classes="text-secondary",
                )
                yield Checkbox("Allow downloading the selected GGUF if missing",
                               id="quality-download", value=False)
            with Collapsible(title="Evaluator setup", collapsed=False, id="quality-setup"):
                yield Checkbox("Allow downloading pinned HellaSwag dataset if missing",
                               id="quality-dataset-download", value=False)
                for key, label, value in (
                    ("dataset", "Evaluation dataset", ""),
                    ("server", "Pinned llama-server executable", ""),
                    ("root", "Trusted pilot checkout", ""),
                    ("image", "Local evaluator image", "jaull-quality-eval:gguf-v1"),
                    ("output", "New output directory",
                     str(user_data_dir("quality-runs") / uuid4().hex)),
                ):
                    yield Static(label, classes="field-label")
                    yield Input(value=value, placeholder=label, id=f"quality-{key}")
            yield Static(
                "Not a full benchmark or general-capability verdict.", classes="warning-line",
            )
            yield Static("", id="quality-result", markup=False)
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
            self.query_one("#quality-setup", Collapsible).collapsed = (
                set(setup) == {"dataset", "server", "root", "image"}
                and all(Path(setup[key]).expanduser().is_file() for key in ("dataset", "server"))
                and Path(setup["root"]).expanduser().is_dir()
            )
        except (OSError, ValueError, QualityEvaluationError) as exc:
            _log.warning("Could not restore quality setup", exc_info=True)
            self._set_status(f"Could not restore evaluator setup: {exc}")

    def on_unmount(self) -> None:
        self._quality_closing.set()
        self._cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _request(self, *, require_artifact: bool = True) -> QualityRunRequest:
        values = {
            key: self.query_one(f"#quality-{key}", Input).value.strip()
            for key in ("artifact", "dataset", "server", "root", "image", "output")
            if key != "artifact" or (self._plan is None and require_artifact)
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
        elif event.button.id == "quality-cancel":
            self._request_cancel()
        elif event.button.id in ("quality-start", "quality-prepare") and not self._busy:
            preparing = event.button.id == "quality-prepare"
            try:
                request = self._request(require_artifact=not preparing)
            except ValueError as exc:
                self._set_status(str(exc))
                self.query_one("#quality-setup", Collapsible).collapsed = False
                for widget in self.query(Input):
                    if not widget.value.strip():
                        widget.focus()
                        break
                return
            self._cancel.clear()
            self._busy = True
            self._set_busy(True)
            if preparing:
                self._set_status("Preparing evaluator infrastructure")
                self._executor.submit(self._prepare, self._app().advisor, request)
                return
            self.query_one("#quality-result", Static).update("")
            self._set_status(
                f"Preparing {request.profile.value}; output: {request.output}"
            )
            allow_download = (
                self.query_one("#quality-download", Checkbox).value
                if self._plan is not None else False
            )
            self._executor.submit(self._run, self._app().advisor, request, allow_download)

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
            message = _QualityFinished(stored, readout)
        except QualityEvaluationCancelled as exc:
            message = _QualityFinished(None, str(exc))
        except Exception as exc:
            # Worker boundary: preserve unexpected failures, never import a success.
            _log.exception("Quality evaluation failed")
            stage = "Result display failed" if stored is not None else "Evaluation failed"
            message = _QualityFinished(stored, f"{stage}: {exc}\nOutput: {request.output}")
        if not self._quality_closing.is_set():
            self.post_message(message)

    @on(_QualityProgress)
    def _progress(self, message: _QualityProgress) -> None:
        if not self._cancel.is_set():
            self._set_status(message.text)

    @on(_QualityFinished)
    def _finished(self, message: _QualityFinished) -> None:
        self._busy = False
        self._stored = message.path
        self._set_busy(False)
        self._set_status(
            f"Saved: {message.path}" if message.path else message.readout.splitlines()[0]
        )
        result = self.query_one("#quality-result", Static)
        result.update(message.readout)
        result.scroll_visible(animate=False)
        if self._quit:
            self.app.exit()
        elif self._leave:
            self.dismiss(self._stored)

    def _set_busy(self, busy: bool) -> None:
        for widget in self.query(Input):
            widget.disabled = busy
        for checkbox in self.query(Checkbox):
            checkbox.disabled = busy
        self.query_one("#quality-profile", Select).disabled = busy
        self.query_one("#quality-start", Button).disabled = busy
        self.query_one("#quality-prepare", Button).disabled = busy
        self.query_one("#quality-start", Button).tooltip = (
            "Evaluation in progress." if busy else None
        )
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
        if self._busy:
            self._leave = True
            self._request_cancel()
        else:
            self.dismiss(self._stored)

    def action_quit(self) -> None:
        if self._busy:
            self._quit = True
            self._request_cancel()
        else:
            self.app.exit()

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app
