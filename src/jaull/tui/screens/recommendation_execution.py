from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, LoadingIndicator, Static

from jaull.artifacts.errors import ArtifactError
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.execution import InferenceResult
from jaull.domain.runtime import RuntimeName, RuntimeRecommendation
from jaull.exceptions import InvalidModelReferenceError, QuantizationNotFoundError
from jaull.execution.errors import ExecutionError
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowStep

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp


class RecommendationExecutionScreen(Screen[None]):
    """Prepare and run one selected recommendation with the app facade."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._artifact: ModelArtifact | None = None
        self._log_messages: list[str] = []

    @property
    def recommendation(self) -> ModelRecommendation:
        return self._recommendation

    @property
    def log_messages(self) -> tuple[str, ...]:
        return tuple(self._log_messages)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield WorkflowHeader(
            WorkflowStep.RANKING,
            "Run selected model",
            "Download, verify and execute this recommendation without leaving jaull.",
        )
        yield LoadingIndicator(id="run-loading")
        with VerticalScroll(id="run-body"):
            yield SummaryCard("Selected recommendation", self._review_rows())
            yield Static("Status: Not prepared", id="run-status", classes="card")
            with Vertical(classes="card", id="run-log"):
                yield Static("Execution steps", classes="card-title")
                yield Static("Waiting for user action.", classes="text-muted")
            yield Vertical(id="run-error")
            with Horizontal(id="run-actions"):
                yield Button("Prepare model", id="run-prepare", classes="-primary")
                yield Button("Back", id="run-back")
            yield Vertical(id="run-prompt")
            yield Vertical(id="run-result")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#run-loading", LoadingIndicator).display = False
        if self._runtime() is None:
            self._render_error(
                "This recommendation has no executable llama.cpp runtime configuration."
            )
            self.query_one("#run-prepare", Button).disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "run-back":
                self.app.pop_screen()
            case "run-prepare":
                self._start_prepare()
            case "run-execute":
                self._start_run()
            case "run-another":
                self._render_prompt()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "run-prompt-input":
            self._start_run()

    # ------------------------------------------------------------------
    # Prepare artifact
    # ------------------------------------------------------------------
    def _start_prepare(self) -> None:
        self._clear_error()
        self._reset_log("Preparing artifact")
        self._set_busy(True, "Resolving artifact...")
        self.query_one("#run-prepare", Button).disabled = True
        self.run_worker(self._prepare_worker, thread=True)

    def _prepare_worker(self) -> None:
        app = self._app()
        rec = self._recommendation
        config = rec.evaluated.selected_configuration
        quantization = config.quantization if config is not None else None
        try:
            app.call_from_thread(
                self._record_step,
                f"Resolve artifact for {rec.repo_id}"
                + (f" ({quantization})" if quantization else ""),
            )
            artifact = app.advisor.resolve_artifact(
                rec.repo_id,
                quantization=quantization,
                revision=None,
            )
            if not artifact.is_downloaded:
                app.call_from_thread(
                    self._record_step,
                    f"Download {artifact.filename} into artifact storage",
                )
                artifact = app.advisor.download_artifact(artifact)
            else:
                app.call_from_thread(
                    self._record_step,
                    f"Reuse local artifact {artifact.filename}",
                )
            app.call_from_thread(
                self._record_step,
                "Verify size and SHA-256 sidecar",
            )
            artifact = app.advisor.verify_artifact(artifact)
        except (
            InvalidModelReferenceError,
            QuantizationNotFoundError,
            ArtifactError,
        ) as exc:
            app.call_from_thread(self._prepare_failed, str(exc))
            return
        app.call_from_thread(self._prepare_done, artifact)

    def _prepare_done(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact
        self._record_step("Artifact ready for execution")
        self._set_busy(False, "Artifact verified")
        self.query_one("#run-prepare", Button).disabled = False
        self._render_prompt()

    def _prepare_failed(self, message: str) -> None:
        self._set_busy(False, "Artifact preparation failed")
        self.query_one("#run-prepare", Button).disabled = False
        self._render_error(message)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    def _start_run(self) -> None:
        prompt = self.query_one("#run-prompt-input", Input).value.strip()
        if not prompt:
            self._render_error("Prompt must not be empty.")
            return
        if self._artifact is None:
            self._render_error("Prepare the model before running inference.")
            return
        runtime = self._runtime()
        if runtime is None:
            self._render_error(
                "This recommendation has no executable llama.cpp runtime configuration."
            )
            return

        self._clear_error()
        self._record_step("Running inference")
        self._record_step("Use selected RuntimeRecommendation")
        self._record_step("Start llama.cpp single-turn generation")
        self.query_one("#run-execute", Button).disabled = True
        self._set_busy(True, "Loading model and generating...")
        self.run_worker(lambda: self._run_worker(prompt, runtime), thread=True)

    def _run_worker(self, prompt: str, runtime: RuntimeRecommendation) -> None:
        app = self._app()
        assert self._artifact is not None
        try:
            result = app.advisor.run_artifact(
                artifact=self._artifact,
                prompt=prompt,
                runtime=runtime,
            )
        except ExecutionError as exc:
            app.call_from_thread(self._run_failed, str(exc))
            return
        app.call_from_thread(self._run_done, result)

    def _run_done(self, result: InferenceResult) -> None:
        self._set_busy(False, "Inference complete")
        self._record_step("Generation finished")
        self.query_one("#run-execute", Button).disabled = False
        result_box = self.query_one("#run-result", Vertical)
        result_box.remove_children()
        result_box.mount(Static("Model response", classes="card-title"))
        result_box.mount(
            Static(
                Text(_display_text(result.text)),
                id="run-response",
                classes="card",
            )
        )
        result_box.mount(
            SummaryCard(
                "Execution",
                [
                    ("Model", self._recommendation.repo_id),
                    ("Runtime", result.runtime),
                    ("Artifact", str(result.model_path)),
                ],
            )
        )
        result_box.mount(Button("Run another prompt", id="run-another"))

    def _run_failed(self, message: str) -> None:
        self._set_busy(False, "Inference failed")
        self.query_one("#run-execute", Button).disabled = False
        self._render_error(message)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_prompt(self) -> None:
        prompt_box = self.query_one("#run-prompt", Vertical)
        prompt_box.remove_children()
        prompt_box.mount(Static("Prompt", classes="card-title"))
        prompt_box.mount(
            Input(placeholder="Ask a single-turn question", id="run-prompt-input")
        )
        prompt_box.mount(
            Horizontal(
                Button("Run", id="run-execute", classes="-primary"),
                Button("Back", id="run-back"),
            )
        )
        self.query_one("#run-prompt-input", Input).focus()

    def _render_error(self, message: str) -> None:
        error_box = self.query_one("#run-error", Vertical)
        error_box.remove_children()
        error_box.mount(WarningsPanel([message]))
        error_box.mount(Static(message, id="run-error-message", classes="warning-line"))

    def _clear_error(self) -> None:
        self.query_one("#run-error", Vertical).remove_children()

    def _set_busy(self, busy: bool, status: str) -> None:
        self.query_one("#run-loading", LoadingIndicator).display = busy
        self._set_status(status)

    def _set_status(self, status: str) -> None:
        self.query_one("#run-status", Static).update(f"Status: {status}")

    def _reset_log(self, title: str) -> None:
        self._log_messages = []
        log = self.query_one("#run-log", Vertical)
        log.remove_children()
        log.mount(Static(title, classes="card-title"))

    def _record_step(self, message: str) -> None:
        self._log_messages.append(message)
        self._set_status(message)
        self.query_one("#run-log", Vertical).mount(Static(f"• {message}"))

    def _review_rows(self) -> list[tuple[str, str]]:
        rec = self._recommendation
        config = rec.evaluated.selected_configuration
        estimate = rec.evaluated.memory_estimate
        runtime = self._runtime()
        rows: list[tuple[str, str]] = [
            ("Model", rec.repo_id),
            ("Artifact", "GGUF" if config is not None and config.quantization else "unknown"),
            (
                "Quantization",
                config.quantization if config is not None and config.quantization else "unknown",
            ),
            ("Runtime", runtime.runtime.value if runtime is not None else "unavailable"),
            (
                "Context",
                f"{config.context_length} tokens" if config is not None else "unknown",
            ),
            ("GPU offload", _flag_value(runtime, "--n-gpu-layers") or "unknown"),
        ]
        if estimate is not None and estimate.total_bytes is not None:
            rows.append(("Estimated memory", f"{estimate.total_bytes / 1024**3:.2f} GiB"))
        return rows

    def _runtime(self) -> RuntimeRecommendation | None:
        estimate = self._recommendation.evaluated.memory_estimate
        if estimate is None or estimate.runtime_recommendation is None:
            return None
        runtime = estimate.runtime_recommendation
        if runtime.runtime is not RuntimeName.LLAMA_CPP:
            return None
        return runtime

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


def _flag_value(runtime: RuntimeRecommendation | None, name: str) -> str | None:
    if runtime is None:
        return None
    return next((flag.value for flag in runtime.flags if flag.name == name), None)


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _display_text(raw: str) -> str:
    text = _ANSI_RE.sub("", raw)
    text = "".join(
        char
        for char in text
        if char in {"\n", "\t"} or ord(char) >= 32
    ).strip()
    return text or "(empty response)"


__all__ = ["RecommendationExecutionScreen"]
