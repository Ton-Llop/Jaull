from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Static

from jaull.artifacts.errors import ArtifactError
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.experiments import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentRunResult,
    ExperimentWorkload,
    RequestedComputeBackend,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    ExecutionReadinessStatus,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.exceptions import (
    InvalidModelReferenceError,
    JaullError,
    QuantizationNotFoundError,
)
from jaull.experiments.errors import (
    ExperimentConfigurationError,
    ExperimentNotReadyError,
    ExperimentPersistenceError,
    ExperimentRunnerError,
)
from jaull.presentation.console import format_bytes
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.artifact_preparation import prepare_recommendation_artifact
from jaull.tui.widgets.summary_card import SummaryCard

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


VALIDATION_PROMPT = "Explain briefly what artificial intelligence is."


class _ValidationStep(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class _ValidationDone(Message):
    def __init__(self, result: ExperimentRunResult) -> None:
        super().__init__()
        self.result = result


class _ValidationNotReady(Message):
    def __init__(self, error: ExperimentNotReadyError) -> None:
        super().__init__()
        self.error = error


class _ValidationPersistenceFailed(Message):
    def __init__(self, error: ExperimentPersistenceError) -> None:
        super().__init__()
        self.error = error


class _ValidationFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class RecommendationValidationScreen(Screen[None]):
    """Run one controlled validation experiment for a recommendation."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, recommendation: ModelRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._log_messages: list[str] = []
        self._last_result: ExperimentRunResult | None = None
        self._last_record: ExperimentRecord | None = None
        self._last_persisted_path: Path | None = None
        self._validation_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-validation",
        )
        self._future: Future[None] | None = None

    @property
    def recommendation(self) -> ModelRecommendation:
        return self._recommendation

    def compose(self) -> ComposeResult:
        yield _ValidationHeader(self._recommendation)
        with VerticalScroll(id="validation-body"):
            yield Static(
                "Validate runs a fixed, controlled prompt and stores one "
                "experiment record. Normal Run remains separate.",
                id="validation-intro",
                classes="text-muted",
            )
            yield Static("", id="validation-status", classes="status-line")
            yield Static("", id="validation-error", classes="warning-line")
            with Horizontal(id="validation-actions"):
                yield Button(
                    "Validate on this machine",
                    id="validation-start",
                    classes="-primary",
                )
                yield Button(
                    "Technical details",
                    id="validation-details",
                    disabled=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        runtime = self._runtime()
        start = self.query_one("#validation-start", Button)
        if runtime is None:
            start.disabled = True
            self._set_error("Validation requires a llama.cpp runtime recommendation.")
        else:
            start.focus()
        self._set_status("")

    def on_unmount(self) -> None:
        self._validation_closing.set()
        self._shutdown_executor()

    def _shutdown_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "validation-start":
                self._start_validation()
            case "validation-details":
                if self._last_record is not None:
                    self.app.push_screen(
                        ValidationDetailsScreen(
                            self._last_record,
                            self._last_persisted_path,
                        )
                    )

    def _start_validation(self) -> None:
        if self._future is not None and not self._future.done():
            return
        estimate = self._estimate()
        runtime = self._runtime()
        if estimate is None or runtime is None:
            self._set_error("Validation requires a complete memory/runtime estimate.")
            return

        app = self._app()
        self._clear_result()
        self._log_messages.clear()
        self._set_error("")
        self._set_busy(True)
        self._record_step("Preparing validation")
        self._future = self._executor.submit(
            self._validation_worker,
            app.advisor,
            app.hardware_profile,
            estimate,
            runtime,
        )

    def _validation_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        estimate: MemoryEstimate,
        runtime: RuntimeRecommendation,
    ) -> None:
        try:
            artifact = prepare_recommendation_artifact(
                advisor=advisor,
                recommendation=self._recommendation,
                report_step=self._post_step,
            )
            if hardware is None:
                self._post_step("Scanning hardware")
                hardware = advisor.scan_hardware()
            self._post_step("Selecting compute backend")
            selection = advisor.select_runtime_backend(hardware)
            self._post_step("Checking runtime readiness")
            request = ExperimentRequest(
                hardware=hardware,
                artifact=artifact,
                runtime=runtime,
                prediction=estimate,
                backend_selection=selection,
                requested_backend=RequestedComputeBackend.AUTO,
                workload=ExperimentWorkload(prompt=VALIDATION_PROMPT),
                persist=True,
            )
            self._post_step("Running controlled validation")
            result = advisor.run_experiment(request)
        except ExperimentNotReadyError as exc:
            self._post_not_ready(exc)
            return
        except ExperimentPersistenceError as exc:
            self._post_persistence_failed(exc)
            return
        except (
            InvalidModelReferenceError,
            QuantizationNotFoundError,
            ArtifactError,
            ExperimentConfigurationError,
            ExperimentRunnerError,
            JaullError,
            OSError,
            ValueError,
        ) as exc:
            self._post_failed(str(exc))
            return
        if not self._validation_closing.is_set():
            self.post_message(_ValidationDone(result))

    @on(_ValidationStep)
    def _validation_step_message(self, message: _ValidationStep) -> None:
        if self._validation_closing.is_set():
            return
        self._record_step(message.message)

    @on(_ValidationDone)
    def _validation_done_message(self, message: _ValidationDone) -> None:
        if self._validation_closing.is_set():
            return
        self._last_result = message.result
        self._last_record = message.result.record
        self._last_persisted_path = message.result.persisted_path
        self._set_busy(False)
        self._set_error("")
        self._render_record(message.result.record, message.result.persisted_path)

    @on(_ValidationNotReady)
    def _validation_not_ready_message(self, message: _ValidationNotReady) -> None:
        if self._validation_closing.is_set():
            return
        self._set_busy(False)
        self._last_result = None
        self._last_record = None
        self._last_persisted_path = None
        readiness = message.error.readiness
        if readiness.status is ExecutionReadinessStatus.UNKNOWN:
            title = "Could not verify runtime readiness."
        else:
            title = "Validation unavailable."
        detail = readiness.message or str(message.error)
        self._set_error(f"{title} {detail}")

    @on(_ValidationPersistenceFailed)
    def _validation_persistence_failed_message(
        self, message: _ValidationPersistenceFailed
    ) -> None:
        if self._validation_closing.is_set():
            return
        self._set_busy(False)
        record = message.error.record
        self._last_result = None
        self._last_record = record
        self._last_persisted_path = None
        self._set_error(
            "Validation completed, but the experiment could not be saved."
        )
        self._render_record(record, None, persistence_failed=True)

    @on(_ValidationFailed)
    def _validation_failed_message(self, message: _ValidationFailed) -> None:
        if self._validation_closing.is_set():
            return
        self._set_busy(False)
        self._set_error(message.message)

    def _post_step(self, message: str) -> None:
        if not self._validation_closing.is_set():
            self.post_message(_ValidationStep(message))

    def _post_not_ready(self, error: ExperimentNotReadyError) -> None:
        if not self._validation_closing.is_set():
            self.post_message(_ValidationNotReady(error))

    def _post_persistence_failed(self, error: ExperimentPersistenceError) -> None:
        if not self._validation_closing.is_set():
            self.post_message(_ValidationPersistenceFailed(error))

    def _post_failed(self, message: str) -> None:
        if not self._validation_closing.is_set():
            self.post_message(_ValidationFailed(message))

    def _record_step(self, message: str) -> None:
        self._log_messages.append(message)
        self._set_status("\n".join(f"- {item}" for item in self._log_messages[-8:]))

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#validation-start", Button).disabled = busy
        self.query_one("#validation-details", Button).disabled = (
            busy or self._last_record is None
        )

    def _set_error(self, message: str) -> None:
        widget = self.query_one("#validation-error", Static)
        widget.update(message)
        widget.display = bool(message)

    def _set_status(self, message: str) -> None:
        widget = self.query_one("#validation-status", Static)
        widget.update(message)
        widget.display = bool(message)

    def _clear_result(self) -> None:
        for node in list(self.query(".validation-result-node")):
            node.remove()
        self._last_result = None
        self._last_record = None
        self._last_persisted_path = None
        self.query_one("#validation-details", Button).disabled = True

    def _render_record(
        self,
        record: ExperimentRecord,
        persisted_path: Path | None,
        *,
        persistence_failed: bool = False,
    ) -> None:
        self._clear_result()
        self._last_record = record
        self._last_persisted_path = persisted_path
        body = self.query_one("#validation-body", VerticalScroll)
        heading = (
            "Configuration validated"
            if record.observation.success
            else "Validation failed during execution"
        )
        if persistence_failed:
            heading = f"{heading} (not saved)"
        body.mount(Static(heading, classes="section-title validation-result-node"))
        body.mount(
            SummaryCard(
                "Backend",
                _backend_rows(record),
                plain=True,
            ).add_class("validation-result-node")
        )
        body.mount(
            SummaryCard(
                "Prediction vs Observation",
                _comparison_rows(record),
                plain=True,
            ).add_class("validation-result-node")
        )
        body.mount(
            SummaryCard(
                "Experiment",
                [
                    ("ID", record.identity.experiment_id),
                    ("Status", "success" if record.observation.success else "failure"),
                    ("Duration", _duration(record.observation.duration_seconds)),
                    ("Saved", str(persisted_path) if persisted_path else "not saved"),
                ],
                plain=True,
            ).add_class("validation-result-node")
        )
        self.query_one("#validation-details", Button).disabled = False

    def _estimate(self) -> MemoryEstimate | None:
        return self._recommendation.evaluated.memory_estimate

    def _runtime(self) -> RuntimeRecommendation | None:
        estimate = self._estimate()
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


class ValidationDetailsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, record: ExperimentRecord, path: Path | None) -> None:
        super().__init__()
        self._record = record
        self._path = path

    def compose(self) -> ComposeResult:
        yield Static("Validation details", classes="section-title")
        with VerticalScroll(id="validation-details-body"):
            yield SummaryCard(
                "Experiment",
                _technical_experiment_rows(self._record, self._path),
                plain=True,
            )
            yield SummaryCard("Artifact", _technical_artifact_rows(self._record), plain=True)
            yield SummaryCard("Runtime", _technical_runtime_rows(self._record), plain=True)
            yield SummaryCard("Observation", _technical_observation_rows(self._record), plain=True)
            yield Button("Back", id="validation-details-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "validation-details-back":
            self.app.pop_screen()


class _ValidationHeader(Static):
    def __init__(self, recommendation: ModelRecommendation) -> None:
        config = recommendation.evaluated.selected_configuration
        quantization = config.quantization if config is not None else "default"
        estimate = recommendation.evaluated.memory_estimate
        runtime = estimate.runtime_recommendation if estimate is not None else None
        runtime_label = runtime.runtime.value if runtime is not None else "unknown"
        super().__init__(
            f"[bold]{recommendation.repo_id}[/bold]\n"
            f"{quantization} · {runtime_label}",
            id="validation-header",
        )


def _backend_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    readiness = record.preflight.execution_readiness
    accelerator = readiness.selection.selected_accelerator
    return [
        ("Requested", record.backend_trace.requested_backend.value),
        ("Selected", readiness.selection.selected_backend.value),
        (
            "Accelerator",
            accelerator.name if accelerator is not None else "CPU fallback",
        ),
        (
            "Observed",
            record.backend_trace.observed_backend.value
            if record.backend_trace.observed_backend is not None
            else "Not verified",
        ),
        ("Readiness", readiness.status.value),
    ]


def _comparison_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    comparison = record.comparison
    rows = [
        ("Predicted RAM", format_bytes(comparison.ram.predicted_bytes)),
        ("Observed RAM", format_bytes(comparison.ram.measured_bytes)),
        ("RAM error", _metric_error(comparison.ram.error_percent)),
        ("Peak VRAM", format_bytes(record.observation.peak_vram_bytes)),
        ("Compatibility", comparison.compatibility.outcome.value),
    ]
    if comparison.ram.unavailable_reason is not None:
        rows.append(("RAM comparison", comparison.ram.availability.value))
    if comparison.vram.unavailable_reason is not None:
        rows.append(("VRAM comparison", comparison.vram.availability.value))
    return rows


def _technical_experiment_rows(
    record: ExperimentRecord,
    path: Path | None,
) -> list[tuple[str, str]]:
    return [
        ("ID", record.identity.experiment_id),
        ("Created", record.identity.created_at.isoformat()),
        ("Schema", str(record.schema_version)),
        ("Jaull", record.environment.jaull_version or "unknown"),
        ("Python", record.environment.python_version or "unknown"),
        ("Saved", str(path) if path else "not saved"),
    ]


def _technical_artifact_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    artifact = record.artifact
    return [
        ("Repository", artifact.repo_id),
        ("Revision", artifact.revision or "unknown"),
        ("Filename", artifact.filename),
        ("Format", artifact.format),
        ("SHA-256", artifact.sha256 or "unknown"),
        ("Local path", str(artifact.local_path) if artifact.local_path else "unknown"),
    ]


def _technical_runtime_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    runtime = record.runtime
    capability = record.preflight.runtime_capability
    flags = ", ".join(f"{flag.name}={flag.value}" for flag in runtime.flags) or "none"
    return [
        ("Runtime", runtime.runtime.value),
        ("Binary", capability.binary_path or "unknown"),
        ("Binary status", capability.binary_status.value),
        ("Version", capability.version_text or "unknown"),
        ("Requested", record.backend_trace.requested_backend.value),
        (
            "Selected",
            record.preflight.execution_readiness.selection.selected_backend.value,
        ),
        (
            "Observed",
            record.backend_trace.observed_backend.value
            if record.backend_trace.observed_backend is not None
            else "Not verified",
        ),
        ("Flags", flags),
    ]


def _technical_observation_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    observation = record.observation
    return [
        ("Success", "yes" if observation.success else "no"),
        (
            "Exit code",
            str(observation.exit_code)
            if observation.exit_code is not None
            else "unknown",
        ),
        ("Failure", observation.failure_reason.value if observation.failure_reason else "none"),
        ("Duration", _duration(observation.duration_seconds)),
        ("Peak RAM", format_bytes(observation.peak_ram_bytes)),
        ("Peak VRAM", format_bytes(observation.peak_vram_bytes)),
    ]


def _metric_error(value: float | None) -> str:
    if value is None:
        return "comparison unavailable"
    return f"{value:+.1f}%"


def _duration(value: float) -> str:
    return f"{value:.2f} s"


__all__ = [
    "VALIDATION_PROMPT",
    "RecommendationValidationScreen",
    "ValidationDetailsScreen",
]
