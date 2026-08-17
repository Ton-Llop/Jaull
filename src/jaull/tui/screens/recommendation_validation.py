from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Static

from jaull.artifacts.errors import ArtifactError
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution_plans import ExecutionPlan, ModelIdentity
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
    LlamaCppRuntimeCapability,
    PyTorchRuntimeCapability,
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
from jaull.presentation.plan_labels import model_display_name, plan_backend
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.artifact_preparation import (
    prepare_recommendation_artifact,
    transformers_recommendation_artifact,
)
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.context_bar import ContextBar
from jaull.tui.widgets.metric_list import MetricList, MetricRow
from jaull.tui.widgets.progress_step import format_step_log
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.technical_details import TechnicalDetails

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

    def __init__(
        self,
        recommendation: ModelRecommendation,
        execution_plan: ExecutionPlan | None = None,
    ) -> None:
        super().__init__()
        self._recommendation = recommendation
        self._execution_plan = execution_plan
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
        yield _ValidationHeader(self._recommendation, self._execution_plan)
        with VerticalScroll(id="validation-body"):
            yield Static(
                "Validate runs a fixed, controlled prompt and stores one "
                "experiment record. Normal Run remains separate.",
                id="validation-intro",
                classes="text-muted",
            )
            yield Static("", id="validation-status", classes="status-line")
            yield Static("", id="validation-error", classes="warning-line")
            # The result lands above the actions. It used to be mounted at the
            # end of the body, so the outcome of the run sat below the button
            # that started it, under a fully expanded progress log.
            yield Vertical(id="validation-result")
            with Horizontal(id="validation-actions"):
                yield Button(
                    "Validate on this machine",
                    id="validation-start",
                    classes="-primary",
                )
                yield ActionButton(
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
            self._set_error("Validation requires an executable runtime recommendation.")
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
        if runtime is None or (estimate is None and self._execution_plan is None):
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
            self._execution_plan,
        )

    def _validation_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        estimate: MemoryEstimate | None,
        runtime: RuntimeRecommendation,
        execution_plan: ExecutionPlan | None,
    ) -> None:
        try:
            if execution_plan is not None:
                prepared = advisor.prepare_execution_plan(
                    execution_plan,
                    hardware=hardware,
                    on_progress=self._post_step,
                )
                artifact = prepared.artifact
                runtime = prepared.plan.runtime
                estimate = prepared.plan.memory_prediction
                hardware = prepared.plan.hardware or hardware
                selection = prepared.plan.backend_selection
                if estimate is None or hardware is None or selection is None:
                    raise ExperimentConfigurationError(
                        "Prepared execution plan is missing required validation data."
                    )
            elif runtime.runtime is RuntimeName.TRANSFORMERS:
                self._post_step("Preparing Transformers runtime")
                artifact = transformers_recommendation_artifact(self._recommendation)
            else:
                artifact = prepare_recommendation_artifact(
                    advisor=advisor,
                    recommendation=self._recommendation,
                    report_step=self._post_step,
                )
            if hardware is None:
                self._post_step("Scanning hardware")
                hardware = advisor.scan_hardware()
            if execution_plan is None:
                assert estimate is not None
                self._post_step("Selecting compute backend")
                selection = advisor.select_runtime_backend(hardware)
            if estimate is None or selection is None:
                raise ExperimentConfigurationError(
                    "Selected execution path is missing prediction or backend data."
                )
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
        self._set_status(format_step_log(self._log_messages))

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
        body = self.query_one("#validation-result", Vertical)
        self.query_one("#validation-intro", Static).display = False
        succeeded = record.observation.success
        heading = (
            "Configuration validated" if succeeded else "Validation failed during execution"
        )
        if persistence_failed:
            heading = f"{heading} (not saved)"
        body.mount(
            Static(
                heading,
                classes=(
                    "result-headline validation-result-node "
                    f"{'-ok' if succeeded else '-fail'}"
                ),
            )
        )
        # Result: what actually happened, in the units a person cares about.
        body.mount(
            MetricList("Result", _result_rows(record), emphasis="measured").add_class(
                "validation-result-node"
            )
        )
        body.mount(
            _prediction_block(record).add_class("validation-result-node")
        )
        body.mount(
            MetricList("Execution", _execution_rows(record)).add_class(
                "validation-result-node"
            )
        )
        # Record id, storage path and the step log are reproducibility detail,
        # not the outcome. They stay one keystroke away rather than competing
        # with the result for the top of the screen.
        detail_rows = [
            ("Experiment ID", record.identity.experiment_id),
            ("Saved", str(persisted_path) if persisted_path else "not saved"),
        ]
        detail_rows.extend(
            (f"Step {index}", step)
            for index, step in enumerate(self._log_messages, start=1)
        )
        body.mount(TechnicalDetails(detail_rows).add_class("validation-result-node"))
        self._set_status("")
        self.query_one("#validation-details", Button).disabled = False

    def _estimate(self) -> MemoryEstimate | None:
        if self._execution_plan is not None:
            return self._execution_plan.memory_prediction
        return self._recommendation.evaluated.memory_estimate

    def _runtime(self) -> RuntimeRecommendation | None:
        if self._execution_plan is not None:
            return self._execution_plan.runtime
        estimate = self._estimate()
        if estimate is None or estimate.runtime_recommendation is None:
            return None
        runtime = estimate.runtime_recommendation
        if runtime.runtime not in {RuntimeName.LLAMA_CPP, RuntimeName.TRANSFORMERS}:
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
            )
            yield SummaryCard("Artifact", _technical_artifact_rows(self._record))
            yield SummaryCard("Runtime", _technical_runtime_rows(self._record))
            yield SummaryCard("Observation", _technical_observation_rows(self._record))
            yield ActionButton("Back", id="validation-details-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "validation-details-back":
            self.app.pop_screen()


class _ValidationHeader(ContextBar):
    """The model and the execution path being validated."""

    def __init__(
        self,
        recommendation: ModelRecommendation,
        execution_plan: ExecutionPlan | None = None,
    ) -> None:
        if execution_plan is not None:
            title = model_display_name(execution_plan.model_identity)
            subtitle = " · ".join(
                [
                    execution_plan.runtime.runtime.value,
                    plan_backend(execution_plan),
                    execution_plan.artifact.label,
                ]
            )
        else:
            config = recommendation.evaluated.selected_configuration
            quantization = config.quantization if config is not None else "default"
            estimate = recommendation.evaluated.memory_estimate
            runtime = estimate.runtime_recommendation if estimate is not None else None
            runtime_label = runtime.runtime.value if runtime is not None else "unknown"
            title = model_display_name(
                ModelIdentity(model_name=recommendation.repo_id.split("/")[-1]),
                fallback=recommendation.repo_id,
            )
            subtitle = f"{quantization} · {runtime_label}"
        super().__init__(title, subtitle, aside="Validate", id="validation-header")


def _humanise(value: str) -> str:
    """An enum value as a phrase — "correct_success" is a wire format."""
    return value.replace("_", " ").capitalize()


def _result_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    """What the run actually did, before anything about how it was configured."""
    observation = record.observation
    rows = [
        ("Outcome", "Success" if observation.success else "Failed"),
        ("Duration", _duration(observation.duration_seconds)),
    ]
    if observation.peak_ram_bytes is not None:
        rows.append(("Peak RAM", format_bytes(observation.peak_ram_bytes)))
    if observation.peak_vram_bytes is not None:
        rows.append(("Peak VRAM", format_bytes(observation.peak_vram_bytes)))
    if observation.failure_reason is not None:
        rows.append(("Failure", observation.failure_reason.value))
    return rows


def _prediction_block(record: ExperimentRecord) -> Widget:
    """Estimated against measured, and why the two sometimes cannot be compared.

    The domain writes an explanatory sentence into ``unavailable_reason`` — VRAM
    is never predicted, and RAM is not predicted under GPU offload. The screen
    used to show only the bare enum, so "comparison unavailable" appeared with
    no way to find out why.
    """
    comparison = record.comparison
    children: list[Widget] = [Static("Prediction", classes="section-title")]
    children.append(
        MetricRow(
            "Estimated RAM",
            format_bytes(comparison.ram.predicted_bytes),
            emphasis="estimated",
        )
    )
    children.append(
        MetricRow(
            "Measured RAM",
            format_bytes(comparison.ram.measured_bytes),
            emphasis="measured",
        )
    )
    children.append(MetricRow("Difference", _metric_error(comparison.ram.error_percent)))
    children.append(
        MetricRow("Compatibility", _humanise(comparison.compatibility.outcome.value))
    )
    for reason in (comparison.ram.unavailable_reason, comparison.vram.unavailable_reason):
        if reason:
            children.append(Static(reason, classes="text-muted-tight"))
    return Vertical(*children, classes="section")


def _execution_rows(record: ExperimentRecord) -> list[tuple[str, str]]:
    """How it ran. Raw capability and readiness reasons stay in the details."""
    readiness = record.preflight.execution_readiness
    accelerator = readiness.selection.selected_accelerator
    return [
        ("Runtime", record.runtime.runtime.value),
        ("Backend", readiness.selection.selected_backend.value),
        (
            "Accelerator",
            accelerator.name if accelerator is not None else "CPU fallback",
        ),
        ("Readiness", readiness.status.value),
    ]


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
    rows = [
        ("Runtime", runtime.runtime.value),
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
    if isinstance(capability, LlamaCppRuntimeCapability):
        rows[1:1] = [
            ("Binary", capability.binary_path or "unknown"),
            ("Binary status", capability.binary_status.value),
            ("Version", capability.version_text or "unknown"),
        ]
    elif isinstance(capability, PyTorchRuntimeCapability):
        rows[1:1] = [
            ("Python", capability.python_executable or "unknown"),
            ("Runtime status", capability.runtime_status.value),
            ("Torch", capability.torch_version or "unknown"),
            ("Transformers", capability.transformers_version or "unknown"),
        ]
    return rows


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
