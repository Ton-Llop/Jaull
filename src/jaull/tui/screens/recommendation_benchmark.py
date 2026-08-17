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
from textual.widgets import Button, DataTable, Footer, Static

from jaull.artifacts.errors import ArtifactError
from jaull.benchmarks.errors import BenchmarkRunnerError
from jaull.benchmarks.matrix import (
    BenchmarkMatrixFailure,
    BenchmarkMatrixRequest,
    BenchmarkMatrixResult,
)
from jaull.domain.benchmarks import (
    BenchmarkGpuLayers,
    BenchmarkMeasurementKind,
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkRunResult,
)
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution_plans import ExecutionPlan, ModelIdentity
from jaull.domain.hardware import ComputeBackend, HardwareProfile
from jaull.domain.runtime import (
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.benchmarks import aggregate_benchmark_records
from jaull.exceptions import (
    InvalidModelReferenceError,
    JaullError,
    QuantizationNotFoundError,
)
from jaull.presentation.plan_labels import backend_hint, model_display_name
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.artifact_preparation import (
    prepare_recommendation_artifact,
    transformers_recommendation_artifact,
)
from jaull.tui.widgets.action_button import ActionButton
from jaull.tui.widgets.bars import ratio_bar
from jaull.tui.widgets.context_bar import ContextBar
from jaull.tui.widgets.metric_list import MetricList, MetricRow
from jaull.tui.widgets.progress_step import format_step_log
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.technical_details import TechnicalDetails
from jaull.tui.widgets.warnings_panel import WarningsPanel

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.tui.app import JaullApp


class _BenchmarkStep(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class _BenchmarkDone(Message):
    def __init__(self, result: BenchmarkMatrixResult) -> None:
        super().__init__()
        self.result = result


class _BenchmarkFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class RecommendationBenchmarkScreen(Screen[None]):
    """Run reproducible runtime-specific measurements for a recommendation."""

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
        self._last_result: BenchmarkMatrixResult | None = None
        self._benchmark_closing = Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jaull-benchmark",
        )
        self._future: Future[None] | None = None

    @property
    def recommendation(self) -> ModelRecommendation:
        return self._recommendation

    def compose(self) -> ComposeResult:
        yield _BenchmarkHeader(self._recommendation, self._execution_plan)
        with VerticalScroll(id="benchmark-body"):
            yield Static(
                "Benchmark runs a runtime-specific controlled measurement and "
                "stores reproducible performance records. Run and Validate remain "
                "separate workflows.",
                id="benchmark-intro",
                classes="text-muted",
            )
            yield Static("", id="benchmark-status", classes="status-line")
            yield Static("", id="benchmark-error", classes="warning-line")
            # Results land here, above the actions: the outcome is what the
            # user came back for, and it used to appear below the buttons and
            # below the fold.
            yield Vertical(id="benchmark-result")
            with Horizontal(id="benchmark-actions"):
                yield Button(
                    "Benchmark on this machine",
                    id="benchmark-start",
                    classes="-primary",
                )
                yield ActionButton(
                    "Technical details",
                    id="benchmark-details",
                    disabled=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        start = self.query_one("#benchmark-start", Button)
        if self._runtime() is None:
            start.disabled = True
            self._set_error("Benchmark requires an executable runtime recommendation.")
        else:
            start.focus()
        self._set_status("")

    def on_unmount(self) -> None:
        self._benchmark_closing.set()
        self._shutdown_executor()

    def _shutdown_executor(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "benchmark-start":
                self._start_benchmark()
            case "benchmark-details":
                if self._last_result is not None:
                    self.app.push_screen(BenchmarkDetailsScreen(self._last_result))

    def _start_benchmark(self) -> None:
        if self._future is not None and not self._future.done():
            return
        runtime = self._runtime()
        if runtime is None:
            self._set_error("Benchmark requires an executable runtime recommendation.")
            return

        app = self._app()
        self._clear_result()
        self._log_messages.clear()
        self._set_error("")
        self._set_busy(True)
        self._record_step("Preparing benchmark")
        self._future = self._executor.submit(
            self._benchmark_worker,
            app.advisor,
            app.hardware_profile,
            runtime,
            self._execution_plan,
        )

    def _benchmark_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        runtime: RuntimeRecommendation,
        execution_plan: ExecutionPlan | None,
    ) -> None:
        try:
            if execution_plan is not None:
                result = self._plan_benchmark_worker(
                    advisor,
                    hardware,
                    execution_plan,
                )
            elif runtime.runtime is RuntimeName.TRANSFORMERS:
                result = self._transformers_benchmark_worker(
                    advisor,
                    hardware,
                    runtime,
                )
            else:
                result = self._llama_cpp_benchmark_worker(advisor, hardware, runtime)
        except (
            InvalidModelReferenceError,
            QuantizationNotFoundError,
            ArtifactError,
            BenchmarkRunnerError,
            JaullError,
            OSError,
            ValueError,
        ) as exc:
            self._post_failed(str(exc))
            return
        if not self._benchmark_closing.is_set():
            self.post_message(_BenchmarkDone(result))

    def _plan_benchmark_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        execution_plan: ExecutionPlan,
    ) -> BenchmarkMatrixResult:
        self._post_step("Preparing selected execution path")
        prepared = advisor.prepare_execution_plan(
            execution_plan,
            hardware=hardware,
            on_progress=self._post_step,
        )
        plan = prepared.plan
        if plan.hardware is None or plan.backend_selection is None:
            raise BenchmarkRunnerError("Prepared execution plan is incomplete.")
        backend = _benchmark_backend(plan.runtime, plan.backend_selection.selected_backend)
        device = _benchmark_device(backend, plan)
        self._post_step(f"Running selected benchmark: {backend.value}")
        request = BenchmarkRequest(
            artifact=prepared.artifact,
            runtime=plan.runtime,
            backend=backend,
            device=device,
            gpu_layers=_benchmark_gpu_layers(backend, plan.runtime),
            prefill_sizes=(128, 512),
            generation_sizes=(64,),
            repetitions=3,
            timeout_seconds=900.0,
        )
        run = advisor.run_benchmark(request, hardware=plan.hardware)
        if run.record.observation.success:
            return BenchmarkMatrixResult(completed=[run], records=[run.record])
        return BenchmarkMatrixResult(
            failed=[
                BenchmarkMatrixFailure(
                    request=request,
                    message=run.record.observation.message or "Benchmark failed.",
                    record=run.record,
                    persisted_path=run.persisted_path,
                )
            ],
            records=[run.record],
        )

    def _llama_cpp_benchmark_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        runtime: RuntimeRecommendation,
    ) -> BenchmarkMatrixResult:
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
        self._post_step("Checking llama.cpp devices")
        runtime_capability = advisor.inspect_llama_cpp_runtime(selection=selection)
        self._post_step(
            "Running llama-bench: "
            + _benchmark_configuration_label(selection, runtime_capability)
        )
        request = BenchmarkMatrixRequest(
            hardware=hardware,
            artifact=artifact,
            runtime=runtime,
            backend_selection=selection,
            runtime_capability=runtime_capability,
            include_cpu=selection.selected_backend is ComputeBackend.CPU,
            include_selected_backend=True,
            selected_gpu_layers=(
                ()
                if selection.selected_backend is ComputeBackend.CPU
                else (BenchmarkGpuLayers.full(),)
            ),
        )
        return advisor.run_benchmark_matrix(request)

    def _transformers_benchmark_worker(
        self,
        advisor: AdvisorService,
        hardware: HardwareProfile | None,
        runtime: RuntimeRecommendation,
    ) -> BenchmarkMatrixResult:
        self._post_step("Preparing Transformers runtime")
        artifact = transformers_recommendation_artifact(self._recommendation)
        if hardware is None:
            self._post_step("Scanning hardware")
            hardware = advisor.scan_hardware()
        self._post_step("Selecting compute backend")
        selection = advisor.select_runtime_backend(hardware)
        backend = _transformers_backend(runtime, selection.selected_backend)
        self._post_step(f"Running Transformers benchmark: {backend.value}")
        request = BenchmarkRequest(
            artifact=artifact,
            runtime=runtime,
            backend=backend,
            device="none" if backend is ComputeBackend.CPU else None,
            gpu_layers=BenchmarkGpuLayers.count_layers(0),
            prefill_sizes=(128, 512),
            generation_sizes=(64,),
            repetitions=3,
            timeout_seconds=900.0,
        )
        run = advisor.run_benchmark(request, hardware=hardware)
        if run.record.observation.success:
            return BenchmarkMatrixResult(completed=[run], records=[run.record])
        return BenchmarkMatrixResult(
            failed=[
                BenchmarkMatrixFailure(
                    request=request,
                    message=run.record.observation.message
                    or "Transformers benchmark failed.",
                    record=run.record,
                    persisted_path=run.persisted_path,
                )
            ],
            records=[run.record],
        )

    @on(_BenchmarkStep)
    def _benchmark_step_message(self, message: _BenchmarkStep) -> None:
        if self._benchmark_closing.is_set():
            return
        self._record_step(message.message)

    @on(_BenchmarkDone)
    def _benchmark_done_message(self, message: _BenchmarkDone) -> None:
        if self._benchmark_closing.is_set():
            return
        self._last_result = message.result
        self._set_busy(False)
        self._render_result(message.result)

    @on(_BenchmarkFailed)
    def _benchmark_failed_message(self, message: _BenchmarkFailed) -> None:
        if self._benchmark_closing.is_set():
            return
        self._set_busy(False)
        self._set_error(message.message)

    def _post_step(self, message: str) -> None:
        if not self._benchmark_closing.is_set():
            self.post_message(_BenchmarkStep(message))

    def _post_failed(self, message: str) -> None:
        if not self._benchmark_closing.is_set():
            self.post_message(_BenchmarkFailed(message))

    def _record_step(self, message: str) -> None:
        self._log_messages.append(message)
        self._set_status(format_step_log(self._log_messages))

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#benchmark-start", Button).disabled = busy
        self.query_one("#benchmark-details", Button).disabled = (
            busy or self._last_result is None
        )

    def _set_error(self, message: str) -> None:
        widget = self.query_one("#benchmark-error", Static)
        widget.update(message)
        widget.display = bool(message)

    def _set_status(self, message: str) -> None:
        widget = self.query_one("#benchmark-status", Static)
        widget.update(message)
        widget.display = bool(message)

    def _clear_result(self) -> None:
        for node in list(self.query(".benchmark-result-node")):
            node.remove()
        self._last_result = None
        self.query_one("#benchmark-details", Button).disabled = True

    def _render_result(self, result: BenchmarkMatrixResult) -> None:
        """Result first, mechanism last.

        The order is deliberate: whether it worked, how fast it was, what it
        cost in memory, and only then the reproducibility fields. The step log
        that dominated this screen while the benchmark ran collapses into the
        technical details once there is a real result to show.
        """
        self._clear_result()
        self._last_result = result
        body = self.query_one("#benchmark-result", Vertical)
        # The blurb explained what the button would do; it has now done it.
        self.query_one("#benchmark-intro", Static).display = False
        succeeded = bool(result.completed)
        heading = (
            "Benchmark complete"
            if succeeded
            else "Benchmark finished without successful configurations"
        )
        body.mount(
            Static(
                heading,
                classes=(
                    "result-headline benchmark-result-node "
                    f"{'-ok' if succeeded else '-fail'}"
                ),
            )
        )
        if succeeded:
            body.mount(
                _performance_block(result.completed).add_class("benchmark-result-node")
            )
            memory_block = _memory_block(result.completed, self._estimate())
            if memory_block is not None:
                body.mount(memory_block.add_class("benchmark-result-node"))

            startup_rows = _startup_rows(result.completed)
            if startup_rows:
                body.mount(
                    MetricList("Startup", startup_rows).add_class(
                        "benchmark-result-node"
                    )
                )

            aggregate = aggregate_benchmark_records([item.record for item in result.completed])
            if aggregate.comparisons:
                body.mount(
                    MetricList(
                        "Measured differences",
                        [
                            (
                                _metric_label(comparison.kind, comparison.tokens),
                                f"~{comparison.speedup:.1f}x",
                            )
                            for comparison in aggregate.comparisons
                        ],
                    ).add_class("benchmark-result-node")
                )
        warnings = _warnings(result)
        if warnings:
            body.mount(
                WarningsPanel(warnings).add_class("benchmark-result-node")
            )
        # The verbose progress log has served its purpose; it is reproducibility
        # detail now, not the headline.
        if self._log_messages:
            body.mount(
                TechnicalDetails(
                    [(str(index), step) for index, step in enumerate(self._log_messages, 1)],
                    title="Run log",
                ).add_class("benchmark-result-node")
            )
            self._set_status("")
        self.query_one("#benchmark-details", Button).disabled = False

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

    def _estimate(self) -> MemoryEstimate | None:
        return self._recommendation.evaluated.memory_estimate

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


class BenchmarkDetailsScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def __init__(self, result: BenchmarkMatrixResult) -> None:
        super().__init__()
        self._result = result

    def compose(self) -> ComposeResult:
        yield Static("Benchmark details", classes="section-title")
        with VerticalScroll(id="benchmark-details-body"):
            for run in self._result.completed:
                yield SummaryCard(
                    _config_label(run.record),
                    _technical_rows(run.record, run.persisted_path),
                )
            for failure in self._result.failed:
                rows = [("Error", failure.message)]
                if failure.record is not None:
                    rows.extend(_technical_rows(failure.record, failure.persisted_path))
                title = _request_label(
                    failure.request.backend.value,
                    failure.request.device,
                )
                yield SummaryCard(
                    f"Failed: {title}",
                    rows,
                )
            if self._result.skipped:
                yield SummaryCard(
                    "Skipped",
                    [
                        (
                            _request_label(skip.backend.value, skip.device),
                            skip.reason,
                        )
                        for skip in self._result.skipped
                    ],
                )
            yield ActionButton("Back", id="benchmark-details-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "benchmark-details-back":
            self.app.pop_screen()


class _BenchmarkHeader(ContextBar):
    """The model and the execution path being measured.

    Whatever is on screen below must describe the plan named here — that is the
    first of `PRODUCT.md`'s design principles, and the reason every screen now
    opens with the same bar rather than its own header.
    """

    def __init__(
        self,
        recommendation: ModelRecommendation,
        execution_plan: ExecutionPlan | None = None,
    ) -> None:
        if execution_plan is not None:
            title = model_display_name(execution_plan.model_identity)
            subtitle = _execution_plan_label(execution_plan)
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
        super().__init__(title, subtitle, aside="Benchmark", id="benchmark-header")


def _execution_plan_label(plan: ExecutionPlan) -> str:
    runtime = (
        "Transformers · PyTorch"
        if plan.runtime.runtime is RuntimeName.TRANSFORMERS
        else plan.runtime.runtime.value
    )
    artifact = plan.artifact.quantization or plan.artifact.precision or plan.artifact.format.value
    return f"{runtime} · {backend_hint(plan.runtime)} · {artifact}"


_BAR_WIDTH = 18


def _performance_block(results: list[BenchmarkRunResult]) -> Widget:
    """Throughput, grouped by what is being measured.

    One configuration gets bars on a single scale across prompt processing and
    generation, so the gap between them — usually the most useful thing on the
    screen — is visible without arithmetic. Several configurations get the
    table instead, because comparing columns is what a table is for.

    Bars never replace the figure. Every row carries its exact tok/s.
    """
    children: list[Widget] = [Static("Performance", classes="section-title")]
    measurements = (
        sorted(
            results[0].record.observation.measurements,
            key=lambda m: (
                0 if m.kind is BenchmarkMeasurementKind.PREFILL else 1,
                m.tokens,
            ),
        )
        if len(results) == 1
        else []
    )
    if not measurements:
        children.append(_result_table(results))
        return Vertical(*children, classes="section")

    peak = max(m.mean_tokens_per_second for m in measurements)
    current_kind: BenchmarkMeasurementKind | None = None
    for measurement in measurements:
        if measurement.kind is not current_kind:
            first_group = current_kind is None
            current_kind = measurement.kind
            heading = Static(_kind_label(measurement.kind), classes="metric-group")
            if not first_group:
                heading.add_class("-spaced")
            children.append(heading)
        children.append(
            MetricRow(
                f"{measurement.tokens} tokens",
                f"{measurement.mean_tokens_per_second:.1f} tok/s",
                bar=ratio_bar(measurement.mean_tokens_per_second, peak, _BAR_WIDTH),
                emphasis="measured",
            ).add_class("-indented")
        )
    return Vertical(*children, classes="section")


def _kind_label(kind: BenchmarkMeasurementKind) -> str:
    return "Prompt processing" if kind is BenchmarkMeasurementKind.PREFILL else "Generation"


def _memory_block(
    results: list[BenchmarkRunResult],
    estimate: MemoryEstimate | None,
) -> Widget | None:
    """Predicted against observed, with the observed figure carrying weight.

    A prediction and a measurement are not the same kind of claim, so they do
    not get the same typography: the estimate reads quiet, the measured peak
    reads bold. The delta between them is the point of the block.
    """
    rows = _memory_rows(results, estimate)
    if not rows:
        return None
    children: list[Widget] = [Static("Memory", classes="section-title")]
    for label, value in rows:
        emphasis = None
        if label == "Estimated":
            emphasis = "estimated"
        elif label in {"Measured"} or label.endswith("peak RAM"):
            emphasis = "measured"
        children.append(MetricRow(label, value, emphasis=emphasis))
    return Vertical(*children, classes="section")


def _startup_rows(results: list[BenchmarkRunResult]) -> list[tuple[str, str]]:
    """Load and first-token timings — Transformers only.

    ``llama_bench_runner`` never populates these, so for a GGUF benchmark they
    are absent rather than "unavailable": a row of blanks would imply the
    measurement was attempted and failed.
    """
    rows: list[tuple[str, str]] = []
    for result in results:
        observation = result.record.observation
        if observation.model_load_seconds is None and (
            observation.time_to_first_token_seconds is None
        ):
            continue
        prefix = f"{_config_label(result.record)} " if len(results) > 1 else ""
        if observation.model_load_seconds is not None:
            rows.append((f"{prefix}Model load", _seconds(observation.model_load_seconds)))
        if observation.time_to_first_token_seconds is not None:
            rows.append(
                (
                    f"{prefix}Time to first token",
                    _seconds(observation.time_to_first_token_seconds),
                )
            )
    return rows


def _result_table(results: list[BenchmarkRunResult]) -> DataTable[str]:
    table: DataTable[str] = DataTable()
    labels = [_config_label(result.record) for result in results]
    table.add_columns("", *labels)
    keys = sorted(
        {
            (measurement.kind, measurement.tokens)
            for result in results
            for measurement in result.record.observation.measurements
        },
        key=lambda item: (0 if item[0] is BenchmarkMeasurementKind.PREFILL else 1, item[1]),
    )
    maps = [
        {
            (measurement.kind, measurement.tokens): measurement
            for measurement in result.record.observation.measurements
        }
        for result in results
    ]
    for key in keys:
        table.add_row(
            _metric_label(key[0], key[1]),
            *[
                (
                    f"{measurement.mean_tokens_per_second:.1f} tok/s"
                    if (measurement := mapping.get(key)) is not None
                    else "-"
                )
                for mapping in maps
            ],
        )
    return table


def _warnings(result: BenchmarkMatrixResult) -> list[str]:
    warnings = [skip.reason for skip in result.skipped]
    warnings.extend(failure.message for failure in result.failed)
    return warnings


def _technical_rows(record: BenchmarkRecord, path: Path | None) -> list[tuple[str, str]]:
    artifact = record.artifact
    capability = record.llama_bench_capability
    rows = [
        ("ID", record.identity.benchmark_id),
        ("Created", record.identity.created_at.isoformat()),
        ("Schema", str(record.schema_version)),
        ("Artifact", f"{artifact.repo_id} / {artifact.filename}"),
        ("Revision", artifact.revision or "unknown"),
        ("Quantization", artifact.quantization or "unknown"),
        ("SHA-256", artifact.sha256 or "unknown"),
        ("Backend", record.requested_backend.value),
        ("Device", record.requested_device or "none"),
        ("GPU layers", record.gpu_layers.label),
        ("Repetitions", str(record.request.repetitions)),
        ("Methodology", record.observation.methodology or "unknown"),
        ("Model load", _seconds(record.observation.model_load_seconds)),
        ("Warmup", _seconds(record.observation.warmup_seconds)),
        (
            "TTFT",
            _seconds_with_stddev(
                record.observation.time_to_first_token_seconds,
                record.observation.time_to_first_token_stddev_seconds,
            ),
        ),
        ("Generation latency", _seconds(record.observation.generation_latency_seconds)),
        (
            "Generation latency stddev",
            _seconds(record.observation.generation_latency_stddev_seconds),
        ),
        ("Peak RAM", _bytes(record.observation.peak_ram_bytes)),
        ("Peak VRAM", _bytes(record.observation.peak_vram_bytes)),
        ("Status", "success" if record.observation.success else "failure"),
        ("Saved", str(path) if path else "not saved"),
    ]
    if capability is not None:
        rows[8:8] = [
            ("llama-bench", capability.binary_path or "unknown"),
            ("llama-bench status", capability.binary_status.value),
            ("llama-bench version", capability.version_text or "unknown"),
        ]
    elif record.runtime_capability is not None:
        rows[8:8] = [
            ("Runtime capability", record.runtime_capability.runtime_family.value),
        ]
    return rows


def _config_label(record: BenchmarkRecord) -> str:
    if record.requested_backend.value == "cpu":
        return "CPU"
    return _request_label(record.requested_backend.value, record.requested_device)


def _request_label(backend: str, device: str | None) -> str:
    if backend == "cpu":
        return "CPU"
    return f"{backend.title()} {device or 'unknown'}"


def _metric_label(kind: BenchmarkMeasurementKind, tokens: int) -> str:
    prefix = "Prompt processing" if kind is BenchmarkMeasurementKind.PREFILL else "Generation"
    return f"{prefix} {tokens}"


def _memory_rows(
    results: list[BenchmarkRunResult],
    estimate: MemoryEstimate | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    measured_values = [
        result.record.observation.peak_ram_bytes
        for result in results
        if result.record.observation.peak_ram_bytes is not None
    ]
    if estimate is not None and estimate.total_bytes is not None and measured_values:
        measured = max(measured_values)
        rows.extend(
            [
                ("Estimated", _bytes(estimate.total_bytes)),
                ("Measured", _bytes(measured)),
                ("Difference", _memory_delta(estimate.total_bytes, measured)),
            ]
        )
    for result in results:
        observation = result.record.observation
        if observation.peak_ram_bytes is None and observation.peak_vram_bytes is None:
            continue
        label = _config_label(result.record)
        # A row reading "unavailable" implies the measurement was attempted and
        # failed. On a CPU-only run there is no VRAM to measure, so the row
        # simply does not exist.
        if observation.peak_ram_bytes is not None:
            rows.append((f"{label} peak RAM", _bytes(observation.peak_ram_bytes)))
        if observation.peak_vram_bytes is not None:
            rows.append((f"{label} peak VRAM", _bytes(observation.peak_vram_bytes)))
    return rows


def _memory_delta(estimated: int, measured: int) -> str:
    if estimated <= 0:
        return "unavailable"
    delta = (measured - estimated) / estimated
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta) * 100:.0f}%"


def _benchmark_configuration_label(
    selection: RuntimeBackendSelection,
    capability: LlamaCppRuntimeCapability,
) -> str:
    backend = selection.selected_backend
    if backend is ComputeBackend.CPU:
        return "CPU"
    if any(
        runtime_device.runtime_id
        for backend_capability in capability.backend_capabilities
        if backend_capability.backend is backend
        for runtime_device in backend_capability.devices
    ):
        return backend.value
    return f"{backend.value} · no ready device"


def _transformers_backend(
    runtime: RuntimeRecommendation,
    selected_backend: ComputeBackend,
) -> ComputeBackend:
    device_map = next(
        (flag.value for flag in runtime.flags if flag.name == "device_map"),
        "cpu",
    )
    if device_map == "cuda":
        return ComputeBackend.CUDA
    if selected_backend in {ComputeBackend.CPU, ComputeBackend.CUDA, ComputeBackend.HIP}:
        return selected_backend
    return ComputeBackend.CPU


def _benchmark_backend(
    runtime: RuntimeRecommendation,
    selected_backend: ComputeBackend,
) -> ComputeBackend:
    if runtime.runtime is RuntimeName.TRANSFORMERS:
        return _transformers_backend(runtime, selected_backend)
    return selected_backend


def _benchmark_device(backend: ComputeBackend, plan: ExecutionPlan) -> str | None:
    if backend is ComputeBackend.CPU:
        return "none"
    readiness = plan.execution_readiness
    if readiness is None or readiness.selected_backend_capability is None:
        return None
    devices = readiness.selected_backend_capability.devices
    if not devices:
        return None
    return devices[0].runtime_id


def _benchmark_gpu_layers(
    backend: ComputeBackend,
    runtime: RuntimeRecommendation,
) -> BenchmarkGpuLayers:
    if backend is ComputeBackend.CPU:
        return BenchmarkGpuLayers.count_layers(0)
    raw = next(
        (flag.value for flag in runtime.flags if flag.name == "--n-gpu-layers"),
        None,
    )
    if raw is None:
        return BenchmarkGpuLayers.full()
    if raw in {"-1", "all"}:
        return BenchmarkGpuLayers.full()
    return BenchmarkGpuLayers.count_layers(max(0, int(raw)))


def _seconds(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f} s"


def _seconds_with_stddev(value: float | None, stddev: float | None) -> str:
    if value is None:
        return "unavailable"
    if stddev is None:
        return f"{value:.2f} s"
    return f"{value:.2f} ± {stddev:.2f} s"


def _bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    gib = value / 1024**3
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = value / 1024**2
    return f"{mib:.1f} MiB"


__all__ = [
    "BenchmarkDetailsScreen",
    "RecommendationBenchmarkScreen",
]
