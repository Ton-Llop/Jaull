from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    Select,
    Static,
)

from local_ai_check.domain.estimation import MemoryEstimate
from local_ai_check.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from local_ai_check.domain.model import ModelAnalysis
from local_ai_check.estimator import service as estimator_service
from local_ai_check.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from local_ai_check.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    LocalAiCheckError,
    ModelAccessDeniedError,
    ModelNotFoundError,
    QuantizationNotFoundError,
)
from local_ai_check.hardware.detector import detect_hardware
from local_ai_check.huggingface.client import HfClient
from local_ai_check.huggingface.repository import inspect_model
from local_ai_check.huggingface.url_parser import normalize_repo_id
from local_ai_check.metadata.range_reader import HttpxRangeClient
from local_ai_check.tui.widgets.assessment_badge import AssessmentBadge
from local_ai_check.tui.widgets.banner import Banner
from local_ai_check.tui.widgets.cli_equivalent import CliEquivalent
from local_ai_check.tui.widgets.memory_usage_bar import MemoryUsageBar
from local_ai_check.tui.widgets.summary_card import SummaryCard
from local_ai_check.tui.widgets.warnings_panel import WarningsPanel

_GIB = 1024 * 1024 * 1024
_CONTEXT_PRESETS = ("4096", "8192", "16384", "32768")


@dataclass
class _EstimateFormState:
    reference: str = ""
    analysis: ModelAnalysis | None = None
    quantization: str | None = None
    dtype: WeightPrecision | None = None
    context: int = 4096
    batch_size: int = 1
    device: TargetDevice = TargetDevice.AUTO


class EstimateScreen(Screen[None]):
    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    state: _EstimateFormState

    def __init__(self) -> None:
        super().__init__()
        self.state = _EstimateFormState()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Banner(
            "Estimate model memory",
            "Guided flow — enter a model, pick options, get a full breakdown.",
        )
        with Vertical(classes="card"):
            yield Static("1. Model", classes="card-title")
            yield Input(placeholder="user/model or huggingface.co URL", id="est-input")
            with Horizontal():
                yield Button("Detect", id="est-detect", classes="-primary")
        yield LoadingIndicator(id="est-loading")
        yield VerticalScroll(Vertical(id="est-form"))
        yield VerticalScroll(Vertical(id="est-result"))
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#est-loading", LoadingIndicator).display = False
        self.query_one("#est-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "est-detect":
            self._start_detect()
        elif event.button.id == "est-run":
            self._start_estimate()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "est-input":
            self._start_detect()

    def _start_detect(self) -> None:
        reference = self.query_one("#est-input", Input).value.strip()
        self.state.reference = reference
        form = self.query_one("#est-form", Vertical)
        result = self.query_one("#est-result", Vertical)
        form.remove_children()
        result.remove_children()
        if not reference:
            form.mount(WarningsPanel(["Please enter a repo_id or URL."]))
            return
        try:
            repo_id = normalize_repo_id(reference)
        except InvalidModelReferenceError as exc:
            form.mount(WarningsPanel([str(exc)]))
            return
        self.query_one("#est-loading", LoadingIndicator).display = True
        self.run_worker(lambda: self._detect_worker(repo_id), thread=True)

    def _detect_worker(self, repo_id: str) -> None:
        try:
            analysis = inspect_model(repo_id, client=HfClient())
        except (
            ModelNotFoundError,
            ModelAccessDeniedError,
            HuggingFaceUnavailableError,
            LocalAiCheckError,
        ) as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        self.app.call_from_thread(self._render_form, analysis)

    def _render_error(self, message: str) -> None:
        self.query_one("#est-loading", LoadingIndicator).display = False
        form = self.query_one("#est-form", Vertical)
        form.mount(WarningsPanel([message]))

    def _render_form(self, analysis: ModelAnalysis) -> None:
        self.query_one("#est-loading", LoadingIndicator).display = False
        self.state.analysis = analysis
        default_ctx = (
            analysis.config.max_position_embeddings
            if analysis.config and analysis.config.max_position_embeddings
            else 4096
        )
        self.state.context = min(default_ctx, 8192)

        form = self.query_one("#est-form", Vertical)
        form.mount(
            SummaryCard(
                "Detected",
                [
                    ("Repository", analysis.repo.repo_id),
                    ("Type", analysis.classification.primary_type.value),
                ],
            )
        )

        # Variant / dtype selector
        variant_options: list[tuple[str, str]] = []
        if analysis.classification.gguf_variants:
            variant_options = [
                (v.quantization, v.quantization)
                for v in analysis.classification.gguf_variants
            ]
            with_default = "Q4_K_M" if any(
                q == "Q4_K_M" for q, _ in variant_options
            ) else variant_options[0][0]
            self.state.quantization = with_default
            form.mount(
                _labelled_select(
                    "2. GGUF variant", "est-variant", variant_options, with_default
                )
            )
        else:
            dtype_options = [(p.value, p.value) for p in WeightPrecision]
            self.state.dtype = WeightPrecision.FLOAT16
            form.mount(
                _labelled_select(
                    "2. Weight precision (dtype)",
                    "est-dtype",
                    dtype_options,
                    WeightPrecision.FLOAT16.value,
                )
            )

        # Context
        ctx_options = [(f"{p} tokens", p) for p in _CONTEXT_PRESETS]
        form.mount(
            _labelled_select(
                "3. Context length",
                "est-context",
                ctx_options,
                str(self.state.context),
            )
        )

        # Batch
        form.mount(
            _labelled_input(
                "4. Batch size", "est-batch", str(self.state.batch_size)
            )
        )

        # Device
        device_options = [(d.value, d.value) for d in TargetDevice]
        form.mount(
            _labelled_select(
                "5. Target device",
                "est-device",
                device_options,
                TargetDevice.AUTO.value,
            )
        )

        form.mount(_RunCard())

    def on_select_changed(self, event: Select.Changed) -> None:
        widget_id = event.select.id or ""
        value = event.value
        if value is Select.BLANK:
            return
        if widget_id == "est-variant":
            self.state.quantization = str(value)
        elif widget_id == "est-dtype":
            self.state.dtype = WeightPrecision(str(value))
        elif widget_id == "est-context":
            self.state.context = int(str(value))
        elif widget_id == "est-device":
            self.state.device = TargetDevice(str(value))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "est-batch":
            try:
                value = int(event.value)
                if value >= 1:
                    self.state.batch_size = value
            except ValueError:
                pass

    def _start_estimate(self) -> None:
        if self.state.analysis is None:
            return
        self.query_one("#est-loading", LoadingIndicator).display = True
        self.run_worker(self._estimate_worker, thread=True)

    def _estimate_worker(self) -> None:
        assert self.state.analysis is not None
        cfg = InferenceConfiguration(
            context_length=self.state.context,
            batch_size=self.state.batch_size,
            target_device=self.state.device,
            precision=self.state.dtype,
            quantization=self.state.quantization,
            safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
            device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        )
        hardware = detect_hardware()
        client = HfClient()
        range_client = HttpxRangeClient()
        try:
            estimate = estimator_service.estimate_memory(
                analysis=self.state.analysis,
                hardware=hardware,
                inference_cfg=cfg,
                client=client,
                range_client=range_client,
            )
        except QuantizationNotFoundError as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        except LocalAiCheckError as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        self.app.call_from_thread(self._render_estimate, estimate)

    def _render_estimate(self, estimate: MemoryEstimate) -> None:
        self.query_one("#est-loading", LoadingIndicator).display = False
        result = self.query_one("#est-result", Vertical)
        result.remove_children()

        breakdown_rows: list[tuple[str, str]] = [
            ("Weights", _fmt(estimate.weights.component.bytes)),
            ("KV cache", _fmt(estimate.kv_cache.component.bytes)),
            ("Runtime overhead", _fmt(estimate.runtime_overhead.component.bytes)),
            ("Device reserve", _fmt(estimate.device_reserve.bytes)),
        ]
        if estimate.safety_margin is not None:
            breakdown_rows.append(("Safety margin", _fmt(estimate.safety_margin.bytes)))
        breakdown_rows.append(("Total required", _fmt(estimate.total_bytes)))
        result.mount(SummaryCard("Memory breakdown", breakdown_rows))

        result.mount(
            MemoryUsageBar(
                "Available VRAM",
                estimate.total_bytes,
                estimate.assessment.available_vram_bytes,
            )
        )
        result.mount(
            MemoryUsageBar(
                "Available RAM",
                estimate.total_bytes,
                estimate.assessment.available_ram_bytes,
            )
        )

        result.mount(_AssessmentCard(estimate))

        if estimate.runtime_recommendation is not None:
            result.mount(_RuntimeCard(estimate))

        base = estimate.base_model_resolution
        if base and base.repo_id:
            result.mount(
                SummaryCard(
                    "Base model resolution",
                    [
                        ("Repository", base.repo_id),
                        ("Source", base.source.value),
                        ("Confidence", base.confidence.value),
                    ],
                )
            )

        if estimate.warnings:
            result.mount(WarningsPanel(estimate.warnings))

        result.mount(CliEquivalent(_equivalent_cli(estimate)))

    def _get_state(self) -> _EstimateFormState:
        # Test helper.
        return self.state


def _equivalent_cli(estimate: MemoryEstimate) -> str:
    cfg = estimate.inference_configuration
    parts = ["local-ai-check estimate", estimate.repository.repo_id]
    if cfg.quantization:
        parts.append(f"--quantization {cfg.quantization}")
    if cfg.precision:
        parts.append(f"--dtype {cfg.precision.value}")
    parts.append(f"--context {cfg.context_length}")
    if cfg.batch_size != 1:
        parts.append(f"--batch-size {cfg.batch_size}")
    if cfg.target_device is not TargetDevice.AUTO:
        parts.append(f"--device {cfg.target_device.value}")
    return " ".join(parts)


class _RunCard(Vertical):
    DEFAULT_CLASSES = "card"

    def compose(self) -> ComposeResult:
        yield Static("6. Run estimation", classes="card-title")
        yield Button("Estimate", id="est-run", classes="-primary")


class _AssessmentCard(Vertical):
    DEFAULT_CLASSES = "card"

    def __init__(self, estimate: MemoryEstimate) -> None:
        super().__init__()
        self._estimate = estimate

    def compose(self) -> ComposeResult:
        yield Static("Assessment", classes="card-title")
        yield AssessmentBadge(self._estimate.assessment.status)
        yield Static(f"Confidence: {self._estimate.assessment.confidence.value}")
        for reason in self._estimate.assessment.reasons:
            yield Static(f"- {reason}")


class _RuntimeCard(Vertical):
    DEFAULT_CLASSES = "runtime-card"

    def __init__(self, estimate: MemoryEstimate) -> None:
        super().__init__()
        self._estimate = estimate

    def compose(self) -> ComposeResult:
        rec = self._estimate.runtime_recommendation
        assert rec is not None
        yield Static("Recommended runtime", classes="runtime-title")
        yield Static(f"Runtime: {rec.runtime.value}")
        yield Static(f"Confidence: {rec.confidence.value}")
        if rec.alternatives:
            yield Static(
                "Alternatives: " + ", ".join(a.value for a in rec.alternatives)
            )
        if rec.command_preview:
            yield Static(f"$ {rec.command_preview}", classes="runtime-code")
        if rec.python_snippet:
            yield Static(rec.python_snippet, classes="runtime-code")
        for reason in rec.reasons:
            yield Static(f"- {reason}")
        for warning in rec.warnings:
            yield Static(f"! {warning}")


class _LabelledInput(Vertical):
    DEFAULT_CLASSES = "card"

    def __init__(self, label: str, widget_id: str, default: str) -> None:
        super().__init__()
        self._label = label
        self._widget_id = widget_id
        self._default = default

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="card-title")
        yield Input(value=self._default, id=self._widget_id)


class _LabelledSelect(Vertical):
    DEFAULT_CLASSES = "card"

    def __init__(
        self,
        label: str,
        widget_id: str,
        options: list[tuple[str, str]],
        default: str,
    ) -> None:
        super().__init__()
        self._label = label
        self._widget_id = widget_id
        self._options = options
        self._default = default

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="card-title")
        yield Select(
            options=self._options,
            value=self._default,
            id=self._widget_id,
            allow_blank=False,
        )


def _labelled_input(label: str, widget_id: str, default: str) -> _LabelledInput:
    return _LabelledInput(label, widget_id, default)


def _labelled_select(
    label: str,
    widget_id: str,
    options: list[tuple[str, str]],
    default: str,
) -> _LabelledSelect:
    return _LabelledSelect(label, widget_id, options, default)


def _fmt(byte_count: int | None) -> str:
    if byte_count is None:
        return "unknown"
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"
