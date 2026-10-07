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

from jaull.advisor.service import AdvisorService
from jaull.application.model_reference import normalize_repo_id
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import ModelAnalysis
from jaull.estimator.gguf_selection import select_variant
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    JaullError,
    ModelAccessDeniedError,
    ModelNotFoundError,
    QuantizationNotFoundError,
)
from jaull.tui.evidence import EvidenceIndex, EvidenceState, PlanEvidence
from jaull.tui.widgets.assessment_badge import AssessmentBadge
from jaull.tui.widgets.banner import Banner
from jaull.tui.widgets.cli_equivalent import CliEquivalent
from jaull.tui.widgets.memory_usage_bar import MemoryUsageBar
from jaull.tui.widgets.metric_list import MetricRow
from jaull.tui.widgets.summary_card import SummaryCard
from jaull.tui.widgets.technical_details import TechnicalDetails
from jaull.tui.widgets.warnings_panel import WarningsPanel

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
        yield LoadingIndicator(id="est-loading")
        with VerticalScroll(id="est-body"):
            with Vertical(id="est-parameters"):
                with Vertical(classes="card"):
                    yield Static("1. Model", classes="card-title")
                    yield Input(placeholder="user/model or huggingface.co URL", id="est-input")
                    with Horizontal():
                        yield Button("Detect", id="est-detect", classes="-primary")
                yield Vertical(id="est-form")
            with Vertical(id="est-output"):
                yield Vertical(id="est-result")
                with Horizontal():
                    yield Button("Adjust parameters", id="est-edit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#est-loading", LoadingIndicator).display = False
        self.query_one("#est-output", Vertical).display = False
        self.query_one("#est-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "est-detect":
            self._start_detect()
        elif event.button.id == "est-run":
            self._start_estimate()
        elif event.button.id == "est-edit":
            self.query_one("#est-output", Vertical).display = False
            self.query_one("#est-parameters", Vertical).display = True
            self.query_one("#est-input", Input).focus()
            self.query_one("#est-body", VerticalScroll).scroll_home(animate=False, immediate=True)

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
            analysis = _advisor(self).inspect_model(repo_id, refresh=True)
        except (
            ModelNotFoundError,
            ModelAccessDeniedError,
            HuggingFaceUnavailableError,
            JaullError,
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
        if self.state.analysis is None or self.query_one("#est-loading", LoadingIndicator).display:
            return
        self.query_one("#est-loading", LoadingIndicator).display = True
        self.run_worker(self._estimate_worker, thread=True)

    def _estimate_worker(self) -> None:
        assert self.state.analysis is not None
        analysis = self.state.analysis
        cfg = InferenceConfiguration(
            context_length=self.state.context,
            batch_size=self.state.batch_size,
            target_device=self.state.device,
            precision=self.state.dtype,
            quantization=self.state.quantization,
            safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
            device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        )
        advisor = _advisor(self)
        hardware = advisor.scan_hardware()
        try:
            estimate = advisor.estimate_model(
                analysis=analysis,
                hardware=hardware,
                inference_cfg=cfg,
            )
        except QuantizationNotFoundError as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        except JaullError as exc:
            self.app.call_from_thread(self._render_error, str(exc))
            return
        evidence = PlanEvidence(state=EvidenceState.ESTIMATED)
        if analysis.classification.gguf_variants:
            variant = select_variant(
                analysis.classification.gguf_variants, cfg.quantization,
            ).variant
            evidence = PlanEvidence(
                state=EvidenceState.ESTIMATED,
                quality_records=EvidenceIndex.load(advisor).quality_for_sha(variant.sha256),
            )
        self.app.call_from_thread(self.call_next, self._render_estimate, estimate, evidence)

    async def _render_estimate(
        self, estimate: MemoryEstimate, evidence: PlanEvidence | None = None,
    ) -> None:
        result = self.query_one("#est-result", Vertical)
        await result.remove_children()
        self.query_one("#est-loading", LoadingIndicator).display = False

        if evidence is not None and evidence.quality_records:
            result.mount(_QualitySection(evidence))

        breakdown_rows: list[tuple[str, str]] = [
            ("Estimation context", f"{estimate.inference_configuration.context_length} tokens"),
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
        self.query_one("#est-parameters", Vertical).display = False
        self.query_one("#est-output", Vertical).display = True
        body = self.query_one("#est-body", VerticalScroll)
        body.focus()
        body.scroll_home(animate=False, immediate=True)

    def _get_state(self) -> _EstimateFormState:
        # Test helper.
        return self.state


def _advisor(screen: Screen[None]) -> AdvisorService:
    from jaull.tui.app import JaullApp

    assert isinstance(screen.app, JaullApp)
    return screen.app.advisor


def _equivalent_cli(estimate: MemoryEstimate) -> str:
    cfg = estimate.inference_configuration
    parts = ["jaull estimate", estimate.repository.repo_id]
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


class _QualitySection(Vertical):
    def __init__(self, evidence: PlanEvidence) -> None:
        super().__init__(id="est-quality", classes="section")
        self._evidence = evidence

    def compose(self) -> ComposeResult:
        yield Static("Measured evaluation", classes="section-title")
        yield Static(
            "Historical results for this exact artifact. "
            "The current execution and protocol have not been verified.",
            classes="text-muted", markup=False,
        )
        for record in self._evidence.quality_records:
            yield MetricRow("Benchmark", record.dataset)
            yield MetricRow("Evaluation", f"{record.classification.capitalize()} evaluation")
            yield MetricRow("Samples", f"{record.samples_used} of {record.samples_available}")
            context = f"{record.context_length} tokens" if record.context_length else "unknown"
            yield MetricRow("Evaluation context", context)
            for metric in record.metrics:
                label = {
                    "acc": "Accuracy", "acc_norm": "Length-normalized accuracy",
                }.get(metric.name, metric.name)
                yield MetricRow(
                    label,
                    f"{metric.value:.1%} ({metric.correct}/{metric.samples})",
                    emphasis="measured",
                )
            for limitation in record.limitations:
                yield Static(limitation, classes="text-muted", markup=False)
        yield Static(
            "These benchmark results are not a general capability assessment.",
            classes="text-muted", markup=False,
        )
        yield TechnicalDetails(
            title="Evaluation provenance",
            extra=[Static(self._evidence.quality_details(), markup=False)],
        )


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
