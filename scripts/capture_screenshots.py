"""Regenerate the docs/assets/ TUI screenshots.

Runs the Textual app under :func:`App.run_test` with fake services, navigates
to each documented screen with :class:`Pilot`, and saves an SVG snapshot per
step. SVGs render inline on GitHub via ``<img src=...>`` and keep the
terminal styling (colours, borders, glyphs) intact — a PNG converter would
lose all of that.

Run once and drop the generated files into ``docs/assets/``:

.. code-block:: shell

    uv run python scripts/capture_screenshots.py

Review a narrower terminal without touching the committed assets:

.. code-block:: shell

    uv run python scripts/capture_screenshots.py --size 90x28 --out /tmp/tui-90x28

Requires no network and never starts llama.cpp — the search client, HF client,
hardware detector, inspector and estimator are fakes borrowed from the test
fixtures, and artifact resolution/execution is faked in this module.

Every capture is deterministic: the fakes return fixed values, the logo shimmer
is pinned to its first frame, and the export path is overwritten with a fixed
filename (the real default is timestamped).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from textual.widgets import Button, Input, TextArea  # noqa: E402

from _workflow_fixtures import (  # noqa: E402
    GIB,
    FakeHfClient,
    FakeSearchClient,
    candidate,
    gguf_analysis,
    hardware,
    size_driven_estimator,
)
from jaull.advisor.service import AdvisorService  # noqa: E402
from jaull.domain.artifacts import ModelArtifact  # noqa: E402
from jaull.domain.estimation import MemoryEstimate  # noqa: E402
from jaull.domain.execution import ExecutionObservation, InferenceResult  # noqa: E402
from jaull.domain.runtime import (  # noqa: E402
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution.errors import ExecutionTimeoutError  # noqa: E402
from jaull.tui.app import JaullApp  # noqa: E402
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen  # noqa: E402
from jaull.tui.screens.recommendation_execution import (  # noqa: E402
    RecommendationExecutionScreen,
)
from jaull.tui.screens.recommendation_results import (  # noqa: E402
    ExportReportModal,
    RecommendationResultsScreen,
)
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen  # noqa: E402
from jaull.tui.widgets.logo import Logo  # noqa: E402
from jaull.workflow.container import ServiceContainer  # noqa: E402
from jaull.workflow.progress import HARDWARE_STEPS  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "assets"
TERMINAL_SIZE = (110, 32)

# The scan pauses here so the mid-scan capture is a real intermediate state
# rather than a duplicate of the finished one.
_PAUSE_AFTER_STEPS = 2

SCREENSHOTS = (
    "tui-welcome.svg",
    "tui-hardware-loading.svg",
    "tui-hardware-done.svg",
    "tui-wizard.svg",
    "tui-search.svg",
    "tui-results.svg",
    "tui-export.svg",
    "tui-run-empty.svg",
    "tui-run-loading.svg",
    "tui-run-history.svg",
    "tui-run-error.svg",
    "tui-home.svg",
    "tui-estimate.svg",
)


class Gates:
    """Every latch the fakes can pause on, so transient states are capturable."""

    def __init__(self) -> None:
        self.scan = _Gate()
        self.search = _Gate()
        self.artifact = _Gate()

    def release_all(self) -> None:
        self.scan.release()
        self.search.release()
        self.artifact.release()


class _Gate:
    """A latch a fake can block on, so a transient UI state can be captured."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.held = False

    def hold(self) -> None:
        self.held = True
        self._event.clear()

    def release(self) -> None:
        self.held = False
        self._event.set()

    def wait(self) -> None:
        if self.held:
            # Bounded: a capture must never hang the script.
            self._event.wait(timeout=10)


def _llama_cpp_runtime(estimate: MemoryEstimate) -> RuntimeRecommendation:
    """The runtime block the real estimator attaches to a GGUF repository.

    The fixture estimator does not build one, and without it the run screen has
    nothing executable to show — so the documented flow would stop at Results.
    """
    context = estimate.inference_configuration.context_length
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        command_preview=(
            f"llama-cli -m {estimate.repository.repo_id}.gguf "
            f"--ctx-size {context} --n-gpu-layers 32"
        ),
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value=str(context),
                source=RuntimeFlagSource.ESTIMATE,
                explanation="context length chosen for this use case",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value="32",
                source=RuntimeFlagSource.HARDWARE,
                explanation="every layer fits in the detected VRAM",
            ),
        ],
        reasons=["The repository ships GGUF weights, which llama.cpp runs directly."],
        confidence=estimate.assessment.confidence,
    )


# Three different families, so ranking produces a primary plus alternatives
# rather than one entry with a series ladder hidden inside it. The weights
# differ enough to give each a distinct score and memory verdict.
_MODELS: dict[str, dict[str, int]] = {
    "Qwen/Qwen2.5-7B-Instruct-GGUF": {
        "base_bytes": 4 * GIB,
        "downloads": 250_000,
        "likes": 900,
    },
    "mistralai/Mistral-7B-Instruct-v0.3-GGUF": {
        "base_bytes": 5 * GIB,
        "downloads": 90_000,
        "likes": 380,
    },
    "google/gemma-2-9b-it-GGUF": {
        "base_bytes": 22 * GIB,
        "downloads": 40_000,
        "likes": 120,
    },
}


def _services(scan_gate: _Gate, search_gate: _Gate | None = None) -> ServiceContainer:
    search = FakeSearchClient(
        default=[
            candidate(
                repo_id=repo,
                tags=["text-generation", "code"],
                downloads=spec["downloads"],
                likes=spec["likes"],
            )
            for repo, spec in _MODELS.items()
        ]
    )
    estimator = size_driven_estimator(vram_budget=24 * GIB)

    def detect(**kwargs: object) -> object:
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for index, (key, _) in enumerate(HARDWARE_STEPS):
                if index == _PAUSE_AFTER_STEPS:
                    scan_gate.wait()
                on_step(key)
        return hardware()

    def inspect(repo_id: str, client: object | None = None) -> object:
        del client
        if search_gate is not None:
            # Pauses the pipeline on the "inspect" step so the search screen
            # can be captured part-way through, as the user really sees it.
            search_gate.wait()
        return gguf_analysis(
            repo_id=repo_id,
            base_bytes=int(_MODELS.get(repo_id, {}).get("base_bytes", 4 * GIB)),
        )

    def estimate(**kwargs: object) -> MemoryEstimate:
        result = estimator(kwargs["analysis"], kwargs["inference_cfg"])
        return result.model_copy(
            update={"runtime_recommendation": _llama_cpp_runtime(result)}
        )

    return ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=inspect,  # type: ignore[arg-type]
        estimate_memory=estimate,  # type: ignore[arg-type]
    )


class _ScriptAdvisor(AdvisorService):
    """Real guided pipeline, faked artifacts and execution.

    Subclassing keeps hardware scanning, discovery and ranking on the genuine
    code path — only the parts that would touch the filesystem, the Hub or
    llama.cpp are replaced.
    """

    def __init__(self, services: ServiceContainer, artifact_gate: _Gate) -> None:
        super().__init__(services=services)
        # AdvisorService is a frozen dataclass; the app treats it as immutable,
        # so the capture knobs go in through the same door the class itself
        # uses for its memoised helpers.
        object.__setattr__(self, "_gate", artifact_gate)
        object.__setattr__(self, "fail_run", None)
        object.__setattr__(self, "response_text", "")

    def set_failure(self, error: Exception | None) -> None:
        object.__setattr__(self, "fail_run", error)

    def set_response(self, text: str) -> None:
        object.__setattr__(self, "response_text", text)

    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        self._gate.wait()  # type: ignore[attr-defined]
        return ModelArtifact(
            repo_id=repo_id,
            revision=revision or "main",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            format="gguf",
            quantization=quantization or "Q4_K_M",
            size_bytes=4 * GIB,
            local_path=Path("~/.cache/jaull/qwen2.5-7b-instruct-q4_k_m.gguf"),
            sha256="0" * 64,
            is_downloaded=False,
            is_verified=False,
        )

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        del on_progress
        return artifact.model_copy(update={"is_downloaded": True})

    def verify_artifact(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        del full
        return artifact.model_copy(update={"is_verified": True})

    def run_artifact(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        del prompt, runtime
        failure = self.fail_run  # type: ignore[attr-defined]
        if failure is not None:
            raise failure
        return InferenceResult(
            text=self.response_text,  # type: ignore[attr-defined]
            runtime="llama.cpp",
            model_path=artifact.local_path or Path("model.gguf"),
            observation=ExecutionObservation(
                success=True,
                duration_seconds=1.24,
                peak_ram_bytes=None,
                peak_vram_bytes=None,
                exit_code=0,
                failure_reason=None,
            ),
        )


def build_app(gates: Gates | None = None) -> JaullApp:
    """A `JaullApp` wired to the offline fakes these screenshots run on.

    Shared with the layout tests so both exercise the same stack: a real
    guided pipeline, with only the Hub, the filesystem and llama.cpp faked.
    """
    gates = gates or Gates()
    advisor = _ScriptAdvisor(_services(gates.scan, gates.search), gates.artifact)
    return JaullApp(advisor=advisor)


def _pin_logo_frame() -> None:
    """Freeze the welcome logo on its first frame.

    The shimmer is a timer, so without this the welcome capture depends on how
    long the run took to reach it and the file churns on every regeneration.
    """

    def on_mount(self: Logo) -> None:
        self._refresh_logo()

    Logo.on_mount = on_mount  # type: ignore[method-assign]


async def _wait_for(
    pilot,  # type: ignore[no-untyped-def]
    predicate: Callable[[], bool],
    *,
    timeout: float = 10.0,
) -> None:
    """Pump the event loop until `predicate` holds, then settle once more."""
    waited = 0.0
    while waited < timeout:
        await pilot.pause()
        if predicate():
            await pilot.pause()
            return
        await asyncio.sleep(0.05)
        waited += 0.05
    raise TimeoutError("timed out waiting for the UI to reach the expected state")


async def _settle(pilot, seconds: float = 0.4) -> None:  # type: ignore[no-untyped-def]
    """Let pending mounts and worker callbacks land."""
    steps = max(1, int(seconds / 0.05))
    for _ in range(steps):
        await pilot.pause()
        await asyncio.sleep(0.05)
    await pilot.pause()


def _save(app: JaullApp, out_dir: Path, filename: str) -> None:
    path = app.save_screenshot(filename=filename, path=str(out_dir))
    print(f"  → {path}")


async def _capture_welcome(app: JaullApp, pilot, out_dir: Path) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    _save(app, out_dir, "tui-welcome.svg")


async def _capture_hardware(  # type: ignore[no-untyped-def]
    app: JaullApp,
    pilot,
    out_dir: Path,
    scan_gate: _Gate,
) -> None:
    """Mid-scan and finished: two states that must not look alike."""
    scan_gate.hold()
    app.start_guided_workflow()
    try:
        await _wait_for(pilot, lambda: _steps_done(app) >= _PAUSE_AFTER_STEPS)
        _save(app, out_dir, "tui-hardware-loading.svg")
    finally:
        scan_gate.release()

    await _wait_for(pilot, lambda: bool(app.screen.query("#hw-continue")))
    _save(app, out_dir, "tui-hardware-done.svg")


def _steps_done(app: JaullApp) -> int:
    """How many checklist lines the running stage has already ticked off."""
    return sum(
        1
        for widget in app.screen.query("#progress-steps Static")
        if widget.has_class("step-done")
    )


async def _capture_wizard(app: JaullApp, pilot, out_dir: Path) -> None:  # type: ignore[no-untyped-def]
    app.goto_requirements()
    await _wait_for(pilot, lambda: isinstance(app.screen, RequirementsWizardScreen))
    _save(app, out_dir, "tui-wizard.svg")


async def _capture_search_and_results(  # type: ignore[no-untyped-def]
    app: JaullApp,
    pilot,
    out_dir: Path,
    search_gate: _Gate,
) -> None:
    wizard = app.screen
    assert isinstance(wizard, RequirementsWizardScreen)
    answers = wizard.collect_answers()
    assert answers is not None

    search_gate.hold()
    app.start_discovery(answers)
    try:
        await _wait_for(pilot, lambda: _steps_done(app) >= 3)
        _save(app, out_dir, "tui-search.svg")
    finally:
        search_gate.release()

    await _wait_for(pilot, lambda: isinstance(app.screen, RecommendationResultsScreen))
    await _settle(pilot)
    _save(app, out_dir, "tui-results.svg")


async def _capture_export(app: JaullApp, pilot, out_dir: Path) -> None:  # type: ignore[no-untyped-def]
    app.screen.query_one("#res-export", Button).press()
    await _wait_for(pilot, lambda: isinstance(app.screen, ExportReportModal))
    # The real default is timestamped; pin it so the capture is reproducible.
    app.screen.query_one("#res-export-path", Input).value = "jaull-report.json"
    await pilot.pause()
    _save(app, out_dir, "tui-export.svg")
    app.screen.query_one("#res-export-cancel", Button).press()
    await _wait_for(pilot, lambda: isinstance(app.screen, RecommendationResultsScreen))


async def _capture_run(  # type: ignore[no-untyped-def]
    app: JaullApp,
    pilot,
    out_dir: Path,
    advisor: _ScriptAdvisor,
    artifact_gate: _Gate,
) -> None:
    app.screen.query_one("#res-run-0", Button).press()
    await _wait_for(pilot, lambda: isinstance(app.screen, RecommendationExecutionScreen))
    _save(app, out_dir, "tui-run-empty.svg")

    screen = app.screen
    assert isinstance(screen, RecommendationExecutionScreen)

    # Preparing: the prompt is already in the history, the model is not ready.
    advisor.set_response(
        "Local inference keeps the prompt on this machine: the weights are "
        "read from disk and the tokens are generated by llama.cpp, so nothing "
        "is sent to a third-party API."
    )
    artifact_gate.hold()
    screen.query_one("#run-prompt-input", TextArea).load_text(
        "In two sentences, what is local inference?"
    )
    screen.query_one("#run-generate", Button).press()
    try:
        await _wait_for(pilot, lambda: bool(screen.query(".inference-prompt")))
        _save(app, out_dir, "tui-run-loading.svg")
    finally:
        artifact_gate.release()

    await _wait_for(pilot, lambda: bool(screen.query(".inference-response")))

    advisor.set_response(
        "Q4_K_M keeps most of the quality of the original weights at roughly a "
        "quarter of the size, which is why it fits in this machine's VRAM."
    )
    screen.query_one("#run-prompt-input", TextArea).load_text(
        "And why was Q4_K_M chosen?"
    )
    screen.query_one("#run-generate", Button).press()
    await _wait_for(pilot, lambda: len(screen.query(".inference-response")) == 2)
    await _settle(pilot)
    _save(app, out_dir, "tui-run-history.svg")

    # A failure the user can recover from: history intact, error next to the
    # composer rather than in a modal.
    advisor.set_failure(
        ExecutionTimeoutError("llama-cli did not finish within 300 s.")
    )
    screen.query_one("#run-prompt-input", TextArea).load_text(
        "Summarise the whole documentation."
    )
    screen.query_one("#run-generate", Button).press()
    await _wait_for(pilot, lambda: screen.query_one("#run-error-message").display)
    await _settle(pilot)
    _save(app, out_dir, "tui-run-error.svg")
    advisor.set_failure(None)


async def _capture_advanced(app: JaullApp, pilot, out_dir: Path) -> None:  # type: ignore[no-untyped-def]
    app.restart_workflow()
    await pilot.pause()
    app.push_screen("advanced")
    await _settle(pilot)
    _save(app, out_dir, "tui-home.svg")


async def _capture_estimate(app: JaullApp, pilot, out_dir: Path) -> None:  # type: ignore[no-untyped-def]
    app.push_screen("estimate")
    await _settle(pilot)
    _save(app, out_dir, "tui-estimate.svg")


async def capture_all(out_dir: Path, size: tuple[int, int]) -> list[Path]:
    """Write every documented screenshot into `out_dir`. Returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _pin_logo_frame()

    gates = Gates()
    scan_gate, search_gate, artifact_gate = gates.scan, gates.search, gates.artifact
    app = build_app(gates)
    advisor = app.advisor
    assert isinstance(advisor, _ScriptAdvisor)

    async with app.run_test(size=size) as pilot:
        try:
            print("Capturing welcome…")
            await _capture_welcome(app, pilot, out_dir)

            print("Capturing hardware analysis…")
            await _capture_hardware(app, pilot, out_dir, scan_gate)

            print("Capturing wizard…")
            assert isinstance(app.screen, HardwareAnalysisScreen)
            await _capture_wizard(app, pilot, out_dir)

            print("Capturing search and results…")
            await _capture_search_and_results(app, pilot, out_dir, search_gate)

            print("Capturing export…")
            await _capture_export(app, pilot, out_dir)

            print("Capturing run…")
            await _capture_run(app, pilot, out_dir, advisor, artifact_gate)

            print("Capturing advanced tools…")
            await _capture_advanced(app, pilot, out_dir)

            print("Capturing estimate…")
            await _capture_estimate(app, pilot, out_dir)
        finally:
            # A held gate would keep a worker thread parked past teardown.
            gates.release_all()

    return [out_dir / name for name in SCREENSHOTS]


def _parse_size(raw: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in raw.lower().split("x", maxsplit=1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {raw!r}") from None
    return width, height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help="directory to write the SVGs into (default: docs/assets)",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=TERMINAL_SIZE,
        help="terminal size as WIDTHxHEIGHT (default: 110x32)",
    )
    args = parser.parse_args(argv)

    asyncio.run(capture_all(args.out, args.size))
    print(f"\nDone. Screenshots saved under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
