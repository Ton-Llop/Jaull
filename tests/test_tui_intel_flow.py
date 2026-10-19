"""`uv run jaull` end to end on a simulated Intel Arc Pro B60 (Windows and WSL2).

The user path is walked through the real screens: scan -> hardware -> requirements ->
discovery -> results -> run / benchmark. Only the Hub, the host probes and the
llama.cpp executables are scripted (see ``tests/_intel_flow.py``). This proves the
flow is wired and that states and reasons reach the screens; it does **not** prove
anything about a physical B60.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button

from jaull.domain.benchmarks import BenchmarkRecord
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.hardware import ComputeBackend, MemoryInfo
from jaull.domain.runtime import ExecutionReadinessReason, ExecutionReadinessStatus
from jaull.tui.app import JaullApp
from jaull.tui.screens.execution_paths import ExecutionPathsScreen
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
from jaull.tui.screens.recommendation_execution import RecommendationExecutionScreen
from jaull.tui.screens.recommendation_results import RecommendationResultsScreen
from jaull.tui.screens.requirements_wizard import RequirementsWizardScreen
from tests._intel_fixtures import (
    B60_NAME,
    GIB,
    IGPU_NAME,
    LLVMPIPE_SUMMARY,
    WINDOWS_OS,
    WSL2_OS,
    scan,
    scan_nvidia,
)
from tests._intel_flow import ScriptedLlama, intel_advisor
from tests.test_tui_benchmark import _run_workers_inline as _inline_benchmark
from tests.test_tui_guided_workflow import _run, _wait_for
from tests.test_tui_recommendation_execution import (
    _run_screen_is_ready,
    _run_workers_inline,
    _set_prompt,
    _visible_text,
    _wait_until,
)

TWO_IDENTICAL = (("Vulkan0", B60_NAME), ("Vulkan1", B60_NAME))
ONLY_THE_IGPU = (("Vulkan0", IGPU_NAME),)
SOFTWARE_ONLY = (("Vulkan0", "llvmpipe (LLVM 15.0.7, 256 bits)"),)


async def _walk_to_results(app: JaullApp, pilot: Any) -> RecommendationResultsScreen:
    """Welcome -> guided analysis -> hardware -> requirements -> search -> results."""
    app.start_guided_workflow()
    await _wait_for(
        pilot,
        lambda: isinstance(pilot.app.screen, HardwareAnalysisScreen)
        and bool(pilot.app.screen.query("#hw-continue")),
    )
    pilot.app.screen.query_one("#hw-continue", Button).press()
    await _wait_for(pilot, lambda: isinstance(pilot.app.screen, RequirementsWizardScreen))
    answers = pilot.app.screen.collect_answers()
    assert answers is not None
    app.start_discovery(answers)
    await _wait_for(
        pilot,
        lambda: isinstance(pilot.app.screen, RecommendationResultsScreen)
        and bool(pilot.app.screen.query("#res-run-0")),
        timeout=60.0,
    )
    return pilot.app.screen  # type: ignore[return-value]


def _first_plan(app: JaullApp) -> ExecutionPlan:
    assert app.workflow_state is not None and app.workflow_state.recommendations
    plan = app.workflow_state.recommendations[0].plan
    assert plan is not None
    return plan


def _action_buttons(results: RecommendationResultsScreen) -> dict[str, bool]:
    """Which of the primary actions a user can press (True = enabled)."""
    return {
        name: not results.query_one(f"#res-{name}-0", Button).disabled
        for name in ("run", "validate", "speed")
    }


def _generation_commands(scripted: ScriptedLlama) -> list[tuple[str, ...]]:
    return [c for c in scripted.argv_of("llama-cli") if "--model" in c]


# ---------------------------------------------------------------------------
# A card the runtime lists exactly once: the whole path works
# ---------------------------------------------------------------------------
def test_guided_flow_reaches_a_ready_vulkan_plan_for_the_intel_card(tmp_path: Path) -> None:
    async def scenario() -> None:
        advisor, _ = intel_advisor(tmp_path, scan(WINDOWS_OS))
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)

            plan = _first_plan(app)
            selection = plan.backend_selection
            assert selection is not None and selection.selected_backend is ComputeBackend.VULKAN
            assert selection.selected_accelerator is not None
            assert selection.selected_accelerator.name == B60_NAME
            assert plan.execution_readiness is not None
            assert plan.execution_readiness.status is ExecutionReadinessStatus.READY
            flags = {flag.name: flag.value for flag in plan.runtime.flags}
            assert flags["--n-gpu-layers"] == "-1"  # fits: everything on the card
            text = _visible_text(results)
            assert "Vulkan" in text and "Execution path available" in text
            assert "23.00 GiB" in text  # the budget the estimate used, not the 24 GiB capacity
            assert _action_buttons(results) == {"run": True, "validate": True, "speed": True}

    _run(scenario())


def test_run_launches_llama_cli_on_the_device_matched_to_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor, scripted = intel_advisor(
            tmp_path,
            scan(WINDOWS_OS),
            devices=(("Vulkan0", IGPU_NAME), ("Vulkan1", B60_NAME)),
        )
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)
            results.query_one("#res-run-0", Button).press()
            await _wait_until(pilot, lambda: _run_screen_is_ready(pilot.app.screen))
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_workers_inline(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "Say hi")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(pilot, lambda: "Hello from the Arc." in _visible_text(screen))

            [command] = _generation_commands(scripted)
            # llama.cpp numbered the integrated GPU first; Jaull must not pin Vulkan0.
            assert command[command.index("--device") + 1] == "Vulkan1"
            assert command[command.index("--n-gpu-layers") + 1] == "-1"

    _run(scenario())


def test_benchmark_measures_the_matched_device_and_records_where_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor, scripted = intel_advisor(tmp_path, scan(WINDOWS_OS))
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)
            results.query_one("#res-speed-0", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, RecommendationBenchmarkScreen)
                and bool(pilot.app.screen.query("#benchmark-start")),
            )
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _inline_benchmark(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(pilot, lambda: "Benchmark complete" in _visible_text(screen))

            [command] = [c for c in scripted.argv_of("llama-bench") if "-dev" in c]
            assert command[command.index("-dev") + 1] == "Vulkan0"
            records = [
                advisor.load_benchmark_record(identity)
                for identity in advisor.list_benchmark_ids()
            ]
            assert records, "the run was not stored"
            record: BenchmarkRecord = records[-1]
            assert record.requested_backend is ComputeBackend.VULKAN
            assert record.requested_device == "Vulkan0"
            assert record.hardware.os == WINDOWS_OS
            assert any(
                accelerator.name == B60_NAME for accelerator in record.hardware.accelerators
            )

    _run(scenario())


# ---------------------------------------------------------------------------
# A card the runtime cannot be tied to: nothing runs, and the reason is on screen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("devices", "status", "reason", "fragment"),
    [
        (
            TWO_IDENTICAL,
            ExecutionReadinessStatus.UNKNOWN,
            ExecutionReadinessReason.SELECTED_DEVICE_NOT_IDENTIFIED,
            "no device is selected automatically",
        ),
        (
            ONLY_THE_IGPU,
            ExecutionReadinessStatus.NOT_READY,
            ExecutionReadinessReason.SELECTED_DEVICE_NOT_EXPOSED,
            f"did not list '{B60_NAME}'",
        ),
        (
            SOFTWARE_ONLY,
            ExecutionReadinessStatus.NOT_READY,
            ExecutionReadinessReason.SELECTED_DEVICE_NOT_EXPOSED,
            "only a software vulkan renderer",
        ),
    ],
)
def test_an_unconfirmed_device_blocks_every_action_and_explains_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    devices: tuple[tuple[str, str | None], ...],
    status: ExecutionReadinessStatus,
    reason: ExecutionReadinessReason,
    fragment: str,
) -> None:
    async def scenario() -> None:
        advisor, scripted = intel_advisor(tmp_path, scan(WINDOWS_OS), devices=devices)
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)

            readiness = _first_plan(app).execution_readiness
            assert readiness is not None
            assert (readiness.status, readiness.reason) == (status, reason)
            assert fragment in (readiness.message or "")
            text = _visible_text(results)
            assert fragment in text  # the user is told why, on the Results screen
            assert "Execution path available" not in text
            assert _action_buttons(results) == {
                "run": False, "validate": False, "speed": False,
            }
            assert results.query_one("#res-paths-0", Button).disabled is False
            # Nothing was launched or measured to get here.
            assert _generation_commands(scripted) == []
            assert scripted.argv_of("llama-bench") == []

    _run(scenario())


# ---------------------------------------------------------------------------
# WSL2 and a missing vulkaninfo: say what was found, never invent a route
# ---------------------------------------------------------------------------
def test_wsl2_without_a_vulkan_device_warns_and_plans_on_the_cpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        profile = scan(WSL2_OS, summary=LLVMPIPE_SUMMARY, full="", windows=()).model_copy(
            update={"memory": MemoryInfo(total_bytes=64 * GIB, available_bytes=60 * GIB)}
        )
        advisor, _ = intel_advisor(tmp_path, profile, devices=SOFTWARE_ONLY)
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            app.start_guided_workflow()
            await _wait_for(
                pilot,
                lambda: isinstance(pilot.app.screen, HardwareAnalysisScreen)
                and bool(pilot.app.screen.query("#hw-continue")),
            )
            hardware_text = _visible_text(pilot.app.screen)
            assert "Running inside WSL2" in hardware_text
            assert "does not show that the Windows host cannot" in hardware_text
            assert "CPU inference only" in hardware_text

            pilot.app.screen.query_one("#hw-continue", Button).press()
            await _wait_for(pilot, lambda: isinstance(pilot.app.screen, RequirementsWizardScreen))
            app.start_discovery(pilot.app.screen.collect_answers())
            await _wait_for(
                pilot,
                lambda: isinstance(pilot.app.screen, RecommendationResultsScreen)
                and bool(pilot.app.screen.query("#res-run-0")),
                timeout=60.0,
            )
            selection = _first_plan(app).backend_selection
            assert selection is not None and selection.selected_backend is ComputeBackend.CPU
            assert "Vulkan" not in _visible_text(pilot.app.screen).split("Plan")[0]

    _run(scenario())


def test_windows_without_vulkaninfo_names_the_adapter_and_does_not_claim_vulkan(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        profile = scan(WINDOWS_OS, vulkaninfo_present=False)
        advisor, _ = intel_advisor(tmp_path, profile)
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            app.start_guided_workflow()
            await _wait_for(
                pilot,
                lambda: isinstance(pilot.app.screen, HardwareAnalysisScreen)
                and bool(pilot.app.screen.query("#hw-continue")),
            )
            text = _visible_text(pilot.app.screen)
            assert B60_NAME in text
            assert "not confirmed" in text.lower() and "vulkaninfo is not installed" in text
            assert select_backend(profile) is ComputeBackend.CPU

    _run(scenario())


def select_backend(profile: Any) -> ComputeBackend:
    from jaull.runtime.backend_selection import select_runtime_backend

    return select_runtime_backend(profile).selected_backend


# ---------------------------------------------------------------------------
# NVIDIA keeps its own flow
# ---------------------------------------------------------------------------
def test_nvidia_still_gets_a_ready_cuda_plan_through_the_same_flow(tmp_path: Path) -> None:
    async def scenario() -> None:
        advisor, scripted = intel_advisor(
            tmp_path, scan_nvidia(), devices=(("CUDA0", "NVIDIA GeForce RTX 4070"),)
        )
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)

            plan = _first_plan(app)
            assert plan.backend_selection is not None
            assert plan.backend_selection.selected_backend is ComputeBackend.CUDA
            assert plan.execution_readiness is not None
            assert plan.execution_readiness.status is ExecutionReadinessStatus.READY
            assert _action_buttons(results) == {"run": True, "validate": True, "speed": True}
            assert "CUDA" in _visible_text(results)
            assert scripted.argv_of("llama-bench") == []

    _run(scenario())


def test_the_paths_screen_also_refuses_and_explains_an_unidentified_device(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        advisor, scripted = intel_advisor(tmp_path, scan(WINDOWS_OS), devices=TWO_IDENTICAL)
        app = JaullApp(advisor=advisor)
        async with app.run_test(size=(140, 50)) as pilot:
            results = await _walk_to_results(app, pilot)
            results.query_one("#res-paths-0", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
                and bool(pilot.app.screen.query("#paths-action-reason"))
                and "no device is selected automatically" in _visible_text(pilot.app.screen),
            )

            paths = pilot.app.screen
            assert isinstance(paths, ExecutionPathsScreen)
            for action in ("run", "validate", "benchmark"):
                assert paths.query_one(f"#paths-{action}", Button).disabled is True
            assert _generation_commands(scripted) == []

    _run(scenario())
