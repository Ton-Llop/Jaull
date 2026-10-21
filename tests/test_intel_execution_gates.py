"""A plan whose device or runtime is not confirmed can be neither run nor measured.

The Results screen decides from the plan it was given, but Run and Benchmark
prepare the plan again, and that fresh readiness is the one that matters: a
stale ``READY`` must not let a launch through. All devices and logs here are
synthetic; nothing touches a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button

from jaull.application.recommendation.execution_plans import execution_plan_for_recommendation
from jaull.benchmarks.errors import BenchmarkConfigurationError
from jaull.domain.execution_plans import ExecutionPlan
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    LlamaCppRuntimeDevice,
)
from jaull.presentation.plan_labels import runtime_block_reason
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
from jaull.tui.screens.recommendation_execution import RecommendationExecutionScreen
from tests.test_runtime_device_matching import B60, _selection
from tests.test_tui_benchmark import _BenchmarkAdvisor, _run_workers_inline
from tests.test_tui_recommendation_execution import (
    _backend_selection,
    _FakeAdvisor,
    _recommendation,
    _run,
    _run_screen_is_ready,
    _set_prompt,
    _visible_text,
    _wait_until,
)
from tests.test_tui_recommendation_execution import (
    _run_workers_inline as _run_inline_execution,
)

AMBIGUOUS = (
    "llama-cli lists several vulkan devices (Vulkan0: Intel(R) Arc(TM) Pro B60 Graphics, "
    "Vulkan1: Intel(R) Arc(TM) Pro B60 Graphics) and none can be tied to the selected "
    "accelerator; no device is selected automatically"
)
MISMATCH = (
    "llama-cli did not list 'Intel(R) Arc(TM) Pro B60 Graphics' among its vulkan devices "
    "(Vulkan0: Intel(R) Iris(R) Xe Graphics)"
)


def _readiness(
    status: ExecutionReadinessStatus,
    reason: ExecutionReadinessReason,
    message: str,
    *,
    devices: tuple[str, ...] = ("Vulkan0",),
) -> ExecutionReadiness:
    capability = LlamaCppBackendCapability(
        backend=ComputeBackend.VULKAN,
        state=LlamaCppBackendCapabilityState.CONFIRMED,
        reason=LlamaCppCapabilityReason.BACKEND_EXPOSED,
        devices=[
            LlamaCppRuntimeDevice(
                backend=ComputeBackend.VULKAN,
                runtime_id=device,
                name="Intel(R) Arc(TM) Pro B60 Graphics",
            )
            for device in devices
        ],
    )
    return ExecutionReadiness(
        status=status,
        reason=reason,
        selection=_selection(B60),
        runtime_capability=LlamaCppRuntimeCapability(
            binary_path="llama-cli",
            binary_status=LlamaCppBinaryStatus.AVAILABLE,
            backend_capabilities=[capability],
        ),
        selected_backend_capability=capability,
        message=message,
    )


def _with_readiness(plan: ExecutionPlan, readiness: ExecutionReadiness) -> ExecutionPlan:
    return plan.model_copy(
        update={
            "execution_readiness": readiness,
            "backend_selection": readiness.selection,
        }
    )


NOT_IDENTIFIED = _readiness(
    ExecutionReadinessStatus.UNKNOWN,
    ExecutionReadinessReason.SELECTED_DEVICE_NOT_IDENTIFIED,
    AMBIGUOUS,
    devices=("Vulkan0", "Vulkan1"),
)
NOT_EXPOSED = _readiness(
    ExecutionReadinessStatus.NOT_READY,
    ExecutionReadinessReason.SELECTED_DEVICE_NOT_EXPOSED,
    MISMATCH,
)
READY = _readiness(
    ExecutionReadinessStatus.READY,
    ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED,
    "llama-cli exposes vulkan device(s)",
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def _plan() -> ExecutionPlan:
    return execution_plan_for_recommendation(_recommendation())


def test_an_unidentified_device_blocks_the_plan_and_says_why() -> None:
    reason = runtime_block_reason(_with_readiness(_plan(), NOT_IDENTIFIED))

    assert reason is not None
    assert "no device is selected automatically" in reason
    assert "Vulkan" in reason


def test_a_missing_device_blocks_the_plan_and_says_why() -> None:
    reason = runtime_block_reason(_with_readiness(_plan(), NOT_EXPOSED))

    assert reason is not None and "Iris" in reason


def test_a_confirmed_device_does_not_block() -> None:
    assert runtime_block_reason(_with_readiness(_plan(), READY)) is None


def test_confirmed_vulkan_remains_actionable_while_gguf_needs_preparation() -> None:
    preparable = READY.model_copy(
        update={
            "status": ExecutionReadinessStatus.PREPARABLE,
            "reason": ExecutionReadinessReason.ARTIFACT_PREPARATION_REQUIRED,
            "message": "The runtime is ready; GGUF will be downloaded before execution.",
        }
    )

    assert runtime_block_reason(_with_readiness(_plan(), preparable)) is None


def test_preparable_vulkan_without_device_evidence_remains_blocked() -> None:
    preparable = READY.model_copy(
        update={
            "status": ExecutionReadinessStatus.PREPARABLE,
            "reason": ExecutionReadinessReason.ARTIFACT_PREPARATION_REQUIRED,
            "selected_backend_capability": None,
        }
    )

    assert runtime_block_reason(_with_readiness(_plan(), preparable)) is not None


def test_vulkan_unknown_probe_failure_is_not_executable() -> None:
    probe_failed = _readiness(
        ExecutionReadinessStatus.UNKNOWN,
        ExecutionReadinessReason.PROBE_FAILED,
        "llama-cli capability probe failed",
    )

    assert runtime_block_reason(_with_readiness(_plan(), probe_failed)) is not None


# ---------------------------------------------------------------------------
# Run: the fresh readiness, not the one the screen opened with
# ---------------------------------------------------------------------------
class _StaleReadyAdvisor(_FakeAdvisor):
    """Results showed READY; by the time the plan is prepared it is not."""

    def __init__(self, fresh: ExecutionReadiness) -> None:
        super().__init__(backend_selection=_backend_selection(ComputeBackend.VULKAN))
        self.fresh = fresh

    def prepare_execution_plan(self, plan: Any, **kwargs: Any) -> Any:
        prepared = super().prepare_execution_plan(plan, **kwargs)
        return prepared.model_copy(
            update={"plan": _with_readiness(prepared.plan, self.fresh)}
        )


@pytest.mark.parametrize("fresh", [NOT_IDENTIFIED, NOT_EXPOSED])
def test_run_refuses_a_plan_that_became_unconfirmed_after_preparation(
    fresh: ExecutionReadiness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        recommendation = _recommendation()
        advisor = _StaleReadyAdvisor(fresh)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        plan = _with_readiness(_plan(), READY)

        async with app.run_test(size=(120, 50)) as pilot:
            app.run_recommendation(recommendation, plan)
            await _wait_until(pilot, lambda: _run_screen_is_ready(pilot.app.screen))
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationExecutionScreen)
            _run_inline_execution(pilot.app, screen, monkeypatch)

            _set_prompt(screen, "Say hi")
            screen.query_one("#run-generate", Button).press()
            await _wait_until(
                pilot, lambda: fresh.message and fresh.message[:40] in _visible_text(screen)
            )

            assert "run" not in advisor.operations
            assert advisor.runs == []

    _run(scenario())


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class _FreshReadinessBenchmarkAdvisor(_BenchmarkAdvisor):
    def __init__(self, tmp_path: Path, fresh: ExecutionReadiness) -> None:
        super().__init__(tmp_path)
        self.fresh = fresh

    def prepare_execution_plan(self, plan: Any, **kwargs: Any) -> Any:
        prepared = super().prepare_execution_plan(plan, **kwargs)
        hardware = self.hardware
        updated = _with_readiness(prepared.plan, self.fresh).model_copy(
            update={"hardware": hardware}
        )
        return prepared.model_copy(update={"plan": updated})


@pytest.mark.parametrize("fresh", [NOT_IDENTIFIED, NOT_EXPOSED])
def test_benchmark_refuses_a_plan_that_became_unconfirmed_after_preparation(
    fresh: ExecutionReadiness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        advisor = _FreshReadinessBenchmarkAdvisor(tmp_path, fresh)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        plan = _with_readiness(_plan(), READY)

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationBenchmarkScreen(_recommendation(), plan))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot, lambda: fresh.message and fresh.message[:40] in _visible_text(screen)
            )

            assert advisor.benchmark_calls == 0
            assert advisor.matrix_calls == 0

    _run(scenario())


def test_benchmark_never_measures_an_unnamed_vulkan_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY with two indistinguishable devices cannot happen for Vulkan; if a stale or
    inconsistent plan says so anyway, the measurement itself still refuses to guess."""
    inconsistent = _readiness(
        ExecutionReadinessStatus.READY,
        ExecutionReadinessReason.SELECTED_BACKEND_EXPOSED,
        "llama-cli exposes vulkan device(s)",
        devices=("Vulkan0", "Vulkan1"),
    )

    async def scenario() -> None:
        advisor = _FreshReadinessBenchmarkAdvisor(tmp_path, inconsistent)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        plan = _with_readiness(_plan(), READY)

        async with app.run_test(size=(120, 50)) as pilot:
            app.push_screen(RecommendationBenchmarkScreen(_recommendation(), plan))
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, RecommendationBenchmarkScreen)
            _run_workers_inline(screen, monkeypatch)

            screen.query_one("#benchmark-start", Button).press()
            await _wait_until(
                pilot, lambda: "device" in _visible_text(screen).lower()
                and "identif" in _visible_text(screen).lower()
            )

            assert advisor.benchmark_calls == 0

    _run(scenario())


# ---------------------------------------------------------------------------
# Speed measurement through the service
# ---------------------------------------------------------------------------
def test_speed_measurement_refuses_a_blocked_plan_before_touching_the_artifact() -> None:
    from jaull.advisor.service import AdvisorService
    from tests.test_benchmark_plan_speed import _Advisor
    from tests.test_fastest_v2_applicability import _cuda_case

    plan, _ = _cuda_case()
    blocked = plan.model_copy(update={"execution_readiness": NOT_IDENTIFIED})
    advisor = _Advisor(["CUDA0"])

    with pytest.raises(BenchmarkConfigurationError, match="no device is selected"):
        AdvisorService.benchmark_plan_speed(advisor, blocked)  # type: ignore[arg-type]

    assert advisor.downloads == [] and advisor.requests == []
