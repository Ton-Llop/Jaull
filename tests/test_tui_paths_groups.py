"""The execution-paths list: runtime groups, one line per path, the rest on demand.

Groups are by runtime option (llama.cpp · CPU, Transformers · PyTorch · CPU...), the
recommended group leads and is open, and paths that cannot work here stay behind a
toggle instead of mixing with the ones that can.
"""

from __future__ import annotations

from typing import Any

from textual.widgets import Button

from jaull.application.recommendation.execution_plans import execution_plan_for_recommendation
from jaull.domain.execution_plans import (
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    PlanCompatibilityStatus,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessReason,
    ExecutionReadinessStatus,
    RuntimeName,
)
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.app import JaullApp
from jaull.tui.screens.execution_paths import ExecutionPathsScreen
from tests.test_tui_recommendation_execution import (
    _backend_selection,
    _FakeAdvisor,
    _pytorch_runtime_capability,
    _recommendation,
    _recommendation_with_plan,
    _run,
    _runtime_capability,
    _visible_text,
    _wait_until,
)


def _ready() -> ExecutionReadiness:
    return ExecutionReadiness(
        status=ExecutionReadinessStatus.READY,
        reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
        selection=_backend_selection(ComputeBackend.CPU),
        runtime_capability=_runtime_capability(ComputeBackend.CPU),
        message="llama.cpp CPU execution is available",
    )


def _blocked() -> ExecutionReadiness:
    return _ready().model_copy(update={
        "status": ExecutionReadinessStatus.NOT_READY,
        "reason": ExecutionReadinessReason.RUNTIME_MISSING,
        "message": "llama.cpp is not installed",
    })


def _gguf(quantization: str, readiness: ExecutionReadiness | None = None) -> ExecutionPlan:
    plan = execution_plan_for_recommendation(_recommendation(quantization=quantization))
    return plan.model_copy(update={
        "plan_id": f"org/Tiny-GGUF:{quantization}",
        "artifact": plan.artifact.model_copy(update={
            "repo_id": "org/Tiny-GGUF",
            "format": ArtifactVariantFormat.GGUF,
            "quantization": quantization,
            "filename": f"tiny-{quantization.lower()}.gguf",
        }),
        "execution_readiness": readiness or _ready(),
    })


def _transformers(precision: str, *, supported: bool = True) -> ExecutionPlan:
    recommendation = _recommendation_with_plan(
        repo_id="org/Tiny-GGUF",
        identity=ModelIdentity(model_name="Tiny-GGUF"),
        format_=ArtifactVariantFormat.SAFETENSORS,
        precision=precision,
        runtime=RuntimeName.TRANSFORMERS,
    )
    plan = recommendation.plan
    assert plan is not None
    readiness = ExecutionReadiness(
        status=ExecutionReadinessStatus.READY,
        reason=ExecutionReadinessReason.RUNTIME_AVAILABLE,
        selection=_backend_selection(ComputeBackend.CPU),
        runtime_capability=_pytorch_runtime_capability(),
        message="Transformers runtime is ready",
    )
    update: dict[str, Any] = {
        "plan_id": f"transformers:{precision}",
        "execution_readiness": readiness,
    }
    if not supported:
        update["compatibility"] = PlanCompatibilityStatus.NOT_COMPATIBLE
        update["warnings"] = [f"Not selected: {precision} needs a CUDA backend."]
    return plan.model_copy(update=update)


def _recommended(plan: ExecutionPlan) -> ModelRecommendation:
    return _recommendation().model_copy(update={"plan": plan})


async def _open(
    pilot: Any, app: JaullApp, plans: list[ExecutionPlan], recommended: ExecutionPlan,
) -> ExecutionPathsScreen:
    app.push_screen(ExecutionPathsScreen(_recommended(recommended)))
    await _wait_until(
        pilot,
        lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
        and bool(pilot.app.screen.query(".path-option"))
        and not pilot.app.screen.query_one("#paths-run", Button).disabled,
    )
    screen = pilot.app.screen
    assert isinstance(screen, ExecutionPathsScreen)
    return screen


def _headers(screen: ExecutionPathsScreen) -> list[str]:
    return [str(h.content) for h in screen.query(".path-group-head")]


def _shown_names(screen: ExecutionPathsScreen) -> list[str]:
    # A row just mounted by a rebuild has no children until its compose has run; it
    # is not shown yet, so polling must not fail on it.
    return [
        str(name.content)
        for row in screen.query(".path-option")
        if all(node.display for node in (row, *row.ancestors))
        for name in row.query(".po-name")
    ]


def _app(plans: list[ExecutionPlan]) -> JaullApp:
    return JaullApp(advisor=_FakeAdvisor(execution_plans=plans))  # type: ignore[arg-type]


def test_the_recommended_group_leads_open_and_the_rest_start_closed() -> None:
    async def scenario() -> None:
        q4, q5 = _gguf("Q4_K_M"), _gguf("Q5_K_M")
        plans = [_transformers("float16"), q5, q4]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            headers = _headers(screen)
            assert "llama.cpp" in headers[0] and "★ recommended" in headers[0]
            assert "Transformers" in headers[1] and "★" not in headers[1]
            assert screen._selected_plan_id == q4.plan_id
            # The open group lists the suggested quantization first; the closed one is hidden.
            assert _shown_names(screen) == ["Q4_K_M", "Q5_K_M"]

    _run(scenario())


def test_a_path_without_a_problem_takes_one_line() -> None:
    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        plans = [q4, _gguf("Q5_K_M")]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            assert all(row.region.height == 1 for row in screen.query(".path-option"))

    _run(scenario())


def test_paths_that_do_not_fit_wait_behind_a_toggle_with_their_reason() -> None:
    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        plans = [q4, _transformers("int8", supported=False)]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            assert _shown_names(screen) == ["Q4_K_M"]
            toggle = screen.query_one("#paths-toggle-unfit", Button)
            assert "1" in str(toggle.label) and "fit" in str(toggle.label)

            toggle.press()
            await _wait_until(pilot, lambda: len(_shown_names(screen)) == 2)
            text = _visible_text(screen.query_one("#paths-list"))
            assert "Transformers · safetensors · int8" in text
            assert "needs a CUDA backend" in text
            assert "Hide" in str(screen.query_one("#paths-toggle-unfit", Button).label)

            screen.query_one("#paths-toggle-unfit", Button).press()
            await _wait_until(pilot, lambda: len(_shown_names(screen)) == 1)

    _run(scenario())


def test_when_nothing_fits_the_list_is_not_left_empty() -> None:
    async def scenario() -> None:
        plan = _transformers("int8", supported=False)
        async with _app([plan]).run_test(size=(120, 50)) as pilot:
            app = pilot.app
            app.push_screen(ExecutionPathsScreen(_recommended(plan)))
            await _wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
                and bool(pilot.app.screen.query(".path-option")),
            )
            screen = pilot.app.screen
            assert isinstance(screen, ExecutionPathsScreen)
            assert len(_shown_names(screen)) == 1
            assert not screen.query("#paths-toggle-unfit")

    _run(scenario())


def test_a_runtime_that_is_merely_missing_is_not_hidden_as_unfit() -> None:
    async def scenario() -> None:
        blocked = _gguf("Q4_K_M", _blocked())
        async with _app([blocked]).run_test(size=(120, 50)) as pilot:
            screen = await _open_blocked(pilot, blocked)
            assert _shown_names(screen) == ["Q4_K_M"]
            assert not screen.query("#paths-toggle-unfit")
            assert "llama.cpp is not installed" in _visible_text(screen)

    async def _open_blocked(pilot: Any, plan: ExecutionPlan) -> ExecutionPathsScreen:
        pilot.app.push_screen(ExecutionPathsScreen(_recommended(plan)))
        await _wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, ExecutionPathsScreen)
            and bool(pilot.app.screen.query(".path-option"))
            and "llama.cpp is not installed" in _visible_text(pilot.app.screen),
        )
        screen = pilot.app.screen
        assert isinstance(screen, ExecutionPathsScreen)
        return screen

    _run(scenario())


def test_advanced_quantizations_sit_inside_their_group() -> None:
    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        plans = [q4, _gguf("Q5_K_M"), _gguf("Q8_0"), _gguf("Q2_K")]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            assert _shown_names(screen) == ["Q4_K_M", "Q5_K_M"]
            toggle = screen.query_one("#paths-toggle-advanced", Button)
            assert "2" in str(toggle.label)
            assert not screen.query("#paths-quant")

            toggle.press()
            await _wait_until(pilot, lambda: len(_shown_names(screen)) == 4)
            assert _shown_names(screen)[:2] == ["Q4_K_M", "Q5_K_M"]

    _run(scenario())


def test_rebuilding_the_list_keeps_focus_on_the_toggle_that_was_pressed() -> None:
    """Regression: the focused toggle was removed and focus jumped to a filter chip."""

    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        plans = [q4, _gguf("Q5_K_M"), _gguf("Q8_0"), _transformers("int8", supported=False)]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            for toggle_id in ("paths-toggle-advanced", "paths-toggle-unfit"):
                screen.query_one(f"#{toggle_id}").focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause(0.3)
                assert screen.focused is not None and screen.focused.id == toggle_id

    _run(scenario())


def test_choosing_a_path_updates_the_detail_without_rebuilding_the_list() -> None:
    async def scenario() -> None:
        q4, q5 = _gguf("Q4_K_M"), _gguf("Q5_K_M")
        advisor = _FakeAdvisor(execution_plans=[q4, q5])
        async with JaullApp(advisor=advisor).run_test(size=(120, 50)) as pilot:  # type: ignore[arg-type]
            screen = await _open(pilot, pilot.app, [q4, q5], recommended=q4)
            rows = list(screen.query(".path-option"))
            rows[1].action_choose()
            await _wait_until(pilot, lambda: screen._selected_plan_id == q5.plan_id)
            assert list(screen.query(".path-option")) == rows
            assert rows[1].has_class("-selected") and not rows[0].has_class("-selected")
            assert "Q5_K_M" in str(screen.query_one("#paths-selected").content)
            assert advisor.operations == ["plans"]

    _run(scenario())


def test_transformers_variants_can_be_told_apart() -> None:
    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        plans = [q4, _transformers("float16"), _transformers("bfloat16")]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=q4)
            header = next(
                h for h in screen.query(".path-group-head") if "Transformers" in str(h.content)
            )
            header.action_open_close()
            await pilot.pause()
            names = [n for n in _shown_names(screen) if n != "Q4_K_M"]
            assert len(names) == 2 and len(set(names)) == 2
            assert any("float16" in n for n in names) and any("bfloat16" in n for n in names)

    _run(scenario())


def test_collapsing_a_group_keeps_the_selection_and_its_actions() -> None:
    async def scenario() -> None:
        q4 = _gguf("Q4_K_M")
        async with _app([q4, _gguf("Q5_K_M")]).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, [q4], recommended=q4)
            header = screen.query_one(".path-group-head")
            header.action_open_close()
            await pilot.pause()
            assert _shown_names(screen) == []
            assert str(header.content).startswith("▸")
            assert screen._selected_plan_id == q4.plan_id
            assert not screen.query_one("#paths-run", Button).disabled
            header.action_open_close()
            await pilot.pause()
            assert str(header.content).startswith("▾")
            assert len(_shown_names(screen)) == 2

    _run(scenario())


def test_arrow_keys_walk_the_list() -> None:
    async def scenario() -> None:
        q4, q5 = _gguf("Q4_K_M"), _gguf("Q5_K_M")
        async with _app([q4, q5]).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, [q4, q5], recommended=q4)
            screen.query_one(".path-group-head").focus()
            await pilot.pause()
            await pilot.press("down")
            rows = list(screen.query(".path-option"))
            assert screen.focused is rows[0]
            await pilot.press("down")
            assert screen.focused is rows[1]
            await pilot.press("up")
            assert screen.focused is rows[0]

    _run(scenario())


def test_advanced_quantizations_and_focus_belong_to_their_own_runtime_group() -> None:
    from tests.test_runtime_device_matching import B60, _selection

    async def scenario() -> None:
        cpu = [_gguf("Q4_K_M"), _gguf("Q8_0")]
        vulkan = [plan.model_copy(update={
            "plan_id": plan.plan_id + "-vulkan", "backend_selection": _selection(B60),
        }) for plan in cpu]
        plans = [*cpu, *vulkan]
        async with _app(plans).run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, pilot.app, plans, recommended=cpu[0])
            toggles = list(screen.query(".path-group-body Button"))
            assert len(toggles) == 2
            assert len({toggle.id for toggle in toggles}) == 2
            header = next(h for h in screen.query(".path-group-head") if "Vulkan" in str(h.content))
            header.action_open_close()
            await pilot.pause()
            toggle = screen.query_one("#paths-group-1").query_one(Button)
            toggle_id = toggle.id
            toggle.focus()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(screen.query(".path-option")) == 3)
            assert not screen.query_one("#paths-group-0").query("#paths-plan-1")
            assert screen.focused is not None and screen.focused.id == toggle_id
            assert screen.query_one("#paths-group-1").query("#paths-plan-3")

    _run(scenario())
