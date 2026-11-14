"""TUI Setup: the five steps compose the facade like ``jaull serve start``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Checkbox, Input, RadioButton, Select, Static

from jaull.application.serving import ServerSizingError
from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.artifacts.storage import LocalModelFile
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppInstallation, RuntimeSource
from jaull.execution.server_process import ServerStatus
from jaull.tui.app import JaullApp
from jaull.tui.screens.setup import SetupScreen
from tests.test_cli_serve import ARTIFACT, STATE, _plan


async def _wait_until(pilot: Pilot[Any], condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not condition():
            await pilot.pause(0.02)


def _text(screen: SetupScreen, selector: str) -> str:
    return str(screen.query_one(selector, Static).render())


INSTALLED = LlamaCppInstallation(
    root="/x", llama_cli="/x/llama-cli", source=RuntimeSource.MANAGED, version_text="b11889",
)


class SetupAdvisor:
    services = None

    def __init__(self, *, installed: bool = True, plan_error: Exception | None = None) -> None:
        self.installed = installed
        self.plan_error = plan_error
        self.running = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def scan_hardware(self) -> Any:
        return SimpleNamespace(cpu=SimpleNamespace(model="Test CPU"))

    def select_runtime_backend(self, hardware: Any) -> Any:
        return SimpleNamespace(
            selected_backend=ComputeBackend.VULKAN,
            selected_accelerator=SimpleNamespace(name="Intel Arc Pro B60"),
            alternatives=[SimpleNamespace(backend=ComputeBackend.CPU)],
        )

    def local_models(self) -> list[tuple[LocalModelFile, bool]]:
        return [(LocalModelFile("org/Model-GGUF", "model-Q4_K_M.gguf", Path("/m/a.gguf"),
                                1024**3, None), False)]

    def server_status(self) -> ServerStatus:
        return ServerStatus(state=STATE if self.running else None, running=self.running,
                            healthy=self.running)

    def gguf_quantization(self, filename: str) -> str | None:
        from jaull.huggingface.classifiers import gguf_quantization

        return gguf_quantization(filename)

    def managed_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation | None:
        self.calls.append(("managed", {"backend": backend}))
        return INSTALLED if self.installed else None

    def install_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation:
        self.calls.append(("install", {"backend": backend}))
        self.installed = True
        return INSTALLED

    def server_context_limit(self, repo_id: str, **kwargs: Any) -> Any:
        from jaull.application.serving import ContextLimit

        self.calls.append(("limit", {"repo_id": repo_id, **kwargs}))
        return ContextLimit(users=kwargs["users"], largest_per_user=98304, native=131072)

    def prepare_serve_artifact(self, repo_id: str, **kwargs: Any) -> ModelArtifact:
        self.calls.append(("prepare", {"repo_id": repo_id, **kwargs}))
        kwargs["on_progress"]("Verifying model-Q4_K_M.gguf")
        return ARTIFACT

    def plan_server(self, **kwargs: Any) -> ServerPlan:
        self.calls.append(("plan", kwargs))
        if self.plan_error is not None:
            raise self.plan_error
        return _plan(parallel=kwargs["users"])

    def start_server(self, plan: ServerPlan, **kwargs: Any) -> ServerStart:
        self.calls.append(("start", kwargs))
        self.running = True
        return ServerStart(
            state=STATE, api_key_path=Path("/data/serve/api-key"), new_api_key="KEY123",
            start_script=Path("/data/serve/start-server.sh"), firewall_rule=None, lan_url=None,
        )

    def llama_cpp_update_notice(self) -> str | None:
        return None

    def stop_server(self) -> bool:
        self.calls.append(("stop", {}))
        self.running = False
        return True

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]


async def _open(pilot: Pilot[Any], app: JaullApp) -> SetupScreen:
    screen = SetupScreen()
    await app.push_screen(screen)
    await _wait_until(pilot, lambda: not screen._busy and bool(screen.query("#setup-backend")))
    return screen


@pytest.mark.parametrize("size", [(80, 24), (160, 48)])
def test_setup_starts_a_downloaded_model_and_can_stop_it(size: tuple[int, int]) -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = await _open(pilot, app)
            assert "Intel Arc Pro B60" in _text(screen, "#setup-hardware")
            assert screen.query_one("#setup-backend", Select).value is ComputeBackend.VULKAN
            assert "b11889 (vulkan)" in _text(screen, "#setup-runtime")
            assert not screen.query_one("#setup-install", Button).display

            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            screen.query_one("#setup-users", Input).value = "4"
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            assert advisor.called("prepare")[0]["repo_id"] == "org/Model-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "Q4_K_M"
            plan = advisor.called("plan")[0]
            assert plan["users"] == 4 and plan["backend"] is ComputeBackend.VULKAN
            assert advisor.called("start")[0] == {"lan": False, "port": 8090, "web_ui": False}
            output = _text(screen, "#setup-output")
            assert "4 users at once" in output and "KEY123" in output
            assert screen.query_one("#setup-start", Button).disabled
            stop = screen.query_one("#setup-stop", Button)
            assert not stop.disabled
            for button in screen.query(Button):
                if button.display:
                    assert button.region.right <= size[0]

            stop.press()
            await _wait_until(pilot, lambda: bool(advisor.called("stop")) and not screen._busy)
            assert "Server stopped." in _text(screen, "#setup-output")
            assert not screen.query_one("#setup-start", Button).disabled

    asyncio.run(scenario())


def test_setup_uses_a_typed_repository_and_the_local_network() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-repo", Input).value = "https://huggingface.co/org/Other-GGUF"
            screen.query_one("#setup-quant", Input).value = "IQ4_XS"
            screen.query_one("#setup-lan", RadioButton).value = True
            screen.query_one("#setup-port", Input).value = "8095"
            screen.query_one("#setup-webui", Checkbox).value = True
            screen.query_one("#setup-shared", Checkbox).value = True
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            assert advisor.called("prepare")[0]["repo_id"] == "org/Other-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "IQ4_XS"
            assert advisor.called("start")[0] == {"lan": True, "port": 8095, "web_ui": True}
            assert advisor.called("plan")[0]["shared_context"] is True

    asyncio.run(scenario())


def test_choosing_a_model_shows_the_largest_context() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-users", Input).value = "4"
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            assert advisor.called("limit")[0] == {
                "repo_id": "org/Model-GGUF", "quantization": "Q4_K_M", "users": 4,
                "backend": ComputeBackend.VULKAN,
            }
            assert ("Largest context for each of 4 people on this machine: 98304 tokens. "
                    "The model was trained for 131072.") in _text(screen, "#setup-limit")

            screen.query_one("#setup-backend", Select).value = ComputeBackend.CPU
            await _wait_until(pilot, lambda: len(advisor.called("limit")) == 2
                              and not screen._busy)
            assert advisor.called("limit")[1]["backend"] is ComputeBackend.CPU

    asyncio.run(scenario())


def test_setup_asks_for_what_is_missing() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            start = screen.query_one("#setup-start", Button)
            start.press()
            await pilot.pause()
            assert "Choose a downloaded model" in _text(screen, "#setup-status")
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            screen.query_one("#setup-users", Input).value = "0"
            start.press()
            await pilot.pause()
            assert "Enter a valid users." in _text(screen, "#setup-status")
            assert advisor.called("prepare") == []

    asyncio.run(scenario())


def test_setup_installs_llama_cpp_when_missing() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor(installed=False)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            install = screen.query_one("#setup-install", Button)
            assert install.display and "No llama.cpp installed" in _text(screen, "#setup-runtime")
            install.press()
            await _wait_until(pilot, lambda: bool(advisor.called("install")) and not screen._busy)
            assert advisor.called("install") == [{"backend": ComputeBackend.VULKAN}]
            assert "b11889 (vulkan)" in _text(screen, "#setup-runtime")
            assert not install.display

            checks = len(advisor.called("managed"))
            screen.query_one("#setup-backend", Select).value = ComputeBackend.CPU
            await _wait_until(pilot, lambda: len(advisor.called("managed")) > checks
                              and not screen._busy)
            assert advisor.called("managed")[-1] == {"backend": ComputeBackend.CPU}

    asyncio.run(scenario())


def test_setup_shows_a_planning_failure_and_stays_usable() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor(plan_error=ServerSizingError(
            "The largest that fits is 65536 tokens", largest_context_per_user=65536,
        ))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("plan")) and not screen._busy)
            assert "Failed: The largest that fits is 65536 tokens" in _text(
                screen, "#setup-status")
            assert advisor.called("start") == []
            assert not screen.query_one("#setup-start", Button).disabled

    asyncio.run(scenario())


def test_setup_is_reachable_from_the_welcome_menu() -> None:
    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#welcome-setup")
            await _wait_until(pilot, lambda: isinstance(app.screen, SetupScreen)
                              and not app.screen._busy)
            await pilot.press("escape")
            assert not isinstance(app.screen, SetupScreen)

    asyncio.run(scenario())


def test_an_installed_backend_is_preselected_and_recommended() -> None:
    from jaull.tui.screens.setup import backend_choices

    options, default = backend_choices(
        ComputeBackend.VULKAN, [ComputeBackend.CUDA], {ComputeBackend.CUDA: "b11889"},
    )
    assert default is ComputeBackend.CUDA
    assert options == [
        ("vulkan", ComputeBackend.VULKAN),
        ("cuda  (recommended, llama.cpp b11889 installed)", ComputeBackend.CUDA),
        ("cpu", ComputeBackend.CPU),
    ]
    _, detected = backend_choices(
        ComputeBackend.CUDA, [ComputeBackend.VULKAN],
        {ComputeBackend.CUDA: "b11889", ComputeBackend.VULKAN: "b11889"},
    )
    assert detected is ComputeBackend.CUDA
    options, nothing = backend_choices(ComputeBackend.VULKAN, [], {})
    assert nothing is ComputeBackend.VULKAN
    assert options[0] == ("vulkan  (recommended)", ComputeBackend.VULKAN)


def test_the_screen_preselects_the_installed_backend() -> None:
    class CudaInstalled(SetupAdvisor):
        def managed_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation | None:
            return INSTALLED if backend is ComputeBackend.CPU else None

    async def scenario() -> None:
        app = JaullApp(advisor=CudaInstalled())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert screen.query_one("#setup-backend", Select).value is ComputeBackend.CPU
            assert "b11889 (cpu)" in _text(screen, "#setup-runtime")
            hardware = _text(screen, "#setup-hardware")
            assert "Test CPU" in hardware and "Intel Arc Pro B60" in hardware

    asyncio.run(scenario())


def _recommendation(fmt: str) -> Any:
    from jaull.domain.execution_plans import ArtifactVariantFormat

    artifact = SimpleNamespace(format=ArtifactVariantFormat(fmt), repo_id="org/Found-GGUF",
                               quantization="Q4_K_M")
    return SimpleNamespace(plan=SimpleNamespace(artifact=artifact))


def test_find_a_model_opens_the_search_and_a_chosen_plan_comes_back() -> None:
    from textual.screen import Screen

    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            opened: list[bool] = []
            app.start_guided_workflow = lambda: opened.append(True)  # type: ignore[method-assign]
            screen.query_one("#setup-find", Button).press()
            await pilot.pause()
            assert opened == [True] and app.serve_handoff is not None

            # Stand-ins for the search screens the guided flow pushes.
            await app.push_screen(Screen())
            await app.push_screen(Screen())
            app.run_recommendation(_recommendation("safetensors"))
            await pilot.pause()
            assert not isinstance(app.screen, SetupScreen)  # Only GGUF can be served.

            app.run_recommendation(_recommendation("gguf"))
            await _wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
            await pilot.pause()
            assert screen.query_one("#setup-repo", Input).value == "org/Found-GGUF"
            assert screen.query_one("#setup-quant", Input).value == "Q4_K_M"
            assert "Model from the guided search: org/Found-GGUF (Q4_K_M)" in _text(
                screen, "#setup-status")
            assert app.serve_handoff is None

    asyncio.run(scenario())


def test_backing_out_of_the_search_cancels_the_handoff() -> None:
    from textual.screen import Screen

    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            app.serve_handoff = screen.use_model
            await app.push_screen(Screen())
            app.pop_screen()
            await _wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
            await pilot.pause()
            assert app.serve_handoff is None

    asyncio.run(scenario())


def test_context_changes_during_a_limit_check_are_checked_afterwards() -> None:
    from threading import Event

    entered, release = Event(), Event()

    class SlowLimit(SetupAdvisor):
        def server_context_limit(self, repo_id: str, **kwargs: Any) -> Any:
            if not entered.is_set():
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("Test did not release the context check")
            return super().server_context_limit(repo_id, **kwargs)

    async def scenario() -> None:
        advisor = SlowLimit()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            try:
                await _wait_until(pilot, entered.is_set)
                screen.query_one("#setup-users", Input).value = "7"
                screen._check_limit()
            finally:
                release.set()
            await _wait_until(pilot, lambda: len(advisor.called("limit")) == 2
                              and not screen._busy)
            assert advisor.called("limit")[-1]["users"] == 7
            assert "each of 7 people" in _text(screen, "#setup-limit")

    asyncio.run(scenario())


def test_presets_fill_the_context_and_the_bar_compares_it_with_what_fits() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert screen.query_one("#setup-preset-max", Button).disabled

            screen.query_one("#setup-preset-65536", Button).press()
            await pilot.pause()
            assert screen.query_one("#setup-context", Input).value == "65536"
            assert screen.query_one("#setup-preset-65536", Button).has_class("-active")

            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            assert "fits" in _text(screen, "#setup-capacity")

            screen.query_one("#setup-preset-131072", Button).press()
            await pilot.pause()
            assert "more than this machine can serve" in _text(screen, "#setup-capacity")
            assert "context too large" in _text(screen, "#setup-state-3")

            screen.query_one("#setup-preset-max", Button).press()
            await pilot.pause()
            assert screen.query_one("#setup-context", Input).value == "98304"

    asyncio.run(scenario())


def test_step_bar_follows_the_form() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert "choose one" in _text(screen, "#setup-state-4")
            assert "Only programs on this PC" in _text(screen, "#setup-exposure-note")

            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()
            assert "local network" in _text(screen, "#setup-state-3")
            assert "does not open the firewall" in _text(screen, "#setup-exposure-note")

            screen.query_one("#setup-port", Input).value = "99999"
            await pilot.pause()
            assert "Enter a valid port." in _text(screen, "#setup-state-3")

            screen.query_one("#setup-port", Input).value = "8090"
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            assert "ready to start" in _text(screen, "#setup-state-5")

    asyncio.run(scenario())
