"""TUI Setup: the five steps compose the facade like ``jaull serve start``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Input, RadioButton, Select, Static

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
            screen.query_one("#setup-users", Input).value = "4"
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            assert advisor.called("prepare")[0]["repo_id"] == "org/Model-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "Q4_K_M"
            plan = advisor.called("plan")[0]
            assert plan["users"] == 4 and plan["backend"] is ComputeBackend.VULKAN
            assert advisor.called("start")[0] == {"lan": False, "port": 8090}
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
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            assert advisor.called("prepare")[0]["repo_id"] == "org/Other-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "IQ4_XS"
            assert advisor.called("start")[0] == {"lan": True, "port": 8095}

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

            screen.query_one("#setup-backend", Select).value = ComputeBackend.CPU
            await _wait_until(pilot, lambda: advisor.called("managed")[-1]["backend"]
                              is ComputeBackend.CPU and not screen._busy)

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
