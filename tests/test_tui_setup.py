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

    found: Any = None

    def found_llama_server(self, backend: ComputeBackend, selection: Any = None) -> Any:
        self.calls.append(("found", {"backend": backend}))
        return self.found

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
            await _wait_until(pilot, lambda: not screen.query_one("#setup-start", Button).disabled)
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
            await _wait_until(pilot, lambda: not screen.query_one("#setup-start", Button).disabled)
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
            assert start.disabled
            start.press()
            await pilot.pause()
            assert "Choose a downloaded model" in _text(screen, "#setup-status")
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            screen.query_one("#setup-users", Input).value = "0"
            await pilot.pause()
            assert start.disabled
            assert screen._choice() == "Enter a valid users."
            assert advisor.called("prepare") == []

    asyncio.run(scenario())


def test_setup_installs_llama_cpp_when_missing() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor(installed=False)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            install = screen.query_one("#setup-install", Button)
            assert install.display and "No llama.cpp found" in _text(screen, "#setup-runtime")
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



def test_max_fills_the_context_with_the_largest_that_fits() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            maximum = screen.query_one("#setup-preset-max", Button)
            assert maximum.disabled  # nothing to compare with before a model is chosen

            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: not maximum.disabled and not screen._busy)
            maximum.press()
            await _wait_until(pilot, lambda: screen.query_one(
                "#setup-context", Input).value == "98304")

    asyncio.run(scenario())


def test_a_downloaded_model_wins_over_hidden_typed_fields() -> None:
    # Audit P1: typing a repository, then choosing a downloaded model, started the typed one.
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-repo", Input).value = "org/Typed-GGUF"
            screen.query_one("#setup-quant", Input).value = "Q8_0"
            screen.query_one("#setup-model", Select).value = 0
            await pilot.pause(0.2)
            await _wait_until(pilot, lambda: not screen._busy)
            assert not screen.query_one("#setup-download").display
            assert screen.query_one("#setup-repo", Input).value == ""
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)
            assert advisor.called("prepare")[0]["repo_id"] == "org/Model-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "Q4_K_M"

    asyncio.run(scenario())


def test_a_limit_for_other_inputs_is_never_used_and_is_rechecked_without_enter() -> None:
    # Audit P2: going from 1 to 8 people kept Max enabled with the 1-person limit.
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            maximum = screen.query_one("#setup-preset-max", Button)
            await pilot.pause()
            assert not maximum.disabled

            screen.query_one("#setup-users", Input).value = "8"
            await pilot.pause()
            assert maximum.disabled  # the 1-person limit no longer applies
            assert screen._current_limit() is None
            await _wait_until(pilot, lambda: len(advisor.called("limit")) == 2
                              and not screen._busy)
            assert advisor.called("limit")[1]["users"] == 8
            await pilot.pause()
            assert not maximum.disabled
            assert screen._current_limit() is not None
            assert screen._current_limit().users == 8

    asyncio.run(scenario())


def test_a_model_that_does_not_fit_is_not_ready_to_start() -> None:
    # Audit P2: with no viable context the summary still said "ready to start".
    class NoFit(SetupAdvisor):
        def server_context_limit(self, repo_id: str, **kwargs: Any) -> Any:
            from jaull.application.serving import ContextLimit

            self.calls.append(("limit", {"repo_id": repo_id, **kwargs}))
            return ContextLimit(users=kwargs["users"], largest_per_user=None, native=None)

    async def scenario() -> None:
        advisor = NoFit()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")) and not screen._busy)
            await pilot.pause()
            states = screen._step_states()
            assert states[3] == ("warn", "does not fit here")
            assert states[4][1] != "ready to start"
            assert "does not fit" in _text(screen, "#setup-limit")
            assert screen.query_one("#setup-start", Button).disabled

    asyncio.run(scenario())


def test_a_found_build_without_the_backend_is_not_presented_as_usable() -> None:
    # Audit: "found on this PC" was shown for any backend, even one the build cannot run.
    from jaull.application.serving.models import FoundLlamaServer

    async def scenario() -> None:
        advisor = SetupAdvisor(installed=False)
        server_path = Path("/opt/llama/llama-server")
        advisor.found = FoundLlamaServer(path=server_path, version="b9000",
                                         supports_backend=False)
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert advisor.called("found")[0] == {"backend": ComputeBackend.VULKAN}
            runtime = _text(screen, "#setup-runtime")
            assert "does not run vulkan" in runtime and str(server_path) in runtime
            assert screen.query_one("#setup-install", Button).display
            assert screen._step_states()[1] == ("warn", "no vulkan build")

            advisor.found = FoundLlamaServer(path=server_path,
                                             version="b9000", supports_backend=True)
            screen.query_one("#setup-backend", Select).value = ComputeBackend.CPU
            await _wait_until(pilot, lambda: len(advisor.called("found")) == 2
                              and not screen._busy)
            assert "Using the llama.cpp found on this PC (b9000)" in _text(
                screen, "#setup-runtime")
            assert screen._step_states()[1] == ("done", "found on this PC")

    asyncio.run(scenario())


def test_stale_context_result_does_not_replace_the_current_note() -> None:
    from jaull.application.serving import ContextLimit

    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: screen._current_limit() is not None
                              and not screen._busy)
            old_key = screen._limit_key()
            assert old_key is not None
            screen.query_one("#setup-users", Input).value = "4"
            note = screen.query_one("#setup-limit", Static)
            note.update("Checking the new settings")
            screen._show_limit(ContextLimit(users=1, largest_per_user=123, native=None), old_key)
            assert _text(screen, "#setup-limit") == "Checking the new settings"
            screen._render_progress()
            assert screen.query_one("#setup-start", Button).disabled
            assert screen.query_one("#setup-preset-max", Button).disabled

    asyncio.run(scenario())


def test_start_requires_valid_port_context_and_runtime() -> None:
    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            start = screen.query_one("#setup-start", Button)
            await _wait_until(pilot, lambda: not start.disabled)
            for field, invalid, valid in (("port", "65536", "8090"),
                                           ("context", "98305", "4096")):
                screen.query_one(f"#setup-{field}", Input).value = invalid
                screen._render_progress()
                assert start.disabled
                assert isinstance(screen._choice(), str)
                screen.query_one(f"#setup-{field}", Input).value = valid
                screen._render_progress()
                assert not start.disabled
            screen._installed.clear()
            screen._found.clear()
            screen._render_progress()
            assert start.disabled
            assert screen._choice() == "Install llama.cpp for the selected backend."

    asyncio.run(scenario())


def test_a_pending_limit_check_runs_after_a_failed_one_and_the_error_stays() -> None:
    # Audit: if the running check failed, the check pending for the new inputs never ran.
    import time

    class FailsFirst(SetupAdvisor):
        def server_context_limit(self, repo_id: str, **kwargs: Any) -> Any:
            first = not self.called("limit")
            if first:
                self.calls.append(("limit", {"repo_id": repo_id, **kwargs}))
                time.sleep(1.5)  # long enough for the typing delay to end meanwhile
                raise RuntimeError("Hugging Face did not answer")
            return super().server_context_limit(repo_id, **kwargs)

    async def scenario() -> None:
        advisor = FailsFirst()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: bool(advisor.called("limit")))
            screen.query_one("#setup-users", Input).value = "4"  # pending while busy
            await _wait_until(pilot, lambda: len(advisor.called("limit")) == 2
                              and not screen._busy)
            assert advisor.called("limit")[1]["users"] == 4
            await pilot.pause()
            assert screen._current_limit() is not None
            assert "Failed: Hugging Face did not answer" in _text(screen, "#setup-status")

    asyncio.run(scenario())
