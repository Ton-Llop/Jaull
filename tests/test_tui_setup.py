"""TUI Setup: the five steps compose the facade like ``jaull serve start``."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Checkbox, Input, RadioButton, Select, Static

from jaull.application.serving import ServerSizingError
from jaull.application.serving import policies as serving_policies
from jaull.application.serving.models import ServerPlan, ServerShare, ServerStart
from jaull.artifacts.storage import LocalModelFile
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import LlamaCppInstallation, RuntimeSource
from jaull.execution.opencode_install import OpencodeInstall
from jaull.execution.server_process import ServerStatus
from jaull.tui.app import JaullApp
from jaull.tui.screens.setup import SetupScreen
from tests.test_cli_serve import ARTIFACT, CERTIFICATE, STATE, _plan


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

    #: When set, a download (llama.cpp or the model) waits here, like a slow real one.
    gate: threading.Event | None = None

    def _download(self) -> None:
        if self.gate is not None:
            self.gate.wait(10)

    def install_llama_cpp(self, backend: ComputeBackend) -> LlamaCppInstallation:
        self.calls.append(("install", {"backend": backend}))
        self._download()
        self.installed = True
        return INSTALLED

    def server_context_limit(self, repo_id: str, **kwargs: Any) -> Any:
        from jaull.application.serving import ContextLimit

        self.calls.append(("limit", {"repo_id": repo_id, **kwargs}))
        return ContextLimit(users=kwargs["users"], largest_per_user=98304, native=131072)

    def prepare_serve_artifact(self, repo_id: str, **kwargs: Any) -> ModelArtifact:
        self.calls.append(("prepare", {"repo_id": repo_id, **kwargs}))
        kwargs["on_progress"]("Verifying model-Q4_K_M.gguf")
        self._download()
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

    def create_server_certificate(self, *, renew: bool = False) -> Any:
        self.calls.append(("certificate", {"renew": renew}))
        return CERTIFICATE

    #: The key in the server's key file; None when the file is missing.
    key: str | None = "KEY123"

    def server_api_key(self) -> str | None:
        return self.key

    def server_opencode_config(self) -> str | None:
        return '{"provider": {"jaull": {}}}'

    #: Where the running server is reached; a local one unless a test says otherwise.
    share = ServerShare(base_url="http://127.0.0.1:8090", chat_url=None, shared=False,
                        firewall_rule=None)

    def server_share(self, state: Any) -> ServerShare:
        return self.share

    #: Set to CERTIFICATE to act as a machine where `jaull serve cert` was run before.
    detected: Any = None
    #: Raised by the detection, like an unreadable certificate file.
    detected_error: Exception | None = None

    def detected_server_certificate(self) -> Any:
        self.calls.append(("detected", {}))
        if self.detected_error is not None:
            raise self.detected_error
        return self.detected

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
            # Changing People invalidates the context limit; Start is only accepted once
            # the debounced recheck for 4 people has come back.
            await _wait_until(pilot, lambda: any(
                call["users"] == 4 for call in advisor.called("limit"))
                and not screen._busy
                and not screen.query_one("#setup-start", Button).disabled)
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            assert advisor.called("prepare")[0]["repo_id"] == "org/Model-GGUF"
            assert advisor.called("prepare")[0]["quantization"] == "Q4_K_M"
            plan = advisor.called("plan")[0]
            assert plan["users"] == 4 and plan["backend"] is ComputeBackend.VULKAN
            assert advisor.called("start")[0] == {
                "lan": False, "port": 8090, "web_ui": False,
                "tls_cert": None, "tls_key": None, "prompt_cache": True,
            }
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
            # Shared on the network means HTTPS, and Jaull made its own certificate for it.
            assert advisor.called("start")[0] == {
                "lan": True, "port": 8095, "web_ui": True,
                "tls_cert": CERTIFICATE.cert_path, "tls_key": CERTIFICATE.key_path,
                "prompt_cache": True,
            }
            assert len(advisor.called("certificate")) == 1
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
            await _wait_until(
                pilot,
                lambda: app.screen is screen and not screen._busy
                and screen._current_limit() is not None,
            )
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


def test_sharing_without_https_warns_and_https_files_reach_the_server(tmp_path: Path) -> None:
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text("cert")
    key.write_text("key")

    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            note = screen.query_one("#setup-exposure-note", Static)
            assert not screen.query_one("#setup-tls").display
            assert not note.has_class("-warn")

            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()
            # Sharing on the network is encrypted unless the person says otherwise.
            assert screen.query_one("#setup-https", Checkbox).value
            assert screen.query_one("#setup-tls").display and not note.has_class("-warn")
            assert "Served over HTTPS" in _text(screen, "#setup-exposure-note")

            screen.query_one("#setup-https", Checkbox).value = False
            await pilot.pause()
            assert note.has_class("-warn") and not screen.query_one("#setup-tls").display
            assert "without HTTPS" in _text(screen, "#setup-exposure-note")
            assert "not separate users" in _text(screen, "#setup-exposure-note")

            screen.query_one("#setup-https", Checkbox).value = True
            await pilot.pause()
            screen.query_one("#setup-tls-cert", Input).value = str(cert)
            await pilot.pause()
            assert screen._tls_files() == "HTTPS needs both a certificate and a private key file."
            screen.query_one("#setup-tls-key", Input).value = str(key)
            await pilot.pause()
            assert not note.has_class("-warn")
            assert "Served over HTTPS" in _text(screen, "#setup-exposure-note")
            assert "not separate users" in _text(screen, "#setup-exposure-note")

            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: not screen.query_one("#setup-start", Button).disabled)
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)
            started = advisor.called("start")[0]
            assert (started["tls_cert"], started["tls_key"]) == (cert, key)

    asyncio.run(scenario())


def test_a_missing_https_key_is_named_before_starting(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("cert")

    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-lan", RadioButton).value = True
            screen.query_one("#setup-tls-cert", Input).value = str(cert)
            await pilot.pause()
            assert screen._tls_files() == "HTTPS needs both a certificate and a private key file."
            screen.query_one("#setup-tls-key", Input).value = str(tmp_path / "nope.pem")
            await pilot.pause()
            assert str(screen._tls_files()).startswith("The HTTPS private key file was not found")

    asyncio.run(scenario())


def test_turning_the_prompt_cache_off_explains_the_cost_and_reaches_the_server() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert "Faster replies" in _text(screen, "#setup-cache-note")
            screen.query_one("#setup-cache", Checkbox).value = False
            await pilot.pause()
            assert "reprocesses the whole conversation" in _text(screen, "#setup-cache-note")

            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: not screen.query_one("#setup-start", Button).disabled)
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)
            assert advisor.called("start")[0]["prompt_cache"] is False

    asyncio.run(scenario())


def test_the_certificate_button_fills_the_https_fields() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            button = screen.query_one("#setup-cert", Button)
            assert not screen.query_one("#setup-tls").display  # only when sharing on the network

            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()
            assert screen.query_one("#setup-tls").display and button.region.right <= 120
            button.press()
            await _wait_until(pilot, lambda: bool(advisor.called("certificate"))
                              and not screen._busy)

            assert advisor.called("certificate") == [{"renew": False}]
            assert screen.query_one("#setup-tls-cert", Input).value == str(CERTIFICATE.cert_path)
            assert screen.query_one("#setup-tls-key", Input).value == str(CERTIFICATE.key_path)
            status = _text(screen, "#setup-status")
            assert "self-signed" in status and "warning" in status

    asyncio.run(scenario())


def test_sharing_picks_up_the_certificate_jaull_already_made() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.detected = CERTIFICATE
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            assert screen.query_one("#setup-tls-cert", Input).value == ""

            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()

            assert screen.query_one("#setup-tls-cert", Input).value == str(CERTIFICATE.cert_path)
            assert screen.query_one("#setup-tls-key", Input).value == str(CERTIFICATE.key_path)

    asyncio.run(scenario())


def test_a_typed_certificate_is_not_replaced_by_the_detected_one(tmp_path: Path) -> None:
    mine = tmp_path / "mine.pem"

    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.detected = CERTIFICATE
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-tls-cert", Input).value = str(mine)

            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()

            assert screen.query_one("#setup-tls-cert", Input).value == str(mine)
            assert screen.query_one("#setup-tls-key", Input).value == ""

    asyncio.run(scenario())


def test_nothing_is_filled_in_when_there_is_no_certificate() -> None:
    async def scenario() -> None:
        app = JaullApp(advisor=SetupAdvisor())  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()
            assert screen.query_one("#setup-tls-cert", Input).value == ""

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
def test_a_live_process_is_not_shown_as_ready_until_healthy(size: tuple[int, int]) -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=size) as pilot:
            screen = await _open(pilot, app)
            advisor.running = True
            screen._show_server(ServerStatus(state=STATE, running=True, healthy=False))
            await pilot.pause()
            assert "loading or not responding" in _text(screen, "#setup-status")
            assert screen._step_states()[-1] == ("warn", "not ready")
            assert not screen.query_one("#setup-stop", Button).disabled
            assert not screen.query_one("#setup-monitor", Button).disabled

            screen._show_server(advisor.server_status())
            await pilot.pause()
            assert "ready" in _text(screen, "#setup-status")
            assert screen._step_states()[-1] == ("done", "ready")

            advisor.running = False
            screen._show_server(advisor.server_status())
            await pilot.pause()
            assert screen.query_one("#setup-stop", Button).disabled
            assert screen.query_one("#setup-monitor", Button).disabled
            assert "A Jaull server is ready" not in _text(screen, "#setup-status")

    asyncio.run(scenario())


def test_the_copy_button_sits_next_to_the_key_and_copies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No system tool: the test must never touch the real clipboard of whoever runs it.
    monkeypatch.setattr("jaull.tui.clipboard._system_copy", lambda text: False)

    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        copied: list[str] = []
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _open(pilot, app)
            row = screen.query_one("#setup-key-row")
            assert not row.display  # no server, no key

            advisor.running = True
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            button = screen.query_one("#setup-copy-key", Button)
            assert row.display and button.region.right <= 80
            button.press()
            await pilot.pause()

            assert copied == ["KEY123"]
            assert "KEY123" not in _text(screen, "#setup-status")

            screen.query_one("#setup-copy-opencode", Button).press()
            await pilot.pause()
            assert copied[1].startswith('{"provider"')

    asyncio.run(scenario())



_LOCKED_INPUTS = ("#setup-users", "#setup-context", "#setup-port", "#setup-repo",
                  "#setup-quant", "#setup-tls-cert", "#setup-tls-key")
_LOCKED_BUTTONS = ("#setup-preset-8192", "#setup-other", "#setup-cert", "#setup-find",
                   "#setup-install", "#setup-start", "#setup-stop")


def _assert_form_locked(screen: SetupScreen, locked: bool) -> None:
    for selector in _LOCKED_INPUTS:
        assert screen.query_one(selector, Input).disabled is locked, selector
    for selector in ("#setup-webui", "#setup-shared", "#setup-cache"):
        assert screen.query_one(selector, Checkbox).disabled is locked, selector
    for selector in ("#setup-model", "#setup-backend"):
        assert screen.query_one(selector, Select).disabled is locked, selector
    assert screen.query_one("#setup-exposure").disabled is locked
    if locked:
        for selector in _LOCKED_BUTTONS:
            assert screen.query_one(selector, Button).disabled, selector
    # Leaving is always possible: the download carries on without the screen.
    assert not screen.query_one("#setup-back", Button).disabled


def test_the_form_is_locked_while_a_model_downloads() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.gate = threading.Event()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: not screen._busy
                              and not screen.query_one("#setup-start", Button).disabled)
            _assert_form_locked(screen, False)

            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("prepare")))
            await pilot.pause()
            _assert_form_locked(screen, True)
            assert screen.query_one("#setup-preset-max", Button).disabled

            advisor.gate.set()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)
            _assert_form_locked(screen, False)

    asyncio.run(scenario())


def test_the_form_is_locked_while_llama_cpp_downloads() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor(installed=False)
        advisor.gate = threading.Event()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            await _wait_until(pilot, lambda: not screen._busy
                              and not screen.query_one("#setup-install", Button).disabled)

            screen.query_one("#setup-install", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("install")))
            await pilot.pause()
            _assert_form_locked(screen, True)

            advisor.gate.set()
            await _wait_until(pilot, lambda: advisor.installed and not screen._busy)
            _assert_form_locked(screen, False)

    asyncio.run(scenario())


def test_typing_never_locks_the_form() -> None:
    # The context limit is re-checked in the background while the person types; that quiet
    # work must not disable the fields they are typing in.
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-model", Select).value = 0
            screen.query_one("#setup-users", Input).value = "3"
            for _ in range(20):
                await pilot.pause(0.02)
                assert not screen.query_one("#setup-users", Input).disabled

    asyncio.run(scenario())


def test_the_ready_card_has_a_row_and_a_copy_for_each_thing_to_hand_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaull.tui.clipboard._system_copy", lambda text: False)

    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.share = ServerShare(
            base_url="https://192.168.1.107:8090", chat_url="https://192.168.1.107:8090/",
            shared=True, firewall_rule="sudo ufw allow from 192.168.1.0/24 to any port 8090",
        )
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        copied: list[str] = []
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        async with app.run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, app)
            for row in ("#setup-address-row", "#setup-firewall-row", "#setup-qr-box"):
                assert not screen.query_one(row).display  # nothing runs yet

            advisor.running = True
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            for row in ("#setup-address-row", "#setup-firewall-row", "#setup-qr-box"):
                assert screen.query_one(row).display, row
            assert "https://192.168.1.107:8090/" in _text(screen, "#setup-address")
            assert _text(screen, "#setup-qr").count("\n") > 10  # a real QR, many lines
            transparent = _text(screen, "#setup-qr").split("\n")[0]
            screen.query_one("#setup-qr-style", Button).press()
            await pilot.pause()
            assert len(_text(screen, "#setup-qr").split("\n")[0]) > len(transparent)  # quiet zone
            screen.query_one("#setup-qr-style", Button).press()
            await pilot.pause()
            assert _text(screen, "#setup-qr").split("\n")[0] == transparent

            screen.query_one("#setup-copy-address", Button).press()
            await pilot.pause()
            screen.query_one("#setup-copy-firewall", Button).press()
            await pilot.pause()
            assert copied == ["https://192.168.1.107:8090/", advisor.share.firewall_rule]

            # With HTTPS, opencode needs the certificate: its command is a row of its own,
            # above the config, and the config says when it is already in their opencode.
            assert not screen.query_one("#setup-run-row").display  # this server has no HTTPS
            command = "NODE_EXTRA_CA_CERTS=/data/tls/server-cert.pem opencode"
            secure = ServerStatus(
                state=STATE.model_copy(update={"tls": True, "argv": [
                    "llama-server", "--ssl-cert-file", "/data/tls/server-cert.pem"]}),
                running=True, healthy=True,
            )
            screen._show_server(secure, describe=False)
            await pilot.pause()
            assert screen.query_one("#setup-run-row").display
            assert command in _text(screen, "#setup-run")
            assert screen.query_one("#setup-run-row").region.y < (
                screen.query_one("#setup-opencode-row").region.y)
            screen.query_one("#setup-copy-run", Button).press()
            await pilot.pause()
            assert copied[-1] == command
            assert "ready to paste" in _text(screen, "#setup-opencode-label")
            screen._opencode_added(OpencodeInstall(Path("/x/opencode.json"), "added"))
            await pilot.pause()
            assert "already added" in _text(screen, "#setup-opencode-label")
            screen._opencode_added(OpencodeInstall(Path("/x/opencode.json"), "unreadable"))
            await pilot.pause()
            assert "ready to paste" in _text(screen, "#setup-opencode-label")
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            assert not screen.query_one("#setup-run-row").display  # back to a server without it

            # The key as a QR only while asked for, and never as text.
            hide_timers: list[Any] = []
            real_set_timer = screen.set_timer

            def capture(delay: float, callback: Any = None, **kwargs: Any) -> Any:
                # The auto-hide is checked through its timer, not by waiting for the clock.
                if delay == serving_policies.SERVE_KEY_QR_VISIBLE_SECONDS:
                    hide_timers.append(callback)
                    return SimpleNamespace(stop=lambda: None)
                return real_set_timer(delay, callback, **kwargs)

            screen.set_timer = capture  # type: ignore[method-assign]
            toggle = screen.query_one("#setup-key-qr-toggle", Button)
            assert not screen.query_one("#setup-key-qr").display
            toggle.press()
            await pilot.pause()
            assert screen.query_one("#setup-key-qr").display
            assert "KEY123" not in _text(screen, "#setup-key-qr")
            toggle.press()
            await pilot.pause()
            assert not screen.query_one("#setup-key-qr").display

            # Shown again, it hides itself when its timer fires.
            toggle.press()
            await pilot.pause()
            assert screen.query_one("#setup-key-qr").display and hide_timers
            hide_timers[-1]()
            await pilot.pause()
            assert not screen.query_one("#setup-key-qr").display

            # Leaving Setup (the monitor opens on top) must not leave the key's QR behind.
            toggle.press()
            await pilot.pause()
            assert screen.query_one("#setup-key-qr").display
            screen.on_screen_suspend()
            await pilot.pause()
            assert not screen.query_one("#setup-key-qr").display

            # A local server that serves the chat page: the hint says to share it.
            advisor.share = ServerShare(base_url="http://127.0.0.1:8090",
                                        chat_url="http://127.0.0.1:8090/", shared=False,
                                        firewall_rule=None)
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            assert "Only this PC" in _text(screen, "#setup-share-hint")

            # A server only this PC reaches: no firewall command, no QR, and a hint why.
            advisor.share = ServerShare(base_url="http://127.0.0.1:8090", chat_url=None,
                                        shared=False, firewall_rule=None)
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            assert not screen.query_one("#setup-firewall-row").display
            assert not screen.query_one("#setup-qr-box").display
            assert screen.query_one("#setup-share-hint").display

            # Shared from inside WSL: the address is unreachable, so say so, no QR for it.
            advisor.share = ServerShare(base_url="https://172.17.6.33:8090",
                                        chat_url="https://172.17.6.33:8090/", shared=True,
                                        firewall_rule=None, wsl=True)
            screen._show_server(advisor.server_status(), describe=False)
            await pilot.pause()
            assert not screen.query_one("#setup-qr-box").display
            assert "inside WSL" in _text(screen, "#setup-share-hint")

    asyncio.run(scenario())


def test_an_unreadable_certificate_is_named_instead_of_filled_in() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.detected_error = RuntimeError("The certificate cannot be read. Use --renew.")
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-lan", RadioButton).value = True
            await pilot.pause()

            assert screen.query_one("#setup-tls-cert", Input).value == ""
            assert "--renew" in _text(screen, "#setup-status")

    asyncio.run(scenario())


def test_a_certificate_that_misses_this_pcs_address_says_so() -> None:
    from dataclasses import replace

    async def scenario() -> None:
        advisor = SetupAdvisor()
        advisor.create_server_certificate = (  # type: ignore[method-assign]
            lambda renew=False: replace(CERTIFICATE, missing=("192.168.1.50",)))
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            screen.query_one("#setup-lan", RadioButton).value = True
            screen.query_one("#setup-cert", Button).press()
            await _wait_until(pilot, lambda: not screen._busy
                              and bool(screen.query_one("#setup-tls-cert", Input).value))
            assert "192.168.1.50" in _text(screen, "#setup-status")

    asyncio.run(scenario())


def test_this_pc_alone_is_not_encrypted_unless_asked_and_then_gets_a_certificate() -> None:
    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(pilot, app)
            https = screen.query_one("#setup-https", Checkbox)
            assert not https.value  # nothing leaves the PC, so no certificate to trust

            https.value = True
            screen.query_one("#setup-model", Select).value = 0
            await _wait_until(pilot, lambda: not screen.query_one("#setup-start", Button).disabled)
            screen.query_one("#setup-start", Button).press()
            await _wait_until(pilot, lambda: bool(advisor.called("start")) and not screen._busy)

            started = advisor.called("start")[0]
            assert started["lan"] is False
            assert (started["tls_cert"], started["tls_key"]) == (
                CERTIFICATE.cert_path, CERTIFICATE.key_path)

    asyncio.run(scenario())


def test_on_windows_the_card_also_offers_the_command_for_opencode_in_wsl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaull.presentation.serving._running_on_windows", lambda: True)
    argv = ["llama-server", "--ssl-cert-file", r"C:\Users\p\AppData\Local\jaull\serve\tls\c.pem"]
    shared = ServerStatus(
        state=STATE.model_copy(update={"tls": True, "host": "0.0.0.0", "argv": argv}),
        running=True, healthy=True,
    )

    async def scenario() -> None:
        advisor = SetupAdvisor()
        app = JaullApp(advisor=advisor)  # type: ignore[arg-type]
        copied: list[str] = []
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        monkeypatch.setattr("jaull.tui.clipboard._system_copy", lambda text: False)
        async with app.run_test(size=(120, 50)) as pilot:
            screen = await _open(pilot, app)
            assert not screen.query_one("#setup-run-wsl-row").display
            advisor.running = True
            screen._show_server(shared, describe=False)
            await pilot.pause()
            assert screen.query_one("#setup-run-wsl-row").display
            assert "/mnt/c/Users/p/AppData/Local/jaull/serve/tls/c.pem" in _text(
                screen, "#setup-run-wsl")
            screen.query_one("#setup-copy-run-wsl", Button).press()
            await pilot.pause()
            assert copied[-1].startswith("NODE_EXTRA_CA_CERTS=/mnt/c/Users/p/")

    asyncio.run(scenario())
