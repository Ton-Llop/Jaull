"""What a user on an Intel machine sees while Jaull scans it (synthetic scans)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Digits

from jaull.bootstrap.container import ServiceContainer
from jaull.domain.hardware import AcceleratorType, ComputeBackend
from jaull.runtime.backend_selection import select_runtime_backend
from jaull.tui.app import JaullApp
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen
from jaull.workflow.progress import HARDWARE_STEPS
from tests._intel_fixtures import (
    B60_NAME,
    GIB,
    LINUX_OS,
    LLVMPIPE_SUMMARY,
    WINDOWS_OS,
    WSL2_OS,
    scan,
)
from tests._workflow_fixtures import FakeHfClient, FakeSearchClient
from tests.test_tui_guided_workflow import _run, _wait_for
from tests.test_tui_recommendation_execution import _visible_text


def _services(profile) -> ServiceContainer:  # type: ignore[no-untyped-def]
    def detect(**kwargs: object) -> object:
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return profile

    return ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=FakeSearchClient(default=[]),
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=lambda repo_id, client=None: None,  # type: ignore[arg-type,return-value]
        estimate_memory=lambda **kwargs: None,  # type: ignore[arg-type,return-value]
    )


def _hardware_screen_text(profile) -> tuple[str, list[str]]:  # type: ignore[no-untyped-def]
    texts: list[str] = []
    digits: list[str] = []

    async def scenario() -> None:
        app = JaullApp(services=_services(profile))
        async with app.run_test(size=(120, 50)) as pilot:
            app.start_guided_workflow()
            await _wait_for(
                pilot,
                lambda: isinstance(pilot.app.screen, HardwareAnalysisScreen)
                and bool(pilot.app.screen.query("#hw-continue")),
            )
            screen = pilot.app.screen
            assert screen.query_one("#hw-continue", Button)
            texts.append(_visible_text(screen))
            digits.extend(str(item.value) for item in screen.query(Digits))

    _run(scenario())
    return texts[0], digits


# ---------------------------------------------------------------------------
# Detection: the profile the screen is built from
# ---------------------------------------------------------------------------
def test_windows_scan_finds_one_discrete_intel_device_with_a_vulkan_backend() -> None:
    profile = scan(WINDOWS_OS)

    [accelerator] = profile.accelerators
    assert accelerator.name == B60_NAME
    assert accelerator.type is AcceleratorType.DEDICATED
    assert accelerator.dedicated_memory_bytes == 24 * GIB
    assert accelerator.available_memory_bytes == 23 * GIB
    assert select_runtime_backend(profile).selected_backend is ComputeBackend.VULKAN
    assert not any("WSL2" in warning for warning in profile.warnings)


@pytest.mark.parametrize("os_label", [WSL2_OS])
def test_wsl2_with_only_a_software_renderer_says_no_vulkan_route_was_found(
    os_label: str,
) -> None:
    profile = scan(os_label, summary=LLVMPIPE_SUMMARY, full="", windows=())

    note = next((item for item in profile.warnings if "WSL2" in item), "")
    assert "no usable Vulkan accelerator is exposed here" in note
    # It must not claim the host cannot use the GPU, nor imply a Windows runtime is used.
    assert "does not show that the Windows host cannot accelerate the GPU" in note
    assert select_runtime_backend(profile).selected_backend is ComputeBackend.CPU


def test_wsl2_with_a_real_vulkan_device_gets_no_caution() -> None:
    profile = scan(WSL2_OS, windows=())

    assert not any("WSL2" in warning for warning in profile.warnings)


@pytest.mark.parametrize("os_label", [WINDOWS_OS, LINUX_OS])
def test_no_wsl2_caution_outside_wsl2(os_label: str) -> None:
    profile = scan(os_label, summary=LLVMPIPE_SUMMARY, full="", windows=())

    assert not any("WSL2" in warning for warning in profile.warnings)


def test_vulkan_budget_semantics_travel_with_the_profile() -> None:
    profile = scan(WINDOWS_OS)

    assert any("not a free-VRAM measurement" in warning for warning in profile.warnings)


# ---------------------------------------------------------------------------
# The hardware screen
# ---------------------------------------------------------------------------
def test_hardware_screen_shows_the_intel_card_its_memory_and_its_backend() -> None:
    text, digits = _hardware_screen_text(scan(WINDOWS_OS))

    assert B60_NAME in text
    assert "24.0 GiB" in text  # capacity, as for an NVIDIA card's total VRAM
    assert "Vulkan" in text  # the backend Jaull would use
    assert "Windows 11" in text
    assert "24.0" in digits  # the headline memory figure exists for this machine too


def test_hardware_screen_says_vulkan_is_not_confirmed_when_only_windows_listed_the_card() -> None:
    profile = scan(WINDOWS_OS, vulkaninfo_present=False)

    text, _ = _hardware_screen_text(profile)

    assert B60_NAME in text
    assert "not confirmed" in text.lower()
    assert "vulkaninfo" in text  # the missing probe is named, not hidden


def test_hardware_screen_shows_the_wsl2_caution_in_its_warnings() -> None:
    profile = scan(WSL2_OS, summary=LLVMPIPE_SUMMARY, full="", windows=())

    text, _ = _hardware_screen_text(profile)

    assert "Running inside WSL2" in text
    assert "CPU inference only" in text
    assert "Windows host cannot" in text and "does not show" in text


def test_hardware_screen_shows_the_budget_warning_for_an_intel_scan() -> None:
    text, _ = _hardware_screen_text(scan(WINDOWS_OS))

    assert "not a free-VRAM measurement" in text
