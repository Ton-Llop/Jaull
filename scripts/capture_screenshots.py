"""Regenerate the docs/assets/ TUI screenshots.

Runs the Textual app under :func:`App.run_test` with fake services, navigates
to each documented screen with :class:`Pilot`, and saves an SVG snapshot per
step. SVGs render inline on GitHub via ``<img src=...>`` and keep the
terminal styling (colours, borders, glyphs) intact — a PNG converter would
lose all of that.

Run once and drop the generated files into ``docs/assets/``:

.. code-block:: shell

    uv run python scripts/capture_screenshots.py

Requires no network — the search client, HF client, hardware detector,
inspector and estimator are all fakes borrowed from the test fixtures.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from _workflow_fixtures import (  # noqa: E402
    GIB,
    FakeHfClient,
    FakeSearchClient,
    candidate,
    hardware,
    size_driven_estimator,
    transformers_analysis,
)
from jaull.tui.app import JaullApp  # noqa: E402
from jaull.tui.screens.hardware_analysis import HardwareAnalysisScreen  # noqa: E402
from jaull.workflow.container import ServiceContainer  # noqa: E402
from jaull.workflow.progress import HARDWARE_STEPS  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "assets"
TERMINAL_SIZE = (110, 32)


def _services() -> ServiceContainer:
    search = FakeSearchClient(
        default=[
            candidate(repo_id=repo, tags=["text-generation", "code"])
            for repo in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct")
        ]
    )
    estimator = size_driven_estimator(vram_budget=24 * GIB)

    def detect(**kwargs: object) -> object:
        on_step = kwargs.get("on_step")
        if callable(on_step):
            for key, _ in HARDWARE_STEPS:
                on_step(key)
        return hardware()

    return ServiceContainer(
        hf_client=FakeHfClient(),
        search_client=search,
        detect_hardware=detect,  # type: ignore[arg-type]
        inspect_model=lambda repo_id, client=None: transformers_analysis(  # type: ignore[arg-type]
            repo_id=repo_id
        ),
        estimate_memory=lambda **kwargs: estimator(  # type: ignore[arg-type]
            kwargs["analysis"], kwargs["inference_cfg"]
        ),
    )


async def _settle(pilot, attempts: int = 40) -> None:  # type: ignore[no-untyped-def]
    for _ in range(attempts):
        await pilot.pause()
        await asyncio.sleep(0.05)


async def _save(app: JaullApp, filename: str) -> None:
    path = app.save_screenshot(filename=filename, path=str(OUTPUT_DIR))
    print(f"  → {path}")


async def _capture_welcome(app: JaullApp, pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    await _save(app, "tui-welcome.svg")


async def _capture_hardware(app: JaullApp, pilot) -> None:  # type: ignore[no-untyped-def]
    """Show the loading card mid-scan — that's the new UX being documented."""
    app.start_guided_workflow()
    # Two ticks: the loading card is mounted and the first probe has reported
    # in but not all five, so the progress bar is visibly partial.
    await pilot.pause()
    await asyncio.sleep(0.6)
    await pilot.pause()
    await _save(app, "tui-hardware-loading.svg")

    # And once settled, the summary card in the same slot.
    await _settle(pilot)
    await _save(app, "tui-hardware-done.svg")


async def _capture_wizard(app: JaullApp, pilot) -> None:  # type: ignore[no-untyped-def]
    app.goto_requirements()
    await pilot.pause()
    await _save(app, "tui-wizard.svg")


async def _capture_advanced(app: JaullApp, pilot) -> None:  # type: ignore[no-untyped-def]
    # Reset to welcome, then open advanced tools.
    app.restart_workflow()
    await pilot.pause()
    app.push_screen("advanced")
    await pilot.pause()
    await _save(app, "tui-home.svg")


async def _capture_estimate(app: JaullApp, pilot) -> None:  # type: ignore[no-untyped-def]
    app.push_screen("estimate")
    await pilot.pause()
    await _save(app, "tui-estimate.svg")


async def _capture_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = JaullApp(services=_services())
    async with app.run_test(size=TERMINAL_SIZE) as pilot:
        print("Capturing welcome…")
        await _capture_welcome(app, pilot)

        print("Capturing hardware analysis…")
        await _capture_hardware(app, pilot)
        # After the hardware screen finishes we're still on HardwareAnalysisScreen;
        # advance the guided flow only if we're currently there.
        if isinstance(app.screen, HardwareAnalysisScreen):
            print("Capturing wizard…")
            await _capture_wizard(app, pilot)

        print("Capturing advanced tools…")
        await _capture_advanced(app, pilot)

        print("Capturing estimate…")
        await _capture_estimate(app, pilot)


def main() -> int:
    asyncio.run(_capture_all())
    print(f"\nDone. Screenshots saved under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
