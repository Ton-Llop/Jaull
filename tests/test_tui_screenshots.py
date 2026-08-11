"""Smoke test for the documentation screenshots.

``scripts/capture_screenshots.py`` drives every screen in the guided flow, so
it rots quietly whenever a screen changes shape. Running it into a temporary
directory keeps that honest: no network, no llama.cpp, no writes to
``docs/assets``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from capture_screenshots import SCREENSHOTS, capture_all  # noqa: E402


def test_capture_script_produces_every_documented_screenshot(tmp_path: Path) -> None:
    written = asyncio.run(capture_all(tmp_path, (110, 32)))

    assert [path.name for path in written] == list(SCREENSHOTS)
    for path in written:
        assert path.exists(), f"{path.name} was never written"
        assert path.stat().st_size > 0, f"{path.name} is empty"
        assert path.read_text(encoding="utf-8").startswith("<svg"), (
            f"{path.name} is not an SVG"
        )


def _rendered_text(path: Path) -> str:
    """The words in an SVG capture. Rich pads runs with non-breaking spaces."""
    return path.read_text(encoding="utf-8").replace("&#160;", " ")


def test_the_two_hardware_states_are_not_the_same_picture(tmp_path: Path) -> None:
    """The scanning capture used to be a byte-identical copy of the finished one."""
    asyncio.run(capture_all(tmp_path, (110, 32)))

    loading = _rendered_text(tmp_path / "tui-hardware-loading.svg")
    done = _rendered_text(tmp_path / "tui-hardware-done.svg")
    assert loading != done
    # Mid-scan: the checklist, with at least one probe still pending.
    assert "Analyzing this machine" in loading
    assert "Building hardware profile" in loading
    # Finished: the profile replaced it, and the checklist is gone.
    assert "This machine" in done
    assert "Building hardware profile" not in done


def test_the_readme_only_points_at_screenshots_the_script_writes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = {
        line.split("docs/assets/")[1].split(")")[0]
        for line in readme.splitlines()
        if "docs/assets/" in line and line.strip().startswith("![")
    }
    assert referenced <= set(SCREENSHOTS), sorted(referenced - set(SCREENSHOTS))
