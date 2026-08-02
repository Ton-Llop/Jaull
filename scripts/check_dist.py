"""Verify that `uv build` produced complete, clean artifacts.

Run after `uv build`, locally or in CI:

    uv run python scripts/check_dist.py

Checks that both a wheel and a source distribution exist, that the wheel ships
the non-Python resources the TUI needs (``tui/styles.tcss``) plus the license,
and that no local clutter — caches, virtualenvs, assistant config — leaked into
either artifact. Uses only the standard library so it behaves identically on
Linux and Windows runners.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

DIST = Path("dist")

REQUIRED_IN_WHEEL = (
    "local_ai_check/tui/styles.tcss",
    "local_ai_check/cli/app.py",
    "local_ai_check/__init__.py",
)

# Substrings that must never appear in a published artifact.
FORBIDDEN = (
    ".claude/",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".egg-info/",
    "/home/",
    "C:\\Users",
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}")


def main() -> int:
    errors = 0

    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))

    if not wheels:
        _fail("no wheel found in dist/")
        return 1
    if not sdists:
        _fail("no source distribution found in dist/")
        return 1

    wheel, sdist = wheels[-1], sdists[-1]
    print(f"wheel:  {wheel.name} ({wheel.stat().st_size:,} bytes)")
    print(f"sdist:  {sdist.name} ({sdist.stat().st_size:,} bytes)")

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()

    for required in REQUIRED_IN_WHEEL:
        if required in wheel_names:
            print(f"  ok: wheel contains {required}")
        else:
            _fail(f"wheel is missing {required}")
            errors += 1

    if any(name.endswith("licenses/LICENSE") for name in wheel_names):
        print("  ok: wheel contains the LICENSE")
    else:
        _fail("wheel does not ship a LICENSE")
        errors += 1

    if not any(name.endswith("/LICENSE") for name in sdist_names):
        _fail("source distribution does not ship a LICENSE")
        errors += 1

    for label, names in (("wheel", wheel_names), ("sdist", sdist_names)):
        for name in names:
            for pattern in FORBIDDEN:
                if pattern in name:
                    _fail(f"{label} contains a forbidden path ({pattern}): {name}")
                    errors += 1

    if errors:
        print(f"\n{errors} problem(s) found.")
        return 1

    print("\nAll packaging checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
