#!/usr/bin/env python
"""Render the hardware-fit scenario catalogue as an observation table.

    uv run python scripts/hardware_fit_matrix.py                 # readable table
    uv run python scripts/hardware_fit_matrix.py --format markdown
    uv run python scripts/hardware_fit_matrix.py --format json
    uv run python scripts/hardware_fit_matrix.py --write-snapshot

The scenarios live in ``tests/_hardware_fit_scenarios.py`` and are the same
ones ``tests/test_hardware_fit_scenarios.py`` asserts against, so this script
reports what the battery checks rather than a second, drifting copy of it.

Everything here is read-only and offline: it runs the analyzer over in-memory
fixtures. ``--write-snapshot`` is the one exception, and it only rewrites
``tests/snapshots/hardware_fit_scenarios.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # The catalogue is test data, and tests are not an installed package.
    sys.path.insert(0, str(REPO_ROOT))

from tests._hardware_fit_scenarios import (  # noqa: E402
    GIB,
    OBSERVED_FIELDS,
    SCENARIOS,
)

SNAPSHOT = REPO_ROOT / "tests" / "snapshots" / "hardware_fit_scenarios.json"

# The columns the comparison asks about, shortened to fit a terminal. Byte
# fields are rendered in GiB because that is the unit both Jaull and llmfit
# report memory in.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("mode", "mode"),
    ("placement_method", "placement"),
    ("gpu_required_bytes", "gpu req"),
    ("ram_required_bytes", "ram req"),
    ("gpu_weight_bytes", "gpu wts"),
    ("ram_weight_bytes", "ram wts"),
    ("gpu_overhead_bytes", "gpu ovh"),
    ("ram_overhead_bytes", "ram ovh"),
    ("gpu_safety_margin_bytes", "gpu mrg"),
    ("ram_safety_margin_bytes", "ram mrg"),
    ("gpu_layers", "gpu L"),
    ("total_layers", "tot L"),
)

_BYTE_FIELDS = frozenset(
    name for name, _ in COLUMNS if name.endswith("_bytes")
)


def main() -> int:
    args = _parse_args()

    if args.write_snapshot:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(_snapshot_json(), encoding="utf-8")
        print(f"wrote {SNAPSHOT.relative_to(REPO_ROOT).as_posix()}")
        return 0

    if args.format == "json":
        print(_snapshot_json())
        return 0

    renderer = _markdown if args.format == "markdown" else _table
    print(renderer())
    return 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(field: str, value: object) -> str:
    if value is None:
        return "-"
    if field in _BYTE_FIELDS:
        assert isinstance(value, int)
        return f"{value / GIB:.3f}"
    return str(value)


def _rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for case in SCENARIOS:
        observed = case.observe()
        rows.append(
            [case.name, *(_cell(field, observed[field]) for field, _ in COLUMNS)]
        )
    return rows


def _table() -> str:
    header = ["scenario", *(label for _, label in COLUMNS)]
    rows = _rows()
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows))
        for i in range(len(header))
    ]

    lines = [
        "Hardware fit analyzer — placement observations (memory columns in GiB)",
        "",
        "  ".join(label.ljust(widths[i]) for i, label in enumerate(header)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(header))),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    )
    lines.extend(["", _inputs_block(), "", _notes_block()])
    return "\n".join(lines)


def _inputs_block() -> str:
    lines = ["Inputs (GiB unless stated):", ""]
    for case in SCENARIOS:
        layers = case.total_layers if case.total_layers is not None else "no metadata"
        lines.append(
            f"  {case.name}\n"
            f"    machine   {case.machine.name}\n"
            f"    weights   {case.weights_bytes / GIB:.3f}"
            f"   kv {case.kv_cache_bytes / GIB:.3f}"
            f"   overhead {case.overhead_bytes / GIB:.3f}"
            f"   reserve {case.device_reserve_bytes / GIB:.3f}"
            f"   margin {case.safety_margin_bytes / GIB:.3f}\n"
            f"    layers    {layers}\n"
            f"    asks      {case.question}"
        )
    return "\n".join(lines)


def _notes_block() -> str:
    lines = ["Notes:", ""]
    for case in SCENARIOS:
        for note in case.notes:
            lines.append(f"  {case.name}: {note}")
    return "\n".join(lines)


def _markdown() -> str:
    header = ["scenario", *(label for _, label in COLUMNS)]
    lines = [
        "# Hardware fit analyzer — placement observations",
        "",
        "Memory columns in GiB. Generated by `scripts/hardware_fit_matrix.py`.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in _rows())
    lines.extend(["", "## Inputs", ""])
    lines.extend(
        f"- **{case.name}** — {case.machine.name}; "
        f"weights {case.weights_bytes / GIB:.3f}, "
        f"kv {case.kv_cache_bytes / GIB:.3f}, "
        f"overhead {case.overhead_bytes / GIB:.3f}, "
        f"reserve {case.device_reserve_bytes / GIB:.3f}, "
        f"margin {case.safety_margin_bytes / GIB:.3f}, "
        f"layers {case.total_layers if case.total_layers is not None else 'none'}. "
        f"{case.question}"
        for case in SCENARIOS
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _snapshot_payload() -> dict[str, object]:
    return {
        "observed_fields": list(OBSERVED_FIELDS),
        "scenarios": {
            case.name: {"inputs": case.inputs(), "observed": case.observe()}
            for case in SCENARIOS
        },
    }


def _snapshot_json() -> str:
    return json.dumps(_snapshot_payload(), indent=2, sort_keys=False) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Rewrite tests/snapshots/hardware_fit_scenarios.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
