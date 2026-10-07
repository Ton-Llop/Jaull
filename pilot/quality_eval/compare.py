"""Offline per-task diagnostics. No quality aggregate, ranking or speed evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pilot.quality_eval.evaluate import write_json
from pilot.quality_eval.records import load_record

from jaull.evaluation.quality_comparison import compare_quality_records


def compare_records(left: Path, right: Path) -> dict[str, Any]:
    return compare_quality_records(
        load_record(left), load_record(right),
        left_source=str(left.resolve()), right_source=str(right.resolve()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, compare_records(args.left, args.right))


if __name__ == "__main__":
    main()
