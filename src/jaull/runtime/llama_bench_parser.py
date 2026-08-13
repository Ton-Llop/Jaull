"""Pure parser for the llama-bench markdown-style result table."""

from __future__ import annotations

import re

from jaull.benchmarks.errors import BenchmarkParseError
from jaull.domain.benchmarks import (
    BenchmarkMeasurement,
    BenchmarkMeasurementKind,
)

_TEST_RE = re.compile(r"^(?P<prefix>pp|tg)(?P<tokens>\d+)$", re.IGNORECASE)
_TPS_RE = re.compile(
    r"^\s*(?P<mean>\d+(?:\.\d+)?)\s*(?:±|\+/-)\s*"
    r"(?P<stddev>\d+(?:\.\d+)?)\s*$"
)


def parse_llama_bench_output(
    output: str,
    *,
    repetitions: int | None = None,
) -> list[BenchmarkMeasurement]:
    """Parse benchmark measurements from llama-bench output.

    llama-bench emits normal logs around a pipe-delimited table. This parser
    intentionally ignores everything except rows from the table that have a
    ``test`` label shaped like ``pp512`` or ``tg128``.
    """

    headers: dict[str, int] | None = None
    measurements: list[BenchmarkMeasurement] = []
    relevant_rows_seen = 0

    for line in output.splitlines():
        cells = _split_table_row(line)
        if cells is None:
            continue
        if _is_separator(cells):
            continue
        normalized = [_normalize_header(cell) for cell in cells]
        if "test" in normalized and "t/s" in normalized:
            headers = {name: index for index, name in enumerate(normalized)}
            continue
        if headers is None:
            continue
        measurement = _parse_measurement_row(cells, headers, repetitions=repetitions)
        if measurement is None:
            continue
        relevant_rows_seen += 1
        measurements.append(measurement)

    if relevant_rows_seen == 0:
        raise BenchmarkParseError("No llama-bench pp/tg measurements were found.")
    if not measurements:
        raise BenchmarkParseError("No parseable llama-bench measurements were found.")
    return measurements


def _parse_measurement_row(
    cells: list[str],
    headers: dict[str, int],
    *,
    repetitions: int | None,
) -> BenchmarkMeasurement | None:
    test = _cell(cells, headers, "test")
    if test is None:
        return None
    test_match = _TEST_RE.match(test.strip())
    if test_match is None:
        return None

    raw_tps = _cell(cells, headers, "t/s")
    if raw_tps is None:
        raise BenchmarkParseError(f"llama-bench row {test!r} has no t/s value.")
    tps_match = _TPS_RE.match(raw_tps)
    if tps_match is None:
        raise BenchmarkParseError(
            f"Could not parse llama-bench t/s value for {test!r}: {raw_tps!r}."
        )

    kind = (
        BenchmarkMeasurementKind.PREFILL
        if test_match.group("prefix").casefold() == "pp"
        else BenchmarkMeasurementKind.GENERATION
    )
    return BenchmarkMeasurement(
        kind=kind,
        tokens=int(test_match.group("tokens")),
        mean_tokens_per_second=float(tps_match.group("mean")),
        stddev_tokens_per_second=float(tps_match.group("stddev")),
        repetitions=repetitions,
        source_label=test.strip(),
        raw_backend=_cell(cells, headers, "backend"),
        raw_device=_cell(cells, headers, "dev"),
        raw_ngl=_cell(cells, headers, "ngl"),
    )


def _split_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    parts = [part.strip() for part in stripped.strip("|").split("|")]
    if len(parts) < 2:
        return None
    return parts


def _is_separator(cells: list[str]) -> bool:
    return all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _cell(cells: list[str], headers: dict[str, int], name: str) -> str | None:
    index = headers.get(name)
    if index is None or index >= len(cells):
        return None
    value = cells[index].strip()
    return value or None


__all__ = ["parse_llama_bench_output"]
