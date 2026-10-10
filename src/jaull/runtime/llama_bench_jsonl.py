"""Read reported test settings from llama-bench's own JSONL output.

The markdown table omits every column still at its default, so threads, batch
sizes, cache types and load mode are simply absent from it. ``-oe jsonl`` asks
llama-bench to restate the full settings of each test on stderr, which is the
only place they can be read without guessing.

These are the test-instance settings, not a snapshot of the runtime context.
In particular, runtime batch limits and flash attention can differ from them.
Warmup has no field at all.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from jaull.domain.benchmarks import LlamaBenchProtocol

# Only these are read. Anything else llama-bench prints stays in the raw output,
# where a later question can still reach it without a schema change.
_FIELDS = {
    "build_commit": "build_commit",
    "build_number": "build_number",
    "n_threads": "n_threads",
    "n_batch": "n_batch",
    "n_ubatch": "n_ubatch",
    "type_k": "cache_type_k",
    "type_v": "cache_type_v",
    "load_mode": "load_mode",
}

# llama-bench echoes the request, so -1 means "auto, never resolved in output".
_FLASH_ATTN = {1: "on", 0: "off", -1: "auto_unresolved"}

# Recognize settings rows, including workload rows with missing identity/settings.
_REQUIRED_KEYS = ("build_commit", "n_threads")


def parse_llama_bench_protocol(
    output: str, *, warmup: bool | None = None,
) -> LlamaBenchProtocol | None:
    """Project the settings every test row agrees on, or ``None`` if unknown.

    ``output`` may be stdout and stderr joined: non-JSON log lines and the
    markdown table are skipped. ``warmup`` comes from the caller's own command,
    not from llama-bench, because the tool reports nothing about it.

    Rows that disagree on a setting mean a sweep this caller did not request, so
    the protocol is reported as unknown instead of picking one row's value.
    """
    rows = _result_rows(output)
    if not rows:
        return None
    try:
        protocols = [LlamaBenchProtocol.model_validate({
            **{name: row.get(key) for key, name in _FIELDS.items()},
            "flash_attn": (
                _FLASH_ATTN.get(flash)
                if (flash := _as_int(row.get("flash_attn"))) is not None else None
            ),
        }, strict=True) for row in rows]
    except ValidationError:
        # Malformed optional metadata must not crash a completed benchmark or
        # become trusted evidence via type coercion / repr(). Raw output remains.
        return None
    fields: dict[str, Any] = {}
    for name in (*_FIELDS.values(), "flash_attn"):
        values = {getattr(protocol, name) for protocol in protocols}
        if len(values) == 1:
            fields[name] = values.pop()
    if warmup is not None:
        fields["warmup"] = "ran" if warmup else "skipped"
    protocol = LlamaBenchProtocol.model_validate(
        {name: value for name, value in fields.items() if value is not None}
    )
    # An all-unknown protocol says nothing a missing one does not.
    return protocol if protocol.model_dump(exclude_none=True) else None


def _result_rows(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(row, dict) and (
            all(key in row for key in _REQUIRED_KEYS)
            or all(key in row for key in ("n_prompt", "n_gen"))
        ):
            rows.append(row)
    return rows


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ["parse_llama_bench_protocol"]
