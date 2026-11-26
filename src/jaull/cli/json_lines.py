"""One JSON object per line on stdout, for front ends that read the CLI as it runs."""

from __future__ import annotations

import json
import sys
import threading
from typing import Any

JSON_LINES_SCHEMA_VERSION = 1

# Progress can be reported from a worker thread; lines must not interleave.
_LOCK = threading.Lock()


def emit(event: str, **fields: Any) -> None:
    line = json.dumps({"schema_version": JSON_LINES_SCHEMA_VERSION, "event": event, **fields})
    with _LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


__all__ = ["JSON_LINES_SCHEMA_VERSION", "emit"]
