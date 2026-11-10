"""JSON emitter for :class:`jaull.estimator.context_fit.ContextSweep`.

A document of its own, with its own schema version: the ``estimate`` JSON is snapshot tested and
is not touched by the sweep.
"""

from __future__ import annotations

from typing import Any

from jaull.estimator.context_fit import ContextSweep
from jaull.reporting.serialization import confidence_to_string

SCHEMA_VERSION = 1


def context_sweep_to_json_dict(repo_id: str, sweep: ContextSweep) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"repo_id": repo_id},
        "context_sweep": {
            "largest_fitting": sweep.largest_fitting,
            "native_limit": sweep.native_limit,
            "points": [
                {
                    "context_length": point.context_length,
                    "total_bytes": point.total_bytes,
                    "kv_cache_bytes": point.kv_cache_bytes,
                    "status": point.status.value,
                    "confidence": confidence_to_string(point.confidence),
                }
                for point in sweep.points
            ],
            "notes": list(sweep.notes),
        },
    }


__all__ = ["SCHEMA_VERSION", "context_sweep_to_json_dict"]
