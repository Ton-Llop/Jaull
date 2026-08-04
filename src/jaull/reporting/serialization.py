"""Common JSON serialisation helpers reused by the estimation and recommendation reports."""

from __future__ import annotations

from typing import Any

from jaull.domain.estimation import EstimationConfidence, MemoryComponent


def confidence_to_string(confidence: EstimationConfidence) -> str:
    return confidence.value


def component_to_dict(component: MemoryComponent) -> dict[str, Any]:
    return {
        "name": component.name,
        "bytes": component.bytes,
        "source": component.source.value,
        "confidence": confidence_to_string(component.confidence),
        "explanation": component.explanation,
    }


def base_resolution_to_dict(resolution: object) -> dict[str, Any] | None:
    """Serialise a BaseModelResolution without importing the class here.

    Duck-typed on purpose so this module stays a pure dependency of
    ``jaull.domain.estimation`` (which is where the class lives) — the
    reporting helpers are consumed *by* the estimation JSON emitter, so
    importing back into estimation would round-trip the layer.
    """
    if resolution is None:
        return None
    return {
        "repo_id": getattr(resolution, "repo_id", None),
        "source": getattr(getattr(resolution, "source", None), "value", None),
        "confidence": getattr(getattr(resolution, "confidence", None), "value", None),
        "evidence": list(getattr(resolution, "evidence", []) or []),
        "warnings": list(getattr(resolution, "warnings", []) or []),
        "candidates": list(getattr(resolution, "candidates", []) or []),
    }


def runtime_to_dict(recommendation: object) -> dict[str, Any] | None:
    if recommendation is None:
        return None
    # Local import: ``jaull.domain.runtime`` is a leaf module and importing it
    # eagerly at module load time would drag it into every reporting import.
    from jaull.domain.runtime import RuntimeRecommendation

    assert isinstance(recommendation, RuntimeRecommendation)
    return {
        "runtime": recommendation.runtime.value,
        "command_preview": recommendation.command_preview,
        "python_snippet": recommendation.python_snippet,
        "flags": [
            {
                "name": flag.name,
                "value": flag.value,
                "source": flag.source.value,
                "explanation": flag.explanation,
            }
            for flag in recommendation.flags
        ],
        "reasons": list(recommendation.reasons),
        "warnings": list(recommendation.warnings),
        "confidence": recommendation.confidence.value,
        "alternatives": [alt.value for alt in recommendation.alternatives],
    }


__all__ = [
    "base_resolution_to_dict",
    "component_to_dict",
    "confidence_to_string",
    "runtime_to_dict",
]
