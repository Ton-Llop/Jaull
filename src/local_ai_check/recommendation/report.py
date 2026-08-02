"""Export a completed guided run as JSON or Markdown.

JSON reuses :func:`~local_ai_check.presentation.estimation_report.estimate_to_json_dict`
for the memory section, so the estimate serialisation stays defined in exactly
one place. Nothing sensitive is written: no token, no authorization header and
no local filesystem path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from local_ai_check.presentation.estimation_report import estimate_to_json_dict
from local_ai_check.recommendation import policies
from local_ai_check.recommendation.models import ModelRecommendation
from local_ai_check.workflow.state import RecommendationWorkflowState

REPORT_SCHEMA_VERSION = 1


def report_to_dict(state: RecommendationWorkflowState) -> dict[str, Any]:
    """Serialise a run into a stable, JSON-safe dictionary."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": _hardware_to_dict(state),
        "user_requirements": _requirements_to_dict(state),
        "search_strategy": {
            "queries": list(state.search_queries),
            "candidate_count": len(state.candidates),
            "evaluated_count": len(state.evaluated_candidates),
        },
        "evaluated_candidates": [
            _evaluated_to_dict(item) for item in state.evaluated_candidates
        ],
        "recommendations": [
            _recommendation_to_dict(rec) for rec in state.recommendations
        ],
        "warnings": list(state.warnings),
        "assumptions": (
            list(state.requirements.assumptions) if state.requirements else []
        ),
        "no_results_reason": list(state.no_results_reason),
        "disclaimer": policies.LEGAL_DISCLAIMER,
    }


def report_to_json(state: RecommendationWorkflowState, indent: int = 2) -> str:
    return json.dumps(report_to_dict(state), indent=indent, sort_keys=False)


def report_to_markdown(state: RecommendationWorkflowState) -> str:
    """A human-readable summary for pasting into notes or a thesis appendix."""
    lines: list[str] = ["# local-ai-check recommendation report", ""]
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")

    if state.hardware is not None:
        hw = state.hardware
        gpu = hw.gpus[0].name if hw.gpus else "no NVIDIA GPU detected"
        lines += [
            "## Hardware",
            "",
            f"- CPU: {hw.cpu.model or 'unknown'}",
            f"- RAM: {_gib(hw.memory.total_bytes)}",
            f"- GPU: {gpu}",
        ]
        if hw.gpus:
            lines.append(f"- VRAM: {_gib(hw.gpus[0].vram_total_bytes)}")
        lines += [f"- Platform: {hw.os} ({hw.arch})", ""]

    if state.requirements is not None:
        req = state.requirements
        lines += [
            "## Requirements",
            "",
            f"- Use case: {req.use_case.value}",
            f"- Priority: {req.priority.value}",
            f"- Languages: {', '.join(req.languages)}",
            f"- Concurrency: {req.concurrency_range}",
            f"- Context: {req.desired_context} tokens",
            f"- Commercial use required: {_yes_no(req.commercial_use_required)}",
            "",
        ]

    if not state.recommendations:
        lines += ["## Result", ""]
        lines += state.no_results_reason or ["No recommendations were produced."]
        lines.append("")
        return "\n".join(lines)

    lines += ["## Recommendations", ""]
    for rec in state.recommendations:
        heading = "Best match" if rec.is_primary else (rec.alternative_label or "Alternative")
        lines += [
            f"### {rec.rank}. {rec.repo_id} — {heading}",
            "",
            f"- Score: {rec.score.out_of_100}/100",
            f"- Compatibility: {rec.status.value}",
            f"- Confidence: {rec.confidence.value}",
            f"- License: {rec.evaluated.candidate.license or 'not declared'} "
            f"({rec.license_category.value})",
        ]
        config = rec.evaluated.selected_configuration
        if config is not None:
            if config.quantization:
                lines.append(f"- Quantization: {config.quantization}")
            if config.precision:
                lines.append(f"- Precision: {config.precision.value}")
            lines.append(f"- Context: {config.context_length} tokens")
        lines.append("")
        if rec.reasons:
            lines += ["**Why this model?**", ""]
            lines += [f"- {reason}" for reason in rec.reasons]
            lines.append("")
        if rec.warnings:
            lines += ["**Limitations and warnings**", ""]
            lines += [f"- {warning}" for warning in rec.warnings]
            lines.append("")

    lines += ["---", "", policies.LEGAL_DISCLAIMER, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------
def _hardware_to_dict(state: RecommendationWorkflowState) -> dict[str, Any] | None:
    hw = state.hardware
    if hw is None:
        return None
    return {
        "os": hw.os,
        "arch": hw.arch,
        "cpu": hw.cpu.model,
        "physical_cores": hw.cpu.physical_cores,
        "ram_total_bytes": hw.memory.total_bytes,
        "ram_available_bytes": hw.memory.available_bytes,
        "gpus": [
            {
                "name": gpu.name,
                "vram_total_bytes": gpu.vram_total_bytes,
                "vram_available_bytes": gpu.vram_available_bytes,
            }
            for gpu in hw.gpus
        ],
        "warnings": list(hw.warnings),
    }


def _requirements_to_dict(
    state: RecommendationWorkflowState,
) -> dict[str, Any] | None:
    req = state.requirements
    if req is None:
        return None
    return {
        "use_case": req.use_case.value,
        "priority": req.priority.value,
        "languages": list(req.languages),
        "concurrent_users": req.concurrent_users,
        "concurrency_range": req.concurrency_range,
        "desired_context": req.desired_context,
        "commercial_use_required": req.commercial_use_required,
        "pipeline_tag": req.pipeline_tag,
        "preferred_formats": list(req.preferred_formats),
    }


def _evaluated_to_dict(item: Any) -> dict[str, Any]:
    return {
        "repo_id": item.repo_id,
        "failed": item.failed,
        "repository_type": (
            item.candidate.repository_type.value
            if item.candidate.repository_type
            else None
        ),
        "license": item.candidate.license,
        "downloads": item.candidate.downloads,
        "compatibility": (
            item.compatibility.status.value if item.compatibility else None
        ),
        "scores": {
            "hardware_fit": item.hardware_fit_score,
            "task_match": item.task_match_score,
            "language_match": item.language_match_score,
            "license": item.license_score,
            "metadata_quality": item.metadata_quality_score,
            "popularity": item.popularity_score,
        },
        "configuration_reason": item.configuration_reason,
        "alternatives_considered": list(item.alternatives_considered),
        "warnings": list(item.warnings),
    }


def _recommendation_to_dict(rec: ModelRecommendation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rank": rec.rank,
        "repo_id": rec.repo_id,
        "score": rec.score.out_of_100,
        "score_breakdown": {
            "total": rec.score.total,
            "hardware_fit": rec.score.hardware_fit,
            "task_match": rec.score.task_match,
            "language_match": rec.score.language_match,
            "license": rec.score.license,
            "metadata_quality": rec.score.metadata_quality,
            "popularity": rec.score.popularity,
            "weights": dict(rec.score.weights),
        },
        "compatibility": rec.status.value,
        "confidence": rec.confidence.value,
        "license": rec.evaluated.candidate.license,
        "license_category": rec.license_category.value,
        "alternative_label": rec.alternative_label,
        "related_repositories": list(rec.related_repositories),
        "reasons": list(rec.reasons),
        "warnings": list(rec.warnings),
        "memory": None,
    }
    if rec.evaluated.memory_estimate is not None:
        payload["memory"] = estimate_to_json_dict(rec.evaluated.memory_estimate)
    return payload


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**3:.1f} GiB"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "not specified"
    return "yes" if value else "no"


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "report_to_dict",
    "report_to_json",
    "report_to_markdown",
]
