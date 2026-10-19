"""Shared display labels; never compute confidence, scores or ranking."""

from __future__ import annotations

from jaull.domain.estimation import EstimateSource, EstimationConfidence
from jaull.recommendation.models import ModelRecommendation


def confidence_rows(rec: ModelRecommendation) -> list[tuple[str, str]]:
    estimate = rec.plan.memory_prediction if rec.plan is not None else None
    estimate = estimate or rec.evaluated.memory_estimate
    rows: list[tuple[str, str]] = []
    if estimate is None:
        rows.append(("Metadata confidence", rec.evaluated.candidate.metadata_confidence.value))
    else:
        rows.append(("Memory estimation confidence", estimate.assessment.confidence.value))
        if estimate.assessment.confidence in {
            EstimationConfidence.LOW, EstimationConfidence.UNKNOWN,
        }:
            causes = []
            for name, component in (
                ("weights", estimate.weights.component),
                ("KV cache", estimate.kv_cache.component),
                ("runtime overhead", estimate.runtime_overhead.component),
            ):
                if component.confidence in {EstimationConfidence.LOW, EstimationConfidence.UNKNOWN}:
                    source = (
                        "assumed heuristic" if component.source is EstimateSource.ASSUMED
                        else component.source.value
                    )
                    causes.append(f"{name} ({source}): {component.explanation}")
            rows.append(("Memory estimation limitations", "; ".join(causes)
                         or "No component-level cause recorded."))
    assessment = rec.plan_assessment
    if assessment is not None:
        rows.append(("Plan evidence confidence", assessment.confidence.value))
        if assessment.local_experiment_id or assessment.local_benchmark_id:
            basis = "; ".join(
                f"{kind} {identity}" for kind, identity in (
                    ("experiment", assessment.local_experiment_id),
                    ("benchmark", assessment.local_benchmark_id),
                ) if identity
            )
            basis = f"Matching local evidence: {basis}. Not a change to memory-estimate confidence."
        else:
            basis = (
                f"Metadata confidence: {rec.evaluated.candidate.metadata_confidence.value}; "
                "without matching local execution evidence, the plan uses metadata and "
                "memory-estimate confidence, not runtime installation."
            )
            if (
                rec.evaluated.candidate.metadata_confidence is EstimationConfidence.LOW
                and rec.evaluated.candidate.penalties
            ):
                basis += " Metadata limitations: " + "; ".join(rec.evaluated.candidate.penalties)
        rows.append(("Plan evidence basis", basis))
        rows.extend([
            ("Performance evidence", "Comparable benchmark available" if assessment.speed.applicable
             else "No comparable benchmark for ranking"),
            ("Quality evidence", "Applicable evaluation available" if assessment.quality.applicable
             else "No applicable evaluation"),
        ])
    return rows


def score_rows(rec: ModelRecommendation) -> list[tuple[str, str]]:
    if rec.plan is None or rec.plan_assessment is None:
        return [
            ("Composite score (legacy ranking)", f"{rec.score.out_of_100}/100"),
            ("Score role", "Without a plan assessment, the legacy composite orders this list."),
        ]
    return [
        ("Diagnostic composite score", f"{rec.score.out_of_100}/100 "
         f"(memory fit {rec.score.memory_fit * 100:.0f}%, "
         f"capability {rec.score.capability * 100:.0f}%)"),
        ("Score role", "This is not what ordered the list. Position follows the plan criteria "
         "and applicable measured policy, then diversity; a higher score need not rank first."),
    ]


def ranking_axis_label(axis: str) -> str:
    return "plan evidence confidence" if axis == "plan_confidence" else axis.replace("_", " ")
