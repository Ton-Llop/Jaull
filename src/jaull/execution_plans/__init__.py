"""Execution-plan services."""

from jaull.application.discovery.artifact_variants import discover_artifact_variants
from jaull.application.recommendation.execution_plans import (
    execution_plan_for_recommendation,
    variant_from_recommendation,
)
from jaull.execution_plans.service import (
    build_execution_plan,
    resolve_model_identity,
)

__all__ = [
    "build_execution_plan",
    "discover_artifact_variants",
    "execution_plan_for_recommendation",
    "resolve_model_identity",
    "variant_from_recommendation",
]
