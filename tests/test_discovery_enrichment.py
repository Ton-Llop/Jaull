from __future__ import annotations

from jaull.discovery import enrichment
from jaull.domain.enums import RepositoryType
from jaull.domain.model import ModelAnalysis, ModelRepositoryInfo, RepositoryClassification
from jaull.workflow.requirements import build_requirements
from tests._workflow_fixtures import answers, candidate, hardware


def test_inspected_adapter_does_not_reach_estimation_or_recommendations() -> None:
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(repo_id="org/lora"),
        classification=RepositoryClassification(
            primary_type=RepositoryType.ADAPTER,
            detected_types={RepositoryType.ADAPTER},
        ),
    )
    requirements = build_requirements(answers(), hardware())

    def estimate_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("adapters must be rejected before memory estimation")

    result = enrichment.evaluate_candidate(
        candidate("org/lora"),
        requirements,
        hardware(),
        inspect_fn=lambda repo_id: analysis,
        estimate_fn=estimate_must_not_run,
    )

    assert result.failed is True
    assert result.analysis == analysis
    assert result.candidate.repository_type is RepositoryType.ADAPTER
    assert "standalone executable model" in result.warnings[0]
