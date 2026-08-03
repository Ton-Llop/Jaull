from __future__ import annotations

from jaull.discovery.models import EvaluatedCandidate, ModelCandidate
from jaull.recommendation.requirements_gate import (
    evaluate_requirements,
)
from jaull.workflow.models import (
    RecommendationPriority,
    UseCase,
    UserRequirements,
)


def _requirements(
    *,
    commercial: bool | None = True,
    concurrent_users: int = 1,
    concurrency_range: str = "One user",
    languages: list[str] | None = None,
) -> UserRequirements:
    return UserRequirements(
        use_case=UseCase.GENERAL_CHAT,
        priority=RecommendationPriority.BALANCED,
        languages=languages or ["en"],
        concurrent_users=concurrent_users,
        concurrency_range=concurrency_range,
        desired_context=4096,
        commercial_use_required=commercial,
        pipeline_tag="text-generation",
    )


def _evaluated(
    *,
    license: str | None = "apache-2.0",
    languages: list[str] | None = None,
    concurrency_fit: float = 1.0,
) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        candidate=ModelCandidate(
            repo_id="user/model",
            license=license,
            languages=languages or [],
        ),
        concurrency_fit_score=concurrency_fit,
    )


def test_restricted_license_with_commercial_required_eliminates_candidate() -> None:
    # Language matched so the only unmet check is the license veto.
    result = evaluate_requirements(
        _evaluated(license="cc-by-nc-4.0", languages=["en"]),
        _requirements(languages=["en"]),
    )
    assert result.penalty_multiplier == 0.0
    assert any("restricts" in label for label in result.unmet_labels)


def test_unknown_license_with_commercial_required_softens_penalty() -> None:
    result = evaluate_requirements(
        _evaluated(license=None, languages=["en"]),
        _requirements(languages=["en"]),
    )
    # Unknown-license penalty is 0.4 → multiplier = 1 - 0.4 = 0.6.
    assert result.penalty_multiplier == 0.6


def test_permissive_license_satisfies_the_check() -> None:
    result = evaluate_requirements(
        _evaluated(license="apache-2.0", languages=["en"]),
        _requirements(languages=["en"]),
    )
    assert result.penalty_multiplier == 1.0
    assert result.unmet_labels == []


def test_language_missing_is_a_soft_penalty() -> None:
    result = evaluate_requirements(
        _evaluated(license="apache-2.0", languages=[]),
        _requirements(languages=["es"]),
    )
    # Language check is `required=False` → soft penalty. Multiplier drops
    # noticeably but the candidate is not eliminated.
    assert result.penalty_multiplier < 1.0
    assert result.penalty_multiplier > 0.7
    assert any("ES" in label for label in result.unmet_labels)


def test_concurrency_zero_fit_adds_soft_penalty() -> None:
    result = evaluate_requirements(
        _evaluated(concurrency_fit=0.0, languages=["en"]),
        _requirements(concurrent_users=6, concurrency_range="6-20 users"),
    )
    # concurrency check is `required=False` → surfaces as unmet and applies
    # its soft multiplier.
    assert result.penalty_multiplier < 1.0
    assert any(
        "user" in label.lower() or "concurr" in label.lower()
        for label in result.unmet_labels
    )
