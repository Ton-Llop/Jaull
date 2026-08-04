from __future__ import annotations

import pytest

from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
)
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.recommendation import explanations, policies, ranker, scoring
from jaull.recommendation.capability import CapabilitySignal
from jaull.workflow import policies as workflow_policies
from jaull.workflow.ranking import recommend
from jaull.workflow.requirements import build_requirements
from tests._workflow_fixtures import (
    GIB,
    answers,
    candidate,
    hardware,
    memory_estimate,
    transformers_analysis,
)


def _req(
    priority: RecommendationPriority = RecommendationPriority.BALANCED,
    use_case: UseCase = UseCase.CODING,
    languages: list[str] | None = None,
    **kwargs: object,
) -> object:
    return build_requirements(
        answers(priority=priority, use_case=use_case, languages=languages, **kwargs),  # type: ignore[arg-type]
        hardware(),
    )


def _evaluated(
    repo_id: str = "org/Model-7B-Coder-Instruct",
    status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE,
    total_gib: int = 8,
    downloads: int = 10_000,
    license_value: str | None = "apache-2.0",
    languages: list[str] | None = None,
    tags: list[str] | None = None,
    confidence: EstimationConfidence = EstimationConfidence.HIGH,
    base_model: str | None = None,
) -> EvaluatedCandidate:
    analysis = transformers_analysis(repo_id=repo_id, license_value=license_value)
    config = InferenceConfiguration(context_length=4096)
    estimate = memory_estimate(
        analysis,
        config,
        total_bytes=total_gib * GIB,
        status=status,
        confidence=confidence,
    )
    return EvaluatedCandidate(
        candidate=candidate(
            repo_id=repo_id,
            license_value=license_value,
            languages=languages if languages is not None else ["es", "en"],
            tags=tags if tags is not None else ["text-generation", "code"],
            downloads=downloads,
            base_model=base_model,
        ),
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("priority", list(RecommendationPriority))
def test_weights_always_sum_to_one(priority: RecommendationPriority) -> None:
    weights = ranker.weights_for(priority)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_priority_changes_the_weights() -> None:
    balanced = ranker.weights_for(RecommendationPriority.BALANCED)
    memory = ranker.weights_for(RecommendationPriority.MEMORY)
    quality = ranker.weights_for(RecommendationPriority.QUALITY)
    # Memory priority pushes both memory-fit and concurrency-fit above balanced.
    assert memory["memory_fit"] > balanced["memory_fit"]
    assert memory["concurrency_fit"] > balanced["concurrency_fit"]
    assert quality["task_match"] > balanced["task_match"]
    assert quality["capability"] > balanced["capability"]


def test_recommend_uses_injected_capability_analyzer() -> None:
    class FixedAnalyzer:
        def analyze(self, candidate: object, analysis: object) -> CapabilitySignal:
            return CapabilitySignal(score=0.91, parameter_count=7_000_000_000)

    results = recommend(
        [_evaluated()],
        _req(),  # type: ignore[arg-type]
        capability_analyzer=FixedAnalyzer(),
    )

    assert results[0].evaluated.capability_score == 0.91
    assert results[0].score.capability == 0.91


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------
def test_task_match_rewards_matching_keywords() -> None:
    req = _req(use_case=UseCase.CODING)
    coder = candidate(repo_id="org/Qwen-Coder-7B", tags=["text-generation", "code"])
    generic = candidate(repo_id="org/Plain-7B", tags=["text-generation"])
    assert scoring.task_match(coder, req) > scoring.task_match(generic, req)  # type: ignore[arg-type]


def test_task_match_penalises_a_clearly_different_target() -> None:
    req = _req(use_case=UseCase.GENERAL_CHAT)
    coder = candidate(repo_id="org/StarCoder-7B", tags=["text-generation", "coder"])
    chat = candidate(repo_id="org/Chat-7B-Instruct", tags=["text-generation", "chat"])
    assert scoring.task_match(chat, req) > scoring.task_match(coder, req)  # type: ignore[arg-type]


def test_language_match_rewards_declared_languages() -> None:
    req = _req(languages=["Spanish", "English"])
    both = candidate(languages=["es", "en"])
    neither = candidate(languages=["zh"])
    assert scoring.language_match(both, req) == 1.0  # type: ignore[arg-type]
    assert scoring.language_match(neither, req) == 0.0  # type: ignore[arg-type]


def test_undeclared_languages_score_between_match_and_mismatch() -> None:
    """Missing metadata must not be punished harder than a real mismatch."""
    req = _req(languages=["Spanish"])
    silent = candidate(languages=[])
    wrong = candidate(languages=["zh"])
    assert scoring.language_match(wrong, req) < scoring.language_match(silent, req)  # type: ignore[arg-type]
    assert scoring.language_match(silent, req) < 1.0  # type: ignore[arg-type]


def test_commercial_license_scores_above_unknown_and_restricted() -> None:
    req = _req()
    permissive = scoring.license_score(candidate(license_value="apache-2.0"), req)  # type: ignore[arg-type]
    unknown = scoring.license_score(candidate(license_value=None), req)  # type: ignore[arg-type]
    restricted = scoring.license_score(candidate(license_value="cc-by-nc-4.0"), req)  # type: ignore[arg-type]
    assert permissive > unknown > restricted


def test_popularity_is_log_compressed() -> None:
    small = candidate(downloads=100, likes=1)
    huge = candidate(downloads=100_000_000, likes=1)
    base = scoring.max_log_downloads([small, huge])
    # Six orders of magnitude must not become a six-orders-of-magnitude score gap.
    assert scoring.popularity(huge, base) < scoring.popularity(small, base) * 4


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------
def test_hardware_fit_dominates_popularity() -> None:
    """A wildly popular model that barely fits must lose to a comfortable one."""
    popular_tight = _evaluated(
        repo_id="org/Popular-Coder", status=CompatibilityStatus.TIGHT,
        downloads=50_000_000,
    )
    quiet_comfortable = _evaluated(
        repo_id="org/Quiet-Coder", status=CompatibilityStatus.COMFORTABLE,
        downloads=1_000,
    )
    results = recommend([popular_tight, quiet_comfortable], _req())  # type: ignore[arg-type]
    assert results[0].repo_id == "org/Quiet-Coder"


def test_insufficient_model_is_never_the_primary_recommendation() -> None:
    insufficient = _evaluated(
        repo_id="org/Huge", status=CompatibilityStatus.INSUFFICIENT
    )
    fine = _evaluated(repo_id="org/Fits", status=CompatibilityStatus.COMPATIBLE)
    results = recommend([insufficient, fine], _req())  # type: ignore[arg-type]
    assert results[0].repo_id == "org/Fits"
    assert all(r.status is not CompatibilityStatus.INSUFFICIENT for r in results)


def test_insufficient_only_yields_no_recommendations() -> None:
    only_bad = _evaluated(repo_id="org/Huge", status=CompatibilityStatus.INSUFFICIENT)
    assert recommend([only_bad], _req()) == []  # type: ignore[arg-type]


def test_unknown_compatibility_cannot_lead() -> None:
    unknown = _evaluated(
        repo_id="org/Mystery",
        status=CompatibilityStatus.UNKNOWN,
        confidence=EstimationConfidence.UNKNOWN,
    )
    known = _evaluated(repo_id="org/Known", status=CompatibilityStatus.TIGHT)
    results = recommend([unknown, known], _req())  # type: ignore[arg-type]
    assert results[0].repo_id == "org/Known"
    assert not ranker.can_be_primary(unknown)


def test_unknown_compatibility_lowers_the_reported_confidence() -> None:
    unknown = _evaluated(
        repo_id="org/Mystery",
        status=CompatibilityStatus.UNKNOWN,
        confidence=EstimationConfidence.UNKNOWN,
    )
    results = recommend([unknown], _req())  # type: ignore[arg-type]
    assert results == [] or results[0].confidence is EstimationConfidence.UNKNOWN


def test_low_confidence_cannot_outrank_a_well_measured_model() -> None:
    """A model we barely measured must not win on the strength of not knowing.

    A tiny repository with no license and half a config scores well on hardware
    fit precisely because so little about it is known; the confidence multiplier
    is what stops that from becoming a recommendation.
    """
    measured = _evaluated(
        repo_id="org/Known-Coder",
        status=CompatibilityStatus.COMPATIBLE,
        confidence=EstimationConfidence.HIGH,
    )
    guessed = _evaluated(
        repo_id="org/Mystery-Coder",
        status=CompatibilityStatus.COMFORTABLE,
        confidence=EstimationConfidence.UNKNOWN,
        license_value=None,
        languages=[],
    )
    results = recommend([guessed, measured], _req())  # type: ignore[arg-type]
    assert results[0].repo_id == "org/Known-Coder"


def test_confidence_multiplier_is_monotonic() -> None:
    order = [
        EstimationConfidence.HIGH,
        EstimationConfidence.MEDIUM,
        EstimationConfidence.LOW,
        EstimationConfidence.UNKNOWN,
    ]
    values = [policies.CONFIDENCE_MULTIPLIER[c] for c in order]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1.0


def test_ties_break_deterministically() -> None:
    a = _evaluated(repo_id="org/aaa")
    b = _evaluated(repo_id="org/bbb")
    first = [r.repo_id for r in recommend([a, b], _req())]  # type: ignore[arg-type]
    second = [r.repo_id for r in recommend([b, a], _req())]  # type: ignore[arg-type]
    assert first == second


def test_at_most_three_recommendations_are_returned() -> None:
    many = [
        _evaluated(repo_id=f"org/model-{index}", downloads=index * 1000)
        for index in range(10)
    ]
    results = recommend(many, _req())  # type: ignore[arg-type]
    assert len(results) <= workflow_policies.MAX_RECOMMENDATIONS
    assert len(results) == 3


def test_ranks_are_sequential_and_primary_is_first() -> None:
    results = recommend(
        [_evaluated(repo_id=f"org/m{i}") for i in range(3)], _req()  # type: ignore[arg-type]
    )
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].is_primary


def test_families_are_collapsed_into_one_recommendation() -> None:
    """The original repo and its GGUF conversion are the same model."""
    original = _evaluated(repo_id="org/Model-7B")
    conversion = _evaluated(repo_id="other/Model-7B-GGUF", base_model="org/Model-7B")
    results = recommend([original, conversion], _req())  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0].related_repositories


def test_similar_names_alone_do_not_merge_models() -> None:
    a = _evaluated(repo_id="org/Qwen2.5-7B")
    b = _evaluated(repo_id="org/Qwen2.5-7B-Coder")
    results = recommend([a, b], _req())  # type: ignore[arg-type]
    assert len(results) == 2


def test_alternative_label_only_set_when_the_difference_is_real() -> None:
    primary = _evaluated(repo_id="org/A", total_gib=8)
    smaller = _evaluated(repo_id="org/B", total_gib=3)
    identical = _evaluated(repo_id="org/C", total_gib=8)
    assert ranker.alternative_label(primary, smaller) == "Smaller and faster"
    assert ranker.alternative_label(primary, identical) is None


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def test_positive_reasons_are_produced() -> None:
    results = recommend([_evaluated()], _req())  # type: ignore[arg-type]
    reasons = results[0].reasons
    assert any("programming" in r for r in reasons)
    assert any("Apache" in r or "apache" in r for r in reasons)


def test_tight_fit_produces_a_warning() -> None:
    results = recommend(
        [_evaluated(status=CompatibilityStatus.TIGHT)], _req()  # type: ignore[arg-type]
    )
    assert any("limited free VRAM" in w for w in results[0].warnings)


def test_offloading_produces_a_warning() -> None:
    results = recommend(
        [_evaluated(status=CompatibilityStatus.OFFLOADING_REQUIRED)], _req()  # type: ignore[arg-type]
    )
    assert any("offloading" in w for w in results[0].warnings)


def test_unknown_license_produces_a_warning() -> None:
    results = recommend([_evaluated(license_value=None)], _req())  # type: ignore[arg-type]
    assert any("No license is declared" in w for w in results[0].warnings)


def test_custom_license_is_reported_but_not_judged() -> None:
    results = recommend([_evaluated(license_value="llama3.1")], _req())  # type: ignore[arg-type]
    assert results[0].license_category is policies.LicenseCategory.UNKNOWN
    assert any("review its terms" in w for w in results[0].warnings)


def test_unconfirmed_language_produces_a_warning() -> None:
    results = recommend(
        [_evaluated(languages=["zh"])], _req(languages=["Spanish"])  # type: ignore[arg-type]
    )
    assert any("not confirmed" in w for w in results[0].warnings)


def test_no_results_explanation_lists_what_was_missing() -> None:
    lines = explanations.no_results_explanation(
        [_evaluated(status=CompatibilityStatus.INSUFFICIENT)]
    )
    assert "No fully compatible models were found." in lines
    assert any("More RAM" in line for line in lines)
    # It must not tell the user to buy hardware.
    assert not any("buy" in line.lower() for line in lines)


def test_reasons_are_rule_based_not_generated() -> None:
    """Same input, same sentences — no sampling anywhere in the pipeline."""
    req = _req()
    first = recommend([_evaluated()], req)[0].reasons  # type: ignore[arg-type]
    second = recommend([_evaluated()], req)[0].reasons  # type: ignore[arg-type]
    assert first == second
