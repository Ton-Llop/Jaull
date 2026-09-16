"""The report must describe the run it actually performed.

Built from the RTX 4060 case that exposed these defects: an 8 GiB card, the
`coding` use case and `quality` priority, where the published report

* listed every evaluated candidate with all scores at ``0.0`` and
  ``requirement_penalty`` at ``1.0`` with no unmet requirements, while the
  recommendation built from the *same* candidate reported a ``0.51`` penalty
  and two of them;
* printed a composite score beside each rank, so the highest-scoring model
  appeared fourth;
* blamed missing model-card metadata for every low-confidence estimate;
* said "No precision fits the detected memory" for a model whose own
  ``alternatives_considered`` showed three precisions at ``offloading_required``.

Each test below pins one of those. Together with
``test_recommendation_shortlist_regression`` they split the contract: that file
owns the *order*, this one owns what the report *says* about it. The last test
here is the hinge — the explanations may change, the decisions may not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _workflow_fixtures import (
    GIB,
    FakeSearchClient,
    answers,
    candidate,
    gguf_analysis,
    hardware,
    transformers_analysis,
)
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.recommendation.report import report_to_dict, report_to_markdown
from jaull.workflow import orchestrator
from jaull.workflow.state import RecommendationWorkflowState
from test_workflow_orchestrator import _container

VRAM_BYTES = 8 * GIB

# A spread that reaches the branches the report has to describe: one model that
# fits comfortably, one that only just fits and whose licence and language cannot
# be confirmed, and one that does not fit at all. The ladder-exhausted wording is
# covered directly below, because the fixture estimator never returns
# `offloading_required`.
SHORTLIST: tuple[tuple[str, int, list[str], str | None], ...] = (
    ("org/Coder-3B-Instruct-GGUF", int(2.0 * GIB), ["en"], "apache-2.0"),
    ("org/Coder-7B-Instruct-GGUF", int(7.6 * GIB), [], "other"),
    ("org/Coder-32B-Instruct-GGUF", int(40.0 * GIB), ["en"], "apache-2.0"),
)


def _run() -> RecommendationWorkflowState:
    search = FakeSearchClient(
        default=[
            candidate(
                repo_id=repo_id,
                tags=["text-generation", "instruct", "gguf", "code"],
                languages=languages,
                license_value=license_value,
            )
            for repo_id, _bytes, languages, license_value in SHORTLIST
        ]
    )
    analyses = {
        repo_id: gguf_analysis(
            repo_id=repo_id, quantizations=("Q4_K_M",), base_bytes=size
        )
        for repo_id, size, *_rest in SHORTLIST
    }
    return orchestrator.run_workflow(
        answers(
            use_case=UseCase.CODING,
            priority=RecommendationPriority.QUALITY,
            languages=["English"],
        ),
        hardware(vram_gib=8, ram_gib=32),
        _container(search, analyses=analyses, vram_budget=VRAM_BYTES),
    )


# ---------------------------------------------------------------------------
# 1. Evaluated candidates carry the numbers the ranker used
# ---------------------------------------------------------------------------


def test_evaluated_candidates_publish_real_scores() -> None:
    """Not the model defaults. Every score at 0.0 was the symptom."""
    payload = report_to_dict(_run())
    scored = [
        item for item in payload["evaluated_candidates"] if not item["failed"]
    ]

    assert scored
    for item in scored:
        assert item["scores"] is not None
        assert any(value > 0.0 for value in item["scores"].values()), item["repo_id"]


def test_a_candidate_does_not_contradict_its_own_recommendation() -> None:
    """The same repo appeared twice in one document with different penalties."""
    payload = report_to_dict(_run())
    by_repo = {item["repo_id"]: item for item in payload["evaluated_candidates"]}

    for rec in payload["recommendations"]:
        evaluated = by_repo[rec["repo_id"]]
        breakdown = rec["score_breakdown"]

        assert evaluated["requirement_penalty"] == breakdown["hard_penalty"]
        assert evaluated["unmet_requirements"] == breakdown["unmet_requirements"]
        for axis in ("capability", "memory_fit", "task_match", "license"):
            assert evaluated["scores"][axis] == breakdown[axis], (rec["repo_id"], axis)


def test_a_failed_candidate_is_unscored_rather_than_zero() -> None:
    """``null`` says "never evaluated"; 0.0 reads as "scored badly"."""
    search = FakeSearchClient(
        default=[candidate(repo_id="org/Adapter-7B", tags=["text-generation"])]
    )
    state = orchestrator.run_workflow(
        answers(use_case=UseCase.CODING),
        hardware(vram_gib=8),
        _container(
            search,
            analyses={"org/Adapter-7B": transformers_analysis("org/Adapter-7B")},
            vram_budget=VRAM_BYTES,
            inspect_error=RuntimeError("boom"),
        ),
    )
    payload = report_to_dict(state)

    for item in payload["evaluated_candidates"]:
        if item["failed"]:
            assert item["scores"] is None
            assert item["requirement_penalty"] is None
            assert item["unmet_requirements"] is None


# ---------------------------------------------------------------------------
# 2. The score does not pretend to order anything
# ---------------------------------------------------------------------------


def test_the_json_marks_the_composite_as_diagnostic() -> None:
    payload = report_to_dict(_run())

    for rec in payload["recommendations"]:
        assert rec["score_role"] == "diagnostic"
        assert rec["ranking"]["ordered_by"] == "quality"
        assert rec["ranking"]["criteria"]
        assert "does not determine the order" in rec["ranking"]["note"]
        # Kept for compatibility: existing consumers still find both keys.
        assert "score" in rec
        assert "score_breakdown" in rec


def test_the_markdown_explains_the_position_instead_of_the_score() -> None:
    """A bare `Score: 44/100` above `Score: 65/100` read as a broken tool."""
    markdown = report_to_markdown(_run())

    assert "**Why this position?**" in markdown
    assert "This is not what ordered the list." in markdown
    # The old headline line is gone; the composite survives only as diagnostic.
    assert "\n- Score: " not in markdown


def test_published_criteria_are_the_axes_that_ordered() -> None:
    state = _run()
    payload = report_to_dict(state)
    axes = [c["axis"] for c in payload["recommendations"][0]["ranking"]["criteria"]]

    # The full effective order: the viability partition that runs after the
    # sort, then the quality branch of the ranking key, then its tie-breaks.
    assert axes == [
        "viability",
        "plan_constraints",
        "suitability",
        "capability",
        "quantization_quality",
        "execution_fitness",
        "performance_evidence",
        "estimate_confidence",
    ]


# ---------------------------------------------------------------------------
# 3. The explanations say what is true
# ---------------------------------------------------------------------------


def test_low_confidence_is_not_blamed_on_the_model_card() -> None:
    """``runtime_overhead`` is ASSUMED unconditionally, so the old sentence
    fired on models with a complete card and an exact parameter count."""
    payload = report_to_dict(_run())
    warnings = [w for rec in payload["recommendations"] for w in rec["warnings"]]

    assert not any(
        "Confidence is low because part of the model metadata was missing" in warning
        for warning in warnings
    )
    low_confidence = [
        rec for rec in payload["recommendations"] if rec["confidence"] == "low"
    ]
    for rec in low_confidence:
        assert any(
            "Confidence is low" in warning for warning in rec["warnings"]
        ), rec["repo_id"]


def test_the_confidence_warning_names_the_component_that_capped_it() -> None:
    """The fixture above reports high confidence, so exercise the branch directly.

    A complete model card and an exact parameter count still yield LOW, because
    ``runtime_overhead`` is ASSUMED unconditionally. The warning has to say that
    rather than send the reader to fix a model card that is already fine.
    """
    from jaull.recommendation.explanations import _confidence_warning

    evaluated = _run().recommendations[0].evaluated
    estimate = evaluated.memory_estimate
    assert estimate is not None
    assert estimate.runtime_overhead.component.source.value == "assumed"

    warning = _confidence_warning(evaluated)

    assert "runtime overhead" in warning
    assert "heuristic" in warning
    assert "metadata was missing" not in warning


def test_an_offloadable_model_is_not_reported_as_not_fitting() -> None:
    """It fits; it just does not fit in VRAM alone.

    The old text was "No precision fits the detected memory, including int4"
    even when every rung had come back ``offloading_required``.
    """
    from jaull.domain.estimation import CompatibilityStatus
    from jaull.estimator.configuration import _exhausted_ladder_message

    message = _exhausted_ladder_message(
        [CompatibilityStatus.OFFLOADING_REQUIRED, CompatibilityStatus.INSUFFICIENT],
        CompatibilityStatus.OFFLOADING_REQUIRED,
        unit="precision",
        aggressive=", including int4",
    )

    assert "offloading" in message
    assert "including int4" not in message
    # An estimated placement is not a verified run, and the text must not imply one.
    assert "placement estimate, not a verified run" in message


def test_the_message_describes_the_option_it_reports_not_the_best_one_seen() -> None:
    """``best_effort`` keeps the *first* rung tried, not the best one.

    With float16 INSUFFICIENT returned while int8/int4 came back
    ``offloading_required``, the text claimed offloading viability for a
    configuration the estimator had marked as too large.
    """
    from jaull.domain.estimation import CompatibilityStatus
    from jaull.estimator.configuration import _exhausted_ladder_message

    message = _exhausted_ladder_message(
        [
            CompatibilityStatus.INSUFFICIENT,
            CompatibilityStatus.OFFLOADING_REQUIRED,
            CompatibilityStatus.OFFLOADING_REQUIRED,
        ],
        CompatibilityStatus.INSUFFICIENT,
        unit="precision",
        aggressive=", including int4",
    )

    assert message.startswith(
        "The reported configuration is estimated not to fit the detected memory."
    )
    assert "Other options on the ladder" in message
    assert "The reported option is estimated to fit" not in message
    # "No precision fits" is a claim about the whole ladder. Making it here and
    # then reporting that other rungs do fit with offloading contradicts itself.
    assert "No precision fits" not in message


def test_the_global_claim_is_only_made_when_the_whole_ladder_failed() -> None:
    from jaull.domain.estimation import CompatibilityStatus
    from jaull.estimator.configuration import _exhausted_ladder_message

    everything_too_large = _exhausted_ladder_message(
        [CompatibilityStatus.INSUFFICIENT] * 3,
        CompatibilityStatus.INSUFFICIENT,
        unit="precision",
        aggressive=", including int4",
    )

    assert everything_too_large == (
        "No precision fits the detected memory, including int4."
    )


def test_unknown_is_reported_as_unconfirmed_not_as_proven_too_large() -> None:
    from jaull.domain.estimation import CompatibilityStatus
    from jaull.estimator.configuration import _exhausted_ladder_message

    unknown = _exhausted_ladder_message(
        [CompatibilityStatus.UNKNOWN],
        CompatibilityStatus.UNKNOWN,
        unit="GGUF variant",
        aggressive="",
    )
    too_large = _exhausted_ladder_message(
        [CompatibilityStatus.INSUFFICIENT],
        CompatibilityStatus.INSUFFICIENT,
        unit="GGUF variant",
        aggressive="",
    )
    offloadable = _exhausted_ladder_message(
        [CompatibilityStatus.OFFLOADING_REQUIRED],
        CompatibilityStatus.OFFLOADING_REQUIRED,
        unit="GGUF variant",
        aggressive="",
    )

    assert "could be confirmed to fit" in unknown
    assert "not a demonstration that it does not fit" in unknown
    assert unknown != too_large != offloadable
    assert "offloading" in offloadable


# ---------------------------------------------------------------------------
# 4. The hinge: explanations may change, decisions may not
# ---------------------------------------------------------------------------


def test_reporting_changes_did_not_move_a_single_decision() -> None:
    """Everything this block touched is presentation.

    If a future edit to the wording also changes an order, a score, a chosen
    configuration or an estimate, that is a behavioural change wearing a
    reporting commit's clothes, and this fails.
    """
    payload = report_to_dict(_run())

    assert [rec["repo_id"] for rec in payload["recommendations"]] == [
        "org/Coder-7B-Instruct-GGUF",
        "org/Coder-3B-Instruct-GGUF",
    ]

    primary = payload["recommendations"][0]
    assert primary["compatibility"] == "tight"
    assert primary["score"] == 26
    assert primary["score_breakdown"]["hard_penalty"] == 0.51
    assert primary["memory"]["inference_configuration"]["quantization"] == "Q4_K_M"
    assert primary["memory"]["memory"]["total_bytes"] == 8160437862

    second = payload["recommendations"][1]
    assert second["compatibility"] == "comfortable"
    assert second["score"] == 64
    assert second["memory"]["memory"]["total_bytes"] == 2147483648


def test_the_composite_score_really_does_disagree_with_the_order() -> None:
    """This fixture reproduces the confusion, so the fix is not cosmetic.

    ``quality`` compares capability before memory headroom, so the 7B leads on
    26/100 while the 3B sits second on 64/100. Printing those two numbers under
    the ranks, with nothing else, is what made the tool look broken. The order
    is right; the old presentation was not.
    """
    payload = report_to_dict(_run())
    scores = [rec["score"] for rec in payload["recommendations"]]

    assert scores != sorted(scores, reverse=True), (
        "the fixture no longer exercises the score/position disagreement"
    )
