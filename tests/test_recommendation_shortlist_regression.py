"""The shortlist as a whole, not two candidates at a time.

Every other ranking test in this suite compares at most two candidates. That is
enough to prove each ordering rule in isolation, and it is exactly why a real
regression went unnoticed: on an RTX 4060 the displayed Top 5 put three
sub-1.5B models above a 4B, while every pairwise rule behind it was individually
correct. The bug was in the *emergent* order, and nothing looked at it.

So this file ranks a realistic spread in one call and pins the result. The
scenario mirrors the measured 4060 case: an 8B that only fits `tight`, a 4B that
fits `compatible` and whose model card declares no language, three small models
that fit `comfortable` with complete cards and far more downloads, and one model
that does not fit at all.

Everything is offline and deterministic — canned candidates, canned analyses and
the size-driven estimator — so the assertions below describe Jaull's policy and
never the state of the Hugging Face Hub on the day the test ran.
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
)
from jaull.domain.requirements import RecommendationPriority, UseCase
from jaull.workflow import orchestrator
from test_workflow_orchestrator import _container

# An 8 GiB card, like the machine the regression was found on.
VRAM_BYTES = 8 * GIB

# repo_id, artifact bytes, downloads, likes, declared languages.
#
# The sizes are chosen against ``size_driven_estimator``'s thresholds
# (comfortable <= 75 %, compatible <= 90 %, tight <= 100 % of the budget) so the
# shortlist spans every status that can reach the user.
#
# ``org/Medium-4B`` is the model under test: it is the most capable thing that
# actually fits, and it carries the two handicaps that used to sink it — an
# empty ``languages`` list, and fewer downloads than the small models.
SHORTLIST: tuple[tuple[str, int, int, int, list[str]], ...] = (
    ("org/Tiny-0.5B-Instruct-GGUF", int(0.6 * GIB), 8_000_000, 900, ["en"]),
    ("org/Mini-1.1B-Instruct-GGUF", int(1.1 * GIB), 1_500_000, 1_200, ["en"]),
    ("org/Small-1.5B-Instruct-GGUF", int(1.4 * GIB), 400_000, 300, ["en"]),
    ("org/Medium-4B-Instruct-GGUF", int(6.5 * GIB), 3_600_000, 500, []),
    ("org/Large-8B-Instruct-GGUF", int(7.6 * GIB), 300_000, 800, ["en"]),
    ("org/Giant-32B-Instruct-GGUF", int(20 * GIB), 100_000, 400, ["en"]),
)

SMALL_MODELS = frozenset(
    {
        "org/Tiny-0.5B-Instruct-GGUF",
        "org/Mini-1.1B-Instruct-GGUF",
        "org/Small-1.5B-Instruct-GGUF",
    }
)


def _shortlist(priority: RecommendationPriority) -> list[str]:
    """Run the whole guided workflow offline and return the displayed order."""
    search = FakeSearchClient(
        default=[
            candidate(
                repo_id=repo_id,
                tags=["text-generation", "instruct", "gguf"],
                downloads=downloads,
                likes=likes,
                languages=languages,
            )
            for repo_id, _bytes, downloads, likes, languages in SHORTLIST
        ]
    )
    analyses = {
        repo_id: gguf_analysis(
            repo_id=repo_id, quantizations=("Q4_K_M",), base_bytes=size
        )
        for repo_id, size, *_rest in SHORTLIST
    }
    state = orchestrator.run_workflow(
        answers(
            use_case=UseCase.GENERAL_CHAT,
            priority=priority,
            languages=["English"],
        ),
        hardware(vram_gib=8, ram_gib=32),
        _container(search, analyses=analyses, vram_budget=VRAM_BYTES),
    )
    return [recommendation.repo_id for recommendation in state.recommendations]


def test_balanced_shortlist_is_ordered_by_capability() -> None:
    """The golden order. Positions 1 and 2 are policy; 3-5 are a tie.

    All three small models land in the same capability bucket and fit equally
    comfortably, so their relative order falls through to the final tie-break in
    ``_ranking_key`` — ``repo_id``, alphabetically. Update this list without
    ceremony if that tie-break changes; the invariants below are the part that
    carries meaning.
    """
    assert _shortlist(RecommendationPriority.BALANCED) == [
        "org/Large-8B-Instruct-GGUF",
        "org/Medium-4B-Instruct-GGUF",
        "org/Mini-1.1B-Instruct-GGUF",
        "org/Small-1.5B-Instruct-GGUF",
        "org/Tiny-0.5B-Instruct-GGUF",
    ]


def test_a_capable_model_outranks_small_ones_with_better_cards() -> None:
    """The regression itself, stated in domain terms.

    ``org/Medium-4B`` declares no language and has fewer downloads than the
    0.5B. Neither is a capability signal, and neither may put a model with eight
    times the parameters below it.
    """
    order = _shortlist(RecommendationPriority.BALANCED)
    medium = order.index("org/Medium-4B-Instruct-GGUF")

    for repo_id in SMALL_MODELS:
        assert medium < order.index(repo_id), (
            f"{repo_id} outranked the 4B on Balanced: {order}"
        )


def test_a_tight_fit_still_wins_when_it_is_the_most_capable() -> None:
    """Spare VRAM is not a reward in Balanced once a plan fits at all."""
    assert _shortlist(RecommendationPriority.BALANCED)[0] == (
        "org/Large-8B-Instruct-GGUF"
    )


def test_memory_priority_inverts_the_same_shortlist() -> None:
    """The control that proves Balanced's order is a choice, not an accident.

    Same candidates, same hardware, one different answer: the order flips end to
    end. If a future change makes capability dominate everywhere, Balanced keeps
    passing and this fails.
    """
    assert _shortlist(RecommendationPriority.MEMORY) == [
        "org/Tiny-0.5B-Instruct-GGUF",
        "org/Mini-1.1B-Instruct-GGUF",
        "org/Small-1.5B-Instruct-GGUF",
        "org/Medium-4B-Instruct-GGUF",
        "org/Large-8B-Instruct-GGUF",
    ]


def test_a_model_that_does_not_fit_is_never_displayed() -> None:
    """20 GiB on an 8 GiB card, with RAM offload still short of it."""
    for priority in RecommendationPriority:
        assert "org/Giant-32B-Instruct-GGUF" not in _shortlist(priority)
