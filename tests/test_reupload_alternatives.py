"""A model several repositories re-quantize is inspected once, but its siblings stay available."""

from __future__ import annotations

from jaull.application.recommendation.alternatives import rescue_with_reupload_alternatives
from jaull.application.requirements import build_requirements
from jaull.discovery import enrichment
from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
from jaull.domain.model import ModelAnalysis
from tests._workflow_fixtures import (
    GIB,
    answers,
    candidate,
    gguf_analysis,
    hardware,
    size_driven_estimator,
)

_ESTIMATOR = size_driven_estimator(vram_budget=24 * GIB)


def _evaluate(repo_id: str, analysis: ModelAnalysis) -> EvaluatedCandidate:
    requirements = build_requirements(answers(), hardware())
    return enrichment.evaluate_candidate(
        candidate(repo_id=repo_id), requirements, hardware(),
        inspect_fn=lambda _repo: analysis, estimate_fn=_ESTIMATOR,
    )


def _too_large(repo_id: str = "org/Popular-GGUF") -> EvaluatedCandidate:
    """The most downloaded copy publishes only a quantization that does not fit 24 GiB."""
    return _evaluate(repo_id, gguf_analysis(repo_id, ("Q8_0",), base_bytes=40 * GIB))


class _Inspector:
    def __init__(self, fitting: set[str]) -> None:
        self.fitting = fitting
        self.inspected: list[str] = []

    def __call__(self, item: ModelCandidate) -> ModelAnalysis:
        self.inspected.append(item.repo_id)
        base = 5 * GIB if item.repo_id in self.fitting else 40 * GIB
        return gguf_analysis(item.repo_id, ("Q4_K_M",), base_bytes=base)


def _rescue(
    evaluated: list[EvaluatedCandidate],
    siblings: dict[str, list[ModelCandidate]],
    inspector: _Inspector,
    max_inspections: int = 6,
) -> list[EvaluatedCandidate]:
    return rescue_with_reupload_alternatives(
        evaluated,
        alternatives=siblings,
        requirements=build_requirements(answers(), hardware()),
        hardware=hardware(),
        inspect_model=inspector,
        estimate_fn=_ESTIMATOR,
        check_cancelled=lambda: None,
        max_inspections=max_inspections,
    )


def test_a_sibling_is_evaluated_when_the_most_downloaded_copy_has_nothing_that_fits() -> None:
    """Found by an audit: sharing a base model does not mean sharing quantizations. The popular
    copy had no file that fits, the viable one was dropped for having fewer downloads."""
    popular = _too_large()
    siblings = {"org/popular-gguf": [candidate(repo_id="org/Viable-GGUF")]}
    inspector = _Inspector(fitting={"org/Viable-GGUF"})

    result = _rescue([popular], siblings, inspector)

    assert [item.repo_id for item in result] == ["org/Popular-GGUF", "org/Viable-GGUF"]
    assert inspector.inspected == ["org/Viable-GGUF"]


def test_siblings_are_left_alone_when_the_copy_already_has_a_viable_plan() -> None:
    fitting = _evaluate("org/Popular-GGUF", gguf_analysis("org/Popular-GGUF", ("Q4_K_M",)))
    inspector = _Inspector(fitting=set())

    result = _rescue([fitting], {"org/popular-gguf": [candidate(repo_id="org/Other-GGUF")]},
                     inspector)

    assert result == [fitting]
    assert inspector.inspected == []


def test_the_search_for_a_viable_sibling_stops_at_the_first_one_and_at_the_budget() -> None:
    siblings = {"org/popular-gguf": [
        candidate(repo_id=f"org/Sibling-{index}-GGUF") for index in range(5)
    ]}

    found = _Inspector(fitting={"org/Sibling-1-GGUF", "org/Sibling-2-GGUF"})
    _rescue([_too_large()], siblings, found)
    assert found.inspected == ["org/Sibling-0-GGUF", "org/Sibling-1-GGUF"]  # stops at the first

    hopeless = _Inspector(fitting=set())
    _rescue([_too_large()], siblings, hopeless, max_inspections=3)
    assert len(hopeless.inspected) == 3  # the shared inspection budget
