"""IFEval candidate selection from the saved pool. Synthetic models; no Hub/Docker/GPU."""

from __future__ import annotations

from typing import Any

import pytest

from jaull.advisor.quality_candidates import prepare_candidates
from jaull.domain.execution_plans import ArtifactVariantFormat
from jaull.domain.model import ModelConfig
from jaull.domain.recommendation import (
    HardConstraint,
    HardConstraintCode,
    PlanAssessment,
)
from jaull.domain.requirements import UseCase
from jaull.recommendation.engine_v2 import RankedPlan
from jaull.runtime.quality_eval_runner import QualityProfile
from tests.test_quality_candidates import SelectionAdvisor

IFEVAL = QualityProfile.IFEVAL


def _ranked(rec: Any, plan: Any = None, *, rejected: bool = False) -> RankedPlan:
    plan = plan or rec.plan
    constraints = [HardConstraint(code=HardConstraintCode.LICENSE_INCOMPATIBLE,
                                  message="Synthetic rejection")] if rejected else []
    return RankedPlan(rec.evaluated, plan,
                      PlanAssessment(plan_id=plan.plan_id, hard_constraints=constraints))


def _english_chat(advisor: SelectionAdvisor, **changes: Any) -> Any:
    state = advisor.state()
    requirements = state.requirements.model_copy(update={"languages": ["en"], **changes})
    # A real search keeps its ranked pool; selection reads it, never re-ranks.
    pool = tuple(_ranked(rec) for rec in advisor.models)
    return state.model_copy(update={"requirements": requirements, "ranked_plans": pool})


def _as_safetensors(plan: Any) -> Any:
    """The same logical model shown through a Transformers path, as in a real search."""
    return plan.model_copy(update={
        "plan_id": plan.plan_id + "-safetensors",
        "artifact": plan.artifact.model_copy(update={
            "format": ArtifactVariantFormat.SAFETENSORS, "filename": None, "sha256": None,
        }),
    })


@pytest.fixture
def no_reranking(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("IFEval selection must not re-rank the search pool")
    monkeypatch.setattr("jaull.advisor.quality_candidates.service.recommend", forbidden)


def test_ifeval_measures_the_plans_the_search_showed_in_their_order(no_reranking) -> None:
    advisor = SelectionAdvisor()
    contexts: list[int] = []
    estimate = advisor.estimate_model

    def recording(analysis: Any, hw: Any, cfg: Any, **kw: Any) -> Any:
        contexts.append(cfg.context_length)
        return estimate(analysis, hw, cfg, **kw)

    advisor.estimate_model = recording  # type: ignore[method-assign]
    state = _english_chat(advisor)
    before = state.model_dump_json()

    result = prepare_candidates(advisor, state, profile=IFEVAL)  # type: ignore[arg-type]

    shown = [rec.plan for rec in state.recommendations]
    assert [c.plan.plan_id for c in result.candidates] == [plan.plan_id for plan in shown[:3]]
    # Same bytes as the shown plans, so a result can attach to them on Recompare.
    assert [c.plan.artifact.sha256 for c in result.candidates] == [
        plan.artifact.sha256 for plan in shown[:3]
    ]
    assert 4096 in contexts and 2048 not in contexts
    assert sum("not offered" in notice for notice in result.notices) == 2
    assert result.considered == 5
    assert state.model_dump_json() == before


@pytest.mark.parametrize("changes", [
    {"languages": ["es"]}, {"languages": ["en", "ca"]}, {"use_case": UseCase.CODING},
])
def test_ifeval_is_offered_only_to_english_chat_searches(no_reranking, changes) -> None:
    advisor = SelectionAdvisor()
    result = prepare_candidates(  # type: ignore[arg-type]
        advisor, _english_chat(advisor, **changes), profile=IFEVAL,
    )
    assert result.candidates == ()
    assert result.notices == ("IFEval applies only to English general-chat requests.",)


def test_published_bytes_other_than_the_planned_ones_are_refused(no_reranking) -> None:
    advisor = SelectionAdvisor()
    first = advisor.models[0].repo_id
    original = advisor._artifacts

    def drifted() -> Any:
        artifacts = original()
        resolve = artifacts.resolver.resolve

        def resolve_drift(repo: str, **kwargs: Any) -> Any:
            artifact = resolve(repo, **kwargs)
            return artifact.model_copy(update={"sha256": "e" * 64}) if repo == first else artifact
        artifacts.resolver.resolve = resolve_drift
        return artifacts

    advisor._artifacts = drifted  # type: ignore[method-assign]
    result = prepare_candidates(  # type: ignore[arg-type]
        advisor, _english_chat(advisor), profile=IFEVAL,
    )
    assert first not in {c.plan.artifact.repo_id for c in result.candidates}
    # check_selected_artifact compares the digest; a result could never attach otherwise.
    assert any("does not match the selected artifact" in notice for notice in result.notices)


def test_a_plan_without_a_digest_is_explained_not_measured(no_reranking) -> None:
    advisor = SelectionAdvisor()
    state = _english_chat(advisor)
    first = state.recommendations[0]
    undigested_plan = first.plan.model_copy(update={
        "artifact": first.plan.artifact.model_copy(update={"sha256": None}),
    })
    undigested = first.model_copy(update={"plan": undigested_plan})
    # Neither the shown plan nor its copy in the pool has a digest to attach to.
    state = state.model_copy(update={
        "recommendations": [undigested, *state.recommendations[1:]],
        "ranked_plans": (_ranked(first, undigested_plan), *state.ranked_plans[1:]),
    })

    result = prepare_candidates(advisor, state, profile=IFEVAL)  # type: ignore[arg-type]

    assert first.repo_id not in {c.plan.artifact.repo_id for c in result.candidates}
    assert any("no single-file GGUF path with a published digest" in notice
               for notice in result.notices)


def test_a_shown_safetensors_model_is_measured_through_its_gguf_path_in_the_pool(
    no_reranking,
) -> None:
    advisor = SelectionAdvisor()
    state = _english_chat(advisor)
    first = state.recommendations[0]
    gguf = first.plan
    shown = first.model_copy(update={"plan": _as_safetensors(gguf)})
    state = state.model_copy(update={"recommendations": [shown, *state.recommendations[1:]]})

    result = prepare_candidates(advisor, state, profile=IFEVAL)  # type: ignore[arg-type]

    measured = result.candidates[0]
    # The GGUF plan of the pool, by its own id and bytes - never the shown plan's.
    assert measured.plan.plan_id == gguf.plan_id
    assert measured.plan.artifact.sha256 == gguf.artifact.sha256
    assert measured.reason.startswith("GGUF path of the shown model, not its shown")
    assert "attaches to this path" in measured.reason


def test_only_an_accepted_gguf_path_of_the_same_model_is_offered(no_reranking) -> None:
    advisor = SelectionAdvisor()
    state = _english_chat(advisor)
    first = state.recommendations[0]
    shown = first.model_copy(update={"plan": _as_safetensors(first.plan)})
    pool = (
        _ranked(first, rejected=True),        # Same model, but rejected: never offered.
        *(_ranked(rec) for rec in advisor.models[1:]),
    )
    state = state.model_copy(update={
        "recommendations": [shown, *state.recommendations[1:]], "ranked_plans": pool,
    })

    result = prepare_candidates(advisor, state, profile=IFEVAL)  # type: ignore[arg-type]

    # Another model's GGUF never stands in for it: the next shown models get their own.
    assert [c.plan.plan_id for c in result.candidates] == [
        rec.plan.plan_id for rec in state.recommendations[1:4]
    ]
    assert any(first.repo_id in notice and "no single-file GGUF path" in notice
               for notice in result.notices)


def test_a_search_without_a_saved_pool_is_explained(no_reranking) -> None:
    advisor = SelectionAdvisor()
    state = _english_chat(advisor).model_copy(update={"ranked_plans": ()})

    result = prepare_candidates(advisor, state, profile=IFEVAL)  # type: ignore[arg-type]

    assert result.candidates == ()
    assert result.notices == ("This search kept no plan pool; run it again to evaluate.",)


def test_a_model_whose_context_cannot_hold_the_protocol_is_refused_before_download(
    no_reranking,
) -> None:
    advisor = SelectionAdvisor()
    first = advisor.models[0].repo_id
    inspect = advisor.inspect_model

    def short_context(repo: str) -> Any:
        analysis = inspect(repo)
        if repo != first:
            return analysis
        # TinyLlama-like: 2048 positions cannot hold the prompt plus 1280 tokens at 4096.
        return analysis.model_copy(update={"config": ModelConfig(max_position_embeddings=2048)})

    advisor.inspect_model = short_context  # type: ignore[method-assign]
    result = prepare_candidates(  # type: ignore[arg-type]
        advisor, _english_chat(advisor), profile=IFEVAL,
    )
    assert first not in {c.plan.artifact.repo_id for c in result.candidates}
    assert any("model context 2048 is below the evaluation context 4096" in notice
               for notice in result.notices)


def _stored_run(artifact: str, *, reply: str | None = None, audited: bool = True) -> dict:
    from jaull.evaluation.quality_records import digest
    from tests.test_ifeval_applicability import _full_record

    value = _full_record()
    value["identity"]["artifact_sha256"] = artifact
    if not audited:
        value["identity"]["runtime"]["fingerprint"] = "another-build"
    value["identity_sha256"] = digest(value["identity"])
    if reply is not None:
        sample = value["result"]["samples"]["jaull-ifeval-chat-v1"][0]
        sample["resps"], sample["filtered_resps"] = [[reply]], [reply]
    return value


def test_selection_reads_the_store_exactly_as_the_shadow_does(no_reranking) -> None:
    advisor = SelectionAdvisor()
    sha = [rec.plan.artifact.sha256 for rec in advisor.models]
    runs = [
        _stored_run(sha[0]),                                   # One audited result.
        _stored_run(sha[1]), _stored_run(sha[1], reply="No."),  # Repeats that disagree.
        _stored_run(sha[2], audited=False),                    # Full, but off the audited pins.
    ]
    for order in (runs, runs[::-1]):
        advisor._stored_quality_records = lambda order=order: order  # type: ignore[method-assign]
        result = prepare_candidates(  # type: ignore[arg-type]
            advisor, _english_chat(advisor), profile=IFEVAL,
        )
        first, second, third = (c.stored for c in result.candidates)
        assert first.applicable is not None and not first.conflict
        # A conflict shows no value at all, so store order can never pick one.
        assert second.applicable is None and second.conflict and second.distinct == 2
        assert third.applicable is None and not third.conflict  # Reference only.
        assert [c.settled for c in result.candidates] == [True, True, False]
