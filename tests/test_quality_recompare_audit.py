"""Audit output uses the saved pool and does not leak record payloads or mutate it."""

import pytest
from scripts.audit_quality_recompare import forbidden, summarize

from jaull.workflow.state import RecommendationWorkflowState
from tests.test_ifeval_applicability import _full_record
from tests.test_quality_recompare import _state


def test_audit_summarizes_real_recompare_contract_without_changing_the_search() -> None:
    state, _ = _state()
    before = state.model_dump_json()
    report = summarize(state, [_full_record(), _full_record(stronger=True)])
    assert report["original_unchanged"] and report["store_order_invariant"]
    assert report["shadow"]["moves"]
    assert len(report["plans"]) == 2
    assert state.model_dump_json() == before
    assert "result" not in report  # No raw responses/HTTP transcripts in the audit.


def test_audit_rejects_an_empty_search_and_accidental_execution() -> None:
    with pytest.raises(ValueError, match="did not produce a pool"):
        summarize(RecommendationWorkflowState(), [])
    with pytest.raises(AssertionError, match="must not download"):
        forbidden()


def test_audit_claims_hold_under_python_optimize(monkeypatch) -> None:
    from scripts import audit_quality_recompare as audit

    state, _ = _state()
    # A recompare that mutates the search must fail even where asserts are stripped.
    def mutating(ranked, requirements, records, *, limit):
        state.warnings.append("mutated")
        return real(ranked, requirements, records, limit=limit)

    real = audit.recompare_quality
    monkeypatch.setattr(audit, "recompare_quality", mutating)
    with pytest.raises(RuntimeError, match="mutated the original search"):
        summarize(state, [_full_record()])
