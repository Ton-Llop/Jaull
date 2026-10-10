"""The step 6 matrix script: probes degrade honestly and the report says what it saw."""

from __future__ import annotations

import pytest
from scripts.shadow_review_matrix import PartialReplay, forbidden, markdown

from jaull.domain.candidates import SearchQuery


def test_an_uncaptured_probe_query_finds_nothing_and_is_named() -> None:
    replay = PartialReplay([])
    query = SearchQuery(label="general_chat:lang-es", search="es")
    assert replay.search(query) == []
    assert replay.missing == ["general_chat:lang-es"]


def test_the_report_marks_synthetic_hardware_and_cases_without_results() -> None:
    report = {
        "at": "now", "mode": "replay_metadata_only", "quality_records": 2,
        "limitations": ["synthetic"],
        "cases": [
            {"hardware": "cpu", "hardware_kind": "synthetic", "task": "chat",
             "probe": "quality", "applicable_quality": 1, "pool_plans": 4, "groups": 0,
             "moves": [], "top5_changes": [],
             "first_fallback_reasons": {"Quality: no evidence for this artifact.": 3}},
            {"hardware": "cpu", "hardware_kind": "synthetic", "task": "chat",
             "probe": "chat in Spanish", "no_results": ["none"]},
        ],
    }
    text = markdown(report)
    assert "| cpu (synthetic) | chat | quality | 1/4 | 0 | 0 | 0 |" in text
    assert "| cpu | chat | chat in Spanish | no results" in text
    assert "- 3 x Quality: no evidence for this artifact." in text


def test_the_matrix_refuses_downloads_and_evaluations() -> None:
    with pytest.raises(RuntimeError, match="must not download"):
        forbidden()
