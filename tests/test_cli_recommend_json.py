from __future__ import annotations

import json
from typing import Any

import pytest

from jaull.cli.recommend import run_recommend
from jaull.domain.requirements import RecommendationPriority, UseCase, UserAnswers
from jaull.workflow.models import ProgressStep, StepStatus, WorkflowProgress
from jaull.workflow.state import RecommendationWorkflowState

ANSWERS = UserAnswers(use_case=UseCase.CODING, priority=RecommendationPriority.BALANCED)


def _progress(status: StepStatus) -> WorkflowProgress:
    return WorkflowProgress(
        steps=[
            ProgressStep(key="hardware", label="Hardware", status=StepStatus.DONE),
            ProgressStep(key="search", label="Searching", status=status, detail="3 of 10"),
        ],
        current_key="search",
    )


class _Advisor:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.answers: UserAnswers | None = None

    def recommend(self, answers: UserAnswers, *, on_progress: Any = None) -> Any:
        self.answers = answers
        if self.error is not None:
            raise self.error
        on_progress(_progress(StepStatus.RUNNING))
        on_progress(_progress(StepStatus.RUNNING))  # repeated snapshot: not emitted twice
        return RecommendationWorkflowState(answers=answers)


def _lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_recommend_json_streams_progress_then_the_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A front end reads these lines one by one and can stop the search by ending the process.
    advisor = _Advisor()
    assert run_recommend(ANSWERS, advisor=advisor, as_json=True) == 0  # type: ignore[arg-type]

    lines = _lines(capsys)
    assert [line["event"] for line in lines] == ["progress", "result"]
    assert lines[0]["step"] == "search" and lines[0]["done"] == 1 and lines[0]["total"] == 2
    assert lines[0]["detail"] == "3 of 10"
    assert [(s["key"], s["status"]) for s in lines[0]["steps"]] == [
        ("hardware", "done"), ("search", "running"),
    ]
    assert lines[1]["report"]["recommendations"] == []
    assert advisor.answers == ANSWERS


def test_recommend_json_reports_a_failure_as_an_error_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    advisor = _Advisor(error=RuntimeError("Hugging Face is not reachable"))
    assert run_recommend(ANSWERS, advisor=advisor, as_json=True) == 1  # type: ignore[arg-type]

    lines = _lines(capsys)
    assert lines == [
        {"schema_version": 1, "event": "error", "message": "Hugging Face is not reachable"}
    ]
