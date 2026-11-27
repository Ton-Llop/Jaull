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


def test_recommend_command_builds_the_answers_the_tui_wizard_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The desktop app asks the TUI's questions; the command turns them into the same answers.
    from typer.testing import CliRunner

    from jaull.cli.app import app
    from jaull.domain.requirements import CommercialUse, DocumentScale

    seen: list[UserAnswers] = []
    monkeypatch.setattr("jaull.cli.app.run_recommend",
                        lambda answers, **_: seen.append(answers) or 0)
    runner = CliRunner()

    result = runner.invoke(app, [
        "recommend", "--use-case", "document_qa", "--language", "English",
        "--other-language", "french", "--commercial-use", "no", "--json",
    ])
    assert result.exit_code == 0, result.output
    answers = seen[-1]
    assert answers.languages == ["English"] and answers.other_languages == ["fr"]
    assert answers.document_scale is DocumentScale.MEDIUM  # the wizard's default for documents
    assert answers.commercial_use is CommercialUse.NO

    runner.invoke(app, ["recommend", "--use-case", "coding", "--document-scale", "long"])
    assert seen[-1].document_scale is None  # only asked for document work


def test_recommend_json_reports_a_failure_as_an_error_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    advisor = _Advisor(error=RuntimeError("Hugging Face is not reachable"))
    assert run_recommend(ANSWERS, advisor=advisor, as_json=True) == 1  # type: ignore[arg-type]

    lines = _lines(capsys)
    assert lines == [
        {"schema_version": 1, "event": "error", "message": "Hugging Face is not reachable"}
    ]
