from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from jaull.cli.app import app
from jaull.domain.experiments import (
    ExperimentReevaluationResult,
    ExperimentReplayability,
    ExperimentReplayabilityStatus,
)


def test_experiments_reevaluate_command_outputs_json(
    monkeypatch: Any,
) -> None:
    fake = _FakeAdvisor(
        ExperimentReevaluationResult(
            experiment_id="exp-test",
            replayability=ExperimentReplayability(
                status=ExperimentReplayabilityStatus.NOT_REPRODUCIBLE,
                reasons=["missing prediction input snapshot"],
            ),
        )
    )
    monkeypatch.setattr("jaull.cli.experiments.AdvisorService.default", lambda: fake)

    result = CliRunner().invoke(
        app,
        ["experiments", "reevaluate", "exp-test", "--json"],
        catch_exceptions=False,
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert fake.requested == ["exp-test"]
    assert payload["experiment_id"] == "exp-test"
    assert payload["mode"] == "re_evaluation"
    assert payload["replayability"]["status"] == "not_reproducible"


class _FakeAdvisor:
    def __init__(self, result: ExperimentReevaluationResult) -> None:
        self._result = result
        self.requested: list[str] = []

    def reevaluate_experiment(self, experiment_id: str) -> ExperimentReevaluationResult:
        self.requested.append(experiment_id)
        return self._result
