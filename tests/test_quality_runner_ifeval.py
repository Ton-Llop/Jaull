"""The product bridge's IFEval profile. Synthetic files; the pilot is never executed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jaull.runtime import quality_eval_runner as runner
from jaull.runtime.quality_eval_runner import (
    PROFILE_CONTEXT,
    PROFILE_GRADE,
    QualityProfile,
    load_quality_setup,
    prepare_quality_setup,
    save_quality_setup,
)
from tests.test_cli_quality import _request


@pytest.fixture
def user_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "user-data"

    def user_data_dir(*parts: str) -> Path:
        return root.joinpath(*parts)

    monkeypatch.setattr(runner, "user_data_dir", user_data_dir)
    return root


def test_ifeval_is_a_full_profile_with_its_own_context() -> None:
    assert PROFILE_GRADE[QualityProfile.IFEVAL] == "full"
    assert PROFILE_CONTEXT[QualityProfile.IFEVAL] == 4096
    assert PROFILE_CONTEXT[QualityProfile.SMOKE] == 2048


@pytest.mark.parametrize("profile,suite", [
    (QualityProfile.SMOKE, "hellaswag"), (QualityProfile.IFEVAL, "ifeval"),
])
def test_setup_checks_the_dataset_and_image_of_the_profiles_suite(
    tmp_path: Path, user_data: Path, monkeypatch: pytest.MonkeyPatch,
    profile: QualityProfile, suite: str,
) -> None:
    commands: list[list[str]] = []

    def invoke(command: list[str], root: Path, log: Path, *args: Any) -> None:
        commands.append(command)
        report = Path(command[command.index("--output") + 1])
        report.write_text(json.dumps({"status": "ready", "image_id": "sha256:" + "0" * 64}))

    monkeypatch.setattr(runner, "_invoke", invoke)
    prepare_quality_setup(replace(_request(tmp_path), profile=profile))

    command, = commands
    assert command[command.index("--suite") + 1] == suite


def test_an_ifeval_run_never_overwrites_the_remembered_hellaswag_dataset(
    tmp_path: Path, user_data: Path,
) -> None:
    request = _request(tmp_path)
    save_quality_setup(request)
    hellaswag = load_quality_setup()["dataset"]

    ifeval = replace(request, profile=QualityProfile.IFEVAL,
                     dataset_file=tmp_path / "ifeval_input_data.jsonl")
    save_quality_setup(ifeval)

    assert load_quality_setup()["dataset"] == hellaswag
