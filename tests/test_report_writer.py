from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jaull.reporting.writer import (
    default_report_name,
    default_reports_dir,
    write_recommendation_report,
)
from jaull.workflow.state import RecommendationWorkflowState


def _state() -> RecommendationWorkflowState:
    # Empty state is still a valid input to the writer — it produces a report
    # that says "no recommendations", which is exactly what we want to
    # exercise here without dragging in the whole guided-workflow fixtures.
    return RecommendationWorkflowState()


def test_default_report_name_is_timestamped() -> None:
    name = default_report_name(datetime(2026, 8, 4, 12, 30, 45, tzinfo=UTC))
    assert name == "jaull-report-20260804-123045.json"


def test_default_reports_dir_ends_in_jaull_reports() -> None:
    path = default_reports_dir()
    assert path.parts[-2:] == ("jaull", "reports")


def test_explicit_absolute_path_is_respected(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "run-a.json"
    written = write_recommendation_report(_state(), target)

    assert written[0] == target
    assert written[1] == target.with_suffix(".md")
    assert target.is_file()
    assert target.with_suffix(".md").is_file()


def test_bare_filename_lands_in_default_reports_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "jaull.reporting.writer.default_reports_dir", lambda: tmp_path
    )
    written = write_recommendation_report(_state(), Path("run-b.json"))

    assert written[0] == tmp_path / "run-b.json"
    assert (tmp_path / "run-b.md").is_file()


def test_no_target_uses_default_dir_and_timestamped_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "jaull.reporting.writer.default_reports_dir", lambda: tmp_path
    )
    written = write_recommendation_report(_state(), None)

    assert written[0].parent == tmp_path
    assert written[0].name.startswith("jaull-report-")
    assert written[0].suffix == ".json"


def test_non_json_extension_gets_corrected(tmp_path: Path) -> None:
    written = write_recommendation_report(_state(), tmp_path / "out.txt")

    assert written[0] == tmp_path / "out.json"
    assert written[1] == tmp_path / "out.md"


def test_relative_path_with_parent_is_respected(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "jaull.reporting.writer.default_reports_dir",
        lambda: tmp_path / "should_not_be_used",
    )
    written = write_recommendation_report(_state(), Path("nested/report.json"))

    assert written[0] == Path("nested/report.json")
    assert Path("nested/report.md").is_file()
