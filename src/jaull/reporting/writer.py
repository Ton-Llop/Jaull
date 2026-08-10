"""Persist guided-run reports to disk.

Centralises the choice of output location so reports don't scatter across
whatever working directory happened to launch the TUI. Users can still pass
an explicit path; when they don't, the report lands in a per-user
``reports/`` folder under the jaull data root, next to ``models/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jaull.paths import user_data_dir
from jaull.reporting.recommendation import report_to_json, report_to_markdown
from jaull.workflow.state import RecommendationWorkflowState


def default_reports_dir() -> Path:
    """The per-user ``jaull/reports/`` directory (not created here)."""
    return user_data_dir("reports")


def default_report_name(now: datetime | None = None) -> str:
    """Timestamped filename so successive runs don't overwrite each other."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"jaull-report-{stamp}.json"


def write_recommendation_report(
    state: RecommendationWorkflowState,
    target: Path | None = None,
) -> list[Path]:
    """Write the JSON report and its Markdown twin. Returns both paths.

    ``target`` resolution:

    - ``None`` → ``<default_reports_dir>/<default_report_name>``.
    - Bare filename (no parent) → placed under ``default_reports_dir``.
    - Anything else (relative with a parent, or absolute) is respected as-is.

    The ``.json`` suffix is enforced; the Markdown twin uses the same stem
    with ``.md``. Parent directories are created on demand.
    """
    resolved = _resolve_target(target)
    json_path = (
        resolved if resolved.suffix == ".json" else resolved.with_suffix(".json")
    )
    markdown_path = json_path.with_suffix(".md")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report_to_json(state), encoding="utf-8")
    markdown_path.write_text(report_to_markdown(state), encoding="utf-8")
    return [json_path, markdown_path]


def _resolve_target(target: Path | None) -> Path:
    if target is None:
        return default_reports_dir() / default_report_name()
    # A bare filename ``report.json`` has ``parent == Path('.')``; treat that
    # as "use the default dir" so users of the TUI never end up with reports
    # dropped in random working directories.
    if target.parent == Path("."):
        return default_reports_dir() / target.name
    return target


__all__ = [
    "default_report_name",
    "default_reports_dir",
    "write_recommendation_report",
]
