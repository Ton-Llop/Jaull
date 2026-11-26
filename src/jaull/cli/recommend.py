"""`jaull recommend`: the guided search without the TUI.

With ``--json`` it prints one JSON object per line: ``progress`` events while the
search runs and a final ``result`` (the same report the TUI exports) or ``error``.
A front end can read it line by line and cancel by ending the process.
"""

from __future__ import annotations

from jaull.advisor.service import AdvisorService
from jaull.cli.json_lines import emit
from jaull.domain.requirements import UserAnswers
from jaull.presentation.console import make_console
from jaull.reporting.recommendation import report_to_dict
from jaull.workflow.models import StepStatus, WorkflowProgress


class _ProgressLines:
    """Emits the current step, skipping snapshots that repeat the previous one."""

    def __init__(self) -> None:
        self._last: tuple[str, str, str | None] | None = None

    def __call__(self, progress: WorkflowProgress) -> None:
        step = next((s for s in progress.steps if s.key == progress.current_key), None)
        if step is None:
            return
        key = (step.key, step.status.value, step.detail)
        if key == self._last:
            return
        self._last = key
        done = sum(1 for s in progress.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        emit(
            "progress",
            step=step.key,
            label=step.label,
            status=step.status.value,
            detail=step.detail,
            done=done,
            total=len(progress.steps),
            # Every step, so a front end can draw the whole checklist as the TUI does.
            steps=[
                {"key": s.key, "label": s.label, "status": s.status.value, "detail": s.detail}
                for s in progress.steps
            ],
        )


def run_recommend(
    answers: UserAnswers,
    *,
    advisor: AdvisorService | None = None,
    as_json: bool = False,
) -> int:
    resolved = advisor or AdvisorService.default()
    if not as_json:
        console = make_console()
        with console.status("Searching for models that fit this computer…"):
            state = resolved.recommend(answers)
        if state.failed or not state.recommendations:
            reasons = state.errors or state.no_results_reason or ["No recommendation."]
            for reason in reasons:
                console.print(f"[red]{reason}[/red]")
            return 1
        for rec in state.recommendations:
            console.print(
                f"{rec.rank}. {rec.repo_id}  ({rec.displayed_status.value}, "
                f"confidence {rec.confidence.value})"
            )
        return 0

    try:
        state = resolved.recommend(answers, on_progress=_ProgressLines())
    except Exception as exc:  # reported to the front end, not hidden
        emit("error", message=str(exc) or type(exc).__name__)
        return 1
    emit("result", report=report_to_dict(state))
    return 0 if not state.failed else 1
