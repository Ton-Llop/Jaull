"""Progress reporting for long-running workflow stages.

The orchestrator emits progress through a plain callback so it stays independent
of Textual: the TUI adapts the callback into `call_from_thread` updates, and
tests just append the events to a list.
"""

from __future__ import annotations

import contextlib
from typing import Protocol

from local_ai_check.workflow.models import (
    ProgressStep,
    StepStatus,
    WorkflowProgress,
)

# --------------------------------------------------------------------------
# Step catalogues. Labels are user-facing, so they avoid jargon.
# --------------------------------------------------------------------------
HARDWARE_STEPS: tuple[tuple[str, str], ...] = (
    ("os", "Detecting operating system"),
    ("cpu", "Reading CPU and memory"),
    ("gpu", "Detecting GPU"),
    ("storage", "Checking available storage"),
    ("profile", "Building hardware profile"),
)

DISCOVERY_STEPS: tuple[tuple[str, str], ...] = (
    ("queries", "Building search queries"),
    ("search", "Searching Hugging Face"),
    ("filter", "Filtering candidates"),
    ("inspect", "Inspecting shortlisted candidates"),
    ("estimate", "Estimating compatible configurations"),
    ("rank", "Ranking recommendations"),
)


def initial_progress(steps: tuple[tuple[str, str], ...]) -> WorkflowProgress:
    return WorkflowProgress(
        steps=[ProgressStep(key=key, label=label) for key, label in steps]
    )


class ProgressCallback(Protocol):
    """Receives every progress update the orchestrator produces.

    Implementations must be cheap and must not raise: the orchestrator calls
    this from a worker thread and treats reporting as best-effort.
    """

    def __call__(self, progress: WorkflowProgress) -> None: ...


class ProgressReporter:
    """Accumulates step transitions and pushes each new snapshot to a callback."""

    def __init__(
        self,
        steps: tuple[tuple[str, str], ...],
        callback: ProgressCallback | None = None,
    ) -> None:
        self._progress = initial_progress(steps)
        self._callback = callback

    @property
    def progress(self) -> WorkflowProgress:
        return self._progress

    def start(self, key: str, detail: str | None = None) -> None:
        self._set(key, StepStatus.RUNNING, detail)

    def done(self, key: str, detail: str | None = None) -> None:
        self._set(key, StepStatus.DONE, detail)

    def fail(self, key: str, detail: str | None = None) -> None:
        self._set(key, StepStatus.FAILED, detail)

    def skip(self, key: str, detail: str | None = None) -> None:
        self._set(key, StepStatus.SKIPPED, detail)

    def detail(self, key: str, detail: str) -> None:
        """Update the sub-label of a running step (e.g. "7 of 12 inspected")."""
        self._set(key, StepStatus.RUNNING, detail)

    def _set(self, key: str, status: StepStatus, detail: str | None) -> None:
        self._progress = self._progress.with_step(key, status, detail)
        if self._callback is None:
            return
        # Reporting is best-effort: a broken display must never abort a run.
        with contextlib.suppress(Exception):
            self._callback(self._progress)


__all__ = [
    "DISCOVERY_STEPS",
    "HARDWARE_STEPS",
    "ProgressCallback",
    "ProgressReporter",
    "initial_progress",
]
