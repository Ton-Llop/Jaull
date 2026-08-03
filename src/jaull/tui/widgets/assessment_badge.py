from __future__ import annotations

from textual.widgets import Static

from jaull.domain.estimation import CompatibilityStatus

_LABELS: dict[CompatibilityStatus, tuple[str, str]] = {
    CompatibilityStatus.COMFORTABLE: ("Comfortable", "badge-comfortable"),
    CompatibilityStatus.COMPATIBLE: ("Compatible", "badge-compatible"),
    CompatibilityStatus.TIGHT: ("Tight fit", "badge-tight"),
    CompatibilityStatus.OFFLOADING_REQUIRED: ("Offloading required", "badge-offloading"),
    CompatibilityStatus.INSUFFICIENT: ("Insufficient", "badge-insufficient"),
    CompatibilityStatus.UNKNOWN: ("Unknown", "badge-unknown"),
}


class AssessmentBadge(Static):
    """A single-line coloured badge for a CompatibilityStatus."""

    def __init__(self, status: CompatibilityStatus) -> None:
        label, style_class = _LABELS[status]
        super().__init__(f" {label} ", classes=f"badge {style_class}")
