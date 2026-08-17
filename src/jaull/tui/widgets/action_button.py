"""A secondary action, written the way a terminal writes one.

``[ Compare ]`` rather than a filled rectangle. The brackets are the affordance,
so the button needs no face, no border and no second row — which is what makes
a toolbar of four actions read as four actions instead of as four slabs. The
primary action stays a filled accent chip, so a row still has one obvious
entry point.

The brackets live here rather than at the call sites for one specific reason:
Textual parses a button label as console markup, so a literal ``[`` has to be
escaped. A call site that forgets renders an empty button — silently, because
markup swallows the unknown tag. Wrapping it once means no call site can get
it wrong.
"""

from __future__ import annotations

from textual.widgets import Button

from jaull.tui.palette import ACCENT_DEEP


def _bracketed(label: str) -> str:
    """`[ label ]`, with the brackets escaped and tinted."""
    return f"[{ACCENT_DEEP}]\\[[/] {label} [{ACCENT_DEEP}]][/]"


class ActionButton(Button):
    """A non-primary action. Use `Button(classes="-primary")` for the primary."""

    DEFAULT_CLASSES = "action"

    def __init__(
        self,
        label: str,
        *,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        self._action_label = label
        super().__init__(
            _bracketed(label),
            id=id,
            classes=classes,
            disabled=disabled,
        )

    @property
    def action_label(self) -> str:
        """The label without its brackets."""
        return self._action_label

    def set_action_label(self, label: str) -> None:
        """Relabel without losing the brackets or re-escaping by hand."""
        self._action_label = label
        self.label = _bracketed(label)


__all__ = ["ActionButton"]
