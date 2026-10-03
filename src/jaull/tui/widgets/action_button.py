"""Bracketed terminal actions; labels remain literal, never console markup."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Button

from jaull.tui.palette import ACCENT_DEEP


def _bracketed(label: str) -> Text:
    return Text.assemble(("[", ACCENT_DEEP), f" {label} ", ("]", ACCENT_DEEP))


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
        super().__init__(_bracketed(label), id=id, classes=classes, disabled=disabled)

    @property
    def action_label(self) -> str:
        return self._action_label

    def set_action_label(self, label: str) -> None:
        """Relabel without routing raw text through console markup."""
        self._action_label = label
        self.label = _bracketed(label)


__all__ = ["ActionButton"]
