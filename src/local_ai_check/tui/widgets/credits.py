from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

_ASCII_BANNER = r"""
 ┏╸┏╸  ╻  ┏━╸┏━╸┏━┓╻     ┏━┓╻   ┏━╸╻ ╻┏━╸┏━╸╻┏
        ┃  ┃ ┃┃  ┣━┫┃     ┣━┫┃   ┃  ┣━┫┣╸ ┃  ┣┻┓
   ╺┛  ┗━╸┗━┛┗━╸╹ ╹┗━╸╺━ ╹ ╹╹   ┗━╸╹ ╹┗━╸┗━╸╹ ╹
""".strip("\n")


class CreditsPanel(Vertical):
    """Small footer banner shown at the bottom of the Home screen."""

    DEFAULT_CLASSES = "credits"

    def compose(self) -> ComposeResult:
        yield Static(_ASCII_BANNER, classes="credits-title")
        yield Static("Made by Ton — Trabajo de Fin de Grado")
        yield Static("MIT License · https://huggingface.co", classes="credits-subtle")
