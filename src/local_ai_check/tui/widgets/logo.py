from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

_LOGO_TOP = r"""
██╗      ██████╗  ██████╗ █████╗ ██╗      █████╗ ██╗
██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔══██╗██║
██║     ██║   ██║██║     ███████║██║     ███████║██║
██║     ██║   ██║██║     ██╔══██║██║     ██╔══██║██║
███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║  ██║██║
╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝
""".strip("\n")

_LOGO_BOTTOM = r"""
 ██████╗██╗  ██╗███████╗ ██████╗██╗  ██╗
██╔════╝██║  ██║██╔════╝██╔════╝██║ ██╔╝
██║     ███████║█████╗  ██║     █████╔╝
██║     ██╔══██║██╔══╝  ██║     ██╔═██╗
╚██████╗██║  ██║███████╗╚██████╗██║  ██╗
 ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝
""".strip("\n")


class Logo(Vertical):
    """Big two-line ASCII logo used on the Home screen only."""

    DEFAULT_CLASSES = "logo"

    def compose(self) -> ComposeResult:
        yield Static(_LOGO_TOP, classes="logo-ascii")
        yield Static(_LOGO_BOTTOM, classes="logo-ascii")
