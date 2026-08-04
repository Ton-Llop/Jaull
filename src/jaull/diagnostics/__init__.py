"""Environment diagnostics.

Pure inspection of the local Python/HF/GPU/cache setup. Rendering (CLI Rich
table, TUI screen) lives in ``jaull.cli.doctor`` and ``jaull.tui.screens.doctor``
so this package stays free of any presentation concern.
"""

from jaull.diagnostics.service import collect_diagnostics

__all__ = ["collect_diagnostics"]
