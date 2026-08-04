"""Application-layer facade for CLI and TUI.

``AdvisorService`` gathers the four operations the front-ends need — hardware
scan, model inspection, memory estimation, environment diagnostics — plus the
guided run, behind a single object so screens never construct HF clients or
call low-level services directly.
"""

from jaull.advisor.service import AdvisorService

__all__ = ["AdvisorService"]
