from __future__ import annotations

from jaull.advisor.service import AdvisorService
from jaull.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from jaull.huggingface.url_parser import normalize_repo_id
from jaull.presentation.console import make_console
from jaull.presentation.model_report import render_model


def run_inspect(reference: str, advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()

    try:
        repo_id = normalize_repo_id(reference)
    except InvalidModelReferenceError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    try:
        analysis = resolved.inspect_model(repo_id)
    except ModelNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 3
    except ModelAccessDeniedError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "[dim]Set HF_TOKEN in your environment if you have been granted access.[/dim]"
        )
        return 4
    except HuggingFaceUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        return 5

    render_model(analysis, console)
    return 0
