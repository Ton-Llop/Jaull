from __future__ import annotations

from local_ai_check.exceptions import (
    HuggingFaceUnavailableError,
    InvalidModelReferenceError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)
from local_ai_check.huggingface.repository import inspect_model
from local_ai_check.huggingface.url_parser import normalize_repo_id
from local_ai_check.presentation.console import make_console
from local_ai_check.presentation.model_report import render_model


def run_inspect(reference: str) -> int:
    console = make_console()

    try:
        repo_id = normalize_repo_id(reference)
    except InvalidModelReferenceError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    try:
        analysis = inspect_model(repo_id)
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
