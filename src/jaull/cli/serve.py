"""``serve`` subcommands: run a model as a local OpenAI-compatible llama-server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionPlanningError
from jaull.application.model_reference import normalize_repo_id
from jaull.application.serving import ServerSizingError
from jaull.application.serving import policies as serving_policies
from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.artifacts.errors import ArtifactError
from jaull.domain.hardware import ComputeBackend
from jaull.exceptions import InvalidModelReferenceError, JaullError, QuantizationNotFoundError
from jaull.execution.server_process import ServerProcessError
from jaull.presentation.console import make_console
from jaull.presentation.serving import plan_lines, start_lines, status_lines


@dataclass(frozen=True)
class ServeOptions:
    quantization: str | None
    users: int = serving_policies.SERVE_DEFAULT_USERS
    context_per_user: int = serving_policies.SERVE_DEFAULT_CONTEXT_PER_USER
    #: Listen on the local network instead of this PC only.
    lan: bool = False
    port: int | None = None
    backend: ComputeBackend | None = None
    revision: str | None = None
    llama_server_path: Path | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None


def run_serve_start(
    reference: str,
    options: ServeOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()
    if (options.tls_cert is None) != (options.tls_key is None):
        _error(console, "TLS needs both --tls-cert and --tls-key.")
        return 2
    resolved = advisor or AdvisorService.default()
    try:
        repo_id = normalize_repo_id(reference)
        artifact = resolved.prepare_serve_artifact(
            repo_id, quantization=options.quantization, revision=options.revision,
        )
        plan = resolved.plan_server(
            repo_id=repo_id,
            artifact=artifact,
            users=options.users,
            context_per_user=options.context_per_user,
            backend=options.backend,
            llama_server_path=options.llama_server_path,
        )
        _print_plan(console, plan)
        started = resolved.start_server(
            plan, lan=options.lan, port=options.port,
            tls_cert=options.tls_cert, tls_key=options.tls_key,
        )
    except InvalidModelReferenceError as exc:
        _error(console, str(exc))
        return 2
    except QuantizationNotFoundError as exc:
        _error(console, str(exc))
        return 3
    except ArtifactError as exc:
        _error(console, str(exc))
        return 4
    except ServerProcessError as exc:
        _error(console, str(exc))
        return 5
    except (ServerSizingError, ExecutionPlanningError, JaullError) as exc:
        _error(console, str(exc))
        return 6

    _print_started(console, started)
    notice = resolved.llama_cpp_update_notice()
    if notice is not None:
        console.print(f"\nNote: {notice}", highlight=False)
    return 0


def run_serve_stop(advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        stopped = resolved.stop_server()
    except ServerProcessError as exc:
        _error(console, str(exc))
        return 2
    console.print("Server stopped." if stopped else "No Jaull server is running.")
    return 0


def run_serve_status(advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        status = resolved.server_status()
    except ServerProcessError as exc:
        _error(console, str(exc))
        return 2
    for line in status_lines(status):
        console.print(line, highlight=False, markup=False)
    if status.state is None or not status.running:
        return 1
    return 0 if status.healthy else 1


def _print_plan(console: Console, plan: ServerPlan) -> None:
    for line in plan_lines(plan):
        console.print(line, highlight=False, markup=False)


def _print_started(console: Console, started: ServerStart) -> None:
    console.print()
    for line in start_lines(started):
        console.print(line, highlight=False, markup=False, soft_wrap=True)


def _error(console: Console, message: str) -> None:
    # Log tails contain brackets; never read them as markup.
    console.print(message, style="red", highlight=False, markup=False)


__all__ = ["ServeOptions", "run_serve_start", "run_serve_status", "run_serve_stop"]
