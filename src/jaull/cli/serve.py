"""``serve`` subcommands: run a model as a local OpenAI-compatible llama-server."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionPlanningError, RuntimeMissingError
from jaull.application.model_reference import normalize_repo_id
from jaull.application.serving import ServerSizingError
from jaull.application.serving import policies as serving_policies
from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.artifacts.errors import ArtifactError
from jaull.cli.json_lines import emit
from jaull.domain.hardware import ComputeBackend
from jaull.exceptions import InvalidModelReferenceError, JaullError, QuantizationNotFoundError
from jaull.execution.server_process import ServerProcessError
from jaull.presentation.console import make_console
from jaull.presentation.serving import certificate_lines, plan_lines, start_lines, status_lines


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
    #: Serve llama.cpp's chat page at the root URL (off by default).
    web_ui: bool = False
    #: One pool of ``context_per_user`` tokens for everyone, split as it is used.
    shared_context: bool = False
    #: llama.cpp's prompt cache. Off: more private, much slower on long chats.
    prompt_cache: bool = True
    #: Seconds idle before the model leaves the GPU; None = the default for this GPU, 0 = off.
    sleep_idle_seconds: int | None = None
    #: Add the server to the opencode on this PC (keeping what it has).
    opencode_config: bool = True
    #: How llama-server reads the model (-lm); None = decided by Jaull, "default" = llama.cpp's.
    load_mode: str | None = None
    #: Encrypt with HTTPS. None: yes when sharing on the network, no for this PC alone.
    https: bool | None = None
    #: JSON lines on stdout (progress, plan, then ``started`` or ``error``) for a front end.
    json: bool = False


def run_serve_start(
    reference: str,
    options: ServeOptions,
    advisor: AdvisorService | None = None,
) -> int:
    console = make_console()

    def fail(code: int, message: str, **detail: str) -> int:
        if options.json:
            # ``reason`` (and its details) let a front end act without parsing the message.
            emit("error", message=message, code=code, **detail)
        else:
            _error(console, message)
        return code

    if (options.tls_cert is None) != (options.tls_key is None):
        return fail(2, "TLS needs both --tls-cert and --tls-key.")
    if options.https is False and options.tls_cert is not None:
        return fail(2, "--no-https cannot be combined with --tls-cert and --tls-key.")
    resolved = advisor or AdvisorService.default()
    on_progress: Callable[[str], None] | None = (
        (lambda message: emit("progress", message=message)) if options.json else None
    )
    try:
        repo_id = normalize_repo_id(reference)
        artifact = resolved.prepare_serve_artifact(
            repo_id, quantization=options.quantization, revision=options.revision,
            on_progress=on_progress,
        )
        plan = resolved.plan_server(
            repo_id=repo_id,
            artifact=artifact,
            users=options.users,
            context_per_user=options.context_per_user,
            backend=options.backend,
            llama_server_path=options.llama_server_path,
            shared_context=options.shared_context,
            sleep_idle_seconds=options.sleep_idle_seconds,
            load_mode=options.load_mode,
        )
        if options.json:
            emit("plan", lines=plan_lines(plan))
            emit("progress", message="Starting llama-server")
        else:
            _print_plan(console, plan)
        tls_cert, tls_key = options.tls_cert, options.tls_key
        https = options.https if options.https is not None else options.lan
        if https and tls_cert is None:
            certificate = resolved.create_server_certificate()
            tls_cert, tls_key = certificate.cert_path, certificate.key_path
            made = "Created a self-signed certificate" if certificate.created else (
                "Using the certificate from `jaull serve cert`")
            if not options.json:
                console.print(f"{made}, valid until {certificate.expires:%Y-%m-%d}.",
                              highlight=False, markup=False)
        started = resolved.start_server(
            plan, lan=options.lan, port=options.port,
            tls_cert=tls_cert, tls_key=tls_key, web_ui=options.web_ui,
            prompt_cache=options.prompt_cache, opencode=options.opencode_config,
        )
    except InvalidModelReferenceError as exc:
        return fail(2, str(exc))
    except QuantizationNotFoundError as exc:
        return fail(3, str(exc))
    except ArtifactError as exc:
        return fail(4, str(exc))
    except ServerProcessError as exc:
        return fail(5, str(exc))
    except RuntimeMissingError as exc:
        return fail(6, str(exc), reason="runtime_missing", backend=exc.backend)
    except (ServerSizingError, ExecutionPlanningError, JaullError) as exc:
        return fail(6, str(exc))

    if options.json:
        # The key itself is never printed here: a front end reads it from api_key_path.
        emit(
            "started",
            alias=started.state.alias,
            url=started.state.url,
            lan_url=started.lan_url,
            api_key_path=str(started.api_key_path),
            web_ui=started.web_ui,
            # With HTTPS, a client trusts this certificate (it covers 127.0.0.1 too).
            tls_cert=str(tls_cert) if started.state.tls and tls_cert is not None else None,
            firewall_rule=started.firewall_rule,
            users=plan.sizing.parallel,
            context_per_user=plan.sizing.context_per_user,
            total_context=plan.sizing.total_context,
            shared_context=options.shared_context,
        )
        return 0
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


def run_serve_cert(*, renew: bool, advisor: AdvisorService | None = None) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        certificate = resolved.create_server_certificate(renew=renew)
    except ServerProcessError as exc:
        _error(console, str(exc))
        return 2
    for line in certificate_lines(certificate):
        console.print(line, highlight=False, markup=False)
    return 0


def run_serve_status(advisor: AdvisorService | None = None, *, as_json: bool = False) -> int:
    console = make_console()
    resolved = advisor or AdvisorService.default()
    try:
        status = resolved.server_status()
    except ServerProcessError as exc:
        if as_json:
            sys.stdout.write(json.dumps({"schema_version": 1, "error": str(exc)}) + "\n")
        else:
            _error(console, str(exc))
        return 2
    ready = status.state is not None and status.running and status.healthy
    if as_json:
        state = status.state
        cert = None
        if state is not None and state.tls and "--ssl-cert-file" in state.argv:
            index = state.argv.index("--ssl-cert-file") + 1
            if index < len(state.argv):
                cert = state.argv[index]
        payload = {
            "schema_version": 1,
            "running": status.running,
            "healthy": status.healthy,
            "alias": state.alias if state else None,
            "url": state.url if state else None,
            "api_key_path": str(resolved.server_api_key_path()),
            "tls_cert": cert,
            # A chat client keeps its history within this (None in states saved before it).
            "context_per_user": state.context_per_user if state else None,
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0 if ready else 1
    for line in status_lines(status):
        console.print(line, highlight=False, markup=False)
    return 0 if ready else 1


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


__all__ = [
    "ServeOptions", "run_serve_cert", "run_serve_start", "run_serve_status", "run_serve_stop",
]
