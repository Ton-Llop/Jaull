"""Plain-text lines describing a model server, shared by ``jaull serve`` and the TUI Setup."""

from __future__ import annotations

from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.application.serving.security import (
    LAN_WITHOUT_TLS_WARNING,
    SELF_SIGNED_CERTIFICATE_NOTE,
    SHARED_POOL_OVERFLOW_WARNING,
    SLOTS_ARE_NOT_USERS,
)
from jaull.execution.server_certificate import ServerCertificate
from jaull.execution.server_process import ServerStatus
from jaull.presentation.console import format_bytes


def plan_lines(plan: ServerPlan) -> list[str]:
    sizing = plan.sizing
    lines = [
        f"Serving {plan.alias} ({plan.artifact.filename}) with llama-server "
        f"({plan.backend.value}).",
    ]
    if sizing.unified_kv and sizing.total_context == sizing.context_per_user:
        lines.append(
            f"  {sizing.parallel} users at once sharing one pool of {sizing.total_context} "
            "tokens. One person can use all of it, several split it as they use it."
        )
    elif sizing.unified_kv:
        lines.append(
            f"  {sizing.parallel} users at once, up to {sizing.context_per_user} tokens each, "
            f"sharing a pool of {sizing.total_context} tokens."
        )
    else:
        lines.append(f"  1 user, {sizing.context_per_user} tokens of context.")
    if sizing.unified_kv and sizing.total_context < sizing.parallel * sizing.context_per_user:
        lines.append(f"  {SHARED_POOL_OVERFLOW_WARNING}")
    limit = plan.context_limit
    if limit is not None and limit.largest_per_user is not None:
        native = f" (the model was trained for {limit.native})" if limit.native else ""
        lines.append(
            f"  Largest context per person on this machine: {limit.largest_per_user} "
            f"tokens{native}."
        )
    if plan.prompt_cache_bytes is not None:
        free = (f" ({format_bytes(plan.ram_available_bytes)} of RAM free now)"
                if plan.ram_available_bytes is not None else "")
        lines.append(
            f"  Prompt cache in RAM: up to {format_bytes(plan.prompt_cache_bytes)}{free}."
            if plan.prompt_cache_bytes else
            f"  Prompt cache in RAM: off, not enough free RAM{free}."
        )
    fit = sizing.estimate.hardware_fit
    physical = fit.gpu_physical_bytes if fit is not None else None
    if physical is not None and sizing.gpu_budget_bytes is not None:
        lines.append(
            f"  Predicted GPU memory: {format_bytes(physical)} "
            f"(serving limit {format_bytes(sizing.gpu_budget_bytes)})."
        )
    return lines


def certificate_lines(certificate: ServerCertificate) -> list[str]:
    """What `jaull serve cert` reports: where the files are and how to use them."""
    lines = [
        "Created a self-signed certificate." if certificate.created
        else "Already exists, kept as it is. Use --renew to replace it.",
        f"Certificate: {certificate.cert_path}",
        f"Private key: {certificate.key_path}",
        f"It is valid until {certificate.expires:%Y-%m-%d} for: {', '.join(certificate.names)}",
    ]
    if certificate.missing:
        lines.append(
            f"It does not cover {', '.join(certificate.missing)}, this PC's address or name "
            "now. Run `jaull serve cert --renew` to include it; clients will have to trust "
            "the new certificate."
        )
    lines += [
        f"Use it with: jaull serve start ... --lan --tls-cert {certificate.cert_path} "
        f"--tls-key {certificate.key_path}",
        SELF_SIGNED_CERTIFICATE_NOTE,
    ]
    return lines


def start_lines(started: ServerStart) -> list[str]:
    """Everything the person needs after a start. The API key appears only when new."""
    state = started.state
    lines = [f"Ready at {state.url}  (OpenAI API: {state.url}/v1)"]
    if started.lan_url is not None:
        lines.append(f"On the local network: {started.lan_url}/v1")
        if not state.tls:
            lines.append(f"Warning: {LAN_WITHOUT_TLS_WARNING}")
        lines.append(f"Note: {SLOTS_ARE_NOT_USERS}")
    if started.web_ui:
        page = started.lan_url or state.url
        lines.append(f"Chat page: {page}/  (enter the API key in its settings)")
    if started.new_api_key is not None:
        lines += [f"API key (shown once, stored in {started.api_key_path}):",
                  f"  {started.new_api_key}"]
    else:
        lines.append(f"API key: in {started.api_key_path}")
    if started.client_config is not None:
        lines.append(f"opencode config (it contains the API key): {started.client_config}")
        lines.append(
            "  Copy it to ~/.config/opencode/opencode.json on the PC that runs opencode, "
            "not into a project folder."
            if started.lan_url is not None else
            "  The server listens only on this PC, so it only works on this PC. Start with "
            "--lan to use it from another one."
        )
    lines += [f"Start script: {started.start_script}", "Stop it with: jaull serve stop"]
    if started.firewall_rule is not None:
        lines += [
            "",
            "Jaull does not change the firewall. To let other PCs on the local network "
            "connect, an administrator can run:",
            f"  {started.firewall_rule}",
        ]
    return lines


def status_lines(status: ServerStatus) -> list[str]:
    if status.state is None or not status.running:
        return ["No Jaull server is running."]
    state = status.state
    health = "ready" if status.healthy else "loading or not responding"
    return [
        f"{state.alias}: {health}",
        f"  URL:     {state.url}",
        f"  PID:     {state.pid} (started {state.started_at})",
        f"  Log:     {state.log_path}",
    ]


__all__ = ["plan_lines", "start_lines", "status_lines"]
