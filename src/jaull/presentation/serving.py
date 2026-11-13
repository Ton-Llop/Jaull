"""Plain-text lines describing a model server, shared by ``jaull serve`` and the TUI Setup."""

from __future__ import annotations

from jaull.application.serving.models import ServerPlan, ServerStart
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
    limit = plan.context_limit
    if limit is not None and limit.largest_per_user is not None:
        native = f" (the model was trained for {limit.native})" if limit.native else ""
        lines.append(
            f"  Largest context per person on this machine: {limit.largest_per_user} "
            f"tokens{native}."
        )
    fit = sizing.estimate.hardware_fit
    physical = fit.gpu_physical_bytes if fit is not None else None
    if physical is not None and sizing.gpu_budget_bytes is not None:
        lines.append(
            f"  Predicted GPU memory: {format_bytes(physical)} "
            f"(serving limit {format_bytes(sizing.gpu_budget_bytes)})."
        )
    return lines


def start_lines(started: ServerStart) -> list[str]:
    """Everything the person needs after a start. The API key appears only when new."""
    state = started.state
    lines = [f"Ready at {state.url}  (OpenAI API: {state.url}/v1)"]
    if started.lan_url is not None:
        lines.append(f"On the local network: {started.lan_url}/v1")
    if started.web_ui:
        page = started.lan_url or state.url
        lines.append(f"Chat page: {page}/  (enter the API key in its settings)")
    if started.new_api_key is not None:
        lines += [f"API key (shown once, stored in {started.api_key_path}):",
                  f"  {started.new_api_key}"]
    else:
        lines.append(f"API key: in {started.api_key_path}")
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
