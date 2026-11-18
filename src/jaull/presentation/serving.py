"""Plain-text lines describing a model server, shared by ``jaull serve`` and the TUI Setup."""

from __future__ import annotations

import os
import re
import shlex
import subprocess

from jaull.application.serving.models import ServerPlan, ServerStart
from jaull.application.serving.security import (
    LAN_WITHOUT_TLS_WARNING,
    SELF_SIGNED_CERTIFICATE_NOTE,
    SHARED_POOL_OVERFLOW_WARNING,
    SLOTS_ARE_NOT_USERS,
    WSL_NOT_REACHABLE_WARNING,
)
from jaull.execution.server_certificate import ServerCertificate
from jaull.execution.server_process import ServerState, ServerStatus
from jaull.presentation.console import format_bytes
from jaull.runtime.policies import LLAMA_SERVER_LAN_HOST


def plan_lines(plan: ServerPlan) -> list[str]:
    sizing = plan.sizing
    lines = [
        f"Serving {plan.alias} ({plan.artifact.filename}) with llama-server "
        f"({plan.backend.value}).",
    ]
    outdated = plan.artifact.outdated
    if outdated is not None:
        lines.append(
            "  Warning: Hugging Face has a newer version of this file. This copy is an older "
            f"published one, still current on {outdated.still_published_on:%Y-%m-%d}. It is "
            "intact, but it may not behave like the one downloaded today."
        )
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
    if plan.load_mode is not None:
        lines.append(
            f"  The model file is read with -lm {plan.load_mode}"
            + (": it is not also copied into system RAM." if plan.load_mode == "dio" else ".")
        )
    if plan.sleep_idle_seconds is not None:
        lines.append(
            f"  The model leaves the GPU after {plan.sleep_idle_seconds} s without requests "
            "(the first one after a pause takes longer). Change it with --sleep-idle, 0 turns "
            "it off."
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


def _opencode_install_lines(started: ServerStart) -> list[str]:
    """What was done to the opencode on this PC, so nothing changes without being said."""
    install = started.opencode
    if install is None:
        return []
    if install.outcome == "unreadable":
        return [
            f"opencode: {install.path} was left as it is (it has comments or a shape Jaull "
            "does not edit). Add the config above to it yourself.",
        ]
    if install.outcome == "unchanged":
        return [
            f"opencode: {install.path} already has this server.",
            *_certificate_trust_lines(started),
        ]
    verb = {"created": "Created", "added": "Added the jaull provider to",
            "updated": "Updated the jaull provider in"}[install.outcome]
    lines = [
        f"opencode: {verb} {install.path}. Your other providers and settings are as they were."
    ]
    if install.backup is not None:
        lines.append(f"  The file as it was before: {install.backup}")
    lines.append(
        "  It is your default model now." if install.selected else
        "  Your default model is unchanged: pick the server with /models in opencode."
    )
    return lines + _certificate_trust_lines(started)


# What each flag Jaull gives llama-server means, in words for someone who has never seen one.
# (meaning, whether the flag is followed by a value). A flag missing here is shown without text.
FLAG_MEANINGS: dict[str, tuple[str, bool]] = {
    "--model": ("the model file", True),
    "--ctx-size": ("memory for all the chats together, in tokens", True),
    "--parallel": ("chats served at the same time", True),
    "--kv-unified": ("the chats share one pool of memory", False),
    "--kv-unified-per-slot": ("the most one chat can use, in tokens", True),
    "--n-gpu-layers": ("layers of the model on the graphics card (-1: all)", True),
    "--device": ("which graphics card runs it", True),
    "--flash-attn": ("lighter reading of long chats (on, off or auto)", True),
    "--host": ("who can connect (127.0.0.1: only this PC)", True),
    "--port": ("the door the server listens on", True),
    "--api-key-file": ("the password file (the password is never shown)", True),
    "--alias": ("the name the model answers to", True),
    "--metrics": ("lets Jaull read the live counters", False),
    "--slots": ("lets Jaull see how full each chat is", False),
    "--no-slots": ("keeps the per-chat figures hidden", False),
    "--no-webui": ("no built-in chat page", False),
    "--no-cache-prompt": ("remembers nothing between turns", False),
    "--cache-ram": ("memory that may keep past conversations, in MiB", True),
    "-lm": ("how the file is read; dio keeps it out of RAM", True),
    "--sleep-idle-seconds": ("frees the graphics card after this many idle seconds", True),
    "--ssl-cert-file": ("the HTTPS certificate", True),
    "--ssl-key-file": ("the HTTPS private key", True),
}
_SHORTEN_FROM = 34
_FLAG_COLUMN = 46


def _short(value: str) -> str:
    """A long path as its last part, so the lines stay readable; the copy is not shortened."""
    if len(value) <= _SHORTEN_FROM or ("/" not in value and "\\" not in value):
        return value
    separator = "\\" if "\\" in value else "/"
    return f"…{separator}{value.replace(chr(92), '/').rsplit('/', 1)[-1]}"


def command_breakdown(argv: list[str]) -> list[str]:
    """The command one flag per line, each with what it means."""
    if not argv:
        return []
    lines = [f"  {_short(argv[0]):<{_FLAG_COLUMN}}  the program that runs the model"]
    index = 1
    while index < len(argv):
        flag = argv[index]
        meaning, takes_value = FLAG_MEANINGS.get(flag, ("", None))
        has_next = index + 1 < len(argv)
        if takes_value is None:  # unknown: a value is whatever does not look like a flag
            takes_value = has_next and not argv[index + 1].startswith("-")
        if takes_value and has_next:
            text = f"{flag} {_short(argv[index + 1])}"
            index += 2
        else:
            text = flag
            index += 1
        lines.append(f"    {text:<{_FLAG_COLUMN}}  {meaning}".rstrip())
    return lines


def full_command(state: ServerState) -> str | None:
    """The whole command as one line to copy. The key is never in it, only its file."""
    if not state.argv:
        return None
    return subprocess.list2cmdline(state.argv) if os.name == "nt" else shlex.join(state.argv)


def _command_lines(state: ServerState, *, explained: bool = False) -> list[str]:
    """What was run, for whoever wants to see or repeat it."""
    command = full_command(state)
    if command is None:
        return []
    if explained:
        return ["", "How it was started", *command_breakdown(state.argv)]
    return ["Command that was run:", f"  {command}"]


def opencode_run_command(state: ServerState) -> str | None:
    """How to run opencode so it trusts this server's own certificate; None without HTTPS."""
    argv = state.argv
    if not state.tls or "--ssl-cert-file" not in argv:
        return None
    return f"NODE_EXTRA_CA_CERTS={argv[argv.index('--ssl-cert-file') + 1]} opencode"


def _running_on_windows() -> bool:
    return os.name == "nt"


def wsl_run_command(state: ServerState, *, windows: bool | None = None) -> str | None:
    """The same command for opencode inside WSL, where the certificate is at /mnt/<drive>/...

    Only for a Windows server that other systems can reach with HTTPS: WSL does not see the
    PC's 127.0.0.1, and a drive-letter path means nothing to a Linux program.
    """
    on_windows = _running_on_windows() if windows is None else windows
    argv = state.argv
    if (not on_windows or not state.tls or state.host != LLAMA_SERVER_LAN_HOST
            or "--ssl-cert-file" not in argv):
        return None
    drive = re.match(r"^([A-Za-z]):[\\/](.*)$", argv[argv.index("--ssl-cert-file") + 1])
    if drive is None:
        return None
    path = drive.group(2).replace("\\", "/")
    return f"NODE_EXTRA_CA_CERTS=/mnt/{drive.group(1).lower()}/{path} opencode"


def _certificate_trust_lines(started: ServerStart) -> list[str]:
    command = opencode_run_command(started.state)
    if command is None:
        return []
    lines = ["  Run opencode with this command, so it trusts the server's certificate:",
             f"    {command}"]
    wsl = wsl_run_command(started.state)
    if wsl is not None:
        lines += ["  From inside WSL, where that path means nothing, use this one:",
                  f"    {wsl}"]
    return lines


def start_lines(started: ServerStart, *, explained: bool = False) -> list[str]:
    """Everything the person needs after a start. The API key appears only when new."""
    state = started.state
    reach = started.lan_url or state.url
    if started.web_ui:
        # The page is what people are given; the API is for other tools, so it comes second.
        lines = [f"Ready. Chat page: {reach}/  (enter the API key in its settings)",
                 f"API for other tools: {reach}/v1"]
    else:
        lines = [f"Ready at {state.url}  (OpenAI API: {state.url}/v1)"]
        if started.lan_url is not None:
            lines.append(f"On the local network: {started.lan_url}/v1")
    if started.lan_url is not None:
        if not state.tls:
            lines.append(f"Warning: {LAN_WITHOUT_TLS_WARNING}")
        lines.append(f"Note: {SLOTS_ARE_NOT_USERS}")
    if started.wsl:
        lines.append(f"Warning: {WSL_NOT_REACHABLE_WARNING}")
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
    lines += _opencode_install_lines(started)
    lines += [f"Start script: {started.start_script}", "Stop it with: jaull serve stop"]
    lines += _command_lines(state, explained=explained)
    if started.firewall_rule is not None:
        lines += [
            "",
            "Jaull does not change the firewall. To let other PCs on the local network "
            "connect, an administrator can run:",
            f"  {started.firewall_rule}",
        ]
    return lines


def status_lines(status: ServerStatus, *, explained: bool = False) -> list[str]:
    if status.state is None or not status.running:
        return ["No Jaull server is running."]
    state = status.state
    health = "ready" if status.healthy else "loading or not responding"
    return [
        f"{state.alias}: {health}",
        f"  URL:     {state.url}",
        f"  PID:     {state.pid} (started {state.started_at})",
        f"  Log:     {state.log_path}",
        *_command_lines(state, explained=explained),
    ]


__all__ = [
    "FLAG_MEANINGS",
    "command_breakdown",
    "full_command",
    "opencode_run_command",
    "plan_lines",
    "start_lines",
    "status_lines",
    "wsl_run_command",
]
