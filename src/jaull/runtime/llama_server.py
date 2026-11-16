"""The llama-server command line Jaull starts for ``jaull serve``.

Security defaults (docs/TFG/disseny-setup.md): the API key is read from a file
so it never shows in the process list or the start script; the slot monitor and
the bundled chat page (web UI) is off unless asked for, and then it still needs
the API key; the slot monitor is on only for builds whose /slots answer was checked
to carry no conversation text; /metrics (aggregate counters, behind the API key)
feeds Jaull's server monitor; the log level stays at the default, because higher
verbosity writes the prompts to the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaull.domain.runtime import LlamaCppInstallation, RuntimeFlag
from jaull.runtime import policies

# Set from the server sizing and the security defaults; a launch flag with the
# same name (from the single-run planner) is ignored.
_SERVER_OWNED_FLAGS = frozenset({
    "--model", "--ctx-size", "--parallel", "--host", "--port", "--api-key",
    "--api-key-file", "--alias",
})


@dataclass(frozen=True)
class ServerLaunch:
    binary: Path
    model: Path
    parallel: int
    total_context: int
    context_per_user: int
    host: str
    port: int
    api_key_file: Path
    alias: str
    #: Placement flags from the launch planner (--n-gpu-layers, --device, ...).
    launch_flags: tuple[RuntimeFlag, ...] = ()
    tls_cert: Path | None = None
    tls_key: Path | None = None
    #: llama.cpp's own chat page at ``/``. Off by default.
    web_ui: bool = False
    #: Expose /slots (token counts per slot) to key holders. See ``slots_allowed``.
    slots: bool = False
    #: llama.cpp's prompt cache (on by default upstream). Off: nothing kept between requests.
    prompt_cache: bool = True
    #: Ceiling for the prompt cache in system RAM, in MiB; None leaves llama.cpp's default.
    cache_ram_mib: int | None = None

    def __post_init__(self) -> None:
        if (self.tls_cert is None) != (self.tls_key is None):
            raise ValueError("TLS needs both a certificate and a key file.")


def server_argv(launch: ServerLaunch) -> list[str]:
    argv = [
        str(launch.binary),
        "--model", str(launch.model),
        "--ctx-size", str(launch.total_context),
        "--parallel", str(launch.parallel),
    ]
    if launch.parallel > 1:
        argv += ["--kv-unified", "--kv-unified-per-slot", str(launch.context_per_user)]
    for flag in launch.launch_flags:
        if flag.name not in _SERVER_OWNED_FLAGS:
            argv += [flag.name, flag.value]
    argv += [
        "--host", launch.host,
        "--port", str(launch.port),
        "--api-key-file", str(launch.api_key_file),
        "--alias", launch.alias,
        "--metrics",
        "--slots" if launch.slots else "--no-slots",
    ]
    if not launch.web_ui:
        argv.append("--no-webui")
    if not launch.prompt_cache:
        # Both: the per-slot cache and the RAM copy of idle slots (8 GiB by default).
        argv += ["--no-cache-prompt", "--cache-ram", "0"]
    elif launch.cache_ram_mib is not None:
        argv += ["--cache-ram", str(launch.cache_ram_mib)]
    if launch.tls_cert is not None and launch.tls_key is not None:
        argv += ["--ssl-cert-file", str(launch.tls_cert), "--ssl-key-file", str(launch.tls_key)]
    return argv


def slots_allowed(build: str | None) -> bool:
    """Whether this build's /slots was verified to expose numbers only, never text."""
    return build is not None and build in policies.LLAMA_SERVER_SLOTS_VERIFIED_BUILDS


def server_binary_for(installation: LlamaCppInstallation) -> Path | None:
    """llama-server from the same build as the located llama-cli, if it is there."""
    if installation.llama_cli is None:
        return None
    cli = Path(installation.llama_cli)
    server = cli.with_name("llama-server" + cli.suffix)
    return server if server.is_file() else None


__all__ = ["ServerLaunch", "server_argv", "server_binary_for", "slots_allowed"]
