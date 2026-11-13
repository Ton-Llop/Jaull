"""The llama-server command line Jaull starts for ``jaull serve``.

Security defaults (docs/TFG/disseny-setup.md): the API key is read from a file
so it never shows in the process list or the start script; the slot monitor and
the bundled chat page (web UI) is off unless asked for, and then it still needs
the API key; the log level stays at the default, because higher
verbosity writes the prompts to the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaull.domain.runtime import LlamaCppInstallation, RuntimeFlag

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
        "--no-slots",
    ]
    if not launch.web_ui:
        argv.append("--no-webui")
    if launch.tls_cert is not None and launch.tls_key is not None:
        argv += ["--ssl-cert-file", str(launch.tls_cert), "--ssl-key-file", str(launch.tls_key)]
    return argv


def server_binary_for(installation: LlamaCppInstallation) -> Path | None:
    """llama-server from the same build as the located llama-cli, if it is there."""
    if installation.llama_cli is None:
        return None
    cli = Path(installation.llama_cli)
    server = cli.with_name("llama-server" + cli.suffix)
    return server if server.is_file() else None


__all__ = ["ServerLaunch", "server_argv", "server_binary_for"]
