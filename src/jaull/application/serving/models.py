"""What ``serve start`` decides before starting, and what it reports afterwards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaull.application.serving.planner import ContextLimit, ServerSizing
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeBackendSelection, RuntimeRecommendation
from jaull.execution.opencode_install import OpencodeInstall
from jaull.execution.server_process import ServerState


@dataclass(frozen=True)
class ServerPlan:
    artifact: ModelArtifact
    alias: str
    backend: ComputeBackend
    sizing: ServerSizing
    #: Placement from the launch planner (--n-gpu-layers, --device, ...).
    launch: RuntimeRecommendation
    server_binary: Path
    backend_selection: RuntimeBackendSelection
    server_sha256: str
    #: The largest context per person this machine can serve with this model.
    context_limit: ContextLimit | None = None
    #: The llama.cpp build of ``server_binary`` when known (``b11889``).
    build: str | None = None
    #: Ceiling for llama.cpp's prompt cache in system RAM; 0 keeps it off.
    prompt_cache_bytes: int | None = None
    #: Free system RAM when the plan was made, to explain the ceiling.
    ram_available_bytes: int | None = None
    #: Idle seconds after which llama-server releases the model from the GPU; None keeps it.
    sleep_idle_seconds: int | None = None
    #: How llama-server reads the model file (``-lm``); None leaves llama.cpp's default.
    load_mode: str | None = None


@dataclass(frozen=True)
class FoundLlamaServer:
    """A llama-server on this PC that Jaull did not install."""

    path: Path
    version: str | None
    #: Its capability probe confirms the chosen backend. A build that does not is refused.
    supports_backend: bool


@dataclass(frozen=True)
class ServerStart:
    state: ServerState
    api_key_path: Path
    #: Only when the key was created by this start; it is shown once.
    new_api_key: str | None
    start_script: Path
    #: For the administrator to run when the server is shared; never applied.
    firewall_rule: str | None
    lan_url: str | None
    #: True when llama.cpp's chat page is served at the root URL.
    web_ui: bool = False
    #: The opencode configuration written for this server; it holds the API key.
    client_config: Path | None = None
    #: Shared from inside WSL: nothing else on the network can reach it.
    wsl: bool = False
    #: What was done to the opencode on this PC; None when it has none or was not asked.
    opencode: OpencodeInstall | None = None


@dataclass(frozen=True)
class ServerShare:
    """Where other devices reach a running server, and what sharing it needs."""

    #: This PC's network address when shared, the loopback address when not.
    base_url: str
    #: llama.cpp's chat page, when it is served.
    chat_url: str | None
    shared: bool
    #: For the administrator to run when shared; never applied by Jaull.
    firewall_rule: str | None
    #: Running inside WSL: other devices cannot reach this address.
    wsl: bool = False


__all__ = ["FoundLlamaServer", "ServerPlan", "ServerShare", "ServerStart"]
