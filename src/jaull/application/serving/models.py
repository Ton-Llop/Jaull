"""What ``serve start`` decides before starting, and what it reports afterwards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaull.application.serving.planner import ContextLimit, ServerSizing
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import RuntimeRecommendation
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
    #: The largest context per person this machine can serve with this model.
    context_limit: ContextLimit | None = None


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


__all__ = ["ServerPlan", "ServerStart"]
