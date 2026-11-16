"""Client configuration for a running server: what a coding agent needs to connect.

Pure functions; writing the file is the server manager's job.
"""

from __future__ import annotations

from typing import Any

from jaull.application.serving import policies
from jaull.application.serving.planner import ServerSizing

OPENCODE_PROVIDER_ID = "jaull"
OPENCODE_SCHEMA = "https://opencode.ai/config.json"
OPENCODE_CONFIG_FILENAME = "opencode.json"


def client_context_limit(sizing: ServerSizing) -> int:
    """The context a client may use without being able to end the other chats."""
    per_user = sizing.context_per_user
    if not sizing.unified_kv or sizing.total_context >= sizing.parallel * per_user:
        return per_user
    share = int(sizing.total_context * policies.SERVE_CLIENT_POOL_SHARE)
    step = policies.SERVE_CONTEXT_STEP_TOKENS
    return min(per_user, max(step, share // step * step))


def opencode_config(*, base_url: str, api_key: str, alias: str,
                    context_limit: int) -> dict[str, Any]:
    """An ``opencode.json`` that uses the server as an OpenAI-compatible provider."""
    return {
        "$schema": OPENCODE_SCHEMA,
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"Jaull ({alias})",
                "options": {"baseURL": base_url, "apiKey": api_key},
                "models": {
                    alias: {
                        "name": alias,
                        "limit": {
                            "context": context_limit,
                            "output": min(
                                policies.SERVE_CLIENT_OUTPUT_TOKENS,
                                int(context_limit * policies.SERVE_CLIENT_OUTPUT_MAX_SHARE),
                            ),
                        },
                    },
                },
            },
        },
        "model": f"{OPENCODE_PROVIDER_ID}/{alias}",
    }


__all__ = ["OPENCODE_CONFIG_FILENAME", "client_context_limit", "opencode_config"]
