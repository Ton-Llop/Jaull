"""The opencode configuration Jaull writes when it starts a server."""

from __future__ import annotations

from types import SimpleNamespace

from jaull.application.serving.client_config import client_context_limit, opencode_config
from jaull.application.serving.policies import (
    SERVE_CLIENT_OUTPUT_TOKENS,
    SERVE_CLIENT_POOL_SHARE,
)


def _sizing(parallel: int, per_user: int, total: int) -> SimpleNamespace:
    return SimpleNamespace(parallel=parallel, context_per_user=per_user, total_context=total,
                           unified_kv=parallel > 1)


def test_one_user_or_a_pool_that_cannot_overflow_gives_the_full_context() -> None:
    assert client_context_limit(_sizing(1, 32768, 32768)) == 32768  # type: ignore[arg-type]
    assert client_context_limit(_sizing(4, 32768, 4 * 32768)) == 32768  # type: ignore[arg-type]


def test_a_shared_pool_leaves_room_for_the_other_chats() -> None:
    # Biosfer: 8 people, one pool of 128K. opencode gets a share, so it cannot fill the
    # pool alone and end everyone's chat (docs/TFG/concurrencia-b60.md).
    limit = client_context_limit(_sizing(8, 131072, 131072))  # type: ignore[arg-type]
    assert limit == int(131072 * SERVE_CLIENT_POOL_SHARE)
    # Never above what one person may reach.
    capped = client_context_limit(_sizing(8, 32768, 131072))  # type: ignore[arg-type]
    assert capped <= 32768


def test_opencode_config_points_at_the_server_with_its_key_and_model() -> None:
    config = opencode_config(base_url="http://192.168.1.107:8090/v1", api_key="KEY123",
                             alias="Qwen3.6-35B-A3B-GGUF", context_limit=65536)

    provider = config["provider"]["jaull"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {"baseURL": "http://192.168.1.107:8090/v1",
                                   "apiKey": "KEY123"}
    model = provider["models"]["Qwen3.6-35B-A3B-GGUF"]
    assert model["limit"] == {"context": 65536, "output": SERVE_CLIENT_OUTPUT_TOKENS}
    assert config["model"] == "jaull/Qwen3.6-35B-A3B-GGUF"


def test_a_small_context_keeps_most_of_it_for_the_prompt() -> None:
    from jaull.application.serving.policies import SERVE_CLIENT_OUTPUT_MAX_SHARE

    config = opencode_config(base_url="http://x/v1", api_key="k", alias="m", context_limit=4096)
    limit = config["provider"]["jaull"]["models"]["m"]["limit"]
    assert limit == {"context": 4096, "output": int(4096 * SERVE_CLIENT_OUTPUT_MAX_SHARE)}
