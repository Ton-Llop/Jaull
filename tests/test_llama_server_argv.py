"""llama-server command line: sizing, launch flags and the security defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaull.domain.runtime import (
    LlamaCppInstallation,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeSource,
)
from jaull.runtime.llama_server import ServerLaunch, server_argv, server_binary_for


def _flag(name: str, value: str) -> RuntimeFlag:
    return RuntimeFlag(name=name, value=value, source=RuntimeFlagSource.POLICY, explanation="x")


def _launch(**changes: object) -> ServerLaunch:
    values: dict[str, object] = {
        "binary": Path("/opt/llama/llama-server"),
        "model": Path("/models/qwen.gguf"),
        "parallel": 1,
        "total_context": 32768,
        "context_per_user": 32768,
        "host": "127.0.0.1",
        "port": 8090,
        "api_key_file": Path("/data/serve/api-key"),
        "alias": "qwen",
    }
    values.update(changes)
    return ServerLaunch(**values)  # type: ignore[arg-type]


def _native(path: str) -> str:
    """The argv carries OS-native paths (backslashes on Windows), as subprocess expects."""
    return str(Path(path))


def _pairs(argv: list[str]) -> dict[str, str]:
    return {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("-")}


def test_single_user_server_is_local_and_locked_down() -> None:
    argv = server_argv(_launch())

    assert argv[0] == _native("/opt/llama/llama-server")
    pairs = _pairs(argv)
    assert pairs["--model"] == _native("/models/qwen.gguf")
    assert pairs["--ctx-size"] == "32768"
    assert pairs["--parallel"] == "1"
    assert pairs["--host"] == "127.0.0.1"
    assert pairs["--port"] == "8090"
    assert pairs["--api-key-file"] == _native("/data/serve/api-key")
    assert pairs["--alias"] == "qwen"
    assert "--no-slots" in argv
    assert "--no-webui" in argv
    assert "--kv-unified" not in argv
    # The key itself never appears on the command line, and prompts are never logged.
    assert "--api-key" not in argv
    assert not {"-lv", "--verbosity", "--log-verbosity", "-v", "--verbose"} & set(argv)


def test_several_users_share_one_kv_pool_with_a_per_user_cap() -> None:
    argv = server_argv(_launch(parallel=8, total_context=196608, context_per_user=131072))

    pairs = _pairs(argv)
    assert pairs["--parallel"] == "8"
    assert pairs["--ctx-size"] == "196608"
    assert "--kv-unified" in argv
    assert pairs["--kv-unified-per-slot"] == "131072"


def test_launch_flags_are_kept_but_cannot_override_server_owned_ones() -> None:
    argv = server_argv(_launch(launch_flags=(
        _flag("--model", "other.gguf"),
        _flag("--ctx-size", "4096"),
        _flag("--n-gpu-layers", "-1"),
        _flag("--device", "Vulkan1"),
        _flag("--flash-attn", "off"),
    )))

    pairs = _pairs(argv)
    assert argv.count("--model") == 1 and pairs["--model"] == _native("/models/qwen.gguf")
    assert argv.count("--ctx-size") == 1 and pairs["--ctx-size"] == "32768"
    assert pairs["--n-gpu-layers"] == "-1"
    assert pairs["--device"] == "Vulkan1"
    assert pairs["--flash-attn"] == "off"


def test_chat_page_is_only_served_when_asked_for() -> None:
    assert "--no-webui" not in server_argv(_launch(web_ui=True))
    assert "--api-key-file" in server_argv(_launch(web_ui=True))


def test_tls_files_are_passed_when_both_are_given() -> None:
    argv = server_argv(_launch(tls_cert=Path("/tls/cert.pem"), tls_key=Path("/tls/key.pem")))

    pairs = _pairs(argv)
    assert pairs["--ssl-cert-file"] == _native("/tls/cert.pem")
    assert pairs["--ssl-key-file"] == _native("/tls/key.pem")


def test_tls_needs_both_files() -> None:
    with pytest.raises(ValueError, match="both"):
        _launch(tls_cert=Path("/tls/cert.pem"))


def test_server_binary_is_found_next_to_llama_cli(tmp_path: Path) -> None:
    cli = tmp_path / "llama-cli"
    cli.write_text("")
    server = tmp_path / "llama-server"
    server.write_text("")
    installation = LlamaCppInstallation(
        root=str(tmp_path), llama_cli=str(cli), source=RuntimeSource.MANAGED,
    )

    assert server_binary_for(installation) == server


def test_server_binary_missing_returns_none(tmp_path: Path) -> None:
    cli = tmp_path / "llama-cli.exe"
    cli.write_text("")
    installation = LlamaCppInstallation(
        root=str(tmp_path), llama_cli=str(cli), source=RuntimeSource.MANAGED,
    )

    assert server_binary_for(installation) is None
    assert server_binary_for(installation.model_copy(update={"llama_cli": None})) is None
