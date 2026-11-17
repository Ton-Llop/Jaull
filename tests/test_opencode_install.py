"""Adding the server to the opencode the person already has, without disturbing it."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from jaull.application.serving.client_config import opencode_config
from jaull.execution.opencode_install import install_opencode_config, merge_opencode_config

OURS = opencode_config(base_url="http://127.0.0.1:8090/v1", api_key="KEY", alias="qwen",
                       context_limit=32768)
THEIRS: dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "model": "anthropic/claude",
    "theme": "tokyonight",
    "provider": {"ollama": {"npm": "@ai-sdk/openai-compatible", "models": {"x": {}}}},
}


def test_the_server_is_added_beside_what_the_person_has() -> None:
    merged = merge_opencode_config(json.dumps(THEIRS), OURS)

    assert merged.outcome == "added" and merged.config is not None
    assert merged.config["provider"]["ollama"] == THEIRS["provider"]["ollama"]
    assert merged.config["provider"]["jaull"] == OURS["provider"]["jaull"]
    assert merged.config["theme"] == "tokyonight"
    assert merged.config["model"] == "anthropic/claude" and not merged.selected  # theirs stays


def test_it_fills_in_the_default_model_only_when_there_is_none() -> None:
    bare = {"provider": {"ollama": {}}}
    merged = merge_opencode_config(json.dumps(bare), OURS)
    assert merged.config is not None
    assert merged.config["model"] == OURS["model"] and merged.selected
    assert merge_opencode_config(None, OURS).outcome == "created"
    assert merge_opencode_config("  ", OURS).config == OURS


def test_running_it_again_changes_nothing_and_a_new_key_updates_only_ours() -> None:
    first = merge_opencode_config(json.dumps(THEIRS), OURS)
    again = merge_opencode_config(json.dumps(first.config), OURS)
    assert again.outcome == "unchanged"

    renewed = opencode_config(base_url="http://127.0.0.1:8090/v1", api_key="NEW", alias="qwen",
                              context_limit=32768)
    updated = merge_opencode_config(json.dumps(first.config), renewed)
    assert updated.outcome == "updated" and updated.config is not None
    assert updated.config["provider"]["jaull"]["options"]["apiKey"] == "NEW"
    assert updated.config["provider"]["ollama"] == THEIRS["provider"]["ollama"]


@pytest.mark.parametrize("text", ["{ // a comment\n}", "[1, 2]", '{"provider": []}', "not json"])
def test_a_file_it_cannot_read_safely_is_left_alone(text: str) -> None:
    merged = merge_opencode_config(text, OURS)
    assert merged.outcome == "unreadable" and merged.config is None


@pytest.fixture
def opencode_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("jaull.execution.opencode_install.shutil.which", lambda name: None)
    return tmp_path / "opencode"


def test_nothing_is_written_for_someone_who_does_not_use_opencode(opencode_home: Path) -> None:
    assert install_opencode_config(OURS) is None
    assert not opencode_home.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_an_existing_config_is_merged_backed_up_and_kept_private(opencode_home: Path) -> None:
    opencode_home.mkdir()
    target = opencode_home / "opencode.json"
    target.write_text(json.dumps(THEIRS), encoding="utf-8")
    target.chmod(0o644)

    result = install_opencode_config(OURS)

    assert result is not None and result.outcome == "added" and result.path == target
    assert json.loads(target.read_text(encoding="utf-8"))["provider"].keys() == {"ollama", "jaull"}
    assert result.backup is not None and json.loads(result.backup.read_text()) == THEIRS
    assert stat.S_IMODE(target.stat().st_mode) == 0o600  # it now holds the API key

    install_opencode_config(OURS)  # the backup keeps the file as it was, not the last merge
    assert json.loads(result.backup.read_text()) == THEIRS


def test_an_unreadable_config_or_a_jsonc_one_is_not_touched(opencode_home: Path) -> None:
    opencode_home.mkdir()
    target = opencode_home / "opencode.json"
    target.write_text("{ // mine\n}", encoding="utf-8")
    result = install_opencode_config(OURS)
    assert result is not None and result.outcome == "unreadable"
    assert target.read_text(encoding="utf-8") == "{ // mine\n}"

    target.unlink()
    (opencode_home / "opencode.jsonc").write_text("{}", encoding="utf-8")
    result = install_opencode_config(OURS)
    assert result is not None and result.outcome == "unreadable" and not target.exists()


def test_with_opencode_installed_a_missing_config_is_created(
    opencode_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaull.execution.opencode_install.shutil.which",
                        lambda name: "/usr/bin/opencode")
    result = install_opencode_config(OURS)
    assert result is not None and result.outcome == "created" and result.backup is None
    assert json.loads((opencode_home / "opencode.json").read_text()) == OURS
