from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaull.artifacts.storage import LocalModelFile
from jaull.cli.storage import run_storage_list, run_storage_remove

GIB = 1024**3


class _Advisor:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.logs_deleted = False

    def local_models(self) -> list[tuple[LocalModelFile, bool]]:
        return [
            (LocalModelFile("a/b-GGUF", "b-Q4_K_M.gguf", Path("/m/b.gguf"), 4 * GIB, "ab" * 32),
             True),
            (LocalModelFile("c/d-GGUF", "d-Q2_K.gguf", Path("/m/d.gguf"), 2 * GIB, None), False),
        ]

    def gguf_quantization(self, filename: str) -> str | None:
        return filename.removesuffix(".gguf").rsplit("-", 1)[-1]

    def quality_run_folders(self) -> tuple[int, int]:
        return 3, 512 * 1024**2

    def delete_local_model(self, repo_id: str, filename: str) -> int:
        self.deleted.append((repo_id, filename))
        return 4 * GIB

    def delete_quality_run_folders(self) -> int:
        self.logs_deleted = True
        return 512 * 1024**2


def test_storage_list_json_is_what_free_disk_space_shows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The desktop app lists and removes files with this, as the TUI's storage screen does.
    assert run_storage_list(_Advisor(), as_json=True) == 0  # type: ignore[arg-type]

    payload = json.loads(capsys.readouterr().out)
    first, second = payload["models"]
    assert (first["repo_id"], first["filename"]) == ("a/b-GGUF", "b-Q4_K_M.gguf")
    assert first["size_bytes"] == 4 * GIB and first["quantization"] == "Q4_K_M"
    assert first["verified"] and first["measured"]
    assert not second["verified"] and not second["measured"]
    assert payload["evaluation_logs"] == {"folders": 3, "size_bytes": 512 * 1024**2}


def test_storage_remove_takes_one_file_or_the_evaluation_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    advisor = _Advisor()
    assert run_storage_remove("a/b-GGUF", "b-Q4_K_M.gguf", advisor=advisor, as_json=True) == 0  # type: ignore[arg-type]
    assert advisor.deleted == [("a/b-GGUF", "b-Q4_K_M.gguf")]
    assert json.loads(capsys.readouterr().out)["freed_bytes"] == 4 * GIB

    assert run_storage_remove(None, None, evaluation_logs=True, advisor=advisor, as_json=True) == 0  # type: ignore[arg-type]
    assert advisor.logs_deleted and advisor.deleted == [("a/b-GGUF", "b-Q4_K_M.gguf")]

    assert run_storage_remove(None, None, advisor=advisor, as_json=True) == 2  # type: ignore[arg-type]
