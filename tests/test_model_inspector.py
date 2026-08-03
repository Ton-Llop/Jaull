from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jaull.domain.enums import RepositoryType
from jaull.exceptions import ModelAccessDeniedError, ModelNotFoundError
from jaull.huggingface.repository import inspect_model


@dataclass
class _Sibling:
    rfilename: str
    size: int | None = None
    lfs: Any | None = None


@dataclass
class _FakeModelInfo:
    id: str
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int | None = None
    likes: int | None = None
    last_modified: Any = None
    private: bool = False
    gated: bool = False
    card_data: dict[str, Any] = field(default_factory=dict)
    siblings: list[_Sibling] = field(default_factory=list)


class _FakeHfClient:
    def __init__(
        self,
        info: _FakeModelInfo | None = None,
        raises: Exception | None = None,
        config_payload: dict[str, Any] | None = None,
        tmp_path: Path | None = None,
    ) -> None:
        self._info = info
        self._raises = raises
        self._config_payload = config_payload
        self._tmp_path = tmp_path

    def model_info(self, repo_id: str):
        if self._raises is not None:
            raise self._raises
        assert self._info is not None
        return self._info

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        assert self._tmp_path is not None
        target = self._tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._config_payload or {}), encoding="utf-8")
        return target


def test_inspect_normal_transformers_model(tmp_path: Path) -> None:
    info = _FakeModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        author="Qwen",
        tags=["text-generation", "transformers"],
        pipeline_tag="text-generation",
        library_name="transformers",
        downloads=1234,
        likes=42,
        private=False,
        gated=False,
        card_data={"license": "apache-2.0"},
        siblings=[
            _Sibling(rfilename="config.json", size=1234),
            _Sibling(rfilename="generation_config.json", size=567),
            _Sibling(
                rfilename="model-00001-of-00004.safetensors",
                size=4_000_000_000,
                lfs=object(),
            ),
            _Sibling(
                rfilename="model.safetensors.index.json",
                size=8_192,
            ),
        ],
    )
    client = _FakeHfClient(
        info=info,
        config_payload={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 32768,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
        },
        tmp_path=tmp_path,
    )

    analysis = inspect_model("Qwen/Qwen2.5-7B-Instruct", client=client)

    assert analysis.repo.repo_id == "Qwen/Qwen2.5-7B-Instruct"
    assert analysis.repo.license == "apache-2.0"
    assert analysis.classification.primary_type is RepositoryType.TRANSFORMERS
    assert analysis.config is not None
    assert analysis.config.architectures == ["Qwen2ForCausalLM"]
    assert analysis.config.max_position_embeddings == 32768
    assert analysis.total_size_bytes is not None
    assert "config.json" in analysis.relevant_files


def test_inspect_missing_model_raises_model_not_found() -> None:
    client = _FakeHfClient(raises=ModelNotFoundError("Model not found: no/such"))
    with pytest.raises(ModelNotFoundError):
        inspect_model("no/such", client=client)


def test_inspect_gated_model_raises_access_denied() -> None:
    client = _FakeHfClient(
        raises=ModelAccessDeniedError("Model gated/one is gated and requires HF_TOKEN.")
    )
    with pytest.raises(ModelAccessDeniedError):
        inspect_model("gated/one", client=client)


def test_repository_total_includes_all_variants(tmp_path: Path) -> None:
    info = _FakeModelInfo(
        id="user/gguf-model",
        siblings=[
            _Sibling(rfilename="README.md", size=100),
            _Sibling(rfilename="model-Q4_K_M.gguf", size=4_000_000_000, lfs=object()),
            _Sibling(rfilename="model-Q5_K_M.gguf", size=5_000_000_000, lfs=object()),
            _Sibling(rfilename="model-Q8_0.gguf", size=8_000_000_000, lfs=object()),
        ],
    )
    client = _FakeHfClient(info=info, tmp_path=tmp_path)
    analysis = inspect_model("user/gguf-model", client=client)

    assert analysis.classification.primary_type is RepositoryType.GGUF
    # total_size_bytes now represents "sum of every unique file in the repo"; the
    # per-variant size lives in classification.gguf_variants and is consumed by the
    # estimator, not conflated with the download total.
    assert analysis.total_size_bytes == 100 + 4_000_000_000 + 5_000_000_000 + 8_000_000_000
    assert len(analysis.classification.gguf_variants) == 3
    sizes = {v.quantization: v.total_bytes for v in analysis.classification.gguf_variants}
    assert sizes == {"Q4_K_M": 4_000_000_000, "Q5_K_M": 5_000_000_000, "Q8_0": 8_000_000_000}
