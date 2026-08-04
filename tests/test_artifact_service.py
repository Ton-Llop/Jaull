from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import EntryNotFoundError

from jaull.artifacts.errors import (
    ArtifactDownloadError,
    ArtifactFormatNotSupportedError,
    ArtifactNotFoundError,
    ArtifactVerificationError,
)
from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.model import SafetensorsSummary
from jaull.exceptions import QuantizationNotFoundError
from jaull.huggingface.artifact_resolver import HuggingFaceArtifactResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@dataclass
class _Sibling:
    rfilename: str
    size: int | None = None
    lfs: object | None = field(default_factory=lambda: object())


@dataclass
class _FakeInfo:
    sha: str = "deadbeef"
    siblings: list[_Sibling] = field(default_factory=list)


@dataclass
class _FakeHfClient:
    info: _FakeInfo

    def model_info(self, repo_id: str) -> _FakeInfo:
        return self.info

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


def _gguf_repo(files: list[tuple[str, int]]) -> _FakeHfClient:
    siblings = [_Sibling(rfilename=name, size=size) for name, size in files]
    return _FakeHfClient(_FakeInfo(sha="sha_from_hub", siblings=siblings))


def _make_service(
    tmp_path: Path,
    client: _FakeHfClient,
    downloader: Any = None,
) -> ArtifactService:
    return ArtifactService(
        resolver=HuggingFaceArtifactResolver(client),  # type: ignore[arg-type]
        storage=ArtifactStorage(root=tmp_path),
        downloader=downloader or (lambda **kwargs: str(tmp_path / "unused")),
    )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
def test_resolves_requested_quantization(tmp_path: Path) -> None:
    client = _gguf_repo(
        [
            ("model-q4_k_m.gguf", 1_000),
            ("model-q5_k_m.gguf", 1_500),
            ("model-q8_0.gguf", 2_000),
        ]
    )
    service = _make_service(tmp_path, client)

    artifact = service.resolve("owner/repo", quantization="Q5_K_M")

    assert artifact.filename == "model-q5_k_m.gguf"
    assert artifact.quantization == "Q5_K_M"
    assert artifact.size_bytes == 1_500
    assert artifact.revision == "sha_from_hub"
    assert artifact.format == "gguf"
    assert artifact.is_downloaded is False


def test_quantization_matching_is_case_insensitive(tmp_path: Path) -> None:
    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client)

    artifact = service.resolve("owner/repo", quantization="q5_k_m")

    assert artifact.quantization == "Q5_K_M"
    assert artifact.filename == "m-q5_k_m.gguf"


def test_missing_quantization_raises(tmp_path: Path) -> None:
    client = _gguf_repo([("m-q4_k_m.gguf", 100), ("m-q8_0.gguf", 200)])
    service = _make_service(tmp_path, client)

    with pytest.raises(QuantizationNotFoundError) as ctx:
        service.resolve("owner/repo", quantization="Q5_K_M")

    assert "Q5_K_M" in str(ctx.value)
    assert set(ctx.value.available) == {"Q4_K_M", "Q8_0"}


def test_non_gguf_repository_rejected(tmp_path: Path) -> None:
    client = _FakeHfClient(
        _FakeInfo(
            siblings=[
                _Sibling(rfilename="config.json", size=1024),
                _Sibling(rfilename="model.safetensors", size=1_000_000),
            ]
        )
    )
    service = _make_service(tmp_path, client)

    with pytest.raises(ArtifactFormatNotSupportedError):
        service.resolve("owner/repo", quantization=None)


def test_empty_repository_raises_not_found(tmp_path: Path) -> None:
    client = _FakeHfClient(_FakeInfo(siblings=[]))
    service = _make_service(tmp_path, client)

    with pytest.raises(ArtifactNotFoundError):
        service.resolve("owner/repo", quantization=None)


def test_multipart_gguf_variant_rejected(tmp_path: Path) -> None:
    client = _gguf_repo(
        [
            ("m-q5_k_m-00001-of-00002.gguf", 500),
            ("m-q5_k_m-00002-of-00002.gguf", 500),
        ]
    )
    service = _make_service(tmp_path, client)

    with pytest.raises(ArtifactFormatNotSupportedError, match="multipart"):
        service.resolve("owner/repo", quantization="Q5_K_M")


def test_revision_from_caller_wins_over_hub_sha(tmp_path: Path) -> None:
    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client)

    artifact = service.resolve("owner/repo", quantization="Q5_K_M", revision="v1")

    assert artifact.revision == "v1"


# ---------------------------------------------------------------------------
# Download / verify
# ---------------------------------------------------------------------------
def _stub_downloader(payload: bytes) -> Any:
    def _do(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        dst = local_dir / kwargs["filename"]
        dst.write_bytes(payload)
        return str(dst)

    return _do


def test_download_writes_file_and_persists_sha(tmp_path: Path) -> None:
    payload = b"synthetic gguf bytes" * 100
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    artifact = service.resolve("owner/repo", quantization="Q5_K_M")
    downloaded = service.download(artifact)

    assert downloaded.is_downloaded is True
    assert downloaded.local_path is not None
    assert downloaded.local_path.read_bytes() == payload
    assert downloaded.sha256 == hashlib.sha256(payload).hexdigest()
    sidecar = downloaded.local_path.with_name(downloaded.local_path.name + ".sha256")
    assert sidecar.read_text(encoding="ascii").strip() == downloaded.sha256


def test_download_translates_hf_not_found(tmp_path: Path) -> None:
    def _boom(**kwargs: Any) -> str:
        raise EntryNotFoundError("nope")

    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client, downloader=_boom)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")

    with pytest.raises(ArtifactNotFoundError):
        service.download(artifact)


def test_download_translates_oserror(tmp_path: Path) -> None:
    def _boom(**kwargs: Any) -> str:
        raise OSError("disk full")

    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client, downloader=_boom)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")

    with pytest.raises(ArtifactDownloadError, match="disk full"):
        service.download(artifact)


def test_resolve_promotes_already_downloaded(tmp_path: Path) -> None:
    payload = b"already here"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    # First resolve+download primes the local dir.
    first = service.resolve("owner/repo", quantization="Q5_K_M")
    service.download(first)

    # Second resolve should notice the local file and its sidecar.
    second = service.resolve("owner/repo", quantization="Q5_K_M")

    assert second.is_downloaded is True
    assert second.local_path is not None
    assert second.local_path.is_file()
    assert second.sha256 == hashlib.sha256(payload).hexdigest()


def test_verify_fast_path_ok(tmp_path: Path) -> None:
    payload = b"content" * 50
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    verified = service.verify(artifact)

    assert verified.is_verified is True


def test_verify_missing_file_raises(tmp_path: Path) -> None:
    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")

    with pytest.raises(ArtifactVerificationError, match="missing"):
        service.verify(artifact)


def test_verify_size_mismatch_raises(tmp_path: Path) -> None:
    payload = b"actual"
    client = _gguf_repo([("m-q5_k_m.gguf", 999)])  # claimed size wrong on purpose
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))

    with pytest.raises(ArtifactVerificationError, match="Size mismatch"):
        service.verify(artifact)


def test_verify_full_detects_disk_corruption(tmp_path: Path) -> None:
    payload = b"original bytes"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    assert artifact.local_path is not None
    # Corrupt the file *without* touching the sidecar. Match original size so
    # the fast checks still pass and only ``full=True`` catches it.
    corrupt = b"CORRUPTED\x00XXXXX"[: len(payload)]
    artifact.local_path.write_bytes(corrupt)

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        service.verify(artifact, full=True)


def test_verify_fast_missing_sidecar_raises(tmp_path: Path) -> None:
    payload = b"bytes"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    assert artifact.local_path is not None
    sidecar = artifact.local_path.with_name(artifact.local_path.name + ".sha256")
    sidecar.unlink()

    with pytest.raises(ArtifactVerificationError, match="Missing SHA-256"):
        service.verify(artifact)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_model_artifact_roundtrip_json() -> None:
    artifact = ModelArtifact(
        repo_id="owner/repo",
        revision="rev",
        filename="m.gguf",
        format="gguf",
        quantization="Q5_K_M",
        size_bytes=42,
        local_path=Path("/tmp/whatever"),
        sha256="abcd",
        is_downloaded=True,
        is_verified=False,
    )
    dumped = artifact.model_dump_json()
    parsed = json.loads(dumped)
    assert parsed["repo_id"] == "owner/repo"
    assert parsed["quantization"] == "Q5_K_M"
    restored = ModelArtifact.model_validate_json(dumped)
    assert restored == artifact
