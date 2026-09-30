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


@pytest.mark.parametrize("known_digest", [False, True])
def test_download_writes_file_and_persists_sha(tmp_path: Path, known_digest: bool) -> None:
    payload = b"synthetic gguf bytes" * 100
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    artifact = service.resolve("owner/repo", quantization="Q5_K_M")
    if known_digest:
        artifact = artifact.model_copy(
            update={"sha256": hashlib.sha256(payload).hexdigest().upper()}
        )
    downloaded = service.download(artifact)

    assert downloaded.is_downloaded is True
    assert downloaded.local_path is not None
    assert downloaded.local_path.read_bytes() == payload
    assert downloaded.sha256 == hashlib.sha256(payload).hexdigest()
    sidecar = downloaded.local_path.with_name(downloaded.local_path.name + ".sha256")
    assert sidecar.read_text(encoding="ascii").strip() == downloaded.sha256


@pytest.mark.parametrize("existing_sidecar", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_download_rejects_bytes_that_conflict_with_known_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_sidecar: bool, interrupted: bool,
) -> None:
    payload = b"replaced model"
    expected_digest = hashlib.sha256(b"expected model").hexdigest()
    assert len(payload) == len(b"expected model")
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])

    def downloader(**kwargs: Any) -> str:
        result = _stub_downloader(payload)(**kwargs)
        if interrupted:
            raise OSError("download interrupted")
        return str(result)

    service = _make_service(tmp_path, client, downloader=downloader)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M").model_copy(
        update={"sha256": expected_digest}
    )
    path = service.storage.path_for(artifact)
    sidecar = path.with_name(path.name + ".sha256")
    if existing_sidecar:
        service.storage.save_sha256(path, expected_digest)
    monkeypatch.setattr(service.resolver, "resolve", lambda *args, **kwargs: artifact)

    error_type = ArtifactDownloadError if interrupted else ArtifactVerificationError
    with pytest.raises(error_type, match="interrupted" if interrupted else "SHA-256 mismatch"):
        service.download(artifact)

    assert artifact.sha256 == expected_digest
    assert not artifact.is_downloaded
    assert path.read_bytes() == payload
    assert not sidecar.exists()
    resolved = service.resolve("owner/repo", quantization="Q5_K_M")
    assert resolved.sha256 == expected_digest
    assert resolved.is_downloaded
    assert not resolved.is_verified
    with pytest.raises(ArtifactVerificationError, match="Missing SHA-256 sidecar"):
        service.verify(resolved)


def test_download_does_not_start_when_sidecar_cannot_be_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"original model"
    calls: list[object] = []

    def downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(Path(kwargs["local_dir"]) / kwargs["filename"])

    service = _make_service(
        tmp_path, _gguf_repo([("m-q5_k_m.gguf", len(payload))]), downloader=downloader,
    )
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")
    path = service.storage.path_for(artifact)
    service.storage.ensure_parent(path)
    path.write_bytes(payload)
    service.storage.save_sha256(path, hashlib.sha256(payload).hexdigest())
    sidecar = path.with_name(path.name + ".sha256")
    original = sidecar.read_bytes()
    error = PermissionError("sidecar removal denied")

    def deny_removal(self: Path, *args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(Path, "unlink", deny_removal)
    with pytest.raises(ArtifactDownloadError, match="sidecar removal denied") as caught:
        service.download(artifact)
    assert caught.value.__cause__ is error
    assert calls == []
    assert path.read_bytes() == payload
    assert sidecar.read_bytes() == original


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


@pytest.mark.parametrize("stage", ["directory", "hash", "sidecar"])
def test_download_translates_local_io_failures(tmp_path: Path, stage: str) -> None:
    payload = b"content"

    def downloader(**kwargs: Any) -> str:
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        if stage != "hash":
            path.write_bytes(payload)
        if stage == "sidecar":
            path.with_name(path.name + ".sha256").mkdir()
        return str(path)

    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=downloader)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")
    path = service.storage.path_for(artifact)
    if stage == "directory":
        path.parent.parent.mkdir(parents=True)
        path.parent.write_bytes(b"not a directory")

    with pytest.raises(ArtifactDownloadError, match="I/O error") as error:
        service.download(artifact)

    assert isinstance(error.value.__cause__, OSError)
    assert not artifact.is_downloaded
    if stage == "sidecar":
        assert path.read_bytes() == payload


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


@pytest.mark.parametrize("digest_state", ["conflict", "known", "unknown", "uppercase"])
def test_resolve_preserves_expected_digest_when_promoting_local_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest_state: str,
) -> None:
    payload = b"local artifact"
    service = _make_service(tmp_path, _gguf_repo([("m-q5_k_m.gguf", len(payload))]))
    reference = service.resolve("owner/repo", quantization="Q5_K_M")
    digest = hashlib.sha256(payload).hexdigest()
    expected = {
        "conflict": "a" * 64, "known": digest, "unknown": None, "uppercase": digest.upper(),
    }[digest_state]
    reference = reference.model_copy(update={"sha256": expected})
    monkeypatch.setattr(service.resolver, "resolve", lambda *args, **kwargs: reference)
    path = service.storage.path_for(reference)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    service.storage.save_sha256(path, digest)
    sidecar = path.with_name(path.name + ".sha256")
    before = (reference.model_dump_json(), path.read_bytes(), sidecar.read_bytes())

    if digest_state == "conflict":
        with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
            service.resolve("owner/repo", quantization="Q5_K_M")
    else:
        resolved = service.resolve("owner/repo", quantization="Q5_K_M")
        assert resolved.sha256 == digest
        assert resolved.local_path == path
        assert resolved.is_downloaded
        assert not resolved.is_verified
    assert (reference.model_dump_json(), path.read_bytes(), sidecar.read_bytes()) == before


@pytest.mark.parametrize("digest_state", ["known", "unknown", "uppercase"])
def test_verify_fast_path_ok(tmp_path: Path, digest_state: str) -> None:
    payload = b"content" * 50
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))

    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    if digest_state == "unknown":
        artifact = artifact.model_copy(update={"sha256": None})
    elif digest_state == "uppercase":
        assert artifact.sha256 is not None
        artifact = artifact.model_copy(update={"sha256": artifact.sha256.upper()})
    verified = service.verify(artifact)

    assert verified.is_verified is True
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("full", [False, True])
def test_verify_preserves_known_artifact_digest(tmp_path: Path, full: bool) -> None:
    payload = b"original"
    replacement = b"replaced"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    assert artifact.local_path is not None
    artifact.local_path.write_bytes(replacement)
    replacement_hash = hashlib.sha256(replacement).hexdigest()
    service.storage.save_sha256(artifact.local_path, replacement_hash)

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        service.verify(artifact, full=full)

    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.local_path.read_bytes() == replacement
    assert service.storage.load_sha256(artifact.local_path) == replacement_hash


@pytest.mark.parametrize("sidecar_bytes", [b"abcd", b"z" * 64, b"\xff"])
def test_verify_rejects_invalid_sha256_sidecar(
    tmp_path: Path, sidecar_bytes: bytes,
) -> None:
    payload = b"content"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    assert artifact.local_path is not None
    sidecar = artifact.local_path.with_name(artifact.local_path.name + ".sha256")
    sidecar.write_bytes(sidecar_bytes)

    with pytest.raises(ArtifactVerificationError, match="SHA-256"):
        service.verify(artifact)

    assert sidecar.read_bytes() == sidecar_bytes
    assert artifact.local_path.read_bytes() == payload


def test_verify_missing_file_raises(tmp_path: Path) -> None:
    client = _gguf_repo([("m-q5_k_m.gguf", 100)])
    service = _make_service(tmp_path, client)
    artifact = service.resolve("owner/repo", quantization="Q5_K_M")

    with pytest.raises(ArtifactVerificationError, match="missing"):
        service.verify(artifact)


def test_verify_full_accepts_uppercase_sha256_sidecar(tmp_path: Path) -> None:
    payload = b"content"
    client = _gguf_repo([("m-q5_k_m.gguf", len(payload))])
    service = _make_service(tmp_path, client, downloader=_stub_downloader(payload))
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    assert artifact.local_path is not None
    assert artifact.sha256 is not None
    sidecar = artifact.local_path.with_name(artifact.local_path.name + ".sha256")
    sidecar.write_text(artifact.sha256.upper() + "\n", encoding="ascii")

    verified = service.verify(artifact, full=True)

    assert verified.is_verified
    assert verified.sha256 == artifact.sha256


@pytest.mark.parametrize(
    "read_error", [PermissionError("read denied"), FileNotFoundError("removed")],
)
def test_full_verification_read_failure_is_a_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_error: OSError,
) -> None:
    payload = b"model data"
    service = _make_service(
        tmp_path, _gguf_repo([("m-q5_k_m.gguf", len(payload))]),
        downloader=_stub_downloader(payload),
    )
    artifact = service.download(service.resolve("owner/repo", quantization="Q5_K_M"))
    path = artifact.local_path
    assert path is not None
    sidecar = path.with_name(path.name + ".sha256")
    before = (artifact.model_dump_json(), path.read_bytes(), sidecar.read_bytes())
    original_open = Path.open

    def unreadable_model(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            raise read_error
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unreadable_model)
        with pytest.raises(ArtifactVerificationError, match="Could not read") as error:
            service.verify(artifact, full=True)
        assert error.value.__cause__ is read_error
    assert (artifact.model_dump_json(), path.read_bytes(), sidecar.read_bytes()) == before


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
