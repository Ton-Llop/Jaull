"""Normalize a user-provided model reference (repo_id or URL) into a canonical repo_id."""

from __future__ import annotations

from urllib.parse import urlparse

from jaull.exceptions import InvalidModelReferenceError

_ALLOWED_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})
_RESERVED_SEGMENTS = frozenset({"tree", "blob", "resolve", "raw", "commit", "commits"})


def normalize_repo_id(reference: str) -> str:
    """Convert a raw ``owner/name`` or a Hugging Face URL into ``owner/name``.

    Accepts:
        - ``owner/name``
        - ``https://huggingface.co/owner/name``
        - ``https://huggingface.co/owner/name/tree/<rev>``
        - ``https://huggingface.co/owner/name/blob/<rev>/<path>``
        - ``https://huggingface.co/owner/name/resolve/<rev>/<path>``

    Raises :class:`InvalidModelReferenceError` for anything else.
    """
    if reference is None:
        raise InvalidModelReferenceError("Model reference cannot be empty.")

    value = reference.strip()
    if not value:
        raise InvalidModelReferenceError("Model reference cannot be empty.")

    if "://" in value or value.startswith("//"):
        return _from_url(value)

    return _from_repo_id(value)


def _from_repo_id(value: str) -> str:
    parts = [p for p in value.split("/") if p]
    if len(parts) != 2:
        raise InvalidModelReferenceError(
            f"Invalid Hugging Face repo_id: {value!r}. Expected the form 'owner/name'."
        )
    owner, name = parts
    _validate_segment(owner)
    _validate_segment(name)
    return f"{owner}/{name}"


def _from_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidModelReferenceError(
            f"Unsupported URL scheme in {value!r}: {parsed.scheme!r}."
        )

    host = (parsed.netloc or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise InvalidModelReferenceError(
            f"URL host {host!r} is not a Hugging Face domain."
        )

    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) < 2:
        raise InvalidModelReferenceError(
            f"URL {value!r} does not point to a model repository."
        )

    owner, name = segments[0], segments[1]
    if owner in _RESERVED_SEGMENTS:
        raise InvalidModelReferenceError(
            f"URL {value!r} does not point to a model repository."
        )
    _validate_segment(owner)
    _validate_segment(name)
    return f"{owner}/{name}"


def _validate_segment(segment: str) -> None:
    if not segment:
        raise InvalidModelReferenceError("Empty segment in repo_id.")
    for ch in segment:
        if ch.isalnum() or ch in {"-", "_", ".",}:
            continue
        raise InvalidModelReferenceError(
            f"Invalid character in repo_id segment {segment!r}: {ch!r}."
        )
