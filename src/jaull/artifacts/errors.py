"""Errors specific to artifact resolution, download and verification.

All errors inherit from ``jaull.exceptions.JaullError`` so callers can catch
either the base type or a specific subclass. Quantization mismatches keep
using the existing :class:`jaull.exceptions.QuantizationNotFoundError` — no
duplicate exception is introduced.
"""

from __future__ import annotations

from jaull.exceptions import JaullError


class ArtifactError(JaullError):
    """Base class for artifact-related failures."""


class ArtifactNotFoundError(ArtifactError):
    """The requested artifact (repo or file) could not be located."""


class ArtifactFormatNotSupportedError(ArtifactError):
    """The repository format is not (yet) supported by the artifact service.

    Raised for non-GGUF repositories and for multipart GGUF variants — both
    are documented limitations of this phase.
    """


class ArtifactDownloadError(ArtifactError):
    """The artifact download failed (network, permissions, disk)."""


class ArtifactVerificationError(ArtifactError):
    """A downloaded artifact failed size or SHA-256 verification."""


__all__ = [
    "ArtifactDownloadError",
    "ArtifactError",
    "ArtifactFormatNotSupportedError",
    "ArtifactNotFoundError",
    "ArtifactVerificationError",
]
