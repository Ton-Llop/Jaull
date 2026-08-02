"""Domain-level exceptions used across the application."""

from __future__ import annotations


class LocalAiCheckError(Exception):
    """Base class for all application-level errors."""


class InvalidModelReferenceError(LocalAiCheckError):
    """The user-provided model reference is not a valid repo_id or Hugging Face URL."""


class ModelNotFoundError(LocalAiCheckError):
    """The requested model does not exist on Hugging Face."""


class ModelAccessDeniedError(LocalAiCheckError):
    """The requested model is gated or private and cannot be accessed with current credentials."""


class HuggingFaceUnavailableError(LocalAiCheckError):
    """The Hugging Face API could not be reached."""


class ConfigurationNotFoundError(LocalAiCheckError):
    """Expected configuration file (e.g. config.json) was not found in the repository."""


class HardwareDetectionError(LocalAiCheckError):
    """A hardware probe failed in an unrecoverable way."""


class EstimationError(LocalAiCheckError):
    """Base class for errors raised while producing a memory estimate."""


class QuantizationNotFoundError(EstimationError):
    """The requested GGUF quantization variant is not available in the repository."""

    def __init__(self, requested: str, available: list[str]) -> None:
        self.requested = requested
        self.available = available
        pretty = ", ".join(available) if available else "none"
        super().__init__(
            f"Quantization {requested!r} not found. Available variants: {pretty}."
        )


class GgufHeaderIncompleteError(LocalAiCheckError):
    """More bytes are needed to finish parsing the GGUF metadata table."""


class GgufHeaderInvalidError(LocalAiCheckError):
    """The GGUF header is malformed or uses an unsupported version."""


class BaseModelResolutionError(LocalAiCheckError):
    """Base class for failures while resolving a GGUF repository's base model."""
