"""Thin adapter over ``huggingface_hub`` mapping SDK errors to our own exceptions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
    SafetensorsParsingError,
)
from huggingface_hub.hf_api import ModelInfo

from jaull.domain.model import SafetensorsSummary
from jaull.exceptions import (
    ConfigurationNotFoundError,
    HuggingFaceUnavailableError,
    ModelAccessDeniedError,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


class HfClientProtocol(Protocol):
    def model_info(self, repo_id: str) -> ModelInfo: ...
    def download_small_file(self, repo_id: str, filename: str) -> Path: ...
    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None: ...


class HfClient:
    """Real client used at runtime; wraps :class:`HfApi` and ``hf_hub_download``."""

    def __init__(self, token: str | None = None, api: HfApi | None = None) -> None:
        self._token = token or os.environ.get("HF_TOKEN")
        self._api = api or HfApi(token=self._token)

    def model_info(self, repo_id: str) -> ModelInfo:
        try:
            return self._api.model_info(repo_id=repo_id, files_metadata=True)
        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(f"Model not found: {repo_id}") from exc
        except GatedRepoError as exc:
            raise ModelAccessDeniedError(
                f"Model {repo_id} is gated and requires access approval or an HF_TOKEN."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ModelAccessDeniedError(
                    f"Access to {repo_id} was denied (HTTP {status})."
                ) from exc
            if status == 404:
                raise ModelNotFoundError(f"Model not found: {repo_id}") from exc
            raise HuggingFaceUnavailableError(
                f"Hugging Face returned HTTP {status} while fetching {repo_id}."
            ) from exc
        except OSError as exc:
            raise HuggingFaceUnavailableError(
                "Unable to connect to Hugging Face. Check your network connection."
            ) from exc

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self._token,
            )
            return Path(local_path)
        except EntryNotFoundError as exc:
            raise ConfigurationNotFoundError(
                f"File {filename!r} was not found in repository {repo_id}."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(f"Model not found: {repo_id}") from exc
        except GatedRepoError as exc:
            raise ModelAccessDeniedError(
                f"Model {repo_id} is gated and requires access approval or an HF_TOKEN."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ModelAccessDeniedError(
                    f"Access to {repo_id} was denied (HTTP {status})."
                ) from exc
            raise HuggingFaceUnavailableError(
                f"Hugging Face returned HTTP {status} while downloading {filename}."
            ) from exc
        except OSError as exc:
            raise HuggingFaceUnavailableError(
                "Unable to download file from Hugging Face. Check your network connection."
            ) from exc

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        """Aggregate parameter counts + dtype breakdown from safetensors headers.

        Returns ``None`` if the repository is not a safetensors repository (e.g. only
        GGUF or pytorch_model.bin) or if the metadata cannot be parsed. Downloads
        are minimal: only file headers are fetched, never the weights themselves.
        """
        try:
            metadata = self._api.get_safetensors_metadata(repo_id=repo_id)
        except NotASafetensorsRepoError:
            return None
        except SafetensorsParsingError:
            logger.debug("Failed to parse safetensors metadata for %s", repo_id)
            return None
        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(f"Model not found: {repo_id}") from exc
        except GatedRepoError as exc:
            raise ModelAccessDeniedError(
                f"Model {repo_id} is gated and requires access approval or an HF_TOKEN."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ModelAccessDeniedError(
                    f"Access to {repo_id} was denied (HTTP {status})."
                ) from exc
            raise HuggingFaceUnavailableError(
                f"Hugging Face returned HTTP {status} while reading safetensors metadata."
            ) from exc
        except OSError as exc:
            raise HuggingFaceUnavailableError(
                "Unable to read safetensors metadata from Hugging Face."
            ) from exc

        parameters_by_dtype: dict[str, int] = {}
        total = 0
        for file_metadata in metadata.files_metadata.values():
            for tensor in file_metadata.tensors.values():
                count = 1
                for dim in tensor.shape:
                    count *= int(dim)
                total += count
                parameters_by_dtype[tensor.dtype] = (
                    parameters_by_dtype.get(tensor.dtype, 0) + count
                )
        if total == 0:
            return None
        return SafetensorsSummary(
            total_parameters=total,
            parameters_by_dtype=parameters_by_dtype,
        )
