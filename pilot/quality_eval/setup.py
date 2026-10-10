"""Shared infrastructure checks and consented preparation, never model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import urlopen

from pilot.quality_eval.evaluate import (
    ARTIFACT_CONTRACT,
    ARTIFACT_CONTRACT_LABEL,
    DATASET_SHA256,
    DATASET_URL,
    SERVER_SHA256,
    verified_dataset_bytes,
    write_json,
)

SUITES_LABEL = "io.jaull.quality.suites"


def _dataset_pin(suite: str) -> tuple[str, str, Callable[[Path], bytes]]:
    if suite == "ifeval":
        from pilot.quality_eval import ifeval

        return ifeval.DATASET_URL, ifeval.DATASET_SHA256, ifeval.verified_dataset_bytes
    return DATASET_URL, DATASET_SHA256, verified_dataset_bytes


def validate_evaluator_image(image: dict[str, Any], *, suite: str = "hellaswag") -> None:
    config = image.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict) or labels.get(ARTIFACT_CONTRACT_LABEL) != ARTIFACT_CONTRACT:
        raise ValueError(
            "Evaluator image lacks the exact-local-gguf-v1 contract; rebuild "
            "pilot/quality_eval with a new image tag before evaluation."
        )
    if suite == "ifeval" and "ifeval-chat-v1" not in str(labels.get(SUITES_LABEL, "")).split(","):
        raise ValueError(
            "Evaluator image predates the IFEval chat suite; rebuild pilot/quality_eval "
            "with a new image tag before evaluation."
        )


def prepare_dataset(path: Path, *, allow_download: bool = False, suite: str = "hellaswag") -> None:
    url, expected_sha256, verify = _dataset_pin(suite)
    if path.exists():
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Pinned dataset exceeds the 64 MiB preparation limit.")
        verify(path)
        return
    if not allow_download:
        raise ValueError("Pinned dataset missing; enable dataset download permission first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".dataset-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            digest = hashlib.sha256()
            size = 0
            deadline = time.monotonic() + 120
            with urlopen(url, timeout=30) as response:
                while chunk := response.read(64 * 1024):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Pinned dataset preparation exceeded 120 seconds.")
                    size += len(chunk)
                    # Bound this fixed validation split, never an arbitrary dataset download.
                    if size > 64 * 1024**2:
                        raise ValueError("Pinned dataset exceeds the 64 MiB preparation limit.")
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"Pinned {suite} dataset file failed SHA256 verification")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary.replace(path)
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def prepare_infrastructure(
    dataset: Path, server: Path, image_name: str, *, allow_download: bool = False,
    suite: str = "hellaswag",
) -> dict[str, Any]:
    if not server.is_file() or not os.access(server, os.X_OK):
        raise ValueError("Runtime unavailable: select an executable pinned llama-server.")
    with server.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SERVER_SHA256:
            raise ValueError("Runtime unavailable: llama-server differs from the audited pin.")
    try:
        images = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", image_name], text=True, stderr=subprocess.PIPE,
            timeout=15,
        ))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "Evaluator unavailable: Docker must be running and the local image already built."
        ) from exc
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError("Invalid evaluator image inspection response.")
    image = images[0]
    validate_evaluator_image(image, suite=suite)
    if (
        not isinstance(image.get("Id"), str)
        or re.fullmatch("sha256:[0-9a-f]{64}", image["Id"]) is None
    ):
        raise ValueError("Evaluator image has no exact image ID.")
    prepare_dataset(dataset, allow_download=allow_download, suite=suite)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dataset-download", action="store_true")
    # Which pinned dataset and image capability to check; HellaSwag keeps its old call.
    parser.add_argument("--suite", choices=("hellaswag", "ifeval"), default="hellaswag")
    args = parser.parse_args()
    try:
        image = prepare_infrastructure(
            args.dataset_file, args.llama_server, args.image,
            allow_download=args.allow_dataset_download, suite=args.suite,
        )
    except (OSError, ValueError) as exc:
        write_json(args.output, {"status": "blocked", "error": str(exc)})
        raise SystemExit(1) from exc
    write_json(args.output, {"status": "ready", "image_id": image["Id"]})


if __name__ == "__main__":
    main()
