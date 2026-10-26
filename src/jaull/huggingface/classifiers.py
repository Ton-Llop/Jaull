"""Repository classification from a plain listing of files.

Runs offline against ``ModelFile`` objects — no downloads required.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import PurePosixPath

from jaull.domain.enums import Format, RepositoryType
from jaull.domain.model import GgufVariant, ModelFile, RepositoryClassification

_MULTIPART_SUFFIX_RE = re.compile(r"-(?P<index>\d{5})-of-(?P<count>\d{5})$")
_GGUF_QUANT_RE = re.compile(r"(?P<q>IQ\d+[A-Z0-9_]*|Q\d+[A-Z0-9_]*|F16|F32|BF16)", re.IGNORECASE)


def classify_repository(files: Iterable[ModelFile]) -> RepositoryClassification:
    files_list = list(files)
    top_level_names = {
        PurePosixPath(f.path).name for f in files_list if "/" not in f.path.strip("/")
    }
    all_lowered_names = {PurePosixPath(f.path).name.lower() for f in files_list}

    detected: set[RepositoryType] = set()
    formats: set[Format] = set()

    # These sentinel files only mean what they mean when they sit at the repo root.
    # Diffusers pipelines carry a config.json inside every sub-component, which must
    # not be confused with a top-level transformers config.json.
    has_config = "config.json" in top_level_names
    has_adapter_config = "adapter_config.json" in top_level_names
    has_model_index = "model_index.json" in top_level_names

    has_safetensors = any(n.endswith(".safetensors") for n in all_lowered_names)
    has_pytorch_bin = any(
        n.endswith(".bin") and "pytorch_model" in n for n in all_lowered_names
    ) or ("pytorch_model.bin" in all_lowered_names)
    has_gguf = any(n.endswith(".gguf") for n in all_lowered_names)
    has_onnx = any(n.endswith(".onnx") for n in all_lowered_names)

    if has_safetensors:
        formats.add(Format.SAFETENSORS)
    if has_pytorch_bin:
        formats.add(Format.PYTORCH_BIN)
    if has_gguf:
        formats.add(Format.GGUF)
    if has_onnx:
        formats.add(Format.ONNX)

    if has_config and (has_safetensors or has_pytorch_bin):
        detected.add(RepositoryType.TRANSFORMERS)
    if has_gguf:
        detected.add(RepositoryType.GGUF)
    if has_model_index:
        detected.add(RepositoryType.DIFFUSERS)
    if has_onnx:
        detected.add(RepositoryType.ONNX)
    if has_adapter_config:
        detected.add(RepositoryType.ADAPTER)

    primary = _pick_primary(detected)
    variants = _group_gguf_variants(files_list) if has_gguf else []

    return RepositoryClassification(
        primary_type=primary,
        detected_types=detected or {RepositoryType.UNKNOWN},
        formats=formats,
        gguf_variants=variants,
    )


def _pick_primary(detected: set[RepositoryType]) -> RepositoryType:
    # Deterministic priority chosen to match the spec's guidance.
    for candidate in (
        RepositoryType.ADAPTER,
        RepositoryType.TRANSFORMERS,
        RepositoryType.DIFFUSERS,
        RepositoryType.GGUF,
        RepositoryType.ONNX,
    ):
        if candidate in detected:
            return candidate
    return RepositoryType.UNKNOWN


def _group_gguf_variants(files: Iterable[ModelFile]) -> list[GgufVariant]:
    """Keep each whole file and each shard set as an independent representation."""
    buckets: dict[tuple[str, str, str, int], list[ModelFile]] = defaultdict(list)
    for f in files:
        path = PurePosixPath(f.path)
        name = path.name
        if not name.lower().endswith(".gguf"):
            continue
        quant = _detect_quantization(name)
        stem = name[: -len(".gguf")]
        split = _MULTIPART_SUFFIX_RE.search(stem)
        count = int(split.group("count")) if split else -1
        base = _MULTIPART_SUFFIX_RE.sub("", stem)
        buckets[(quant, str(path.parent), base, count)].append(f)

    variants: list[GgufVariant] = []
    for (quant, _parent, _base, count), group in sorted(buckets.items()):
        parts = [_MULTIPART_SUFFIX_RE.search(PurePosixPath(f.path).stem) for f in group]
        is_split = any(part is not None for part in parts)
        indexes = [int(part.group("index")) for part in parts if part is not None]
        complete = not is_split or (
            count > 0 and len(indexes) == count and sorted(indexes) == list(range(1, count + 1))
        )
        sizes = [f.size_bytes for f in group]
        total = (
            sum(size for size in sizes if size is not None)
            if complete and all(size is not None and size > 0 for size in sizes)
            else None
        )
        variants.append(
            GgufVariant(
                quantization=quant,
                files=sorted(group, key=lambda f: f.path),
                total_bytes=total,
                is_complete=complete,
                is_split=is_split,
            )
        )
    return variants


def _detect_quantization(filename: str) -> str:
    stem = filename[: -len(".gguf")]
    stem = _MULTIPART_SUFFIX_RE.sub("", stem)
    match = _GGUF_QUANT_RE.search(stem)
    return match.group("q").upper() if match else "unknown"
