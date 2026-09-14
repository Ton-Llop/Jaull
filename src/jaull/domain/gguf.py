"""Local GGUF tensor descriptors, independent of any execution backend."""

from dataclasses import dataclass

from jaull.domain.artifacts import ModelArtifact


@dataclass(frozen=True)
class GgufTensor:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    offset: int
    size_bytes: int


@dataclass(frozen=True)
class GgufTensorIndex:
    """Stored tensor bytes, not device allocations. No tensor payload is read."""

    artifact: ModelArtifact
    architecture: str | None
    block_count: int | None
    alignment: int
    data_offset: int
    file_size_bytes: int
    tensors: tuple[GgufTensor, ...]
