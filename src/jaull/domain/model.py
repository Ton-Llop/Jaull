from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from jaull.domain.enums import DiagnosticStatus, Format, RepositoryType


class ModelFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    size_bytes: int | None = None
    lfs: bool = False


class ModelRepositoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_id: str
    author: str | None = None
    private: bool = False
    gated: bool = False
    downloads: int | None = None
    likes: int | None = None
    last_modified: datetime | None = None
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None


class ModelConfig(BaseModel):
    """Subset of ``config.json`` fields we care about. All optional to survive exotic configs."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    architectures: list[str] = Field(default_factory=list)
    model_type: str | None = None
    torch_dtype: str | None = None
    max_position_embeddings: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    sliding_window: int | None = None
    rope_scaling: dict[str, object] | None = None
    tie_word_embeddings: bool | None = None
    vocab_size: int | None = None
    # Free-form flags used by the estimator to detect exotic architectures we should
    # warn about (MoE, MLA, custom auto_map, multimodal, ...). Kept as a raw payload
    # so we do not lose information every future addition of a hint would need.
    raw_flags: dict[str, object] = Field(default_factory=dict)


class GgufVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    quantization: str
    files: list[ModelFile]
    total_bytes: int


class SafetensorsSummary(BaseModel):
    """Aggregate view of a safetensors repository's metadata."""

    model_config = ConfigDict(frozen=True)

    total_parameters: int
    parameters_by_dtype: dict[str, int] = Field(default_factory=dict)


class RepositoryClassification(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_type: RepositoryType
    detected_types: set[RepositoryType] = Field(default_factory=set)
    formats: set[Format] = Field(default_factory=set)
    gguf_variants: list[GgufVariant] = Field(default_factory=list)


class ModelAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: ModelRepositoryInfo
    files: list[ModelFile] = Field(default_factory=list)
    classification: RepositoryClassification
    config: ModelConfig | None = None
    relevant_files: list[str] = Field(default_factory=list)
    total_size_bytes: int | None = None
    warnings: list[str] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: DiagnosticStatus
    detail: str
