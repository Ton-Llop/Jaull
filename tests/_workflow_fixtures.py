"""Shared offline doubles for the guided-workflow tests.

Nothing here touches the network: the search client returns canned candidates,
the inspector returns hand-built ``ModelAnalysis`` objects and the estimator is
a pure function of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jaull.discovery.models import ModelCandidate, SearchQuery
from jaull.domain.enums import Format, RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
    KvCacheEstimate,
    MemoryComponent,
    MemoryEstimate,
    RuntimeOverheadEstimate,
    WeightEstimate,
)
from jaull.domain.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
)
from jaull.exceptions import HuggingFaceUnavailableError
from jaull.workflow.models import (
    CommercialUse,
    ConcurrencyLevel,
    RecommendationPriority,
    UseCase,
    UserAnswers,
)

GIB = 1024**3


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def hardware(
    ram_gib: int = 32, vram_gib: int | None = 12, name: str = "NVIDIA RTX 4070"
) -> HardwareProfile:
    gpus = (
        [
            GpuInfo(
                name=name,
                vram_total_bytes=vram_gib * GIB,
                vram_available_bytes=vram_gib * GIB,
                driver_version="550.0",
                cuda_version="12.4",
            )
        ]
        if vram_gib is not None
        else []
    )
    return HardwareProfile(
        os="Linux 6.8",
        os_version="#1 SMP",
        arch="x86_64",
        cpu=CpuInfo(model="AMD Ryzen 5 3600", physical_cores=6, logical_cores=12),
        memory=MemoryInfo(total_bytes=ram_gib * GIB, available_bytes=ram_gib * GIB),
        storage=[],
        gpus=gpus,
        warnings=[] if gpus else ["No NVIDIA GPU detected."],
    )


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
def answers(
    use_case: UseCase = UseCase.CODING,
    priority: RecommendationPriority = RecommendationPriority.BALANCED,
    languages: list[str] | None = None,
    concurrency: ConcurrencyLevel = ConcurrencyLevel.SINGLE,
    commercial: CommercialUse = CommercialUse.YES,
    document_scale: Any = None,
) -> UserAnswers:
    return UserAnswers(
        use_case=use_case,
        priority=priority,
        languages=languages if languages is not None else ["Spanish", "English"],
        concurrency=concurrency,
        commercial_use=commercial,
        document_scale=document_scale,
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def candidate(
    repo_id: str = "org/Model-7B-Instruct",
    *,
    license_value: str | None = "apache-2.0",
    languages: list[str] | None = None,
    tags: list[str] | None = None,
    downloads: int = 10_000,
    likes: int = 100,
    pipeline_tag: str | None = "text-generation",
    gated: bool | str = False,
    private: bool = False,
    base_model: str | None = None,
    queries: list[str] | None = None,
) -> ModelCandidate:
    return ModelCandidate(
        repo_id=repo_id,
        pipeline_tag=pipeline_tag,
        library_name="transformers",
        license=license_value,
        languages=languages if languages is not None else ["es", "en"],
        tags=tags if tags is not None else ["text-generation", "instruct"],
        downloads=downloads,
        likes=likes,
        gated=gated,
        private=private,
        base_model_repo_id=base_model,
        source_queries=queries or ["q1"],
    )


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def transformers_analysis(
    repo_id: str = "org/Model-7B-Instruct",
    license_value: str | None = "apache-2.0",
) -> ModelAnalysis:
    files = [
        ModelFile(path="config.json", size_bytes=800),
        ModelFile(path="model.safetensors", size_bytes=14 * GIB, lfs=True),
    ]
    return ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id=repo_id,
            license=license_value,
            tags=["text-generation"],
            pipeline_tag="text-generation",
            library_name="transformers",
            downloads=10_000,
            likes=100,
        ),
        files=files,
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
        ),
        config=ModelConfig(
            model_type="llama",
            max_position_embeddings=32768,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        relevant_files=["config.json"],
        total_size_bytes=14 * GIB,
    )


def gguf_analysis(
    repo_id: str = "org/Model-7B-Instruct-GGUF",
    quantizations: tuple[str, ...] = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"),
    base_bytes: int = 4 * GIB,
) -> ModelAnalysis:
    variants = []
    for index, quant in enumerate(quantizations):
        size = base_bytes + index * GIB
        variants.append(
            GgufVariant(
                quantization=quant,
                files=[ModelFile(path=f"model-{quant}.gguf", size_bytes=size, lfs=True)],
                total_bytes=size,
            )
        )
    return ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id=repo_id,
            license="apache-2.0",
            tags=["text-generation", "gguf"],
            pipeline_tag="text-generation",
        ),
        files=[v.files[0] for v in variants],
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=variants,
        ),
        config=None,
        total_size_bytes=sum(v.total_bytes for v in variants),
    )


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
def memory_estimate(
    analysis: ModelAnalysis,
    config: InferenceConfiguration,
    total_bytes: int,
    status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE,
    confidence: EstimationConfidence = EstimationConfidence.HIGH,
    available_vram: int | None = 12 * GIB,
    available_ram: int | None = 32 * GIB,
) -> MemoryEstimate:
    component = MemoryComponent(
        name="Weights",
        bytes=total_bytes,
        source=EstimateSource.EXACT,
        confidence=confidence,
        explanation="fixture",
    )
    return MemoryEstimate(
        repository=analysis.repo,
        repository_type=analysis.classification.primary_type,
        inference_configuration=config,
        weights=WeightEstimate(component=component),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV cache",
                bytes=GIB // 2,
                source=EstimateSource.DERIVED,
                confidence=confidence,
                explanation="fixture",
            ),
            context_length=config.context_length,
            batch_size=config.batch_size,
            dtype_bytes=2,
            formula="fixture",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=GIB // 2,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="fixture",
            ),
            base_bytes=GIB // 2,
            weight_fraction=0.1,
            minimum_bytes=GIB // 4,
        ),
        device_reserve=MemoryComponent(
            name="Device reserve",
            bytes=GIB // 2,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="fixture",
        ),
        safety_margin=None,
        total_bytes=total_bytes,
        assessment=CompatibilityAssessment(
            status=status,
            confidence=confidence,
            target_device=config.target_device,
            effective_device=TargetDevice.GPU,
            available_vram_bytes=available_vram,
            available_ram_bytes=available_ram,
            ratio=(total_bytes / available_vram) if available_vram else None,
        ),
    )


def size_driven_estimator(
    vram_budget: int = 12 * GIB,
) -> Any:
    """An estimator whose verdict depends only on the selected variant's size.

    That is enough to exercise the quantization ladder deterministically: a rung
    fits when its bytes are within the budget.
    """

    def estimate(
        analysis: ModelAnalysis, config: InferenceConfiguration
    ) -> MemoryEstimate:
        total = _size_for(analysis, config)
        if total <= vram_budget * 0.75:
            status = CompatibilityStatus.COMFORTABLE
        elif total <= vram_budget * 0.9:
            status = CompatibilityStatus.COMPATIBLE
        elif total <= vram_budget:
            status = CompatibilityStatus.TIGHT
        else:
            status = CompatibilityStatus.INSUFFICIENT
        return memory_estimate(
            analysis, config, total_bytes=total, status=status,
            available_vram=vram_budget,
        )

    return estimate


def _size_for(analysis: ModelAnalysis, config: InferenceConfiguration) -> int:
    if config.quantization:
        for variant in analysis.classification.gguf_variants:
            if variant.quantization == config.quantization:
                return variant.total_bytes
        return 99 * GIB
    if config.precision is not None:
        ratio = {"float16": 1.0, "int8": 0.5, "int4": 0.25}.get(
            config.precision.value, 1.0
        )
        return int((analysis.total_size_bytes or 8 * GIB) * ratio)
    return analysis.total_size_bytes or 8 * GIB


# ---------------------------------------------------------------------------
# Search client doubles
# ---------------------------------------------------------------------------
@dataclass
class FakeSearchClient:
    """Returns canned candidates and records the queries it was given."""

    results: dict[str, list[ModelCandidate]] = field(default_factory=dict)
    default: list[ModelCandidate] = field(default_factory=list)
    raises: Exception | None = None
    raise_on: set[str] = field(default_factory=set)
    seen: list[SearchQuery] = field(default_factory=list)

    def search(self, query: SearchQuery) -> list[ModelCandidate]:
        self.seen.append(query)
        if self.raises is not None and (
            not self.raise_on or query.label in self.raise_on
        ):
            raise self.raises
        return list(self.results.get(query.label, self.default))


@dataclass
class FakeHfClient:
    """Minimal stand-in for HfClientProtocol; never reaches the network."""

    def model_info(self, repo_id: str) -> Any:  # pragma: no cover - unused
        raise HuggingFaceUnavailableError("fake client has no model_info")

    def download_small_file(self, repo_id: str, filename: str) -> Any:
        raise HuggingFaceUnavailableError("fake client has no downloads")

    def safetensors_summary(self, repo_id: str) -> Any:
        return None


__all__ = [
    "GIB",
    "FakeHfClient",
    "FakeSearchClient",
    "answers",
    "candidate",
    "gguf_analysis",
    "hardware",
    "memory_estimate",
    "size_driven_estimator",
    "transformers_analysis",
]
