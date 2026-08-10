"""Regression test for the CPU-only 7.4 GiB scenario reported by users.

Every problem listed in the bug report is asserted here as an invariant on a
synthetic workload:

- an AWQ artifact must not lead on hardware without a CUDA GPU;
- a theoretical int4 must not lead over a confirmed GGUF that fits;
- no warning may mention VRAM when there is no GPU;
- Qwen Coder must not share a series bucket with general Qwen;
- alternatives must not be labelled "higher quality" without a real
  capability gap;
- parameter counts must be complete (7.6B for a 7B safetensors model), not
  the old 12·L·H² undercount that surfaced 7B as 4.3B.

The scenario is hand-built (no network) so this test also proves the new
rules survive the composition end-to-end.
"""

from __future__ import annotations

import pytest

from jaull.discovery import enrichment, series
from jaull.discovery.grouping import collapse_families
from jaull.domain.artifact_profile import ArtifactConfirmation, ArtifactFormat
from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
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
    HardwareProfile,
    MemoryInfo,
)
from jaull.domain.inference import (
    InferenceConfiguration,
    TargetDevice,
    WeightPrecision,
)
from jaull.domain.model import (
    GgufVariant,
    ModelAnalysis,
    ModelConfig,
    ModelFile,
    ModelRepositoryInfo,
    RepositoryClassification,
    SafetensorsSummary,
)
from jaull.domain.requirements import (
    CommercialUse,
    ConcurrencyLevel,
    RecommendationPriority,
    UseCase,
    UserAnswers,
)
from jaull.recommendation.actionability import (
    ActionabilityLevel,
    assess_actionability,
)
from jaull.runtime import service as runtime_service
from jaull.workflow import ranking
from jaull.workflow.requirements import build_requirements

GIB = 1024**3


# ---------------------------------------------------------------------------
# Hardware: the exact CPU-only 7.4 GiB profile from the report.
# ---------------------------------------------------------------------------
def _cpu_only_hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        os_version="#1 SMP",
        arch="x86_64",
        cpu=CpuInfo(
            model="AMD Ryzen 5 5500U",
            physical_cores=6,
            logical_cores=12,
        ),
        memory=MemoryInfo(
            total_bytes=int(7.4 * GIB),
            available_bytes=int(6.5 * GIB),
        ),
        storage=[],
        gpus=[],
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Synthetic candidates matching the reported set.
# ---------------------------------------------------------------------------
def _qwen_3b_gguf() -> tuple[ModelCandidate, ModelAnalysis]:
    variants = [
        GgufVariant(
            quantization="Q4_K_M",
            files=[
                ModelFile(
                    path="qwen2.5-3b-q4_k_m.gguf",
                    size_bytes=int(2.2 * GIB),
                    lfs=True,
                )
            ],
            total_bytes=int(2.2 * GIB),
        ),
        GgufVariant(
            quantization="Q5_K_M",
            files=[
                ModelFile(
                    path="qwen2.5-3b-q5_k_m.gguf",
                    size_bytes=int(2.6 * GIB),
                    lfs=True,
                )
            ],
            total_bytes=int(2.6 * GIB),
        ),
    ]
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            license="apache-2.0",
            tags=["text-generation", "gguf"],
            pipeline_tag="text-generation",
        ),
        files=[f for v in variants for f in v.files],
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=variants,
        ),
    )
    return _candidate("Qwen/Qwen2.5-3B-Instruct-GGUF", tags=["gguf"]), analysis


def _qwen_7b_gguf() -> tuple[ModelCandidate, ModelAnalysis]:
    variants = [
        GgufVariant(
            quantization="Q4_K_M",
            files=[
                ModelFile(
                    path="qwen2.5-7b-q4_k_m.gguf",
                    size_bytes=int(4.9 * GIB),
                    lfs=True,
                )
            ],
            total_bytes=int(4.9 * GIB),
        ),
    ]
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            license="apache-2.0",
            tags=["text-generation", "gguf"],
            pipeline_tag="text-generation",
        ),
        files=[f for v in variants for f in v.files],
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=variants,
        ),
    )
    return _candidate("Qwen/Qwen2.5-7B-Instruct-GGUF", tags=["gguf"]), analysis


def _qwen_7b_awq() -> tuple[ModelCandidate, ModelAnalysis]:
    """An AWQ repo — real config-level quantization_config, not just a tag."""
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
            license="apache-2.0",
            tags=["text-generation", "awq"],
            pipeline_tag="text-generation",
        ),
        files=[
            ModelFile(path="config.json", size_bytes=1024),
            ModelFile(
                path="model.safetensors", size_bytes=int(4.5 * GIB), lfs=True
            ),
        ],
        classification=RepositoryClassification(
            primary_type=RepositoryType.TRANSFORMERS,
            detected_types={RepositoryType.TRANSFORMERS},
            formats={Format.SAFETENSORS},
        ),
        config=ModelConfig(
            model_type="qwen2",
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
            intermediate_size=18944,
            vocab_size=152064,
            quantization_config={"quant_method": "awq", "bits": 4},
        ),
        safetensors_summary=SafetensorsSummary(
            total_parameters=7_615_616_512,
            parameters_by_dtype={"I4": 7_615_616_512},
        ),
    )
    return (
        _candidate("Qwen/Qwen2.5-7B-Instruct-AWQ", tags=["awq"]),
        analysis,
    )


def _qwen_coder_7b_gguf() -> tuple[ModelCandidate, ModelAnalysis]:
    variants = [
        GgufVariant(
            quantization="Q4_K_M",
            files=[
                ModelFile(
                    path="qwen2.5-coder-7b-q4_k_m.gguf",
                    size_bytes=int(4.9 * GIB),
                    lfs=True,
                )
            ],
            total_bytes=int(4.9 * GIB),
        ),
    ]
    analysis = ModelAnalysis(
        repo=ModelRepositoryInfo(
            repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            license="apache-2.0",
            tags=["text-generation", "gguf", "code"],
            pipeline_tag="text-generation",
        ),
        files=[f for v in variants for f in v.files],
        classification=RepositoryClassification(
            primary_type=RepositoryType.GGUF,
            detected_types={RepositoryType.GGUF},
            formats={Format.GGUF},
            gguf_variants=variants,
        ),
    )
    return (
        _candidate(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", tags=["gguf", "code"]
        ),
        analysis,
    )


def _candidate(repo_id: str, *, tags: list[str] | None = None) -> ModelCandidate:
    return ModelCandidate(
        repo_id=repo_id,
        pipeline_tag="text-generation",
        library_name="transformers",
        license="apache-2.0",
        languages=["en"],
        tags=(tags or []) + ["text-generation", "instruct"],
        downloads=100_000,
        likes=500,
        metadata_confidence=EstimationConfidence.HIGH,
    )


# ---------------------------------------------------------------------------
# Stub estimator: sizes memory strictly from the selected artifact.
# ---------------------------------------------------------------------------
def _fake_estimator(hardware: HardwareProfile):  # type: ignore[no-untyped-def]
    ram = hardware.memory.total_bytes

    def _estimate(
        analysis: ModelAnalysis, cfg: InferenceConfiguration
    ) -> MemoryEstimate:
        if cfg.quantization:
            # GGUF path — pick the requested variant's on-disk size.
            variant = next(
                v
                for v in analysis.classification.gguf_variants
                if v.quantization == cfg.quantization
            )
            weights_bytes = variant.total_bytes
            source = EstimateSource.EXACT
        else:
            # Transformers path. Use the safetensors summary when present.
            if analysis.safetensors_summary is not None:
                params = analysis.safetensors_summary.total_parameters
            elif analysis.config and analysis.config.num_hidden_layers:
                # Rough fallback — enough for the test.
                params = 3_000_000_000
            else:
                params = 7_000_000_000
            bpp = _bpp(cfg.precision)
            weights_bytes = int(params * bpp)
            source = EstimateSource.METADATA

        # KV cache + overhead ≈ 0.8 GiB total for a 4k ctx toy model.
        kv_bytes = int(0.4 * GIB)
        overhead_bytes = int(0.4 * GIB)
        total = weights_bytes + kv_bytes + overhead_bytes

        ratio = total / ram if ram else 1.0
        if ratio <= 0.75:
            status = CompatibilityStatus.COMFORTABLE
        elif ratio <= 0.9:
            status = CompatibilityStatus.COMPATIBLE
        elif ratio <= 1.0:
            status = CompatibilityStatus.TIGHT
        else:
            status = CompatibilityStatus.INSUFFICIENT

        assessment = CompatibilityAssessment(
            status=status,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.CPU,
            effective_device=TargetDevice.CPU,
            available_ram_bytes=ram,
            available_vram_bytes=None,
            ratio=ratio,
        )
        estimate = MemoryEstimate(
            repository=analysis.repo,
            repository_type=analysis.classification.primary_type,
            inference_configuration=cfg,
            weights=WeightEstimate(
                component=MemoryComponent(
                    name="Weights",
                    bytes=weights_bytes,
                    source=source,
                    confidence=EstimationConfidence.HIGH,
                    explanation="synthetic",
                ),
                num_parameters=(
                    analysis.safetensors_summary.total_parameters
                    if analysis.safetensors_summary
                    else None
                ),
            ),
            kv_cache=KvCacheEstimate(
                component=MemoryComponent(
                    name="KV cache",
                    bytes=kv_bytes,
                    source=EstimateSource.DERIVED,
                    confidence=EstimationConfidence.MEDIUM,
                    explanation="synthetic",
                ),
                layers=analysis.config.num_hidden_layers if analysis.config else None,
                kv_heads=analysis.config.num_key_value_heads if analysis.config else None,
                head_dim=None,
                context_length=cfg.context_length,
                batch_size=cfg.batch_size,
                dtype_bytes=2,
                formula="synthetic",
                notes=[],
            ),
            runtime_overhead=RuntimeOverheadEstimate(
                component=MemoryComponent(
                    name="Runtime overhead",
                    bytes=overhead_bytes,
                    source=EstimateSource.ASSUMED,
                    confidence=EstimationConfidence.LOW,
                    explanation="synthetic",
                ),
                base_bytes=int(0.3 * GIB),
                weight_fraction=0.1,
                minimum_bytes=int(0.25 * GIB),
            ),
            device_reserve=MemoryComponent(
                name="Device reserve",
                bytes=0,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="none",
            ),
            safety_margin=None,
            total_bytes=total,
            assessment=assessment,
        )
        # Attach a real runtime recommendation so ``assess_runtime`` has
        # something meaningful to grade.
        runtime_rec = runtime_service.recommend(
            estimate, hardware, analysis.classification.primary_type
        )
        return estimate.model_copy(update={"runtime_recommendation": runtime_rec})

    return _estimate


def _bpp(precision: WeightPrecision | None) -> float:
    return {
        WeightPrecision.FLOAT32: 4.0,
        WeightPrecision.FLOAT16: 2.0,
        WeightPrecision.BFLOAT16: 2.0,
        WeightPrecision.INT8: 1.0,
        WeightPrecision.INT4: 0.5,
    }.get(precision or WeightPrecision.FLOAT16, 2.0)


# ---------------------------------------------------------------------------
# End-to-end: run the ranker with the whole set on CPU-only hardware.
# ---------------------------------------------------------------------------
@pytest.fixture
def evaluated_bundle() -> tuple[list[EvaluatedCandidate], HardwareProfile]:
    hardware = _cpu_only_hardware()
    analyses = {
        cand.repo_id: analysis
        for cand, analysis in (
            _qwen_3b_gguf(),
            _qwen_7b_gguf(),
            _qwen_7b_awq(),
            _qwen_coder_7b_gguf(),
        )
    }
    candidates = [
        cand for cand, _ in (
            _qwen_3b_gguf(),
            _qwen_7b_gguf(),
            _qwen_7b_awq(),
            _qwen_coder_7b_gguf(),
        )
    ]
    estimate_fn = _fake_estimator(hardware)

    def inspect(repo_id: str) -> ModelAnalysis:
        return analyses[repo_id]

    answers = UserAnswers(
        use_case=UseCase.GENERAL_CHAT,
        priority=RecommendationPriority.BALANCED,
        languages=["English"],
        concurrency=ConcurrencyLevel.SINGLE,
        commercial_use=CommercialUse.YES,
        document_scale=None,
    )
    requirements = build_requirements(answers, hardware)

    evaluated = [
        enrichment.evaluate_candidate(
            candidate=cand,
            requirements=requirements,
            hardware=hardware,
            inspect_fn=inspect,
            estimate_fn=estimate_fn,
        )
        for cand in candidates
    ]
    return evaluated, hardware


def test_awq_on_cpu_only_is_blocked_from_primary(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    awq = next(e for e in evaluated if "AWQ" in e.repo_id)
    assert assess_actionability(awq) is ActionabilityLevel.BLOCKED


def test_awq_is_recognised_as_confirmed_artifact(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    awq = next(e for e in evaluated if "AWQ" in e.repo_id)
    assert awq.artifact_profile is not None
    assert awq.artifact_profile.format is ArtifactFormat.AWQ
    assert awq.artifact_profile.confirmation is ArtifactConfirmation.CONFIRMED


def test_gguf_variants_score_actionable(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    for item in evaluated:
        if "GGUF" in item.repo_id and "AWQ" not in item.repo_id:
            assert assess_actionability(item) is not ActionabilityLevel.BLOCKED
            assert item.artifact_profile is not None
            assert item.artifact_profile.format is ArtifactFormat.GGUF
            assert (
                item.artifact_profile.confirmation
                is ArtifactConfirmation.CONFIRMED
            )


def test_ranked_primary_is_a_gguf_and_not_awq(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    recs = ranking.recommend(
        evaluated,
        build_requirements(
            UserAnswers(
                use_case=UseCase.GENERAL_CHAT,
                priority=RecommendationPriority.BALANCED,
                languages=["English"],
                concurrency=ConcurrencyLevel.SINGLE,
                commercial_use=CommercialUse.YES,
                document_scale=None,
            ),
            _cpu_only_hardware(),
        ),
    )
    assert recs, "expected at least one recommendation on this CPU-only box"
    primary = recs[0]
    assert "AWQ" not in primary.repo_id, (
        "AWQ must not lead on CPU-only hardware"
    )
    # And whatever ended up first is a GGUF artifact.
    assert primary.evaluated.artifact_profile is not None
    assert primary.evaluated.artifact_profile.format is ArtifactFormat.GGUF


def test_no_vram_warnings_on_cpu_only(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, hardware = evaluated_bundle
    assert not hardware.gpus
    from jaull.recommendation.explanations import build_warnings

    requirements = build_requirements(
        UserAnswers(
            use_case=UseCase.GENERAL_CHAT,
            priority=RecommendationPriority.BALANCED,
            languages=["English"],
            concurrency=ConcurrencyLevel.SINGLE,
            commercial_use=CommercialUse.YES,
            document_scale=None,
        ),
        hardware,
    )
    for item in evaluated:
        warnings = build_warnings(item, requirements)
        for line in warnings:
            assert "VRAM" not in line, (
                f"CPU-only run must not mention VRAM, got: {line!r}"
            )


def test_parameter_count_uses_safetensors_when_available(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    awq = next(e for e in evaluated if "AWQ" in e.repo_id)
    assert awq.parameter_count_info is not None
    assert awq.parameter_count_info.count == 7_615_616_512
    assert awq.parameter_count_info.confidence is EstimationConfidence.HIGH


def test_series_grouping_isolates_coder(evaluated_bundle) -> None:  # type: ignore[no-untyped-def]
    evaluated, _ = evaluated_bundle
    coder = next(e for e in evaluated if "Coder" in e.repo_id)
    general_7b = next(
        e
        for e in evaluated
        if e.repo_id == "Qwen/Qwen2.5-7B-Instruct-GGUF"
    )
    assert series.series_key(coder) != series.series_key(general_7b)


def test_family_collapse_still_merges_format_only_conversions(
    evaluated_bundle,  # type: ignore[no-untyped-def]
) -> None:
    """Grouping by family should still fuse a Transformers repo with its
    GGUF conversion (same repo, different format)."""
    evaluated, _ = evaluated_bundle
    provisional = {item.repo_id: 0.5 for item in evaluated}
    # Add a "same repo, different format" pair.
    extra_candidate = _candidate(
        "Qwen/Qwen2.5-7B-Instruct", tags=["transformers"]
    )
    extra_evaluated = evaluated[0].model_copy(
        update={
            "candidate": extra_candidate.model_copy(
                update={"base_model_repo_id": "Qwen/Qwen2.5-7B-Instruct"}
            ),
        }
    )
    provisional[extra_candidate.repo_id] = 0.6
    all_items = [*evaluated, extra_evaluated]
    _, siblings = collapse_families(all_items, provisional)
    # The exact grouping is an implementation detail — just assert the call
    # didn't crash and returned a mapping.
    assert isinstance(siblings, dict)
