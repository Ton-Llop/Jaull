"""Evidence-based ranking over concrete execution plans."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkMeasurementKind,
    BenchmarkRecord,
)
from jaull.domain.candidates import EvaluatedCandidate
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityStatus,
    EstimationConfidence,
    MemoryEstimate,
)
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    PlanCompatibilityStatus,
)
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration, WeightPrecision
from jaull.domain.model import GgufVariant, ModelAnalysis
from jaull.domain.recommendation import (
    AssessmentLevel,
    ExternalEvaluationEvidence,
    HardConstraint,
    HardConstraintCode,
    PlanAssessment,
    RecommendationEvidence,
    RecommendationEvidenceCategory,
    RecommendationEvidenceSource,
)
from jaull.domain.requirements import RecommendationPriority, UseCase, UserRequirements
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    RuntimeBackendSelection,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.estimator.configuration import select_configuration
from jaull.evaluation.hardware_fingerprint import machine_fingerprint
from jaull.execution_plans import build_execution_plan, resolve_model_identity
from jaull.recommendation.capability import CapabilitySignal, MetadataCapabilityAnalyzer
from jaull.recommendation.policies import LicenseCategory, classify_license

EstimatePlanFn = Callable[[ModelAnalysis, InferenceConfiguration], MemoryEstimate]


@dataclass(frozen=True)
class PlanRankingContext:
    hardware: HardwareProfile | None = None
    backend_selection: RuntimeBackendSelection | None = None
    readiness_by_runtime: dict[RuntimeName, ExecutionReadiness] | None = None
    benchmark_records: Sequence[BenchmarkRecord] = ()
    experiment_records: Sequence[ExperimentRecord] = ()
    external_evaluations: Sequence[ExternalEvaluationEvidence] = ()


@dataclass(frozen=True)
class RankedPlan:
    evaluated: EvaluatedCandidate
    plan: ExecutionPlan
    assessment: PlanAssessment


def rank_execution_plans(
    evaluated: Sequence[EvaluatedCandidate],
    requirements: UserRequirements,
    *,
    context: PlanRankingContext | None = None,
    estimate_fn: EstimatePlanFn | None = None,
    limit: int | None = None,
) -> list[RankedPlan]:
    """Rank concrete execution plans with visible dimensions, not one magic score."""

    ctx = context or PlanRankingContext()
    ranked: list[RankedPlan] = []
    for item in evaluated:
        if item.failed or item.analysis is None:
            continue
        for plan in generate_execution_plans(
            item,
            requirements,
            context=ctx,
            estimate_fn=estimate_fn,
        ):
            assessment = assess_plan(
                item,
                plan,
                requirements,
                context=ctx,
            )
            ranked.append(RankedPlan(evaluated=item, plan=plan, assessment=assessment))
    ordered = sorted(ranked, key=lambda item: _ranking_key(item, requirements.priority))
    if limit is None:
        return ordered
    return ordered[:limit]


def generate_execution_plans(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
    *,
    context: PlanRankingContext | None = None,
    estimate_fn: EstimatePlanFn | None = None,
) -> list[ExecutionPlan]:
    if evaluated.analysis is None:
        return []
    ctx = context or PlanRankingContext()
    identity = resolve_model_identity(
        candidate=evaluated.candidate,
        analysis=evaluated.analysis,
    )
    if evaluated.analysis.classification.primary_type is RepositoryType.GGUF:
        return _gguf_plans(
            evaluated,
            identity,
            requirements,
            context=ctx,
            estimate_fn=estimate_fn,
        )
    return [
        _transformers_plan(
            evaluated,
            identity,
            context=ctx,
        )
    ]


def assess_plan(
    evaluated: EvaluatedCandidate,
    plan: ExecutionPlan,
    requirements: UserRequirements,
    *,
    context: PlanRankingContext | None = None,
) -> PlanAssessment:
    ctx = context or PlanRankingContext()
    evidence = _base_evidence(evaluated, plan)
    capability = MetadataCapabilityAnalyzer().analyze(
        evaluated.candidate,
        evaluated.analysis,
    )
    evidence.extend(_capability_evidence(capability))
    hard = _hard_constraints(evaluated, plan, requirements)
    local_experiment = _matching_experiment(plan, ctx.experiment_records)
    local_benchmark = _matching_benchmark(plan, ctx.benchmark_records)
    if local_experiment is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.LOCAL,
                key="validation",
                value="success" if local_experiment.observation.success else "failed",
                source=RecommendationEvidenceSource.LOCAL_EXPERIMENT,
                confidence=EstimationConfidence.HIGH,
                observed=True,
                provenance={
                    "experiment_id": local_experiment.identity.experiment_id,
                },
            )
        )
    if local_benchmark is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.LOCAL,
                key="benchmark",
                value=local_benchmark.observation.methodology or "unknown",
                source=RecommendationEvidenceSource.LOCAL_BENCHMARK,
                confidence=EstimationConfidence.HIGH,
                observed=True,
                provenance={
                    "benchmark_id": local_benchmark.identity.benchmark_id,
                    "methodology": local_benchmark.observation.methodology,
                },
            )
        )

    measured_memory = _measured_memory(local_experiment, local_benchmark)
    measured_tps = _measured_generation_tps(local_benchmark)
    external = list(ctx.external_evaluations)
    evidence.extend(_external_evidence(external))

    if plan.execution_readiness is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.EXECUTION,
                key="readiness",
                value=plan.execution_readiness.status.value,
                source=RecommendationEvidenceSource.EXECUTION_READINESS,
                confidence=EstimationConfidence.HIGH,
                observed=True,
            )
        )

    reasons = _reasons(evaluated, plan, local_experiment, local_benchmark, measured_memory)
    trade_offs = _trade_offs(plan, external)
    rejection_reasons = [constraint.message for constraint in hard]
    return PlanAssessment(
        plan_id=plan.plan_id,
        hard_constraints=hard,
        capability=_capability_level(capability),
        suitability=_suitability(evaluated, requirements),
        execution_fitness=_execution_fitness(plan),
        feasibility=_feasibility(plan),
        executability=_executability(plan),
        performance_evidence=(
            AssessmentLevel.STRONG if local_benchmark is not None else AssessmentLevel.UNKNOWN
        ),
        resource_evidence=(
            AssessmentLevel.STRONG
            if measured_memory is not None
            else _resource_evidence_from_estimate(plan)
        ),
        license_fit=_license_fit(evaluated, requirements),
        language_fit=_language_fit(evaluated, requirements),
        confidence=_confidence(evaluated, plan, local_experiment, local_benchmark),
        evidence=evidence,
        external_evaluations=external,
        reasons=_with_capability_reasons(reasons, capability),
        trade_offs=_with_capability_trade_offs(trade_offs, capability),
        rejection_reasons=rejection_reasons,
        measured_memory_bytes=measured_memory,
        estimated_memory_bytes=(
            plan.memory_prediction.total_bytes if plan.memory_prediction else None
        ),
        measured_generation_tokens_per_second=measured_tps,
        local_experiment_id=(
            local_experiment.identity.experiment_id if local_experiment is not None else None
        ),
        local_benchmark_id=(
            local_benchmark.identity.benchmark_id if local_benchmark is not None else None
        ),
    )


def _gguf_plans(
    evaluated: EvaluatedCandidate,
    identity: ModelIdentity,
    requirements: UserRequirements,
    *,
    context: PlanRankingContext,
    estimate_fn: EstimatePlanFn | None,
) -> list[ExecutionPlan]:
    analysis = evaluated.analysis
    if analysis is None:
        return []
    variants = _ordered_gguf_variants(
        analysis.classification.gguf_variants,
        requirements.priority,
    )
    if estimate_fn is None:
        selected = (
            evaluated.selected_configuration.quantization
            if evaluated.selected_configuration
            else None
        )
        selected_variants = [
            variant for variant in variants if variant.quantization == selected
        ]
        variants = selected_variants or variants[:1]
    plans: list[ExecutionPlan] = []
    for variant in variants:
        config = _config_for_variant(evaluated, requirements, quantization=variant.quantization)
        estimate = (
            estimate_fn(analysis, config)
            if estimate_fn is not None
            else evaluated.memory_estimate
        )
        artifact = ArtifactVariant(
            model_identity=identity,
            repo_id=evaluated.repo_id,
            revision="main",
            format=ArtifactVariantFormat.GGUF,
            filename=variant.files[0].path if variant.files else None,
            size_bytes=variant.total_bytes,
            quantization=variant.quantization,
            source="analysis",
            compatible_runtimes=[RuntimeName.LLAMA_CPP],
            confidence=identity.confidence,
            evidence=list(identity.evidence),
        )
        plans.append(
            build_execution_plan(
                model_identity=identity,
                artifact=artifact,
                runtime=_runtime_for_artifact(artifact),
                memory_prediction=estimate,
                hardware=context.hardware,
                backend_selection=context.backend_selection,
                execution_readiness=_readiness_for(RuntimeName.LLAMA_CPP, context),
            )
        )
    return plans


def _transformers_plan(
    evaluated: EvaluatedCandidate,
    identity: ModelIdentity,
    *,
    context: PlanRankingContext,
) -> ExecutionPlan:
    analysis = evaluated.analysis
    config = evaluated.selected_configuration
    precision = config.precision.value if config is not None and config.precision else None
    filenames = {file.path for file in analysis.files} if analysis is not None else set()
    format_ = (
        ArtifactVariantFormat.SAFETENSORS
        if any(name.endswith(".safetensors") for name in filenames)
        else ArtifactVariantFormat.TRANSFORMERS
    )
    artifact = ArtifactVariant(
        model_identity=identity,
        repo_id=evaluated.repo_id,
        revision="main",
        format=format_,
        filename=evaluated.repo_id,
        size_bytes=analysis.total_size_bytes if analysis is not None else None,
        precision=precision,
        source="analysis",
        compatible_runtimes=[RuntimeName.TRANSFORMERS],
        confidence=identity.confidence,
        evidence=list(identity.evidence),
    )
    return build_execution_plan(
        model_identity=identity,
        artifact=artifact,
        runtime=_runtime_for_artifact(artifact),
        memory_prediction=evaluated.memory_estimate,
        hardware=context.hardware,
        backend_selection=context.backend_selection,
        execution_readiness=_readiness_for(RuntimeName.TRANSFORMERS, context),
    )


def _config_for_variant(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
    *,
    quantization: str | None = None,
    precision: WeightPrecision | None = None,
) -> InferenceConfiguration:
    if evaluated.selected_configuration is not None:
        return evaluated.selected_configuration.model_copy(
            update={
                "quantization": quantization,
                "precision": precision,
            }
        )
    base = InferenceConfiguration(
        context_length=requirements.desired_context,
        concurrent_users=requirements.concurrent_users,
    )
    if evaluated.analysis is not None and evaluated.memory_estimate is not None:
        estimate = evaluated.memory_estimate

        def reuse_estimate(
            analysis: ModelAnalysis,
            config: InferenceConfiguration,
        ) -> MemoryEstimate:
            del analysis, config
            return estimate

        choice = select_configuration(
            evaluated.analysis,
            requirements,
            reuse_estimate,
        )
        base = choice.configuration or base
    return base.model_copy(update={"quantization": quantization, "precision": precision})


def _ordered_gguf_variants(
    variants: Sequence[GgufVariant],
    priority: RecommendationPriority,
) -> list[GgufVariant]:
    ladder = _quantization_ladder(priority)
    rank = {name: index for index, name in enumerate(ladder)}
    return sorted(
        variants,
        key=lambda variant: (
            rank.get(variant.quantization.upper(), len(rank)),
            variant.total_bytes,
            variant.quantization,
        ),
    )


def _quantization_ladder(priority: RecommendationPriority) -> tuple[str, ...]:
    from jaull.estimator import policies

    return policies.QUANTIZATION_LADDERS[priority]


def _runtime_for_artifact(artifact: ArtifactVariant) -> RuntimeRecommendation:
    runtime = (
        RuntimeName.LLAMA_CPP
        if artifact.format is ArtifactVariantFormat.GGUF
        else RuntimeName.TRANSFORMERS
    )
    return RuntimeRecommendation(
        runtime=runtime,
        reasons=[f"{artifact.format.value} artifact is compatible with {runtime.value}."],
        confidence=artifact.confidence,
    )


def _readiness_for(
    runtime: RuntimeName,
    context: PlanRankingContext,
) -> ExecutionReadiness | None:
    if context.readiness_by_runtime is None:
        return None
    return context.readiness_by_runtime.get(runtime)


def _hard_constraints(
    evaluated: EvaluatedCandidate,
    plan: ExecutionPlan,
    requirements: UserRequirements,
) -> list[HardConstraint]:
    constraints: list[HardConstraint] = []
    if plan.runtime.runtime not in plan.artifact.compatible_runtimes:
        constraints.append(
            HardConstraint(
                code=HardConstraintCode.ARTIFACT_RUNTIME_INCOMPATIBLE,
                message="Artifact is not compatible with the selected runtime.",
                evidence_key="artifact.runtime",
            )
        )
    if (
        plan.memory_prediction is not None
        and plan.memory_prediction.assessment.status is CompatibilityStatus.INSUFFICIENT
    ):
        constraints.append(
            HardConstraint(
                code=HardConstraintCode.MEMORY_INSUFFICIENT,
                message="Estimated memory exceeds available hardware memory.",
                evidence_key="memory.estimate",
            )
        )
    if (
        requirements.commercial_use_required
        and classify_license(evaluated.candidate.license)
        is LicenseCategory.COMMERCIAL_RESTRICTED
    ):
        constraints.append(
            HardConstraint(
                code=HardConstraintCode.LICENSE_INCOMPATIBLE,
                message=f"License {evaluated.candidate.license!r} restricts commercial use.",
                evidence_key="license",
            )
        )
    if _declared_language_mismatch(evaluated, requirements):
        constraints.append(
            HardConstraint(
                code=HardConstraintCode.LANGUAGE_INCOMPATIBLE,
                message="Declared model languages do not include the required language.",
                evidence_key="language",
            )
        )
    if (
        plan.execution_readiness is not None
        and plan.execution_readiness.status is ExecutionReadinessStatus.NOT_READY
    ):
        constraints.append(
            HardConstraint(
                code=HardConstraintCode.RUNTIME_NOT_READY,
                message=plan.execution_readiness.message
                or "Runtime is not ready for the selected backend.",
                evidence_key="execution.readiness",
            )
        )
    return constraints


def _base_evidence(
    evaluated: EvaluatedCandidate,
    plan: ExecutionPlan,
) -> list[RecommendationEvidence]:
    evidence = [
        RecommendationEvidence(
            category=RecommendationEvidenceCategory.IDENTITY,
            key="repository",
            value=evaluated.repo_id,
            source=RecommendationEvidenceSource.SEARCH_METADATA,
            confidence=evaluated.candidate.metadata_confidence,
            provenance={"repo_id": evaluated.repo_id},
        ),
        RecommendationEvidence(
            category=RecommendationEvidenceCategory.METADATA,
            key="artifact",
            value=plan.artifact.label,
            source=RecommendationEvidenceSource.MODEL_ANALYSIS,
            confidence=plan.artifact.confidence,
            provenance={"repo_id": plan.artifact.repo_id, "filename": plan.artifact.filename},
        ),
    ]
    if plan.memory_prediction is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.HARDWARE,
                key="memory_estimate",
                value=str(plan.memory_prediction.total_bytes),
                source=RecommendationEvidenceSource.MEMORY_ESTIMATOR,
                confidence=plan.memory_prediction.assessment.confidence,
                provenance={
                    "status": plan.memory_prediction.assessment.status.value,
                    "context": plan.memory_prediction.inference_configuration.context_length,
                },
            )
        )
    return evidence


def _capability_evidence(capability: CapabilitySignal) -> list[RecommendationEvidence]:
    evidence: list[RecommendationEvidence] = []
    if capability.parameter_count is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.METADATA,
                key="parameter_count",
                value=str(capability.parameter_count),
                source=RecommendationEvidenceSource.MODEL_ANALYSIS,
                confidence=capability.confidence,
            )
        )
    if capability.scale_tier is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.METADATA,
                key="model_scale",
                value=capability.scale_tier,
                source=RecommendationEvidenceSource.POLICY,
                confidence=capability.confidence,
                provenance={"parameter_count": capability.parameter_count},
            )
        )
    if capability.chat_or_base is not None:
        evidence.append(
            RecommendationEvidence(
                category=RecommendationEvidenceCategory.METADATA,
                key="assistant_tuning",
                value=capability.chat_or_base,
                source=RecommendationEvidenceSource.HEURISTIC,
                confidence=EstimationConfidence.LOW,
            )
        )
    return evidence


def _external_evidence(
    external: Sequence[ExternalEvaluationEvidence],
) -> list[RecommendationEvidence]:
    return [
        RecommendationEvidence(
            category=RecommendationEvidenceCategory.EXTERNAL,
            key=f"external:{item.benchmark}:{item.task}",
            value=f"{item.value} {item.metric}",
            source=RecommendationEvidenceSource.EXTERNAL_EVALUATION,
            confidence=item.confidence,
            provenance={
                "benchmark": item.benchmark,
                "task": item.task,
                "source": item.source,
                "revision": item.revision,
            },
        )
        for item in external
    ]


def _suitability(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
) -> AssessmentLevel:
    task_score = evaluated.task_match_score
    if requirements.use_case in {
        UseCase.GENERAL_CHAT,
        UseCase.REASONING,
        UseCase.DOCUMENT_QA,
    }:
        tokens = _candidate_signal_tokens(evaluated)
        has_chat_signal = any(
            _has_signal(tokens, token)
            for token in ("chat", "assistant", "conversational")
        )
        has_instruction_signal = has_chat_signal or any(
            _has_signal(tokens, token) for token in ("instruct", "instruction")
        )
        has_base_signal = any(
            _has_signal(tokens, token) for token in ("base", "pretrain", "foundation")
        )
        if has_instruction_signal:
            task_score = max(task_score, 0.65)
            if not has_chat_signal:
                task_score = min(task_score, 0.70)
        elif has_base_signal:
            task_score = min(task_score, 0.40)
    if task_score >= 0.75:
        return AssessmentLevel.STRONG
    if task_score >= 0.45:
        return AssessmentLevel.ADEQUATE
    if task_score > 0:
        return AssessmentLevel.WEAK
    if evaluated.candidate.metadata_confidence is EstimationConfidence.LOW:
        return AssessmentLevel.UNKNOWN
    return AssessmentLevel.WEAK


def _candidate_signal_tokens(evaluated: EvaluatedCandidate) -> set[str]:
    values = [
        evaluated.repo_id.split("/")[-1],
        evaluated.candidate.pipeline_tag or "",
        evaluated.candidate.library_name or "",
        *evaluated.candidate.tags,
    ]
    tokens: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.add(normalized.replace("_", "-"))
        tokens.update(token for token in re.split(r"[^a-z0-9.]+", normalized) if token)
    return tokens


def _has_signal(tokens: set[str], signal: str) -> bool:
    normalized = signal.lower()
    return normalized in tokens or normalized.replace("_", "-") in tokens


def _capability_level(capability: CapabilitySignal) -> AssessmentLevel:
    if capability.score >= 0.72:
        return AssessmentLevel.STRONG
    if capability.score >= 0.55:
        return AssessmentLevel.ADEQUATE
    if capability.score > 0:
        return AssessmentLevel.WEAK
    return AssessmentLevel.UNKNOWN


def _execution_fitness(plan: ExecutionPlan) -> AssessmentLevel:
    executability = _executability(plan)
    feasibility = _feasibility(plan)
    if AssessmentLevel.BLOCKED in {executability, feasibility}:
        return AssessmentLevel.BLOCKED
    if executability is AssessmentLevel.UNKNOWN:
        return AssessmentLevel.UNKNOWN
    if feasibility is AssessmentLevel.STRONG:
        return AssessmentLevel.STRONG
    if feasibility is AssessmentLevel.ADEQUATE:
        return AssessmentLevel.ADEQUATE
    if feasibility is AssessmentLevel.WEAK:
        return AssessmentLevel.WEAK
    return AssessmentLevel.UNKNOWN


def _feasibility(plan: ExecutionPlan) -> AssessmentLevel:
    if plan.memory_prediction is None:
        return AssessmentLevel.UNKNOWN
    status = plan.memory_prediction.assessment.status
    if status is CompatibilityStatus.COMFORTABLE:
        return AssessmentLevel.STRONG
    if status in {CompatibilityStatus.COMPATIBLE, CompatibilityStatus.TIGHT}:
        return AssessmentLevel.ADEQUATE
    if status is CompatibilityStatus.OFFLOADING_REQUIRED:
        return AssessmentLevel.WEAK
    if status is CompatibilityStatus.INSUFFICIENT:
        return AssessmentLevel.BLOCKED
    return AssessmentLevel.UNKNOWN


def _executability(plan: ExecutionPlan) -> AssessmentLevel:
    if plan.compatibility is PlanCompatibilityStatus.NOT_COMPATIBLE:
        return AssessmentLevel.BLOCKED
    if plan.execution_readiness is None:
        return AssessmentLevel.UNKNOWN
    if plan.execution_readiness.status is ExecutionReadinessStatus.READY:
        return AssessmentLevel.STRONG
    if plan.execution_readiness.status is ExecutionReadinessStatus.NOT_READY:
        return AssessmentLevel.BLOCKED
    return AssessmentLevel.UNKNOWN


def _resource_evidence_from_estimate(plan: ExecutionPlan) -> AssessmentLevel:
    if plan.memory_prediction is None:
        return AssessmentLevel.UNKNOWN
    if plan.memory_prediction.assessment.confidence is EstimationConfidence.HIGH:
        return AssessmentLevel.STRONG
    if plan.memory_prediction.assessment.confidence is EstimationConfidence.MEDIUM:
        return AssessmentLevel.ADEQUATE
    return AssessmentLevel.WEAK


def _license_fit(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
) -> AssessmentLevel:
    category = classify_license(evaluated.candidate.license)
    if category is LicenseCategory.COMMERCIAL_ALLOWED:
        return AssessmentLevel.STRONG
    if category is LicenseCategory.COMMERCIAL_RESTRICTED:
        if requirements.commercial_use_required:
            return AssessmentLevel.BLOCKED
        return AssessmentLevel.WEAK
    return AssessmentLevel.UNKNOWN


def _language_fit(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
) -> AssessmentLevel:
    if not requirements.languages:
        return AssessmentLevel.UNKNOWN
    if _declared_language_mismatch(evaluated, requirements):
        return AssessmentLevel.BLOCKED
    declared = {language.lower().split("-")[0] for language in evaluated.candidate.languages}
    if declared and all(language.split("-")[0] in declared for language in requirements.languages):
        return AssessmentLevel.STRONG
    tags = {tag.lower() for tag in evaluated.candidate.tags}
    if "multilingual" in tags:
        return AssessmentLevel.ADEQUATE
    return AssessmentLevel.UNKNOWN


def _declared_language_mismatch(
    evaluated: EvaluatedCandidate,
    requirements: UserRequirements,
) -> bool:
    if not requirements.languages or not evaluated.candidate.languages:
        return False
    declared = {language.lower().split("-")[0] for language in evaluated.candidate.languages}
    if "multilingual" in {tag.lower() for tag in evaluated.candidate.tags}:
        return False
    return not any(language.split("-")[0] in declared for language in requirements.languages)


def _confidence(
    evaluated: EvaluatedCandidate,
    plan: ExecutionPlan,
    experiment: ExperimentRecord | None,
    benchmark: BenchmarkRecord | None,
) -> EstimationConfidence:
    if experiment is not None or benchmark is not None:
        return EstimationConfidence.HIGH
    if (
        plan.execution_readiness is not None
        and plan.execution_readiness.status is ExecutionReadinessStatus.READY
        and evaluated.candidate.metadata_confidence is EstimationConfidence.UNKNOWN
    ):
        return EstimationConfidence.MEDIUM
    if plan.memory_prediction is not None:
        return _weaker_confidence(
            evaluated.candidate.metadata_confidence,
            plan.memory_prediction.assessment.confidence,
        )
    return evaluated.candidate.metadata_confidence


def _reasons(
    evaluated: EvaluatedCandidate,
    plan: ExecutionPlan,
    experiment: ExperimentRecord | None,
    benchmark: BenchmarkRecord | None,
    measured_memory: int | None,
) -> list[str]:
    reasons: list[str] = []
    if plan.memory_prediction is not None:
        reasons.append(
            f"memory:{plan.memory_prediction.assessment.status.value}"
        )
    if plan.execution_readiness is not None:
        reasons.append(f"runtime:{plan.execution_readiness.status.value}")
    if evaluated.task_match_score >= 0.45:
        reasons.append("use_case:metadata_match")
    if evaluated.language_match_score >= 0.6:
        reasons.append("language:metadata_match")
    if experiment is not None:
        reasons.append("local:validated")
    if benchmark is not None:
        reasons.append("local:benchmarked")
    if measured_memory is not None:
        reasons.append("memory:measured")
    return reasons


def _trade_offs(
    plan: ExecutionPlan,
    external: Sequence[ExternalEvaluationEvidence],
) -> list[str]:
    trade_offs: list[str] = []
    if plan.artifact.quantization is not None:
        trade_offs.append(f"quantized:{plan.artifact.quantization}")
    if plan.artifact.precision in {"int8", "int4"}:
        trade_offs.append(f"runtime_quantization:{plan.artifact.precision}")
    if not external:
        trade_offs.append("external_quality_evidence:incomplete")
    if plan.execution_readiness is None:
        trade_offs.append("runtime_readiness:not_checked")
    return trade_offs


def _with_capability_reasons(
    reasons: list[str],
    capability: CapabilitySignal,
) -> list[str]:
    updated = list(reasons)
    if capability.scale_tier is not None:
        updated.append(f"capability:scale:{capability.scale_tier}")
    if capability.chat_or_base in {"chat", "instruction"}:
        updated.append(f"capability:{capability.chat_or_base}_tuned")
    return updated


def _with_capability_trade_offs(
    trade_offs: list[str],
    capability: CapabilitySignal,
) -> list[str]:
    updated = list(trade_offs)
    if capability.scale_tier == "lightweight":
        updated.append("capability:lightweight_model")
    if capability.chat_or_base == "base":
        updated.append("capability:base_model")
    return updated


def _matching_experiment(
    plan: ExecutionPlan,
    records: Sequence[ExperimentRecord],
) -> ExperimentRecord | None:
    compatible = [
        record
        for record in records
        if _artifact_matches(plan.artifact.to_model_artifact(), record.artifact)
        and record.runtime.runtime is plan.runtime.runtime
        and _hardware_matches(plan.hardware, record.hardware)
    ]
    return max(compatible, key=lambda record: record.identity.created_at, default=None)


def _matching_benchmark(
    plan: ExecutionPlan,
    records: Sequence[BenchmarkRecord],
) -> BenchmarkRecord | None:
    compatible = [
        record
        for record in records
        if _artifact_matches(plan.artifact.to_model_artifact(), record.artifact)
        and record.runtime.runtime is plan.runtime.runtime
        and _hardware_matches(plan.hardware, record.hardware)
        and (
            plan.backend_selection is None
            or record.requested_backend is plan.backend_selection.selected_backend
        )
    ]
    return max(compatible, key=lambda record: record.identity.created_at, default=None)


def _artifact_matches(left: ModelArtifact, right: ModelArtifact) -> bool:
    return (
        left.repo_id == right.repo_id
        and left.filename == right.filename
        and left.format == right.format
        and left.quantization == right.quantization
    )


def _hardware_matches(
    plan_hardware: HardwareProfile | None,
    record_hardware: HardwareProfile,
) -> bool:
    if plan_hardware is None:
        return True
    return machine_fingerprint(plan_hardware) == machine_fingerprint(record_hardware)


def _measured_memory(
    experiment: ExperimentRecord | None,
    benchmark: BenchmarkRecord | None,
) -> int | None:
    values: list[int] = []
    if experiment is not None:
        values.extend(
            value
            for value in (
                experiment.observation.peak_ram_bytes,
                experiment.observation.peak_vram_bytes,
            )
            if value is not None
        )
    if benchmark is not None:
        values.extend(
            value
            for value in (
                benchmark.observation.peak_ram_bytes,
                benchmark.observation.peak_vram_bytes,
            )
            if value is not None
        )
    return max(values) if values else None


def _measured_generation_tps(benchmark: BenchmarkRecord | None) -> float | None:
    if benchmark is None:
        return None
    generation = [
        measurement.mean_tokens_per_second
        for measurement in benchmark.observation.measurements
        if measurement.kind is BenchmarkMeasurementKind.GENERATION
    ]
    return max(generation) if generation else None


_LEVEL_RANK = {
    AssessmentLevel.STRONG: 0,
    AssessmentLevel.ADEQUATE: 1,
    AssessmentLevel.WEAK: 2,
    AssessmentLevel.UNKNOWN: 3,
    AssessmentLevel.BLOCKED: 4,
}

_CONFIDENCE_RANK = {
    EstimationConfidence.HIGH: 0,
    EstimationConfidence.MEDIUM: 1,
    EstimationConfidence.LOW: 2,
    EstimationConfidence.UNKNOWN: 3,
}


def _ranking_key(item: RankedPlan, priority: RecommendationPriority) -> tuple[object, ...]:
    assessment = item.assessment
    if priority is RecommendationPriority.QUALITY:
        priority_axes: tuple[int | float, ...] = (
            _LEVEL_RANK[assessment.suitability],
            _LEVEL_RANK[assessment.capability],
            _quantization_quality_rank(item.plan),
            _LEVEL_RANK[assessment.execution_fitness],
        )
    elif priority is RecommendationPriority.SPEED:
        priority_axes = (
            _LEVEL_RANK[assessment.executability],
            _throughput_rank(assessment),
            _priority_quantization_rank(item.plan, priority),
            _memory_efficiency_rank(assessment),
        )
    elif priority is RecommendationPriority.MEMORY:
        priority_axes = (
            _memory_efficiency_rank(assessment),
            _LEVEL_RANK[assessment.execution_fitness],
            _LEVEL_RANK[assessment.executability],
        )
    else:
        priority_axes = (
            _LEVEL_RANK[assessment.suitability],
            _LEVEL_RANK[assessment.capability],
            _LEVEL_RANK[assessment.execution_fitness],
            _LEVEL_RANK[assessment.executability],
            _comfortable_memory_rank(assessment),
        )
    return (
        1 if assessment.rejected else 0,
        *priority_axes,
        _LEVEL_RANK[assessment.performance_evidence],
        _CONFIDENCE_RANK[assessment.confidence],
        item.evaluated.repo_id,
        item.plan.artifact.quantization or item.plan.artifact.precision or "",
        item.plan.runtime.runtime.value,
    )


def _quantization_quality_rank(plan: ExecutionPlan) -> int:
    quant = plan.artifact.quantization
    if quant is None:
        return 0
    order = (
        "F16",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q5_K_S",
        "Q4_K_M",
        "Q4_K_S",
        "Q3_K_M",
        "Q3_K_S",
        "Q2_K",
    )
    try:
        return order.index(quant.upper())
    except ValueError:
        return len(order)


def _priority_quantization_rank(
    plan: ExecutionPlan,
    priority: RecommendationPriority,
) -> int:
    quant = plan.artifact.quantization
    if quant is None:
        return len(_quantization_ladder(priority))
    ladder = _quantization_ladder(priority)
    try:
        return ladder.index(quant.upper())
    except ValueError:
        return len(ladder)


def _throughput_rank(assessment: PlanAssessment) -> float:
    if assessment.measured_generation_tokens_per_second is None:
        return 10_000.0
    return -assessment.measured_generation_tokens_per_second


def _memory_efficiency_rank(assessment: PlanAssessment) -> int:
    memory = assessment.measured_memory_bytes or assessment.estimated_memory_bytes
    return memory if memory is not None else 10**18


def _comfortable_memory_rank(assessment: PlanAssessment) -> int:
    """Diminishing memory utility once a plan is comfortably runnable.

    Balanced ranking should care a lot about impossible/tight plans, but should
    not keep rewarding every extra unused GiB after the plan already fits well.
    """
    if assessment.feasibility is AssessmentLevel.STRONG:
        return 0
    if assessment.feasibility is AssessmentLevel.ADEQUATE:
        return 1
    if assessment.feasibility is AssessmentLevel.WEAK:
        return 2
    if assessment.feasibility is AssessmentLevel.UNKNOWN:
        return 3
    return 4


def _weaker_confidence(
    left: EstimationConfidence,
    right: EstimationConfidence,
) -> EstimationConfidence:
    return left if _CONFIDENCE_RANK[left] >= _CONFIDENCE_RANK[right] else right


__all__ = [
    "EstimatePlanFn",
    "PlanRankingContext",
    "RankedPlan",
    "assess_plan",
    "generate_execution_plans",
    "rank_execution_plans",
]
