"""Explicit, bounded evaluation selection from an existing search, not a new ranker."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from jaull.advisor.quality import check_selected_artifact
from jaull.application.recommendation import policies, service
from jaull.domain.estimation import CompatibilityStatus, HardwareFitMode
from jaull.domain.execution_plans import (
    ArtifactVariantFormat,
    ExecutionPlan,
    IdentityMatchStatus,
    model_identity_key,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.inference import InferenceConfiguration
from jaull.estimator.policies import (
    DEVICE_RESERVE_DEFAULT_BYTES,
    QUANTIZATION_LADDERS,
    SAFETY_MARGIN_DEFAULT_PERCENT,
)
from jaull.exceptions import JaullError
from jaull.recommendation.policies import CONFIRMED_MEMORY_STATUSES
from jaull.runtime.quality_eval_runner import QualityEvaluationCancelled, QualityRunRequest

if TYPE_CHECKING:
    from jaull.advisor.service import AdvisorService
    from jaull.domain.hardware import HardwareProfile
    from jaull.domain.requirements import UserRequirements
    from jaull.evaluation.quality_records import QualityEvidence
    from jaull.workflow.state import RecommendationWorkflowState

MAX_QUALITY_CANDIDATES = 3
EVALUATION_CONTEXT = 2048


@dataclass(frozen=True)
class QualityCandidate:
    plan: ExecutionPlan
    reason: str
    downloaded: bool
    historical_records: int = 0


@dataclass(frozen=True)
class QualityCandidateSelection:
    candidates: tuple[QualityCandidate, ...] = ()
    notices: tuple[str, ...] = ()
    considered: int = 0


@dataclass(frozen=True)
class QualityCandidateOutcome:
    plan: ExecutionPlan
    path: Path | None = None
    error: str | None = None


def _prepare_candidate(
    advisor: AdvisorService, plan: ExecutionPlan, hardware: HardwareProfile,
    requirements: UserRequirements, seen: set[str], evidence: list[QualityEvidence],
) -> QualityCandidate:
    variant = plan.artifact
    if not variant.filename or variant.file_count != 1:
        raise ValueError("single-file GGUF required")
    artifact = advisor._artifacts().resolver.resolve(
        variant.repo_id, quantization=variant.quantization,
        revision=variant.revision if variant.revision not in (None, "main") else None,
    )
    check_selected_artifact(artifact, variant)
    if re.fullmatch(r"[0-9a-f]{40}", artifact.revision) is None:
        raise ValueError("immutable artifact revision unavailable")
    if not artifact.sha256 or not artifact.size_bytes or artifact.sha256 in seen:
        raise ValueError("missing digest/size or duplicate artifact")
    analysis = advisor.inspect_model(artifact.repo_id)
    exact = [item for item in analysis.classification.gguf_variants
             if item.quantization == artifact.quantization
             and len(item.files) == 1 and item.files[0].path == artifact.filename
             and item.sha256 == artifact.sha256 and item.total_bytes == artifact.size_bytes]
    if len(exact) != 1:
        raise ValueError("exact artifact digest/size metadata unavailable")
    exact_analysis = analysis.model_copy(update={
        "classification": analysis.classification.model_copy(update={"gguf_variants": exact}),
    })
    config = InferenceConfiguration(
        context_length=EVALUATION_CONTEXT, quantization=artifact.quantization,
        device_reserve_bytes=DEVICE_RESERVE_DEFAULT_BYTES,
        safety_margin_percent=SAFETY_MARGIN_DEFAULT_PERCENT,
    )
    estimate = advisor.estimate_model(exact_analysis, hardware, config, recommend_runtime=False)
    if (estimate.hardware_fit is None
            or estimate.hardware_fit.mode is not HardwareFitMode.GPU_RESIDENT
            or estimate.assessment.status not in {
                CompatibilityStatus.COMFORTABLE, CompatibilityStatus.COMPATIBLE,
                CompatibilityStatus.TIGHT,
            }):
        raise ValueError("full-device fit unconfirmed at evaluation context 2048")
    workload_config = config.model_copy(update={
        "context_length": requirements.desired_context,
        "concurrent_users": requirements.concurrent_users,
    })
    workload_estimate = estimate if workload_config == config else advisor.estimate_model(
        exact_analysis, hardware, workload_config, recommend_runtime=False,
    )
    if workload_estimate.assessment.status not in CONFIRMED_MEMORY_STATUSES:
        raise ValueError("fit unconfirmed for the requested search workload")
    pinned = plan.model_copy(update={
        "artifact": variant.model_copy(update={
            "revision": artifact.revision, "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }), "memory_prediction": workload_estimate,
    })
    return QualityCandidate(
        pinned, "Search priority/diversity; workload fits; full-device evaluation fit at ctx 2048.",
        advisor._artifacts().is_downloaded(artifact),
        sum(item.artifact_sha256 == artifact.sha256 for item in evidence),
    )


def prepare_candidates(
    advisor: AdvisorService, state: RecommendationWorkflowState, *,
    is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> QualityCandidateSelection:
    """Inspect metadata only. Never download a model or launch the evaluator."""
    def report(text: str) -> None:
        if is_cancelled is not None and is_cancelled():
            raise QualityEvaluationCancelled("Candidate selection cancelled.")
        if on_progress is not None:
            on_progress(text)

    if state.hardware is None or state.requirements is None:
        return QualityCandidateSelection(notices=("Search hardware/requirements unavailable.",))
    report("Preparing candidates from the inspected search pool")
    context = advisor._plan_ranking_context(state.hardware)
    selection = context.backend_selection
    if (selection is None or selection.selected_backend is not ComputeBackend.CUDA
            or len(state.hardware.gpus) != 1):
        return QualityCandidateSelection(notices=(
            "Quality evaluation requires one CUDA GPU; multi-device selection is unsupported.",
        ))
    # Request every inspected logical model, not just the displayed top five.
    # Existing eligibility, ordering and diversity stay owned by recommendation.
    pool = service.recommend(
        state.evaluated_candidates, state.requirements,
        limit=len(state.evaluated_candidates),
        capability_analyzer=advisor.services.capability_analyzer,
        hardware=state.hardware,
        plan_context=context,
    ) if state.evaluated_candidates else list(state.recommendations)
    evidence = advisor.quality_evidence()
    chosen: list[QualityCandidate] = []
    notices: list[str] = []
    seen: set[str] = set()
    inspected = 0
    for rec in pool:
        if (len(chosen) == MAX_QUALITY_CANDIDATES
                or inspected == policies.MAX_VARIANT_DEEP_INSPECTION):
            break
        inspected += 1
        report(f"Inspecting evaluation paths: {rec.repo_id}")
        try:
            plans = advisor.execution_plans_for_recommendation(rec)
            ladder = QUANTIZATION_LADDERS[state.requirements.priority]
            plans = sorted(plans, key=lambda plan: (
                ladder.index(plan.artifact.quantization)
                if plan.artifact.quantization in ladder else len(ladder), plan.plan_id,
            ))
            failures: list[str] = []
            for plan in plans:
                report(f"Checking {plan.artifact.repo_id} / {plan.artifact.label}")
                variant = plan.artifact
                if (variant.format is not ArtifactVariantFormat.GGUF
                        or variant.identity_match is not IdentityMatchStatus.CONFIRMED
                        or model_identity_key(plan.model_identity) != model_identity_key(
                            advisor.resolve_model_identity(rec))):
                    continue
                try:
                    candidate = _prepare_candidate(
                        advisor, plan, state.hardware, state.requirements, seen, evidence,
                    )
                except (JaullError, OSError, ValueError) as exc:
                    if isinstance(exc, QualityEvaluationCancelled):
                        raise
                    failures.append(f"{variant.label}: {exc}")
                    continue
                chosen.append(candidate)
                assert candidate.plan.artifact.sha256 is not None
                seen.add(candidate.plan.artifact.sha256)
                break
            else:
                notices.append(f"{rec.repo_id}: " + "; ".join(dict.fromkeys(failures or [
                    "no confirmed GGUF path found",
                ])))
        except QualityEvaluationCancelled:
            raise
        except (JaullError, OSError, ValueError) as exc:
            notices.append(f"{rec.repo_id}: {exc}")
    report("Candidate selection ready; no models downloaded")
    if inspected < len(pool):
        notices.append(
            f"{len(pool) - inspected} models not inspected: selection budget reached."
        )
    return QualityCandidateSelection(tuple(chosen), tuple(notices), len(pool))


def run_candidates(
    advisor: AdvisorService, candidates: tuple[QualityCandidate, ...],
    request: QualityRunRequest, *,
    allow_download: bool = False, is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[QualityCandidateOutcome, ...]:
    """Sequential, explicit runs. Each success persists before advancing to the next."""
    if not 1 <= len(candidates) <= MAX_QUALITY_CANDIDATES:
        raise ValueError("Select between one and three evaluation candidates.")
    root = request.output.expanduser()
    if root.exists():
        raise FileExistsError(root)
    advisor.prepare_quality_evaluation_setup(
        request, is_cancelled=is_cancelled, on_progress=on_progress,
    )
    root.mkdir(parents=True, exist_ok=False)
    outcomes: list[QualityCandidateOutcome] = []
    for index, candidate in enumerate(candidates, 1):
        if is_cancelled is not None and is_cancelled():
            break
        prefix = f"{index}/{len(candidates)} {candidate.plan.model_identity.model_name}"
        def report(text: str, prefix: str = prefix) -> None:
            if on_progress is not None:
                on_progress(f"{prefix}: {text}")
        report("Preparing evaluation")
        try:
            path = advisor.run_quality_evaluation_for_plan(
                candidate.plan, replace(request, output=root / f"model-{index:02d}"),
                allow_download=allow_download, is_cancelled=is_cancelled, on_progress=report,
            )
            outcomes.append(QualityCandidateOutcome(candidate.plan, path=path))
            report(f"Saved: {path}")
        except QualityEvaluationCancelled:
            outcomes.append(QualityCandidateOutcome(candidate.plan, error="Cancelled"))
            break
        except (JaullError, OSError, ValueError) as exc:
            outcomes.append(QualityCandidateOutcome(candidate.plan, error=str(exc)))
            report(f"Failed: {exc}")
    return tuple(outcomes)
