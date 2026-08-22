"""Recommendation must not depend on what is installed on this machine.

The Top 5 answers "what are the best execution plans for this hardware?", which
is a question about the hardware and the model, not about whether llama.cpp
happens to be on PATH right now. Installing a runtime may change which actions
are available; it must not change which models are recommended or in what order.

These tests pin that invariant by running the same candidates through two
ranking contexts that differ *only* in readiness, and asserting the outcome is
identical. Launching is tested separately: it still requires the runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

from jaull.domain.execution_plans import (
    ArtifactVariantFormat,
    model_identity_key,
)
from jaull.domain.hardware import ComputeBackend
from jaull.domain.recommendation import (
    AssessmentLevel,
    HardConstraintCode,
    PlanAssessment,
)
from jaull.domain.runtime import ExecutionReadinessStatus, RuntimeName
from jaull.presentation.plan_labels import runtime_block_reason
from jaull.recommendation.engine_v2 import (
    PlanRankingContext,
    assess_plan,
    generate_execution_plans,
)
from jaull.workflow import ranking

# Reusing the engine-v2 builders keeps candidate construction in one place; a
# second copy would drift from the fixtures the rest of the suite ranks against.
from tests._workflow_fixtures import hardware
from tests.test_recommendation_engine_v2 import (
    _evaluated_gguf,
    _evaluated_transformers,
    _readiness,
    _requirements,
    _selection,
)

_READY = ExecutionReadinessStatus.READY
_NOT_READY = ExecutionReadinessStatus.NOT_READY


def _context(status: ExecutionReadinessStatus) -> PlanRankingContext:
    """A CUDA machine where both runtimes are in the same readiness state."""
    selection = _selection(ComputeBackend.CUDA)
    return PlanRankingContext(
        hardware=hardware(),
        backend_selection=selection,
        readiness_by_runtime={
            runtime: _readiness(runtime, selection, status)
            for runtime in (RuntimeName.LLAMA_CPP, RuntimeName.TRANSFORMERS)
        },
    )


def _candidates():
    return [
        _evaluated_gguf(repo_id="org/Coder-7B-GGUF"),
        _evaluated_gguf(repo_id="org/Helper-3B-GGUF"),
        _evaluated_transformers(repo_id="org/Coder-7B"),
    ]


def _semantic_plan(plan) -> tuple[str, ...]:
    """What the user actually chose between, minus anything operational."""
    return (
        model_identity_key(plan.model_identity),
        plan.artifact.format.value,
        plan.artifact.quantization or plan.artifact.precision or "",
        plan.runtime_family.value,
        (
            plan.backend_selection.selected_backend.value
            if plan.backend_selection is not None
            else ""
        ),
    )


def _assessment_axes(assessment: PlanAssessment) -> dict[str, object]:
    return {
        "suitability": assessment.suitability,
        "capability": assessment.capability,
        "execution_fitness": assessment.execution_fitness,
        "feasibility": assessment.feasibility,
        "executability": assessment.executability,
        "confidence": assessment.confidence,
        "rejected": assessment.rejected,
        "hard_constraints": sorted(c.code.value for c in assessment.hard_constraints),
    }


# --------------------------------------------------------------------------
# 1 & 9. Plans exist for a runtime that is not installed
# --------------------------------------------------------------------------


def test_gguf_plan_is_built_and_kept_without_llama_cpp() -> None:
    evaluated = _evaluated_gguf()
    context = _context(_NOT_READY)

    plans = generate_execution_plans(evaluated, _requirements(), context=context)

    assert plans
    plan = plans[0]
    assert plan.runtime_family is RuntimeName.LLAMA_CPP
    assert plan.artifact.format is ArtifactVariantFormat.GGUF
    assert plan.backend_selection is not None
    assert plan.backend_selection.selected_backend is ComputeBackend.CUDA
    assert plan.execution_readiness is not None
    assert plan.execution_readiness.status is _NOT_READY

    assessment = assess_plan(evaluated, plan, _requirements(), context=context)
    assert not assessment.rejected


def test_transformers_plan_is_built_and_kept_without_a_ready_runtime() -> None:
    evaluated = _evaluated_transformers()
    context = _context(_NOT_READY)

    plans = generate_execution_plans(evaluated, _requirements(), context=context)

    assert plans
    plan = plans[0]
    assert plan.runtime_family is RuntimeName.TRANSFORMERS
    assert plan.backend_selection is not None
    assert plan.backend_selection.selected_backend is ComputeBackend.CUDA

    assessment = assess_plan(evaluated, plan, _requirements(), context=context)
    assert not assessment.rejected


# --------------------------------------------------------------------------
# 2. Ranking invariant
# --------------------------------------------------------------------------


def test_top_n_is_identical_with_and_without_the_runtime_installed() -> None:
    ready = ranking.recommend(
        _candidates(),
        _requirements(),
        hardware=hardware(),
        plan_context=_context(_READY),
    )
    missing = ranking.recommend(
        _candidates(),
        _requirements(),
        hardware=hardware(),
        plan_context=_context(_NOT_READY),
    )

    assert ready, "the ready run must produce recommendations to compare against"
    assert [rec.repo_id for rec in missing] == [rec.repo_id for rec in ready]
    assert [_semantic_plan(rec.plan) for rec in missing] == [
        _semantic_plan(rec.plan) for rec in ready
    ]


def test_a_gguf_only_shortlist_still_recommends_without_llama_cpp() -> None:
    """The regression that motivated this: GGUF repos vanished entirely."""
    gguf_only = [
        _evaluated_gguf(repo_id="org/Coder-7B-GGUF"),
        _evaluated_gguf(repo_id="org/Helper-3B-GGUF"),
    ]

    recs = ranking.recommend(
        gguf_only,
        _requirements(),
        hardware=hardware(),
        plan_context=_context(_NOT_READY),
    )

    assert recs
    assert all(rec.plan.runtime_family is RuntimeName.LLAMA_CPP for rec in recs)


# --------------------------------------------------------------------------
# 3 & 4. Assessment invariant, and no runtime hard gate
# --------------------------------------------------------------------------


def test_plan_assessment_differs_only_in_runtime_readiness() -> None:
    evaluated = _evaluated_gguf()
    ready_context = _context(_READY)
    missing_context = _context(_NOT_READY)
    ready_plan = generate_execution_plans(
        evaluated, _requirements(), context=ready_context
    )[0]
    missing_plan = generate_execution_plans(
        evaluated, _requirements(), context=missing_context
    )[0]

    ready = assess_plan(evaluated, ready_plan, _requirements(), context=ready_context)
    missing = assess_plan(
        evaluated, missing_plan, _requirements(), context=missing_context
    )

    assert _assessment_axes(missing) == _assessment_axes(ready)
    assert ready.runtime_readiness is AssessmentLevel.STRONG
    assert missing.runtime_readiness is AssessmentLevel.BLOCKED


def test_missing_runtime_is_never_a_hard_constraint() -> None:
    for evaluated in (_evaluated_gguf(), _evaluated_transformers()):
        context = _context(_NOT_READY)
        plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

        assessment = assess_plan(evaluated, plan, _requirements(), context=context)

        assert HardConstraintCode.RUNTIME_NOT_READY not in {
            constraint.code for constraint in assessment.hard_constraints
        }
        assert not assessment.rejected


def test_executability_reports_technical_coherence_not_installation() -> None:
    evaluated = _evaluated_gguf()
    context = _context(_NOT_READY)
    plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

    assessment = assess_plan(evaluated, plan, _requirements(), context=context)

    assert assessment.executability is AssessmentLevel.STRONG
    assert assessment.execution_fitness is not AssessmentLevel.BLOCKED


# --------------------------------------------------------------------------
# 5, 6 & 7. Launching still requires the runtime
# --------------------------------------------------------------------------
# The controlled failure itself is already covered where it is enforced:
# tests/test_experiment_runner.py asserts ExperimentNotReadyError is raised and
# the runner is never invoked, and tests/test_benchmarks.py asserts
# BenchmarkUnavailableError. What is new here is that the very same plan is
# recommendable, so the block has to come from readiness rather than rejection.


def test_a_recommended_plan_can_still_be_blocked_from_launching() -> None:
    evaluated = _evaluated_gguf()
    context = _context(_NOT_READY)
    plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

    assert not assess_plan(evaluated, plan, _requirements(), context=context).rejected

    reason = runtime_block_reason(plan)
    assert reason is not None
    # Neutral wording on purpose: Run, Validate and Benchmark share this helper.
    assert "This execution plan is not ready on this machine." in reason
    assert RuntimeName.LLAMA_CPP.value in reason
    assert ComputeBackend.CUDA.value.upper() in reason


def test_a_ready_plan_is_not_blocked_from_launching() -> None:
    evaluated = _evaluated_gguf()
    context = _context(_READY)
    plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

    assert runtime_block_reason(plan) is None


def test_unknown_readiness_does_not_block_the_attempt() -> None:
    """An undecided probe is not evidence of absence; let the runtime answer."""
    evaluated = _evaluated_gguf()
    context = _context(ExecutionReadinessStatus.UNKNOWN)
    plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

    assert runtime_block_reason(plan) is None


# --------------------------------------------------------------------------
# 8. Download does not require a runtime
# --------------------------------------------------------------------------


def test_artifact_layer_does_not_depend_on_any_runtime() -> None:
    """Resolving and downloading a GGUF must not need llama.cpp to exist.

    Asserted structurally rather than by mocking a download: the guarantee is
    that no import path from the artifact layer can reach a runtime probe, and
    an import scan proves that for every code path at once.
    """
    root = Path(__file__).resolve().parents[1]
    forbidden = ("jaull.runtime", "jaull.execution")
    offenders: list[str] = []

    for path in (root / "src" / "jaull" / "artifacts").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = node.module
            elif isinstance(node, ast.Import):
                imported = node.names[0].name
            else:
                continue
            if imported.startswith(forbidden):
                offenders.append(f"{path.name} -> {imported}")

    assert not offenders, f"artifact layer reached a runtime: {offenders}"


# --------------------------------------------------------------------------
# 10. The ready path is unchanged
# --------------------------------------------------------------------------


def test_ready_runtime_keeps_the_existing_contract() -> None:
    evaluated = _evaluated_gguf()
    context = _context(_READY)
    plan = generate_execution_plans(evaluated, _requirements(), context=context)[0]

    assessment = assess_plan(evaluated, plan, _requirements(), context=context)

    assert plan.execution_readiness is not None
    assert plan.execution_readiness.status is _READY
    assert assessment.executability is AssessmentLevel.STRONG
    assert assessment.runtime_readiness is AssessmentLevel.STRONG
    assert not assessment.rejected
