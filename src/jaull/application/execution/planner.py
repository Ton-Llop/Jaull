"""Turn hardware + estimate + explicit overrides into a RUN-READY launch plan.

One authority, used by both front-ends:

* :func:`plan_launch` returns the ``RuntimeRecommendation`` the runner consumes.
  For llama.cpp it is guaranteed run-ready: ``--ctx-size`` and ``--n-gpu-layers``
  are always present, or :class:`ExecutionPlanningError` is raised. The runner's
  own defensive defaults are never reached from here.
* :func:`plan_execution` wraps that launch in the existing :class:`ExecutionPlan`
  via ``execution_plans.service.build_execution_plan``. It accepts a pre-resolved
  ``launch`` so a caller that needs the runtime *before* the plan (to prepare an
  artifact) does not evaluate the policy twice.

Backend dispatch is **not** re-implemented here. The automatic base comes from
``estimate.runtime_recommendation`` (already computed by the estimator) or, when
absent, from ``runtime.service.recommend`` -- which owns the
GGUF/Transformers/vLLM branch and calls the per-backend launch policies. This
module only *layers user overrides* on top, as ``USER_INPUT`` flags, and only for
values the user actually set.

Scope note: partial / bare ``RuntimeRecommendation``s built elsewhere for
recommendation ranking and display (``engine_v2._runtime_for_artifact``,
``advisor._runtime_for_variant``) are fine and are out of scope this milestone --
they converge here (via ``AdvisorService.prepare_execution_plan``) before
anything executes. Anything that reaches a runner must come from this module and
be run-ready.

``hardware_fit`` is deliberately not a parameter: the planner already receives
``MemoryEstimate``, which carries ``.hardware_fit``. A launch policy that
actually consumes the placement will be added in the physical-model unification
milestone; until then there is no coupling to add.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution_plans import ArtifactVariant, ExecutionPlan, ModelIdentity
from jaull.domain.hardware import HardwareProfile
from jaull.domain.runtime import (
    ExecutionReadiness,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.execution_plans.service import build_execution_plan
from jaull.runtime import service as runtime_service
from jaull.runtime.policies import LLAMA_CPP_DEFAULT_CONTEXT_SIZE

_LLAMA_CPP_REQUIRED_FLAGS = ("--ctx-size", "--n-gpu-layers")


class ExecutionPlanningError(RuntimeError):
    """Automatic launch planning could not resolve a runnable plan.

    The CLI turns this into a non-zero exit with a message telling the user to
    pass ``--n-gpu-layers 0`` (CPU-only) or ``--n-gpu-layers N`` explicitly.
    """


@dataclass(frozen=True)
class ExecutionOverrides:
    """Values the user set explicitly. ``None`` means "let the planner decide"."""

    context_size: int | None = None
    n_gpu_layers: int | None = None


def plan_launch(
    *,
    runtime: RuntimeName,
    estimate: MemoryEstimate | None,
    hardware: HardwareProfile | None,
    overrides: ExecutionOverrides | None = None,
) -> RuntimeRecommendation:
    base = _automatic_launch(runtime, estimate, hardware)
    resolved = _apply_overrides(base, overrides or ExecutionOverrides())
    _require_runnable(runtime, resolved)
    return resolved


def plan_execution(
    *,
    model_identity: ModelIdentity,
    artifact: ArtifactVariant,
    runtime: RuntimeName,
    estimate: MemoryEstimate | None,
    hardware: HardwareProfile | None,
    overrides: ExecutionOverrides | None = None,
    launch: RuntimeRecommendation | None = None,
    backend_selection: RuntimeBackendSelection | None = None,
    runtime_capability: RuntimeCapability | None = None,
    execution_readiness: ExecutionReadiness | None = None,
) -> ExecutionPlan:
    if launch is None:
        launch = plan_launch(
            runtime=runtime,
            estimate=estimate,
            hardware=hardware,
            overrides=overrides,
        )
    else:
        _require_runnable(runtime, launch)
    return build_execution_plan(
        model_identity=model_identity,
        artifact=artifact,
        runtime=launch,
        memory_prediction=estimate,
        hardware=hardware,
        backend_selection=backend_selection,
        runtime_capability=runtime_capability,
        execution_readiness=execution_readiness,
    )


def _automatic_launch(
    runtime: RuntimeName,
    estimate: MemoryEstimate | None,
    hardware: HardwareProfile | None,
) -> RuntimeRecommendation:
    if estimate is not None:
        existing = estimate.runtime_recommendation
        if existing is not None and existing.runtime is runtime:
            return existing
        if hardware is not None:
            recomputed = runtime_service.recommend(
                estimate, hardware, estimate.repository_type
            )
            if recomputed.runtime is runtime:
                return recomputed

    flags: list[RuntimeFlag] = []
    if runtime is RuntimeName.LLAMA_CPP:
        # Keep a no-estimate llama.cpp launch run-ready: fill --ctx-size at the
        # documented default. --n-gpu-layers is left for an explicit override;
        # _require_runnable raises if none is given.
        flags.append(
            RuntimeFlag(
                name="--ctx-size",
                value=str(LLAMA_CPP_DEFAULT_CONTEXT_SIZE),
                source=RuntimeFlagSource.POLICY,
                explanation="llama.cpp default context; no memory estimate was run.",
            )
        )
    return RuntimeRecommendation(
        runtime=runtime,
        confidence=EstimationConfidence.UNKNOWN,
        reasons=["No memory estimate available; the launch plan is user-specified."],
        flags=flags,
    )


def _apply_overrides(
    rec: RuntimeRecommendation, overrides: ExecutionOverrides
) -> RuntimeRecommendation:
    flags = list(rec.flags)
    if overrides.context_size is not None:
        flags = _upsert_flag(
            flags,
            "--ctx-size",
            str(overrides.context_size),
            "Context length set explicitly on the command line.",
        )
    if overrides.n_gpu_layers is not None:
        flags = _upsert_flag(
            flags,
            "--n-gpu-layers",
            str(overrides.n_gpu_layers),
            "GPU layer count set explicitly on the command line.",
        )
    if flags == list(rec.flags):
        return rec
    return rec.model_copy(update={"flags": flags})


def _upsert_flag(
    flags: list[RuntimeFlag], name: str, value: str, explanation: str
) -> list[RuntimeFlag]:
    replacement = RuntimeFlag(
        name=name,
        value=value,
        source=RuntimeFlagSource.USER_INPUT,
        explanation=explanation,
    )
    if any(flag.name == name for flag in flags):
        return [replacement if flag.name == name else flag for flag in flags]
    return [*flags, replacement]


def _require_runnable(runtime: RuntimeName, rec: RuntimeRecommendation) -> None:
    if runtime is not RuntimeName.LLAMA_CPP:
        return
    present = {flag.name for flag in rec.flags}
    missing = [name for name in _LLAMA_CPP_REQUIRED_FLAGS if name not in present]
    if missing:
        raise ExecutionPlanningError(
            "Could not resolve an automatic llama.cpp launch plan "
            f"(missing {', '.join(missing)}). Re-run with --n-gpu-layers 0 for "
            "CPU-only, or --n-gpu-layers N to choose the offload explicitly."
        )


__all__ = [
    "ExecutionOverrides",
    "ExecutionPlanningError",
    "plan_execution",
    "plan_launch",
]
