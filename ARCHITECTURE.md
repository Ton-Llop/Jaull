# Jaull Architecture

Jaull is a modular monolith. The intended dependency direction is:

```text
Presentation
     ↓
Advisor
     ↓
Application
     ↓
Domain / Ports
          ↑
       Adapters
```

This is enforced by `tests/test_architecture_dependencies.py`, which scans Python
imports with `ast`. The allowlist is intentionally exact and currently empty.

## Layers

### Domain

`src/jaull/domain/` contains the domain model: hardware profiles, model metadata,
requirements, execution plans, benchmark and experiment records, recommendation
assessments, and value objects such as model identity keys.

Domain code must not import application, advisor, presentation, TUI, workflow,
runtime implementations, Hugging Face clients, Textual, PyTorch, or subprocess
infrastructure.

### Application

`src/jaull/application/` contains use-case orchestration that is not UI and not
infrastructure:

```text
application/
├── discovery/artifact_variants.py
├── recommendation/execution_plans.py
├── recommendation/policies.py
├── recommendation/service.py
└── requirements.py
```

The recommendation application service enriches candidates, invokes the v2 plan
ranker, applies diversity, and builds `ModelRecommendation` results. The normal
hardware-aware path ranks `ExecutionPlan` objects through PlanAssessment; legacy
score construction remains only for compatibility fields on the final result and
for the explicit no-hardware fallback.

### Ports

`src/jaull/ports/` contains boundary protocols only when there is a real
replaceable infrastructure concern. Currently this includes
`ModelAnalysisCacheProtocol` and its stats DTO.

### Adapters

`src/jaull/adapters/` contains concrete infrastructure implementations. The
persistent model-analysis cache now lives in `adapters/cache/` and implements the
cache port.

Existing infrastructure packages such as `huggingface/`, `runtime/`, `hardware/`,
`benchmarks/`, and `experiments/` still contain concrete adapters. They are not
renamed wholesale because the refactor keeps public imports stable.

### Advisor

`AdvisorService` remains the public facade used by CLI and TUI. It delegates to
the guided workflow, runtime services, execution runners, validation, benchmark,
and experiment helpers. It should stay a facade; new recommendation policy should
not be added there.

### Bootstrap

`src/jaull/bootstrap/container.py` is the production composition root for the
guided workflow. It is allowed to know concrete adapters and construct them.
`workflow.container` is a compatibility shim.

### Presentation

Presentation modules render already-computed semantics. TUI screens must not
import Hugging Face adapters directly. Pure model-reference parsing lives in
`domain.model_reference`; `huggingface.url_parser` remains as an import
compatibility shim.

## Recommendation Pipeline

The current recommendation pipeline is:

```text
EvaluatedCandidate[]
    ↓ application.recommendation.service.enrich_candidate_features
Enriched candidates
    ↓ recommendation.engine_v2.rank_execution_plans
RankedPlan[]
    ↓ recommendation.diversity.diversify_ranked_plans
DiversifiedRecommendation[]
    ↓ application.recommendation.service.recommend
ModelRecommendation[]
```

Inside `recommendation.engine_v2`, the v2 responsibilities are still colocated:
plan generation, assessment, local evidence matching, ranking key, and ranking
helpers. This is an intentional residual seam for a later mechanical split; the
policy was not changed during the architectural refactor.

## Execution Plans

`execution_plans/service.py` owns logical model identity resolution and pure
`ExecutionPlan` construction. It no longer imports `workflow` or
`recommendation.models`.

Functions that require recommendation DTOs live in
`application/recommendation/execution_plans.py`. Variant discovery, which needs
search, inspection, cache, and telemetry, lives in
`application/discovery/artifact_variants.py`.

`jaull.execution_plans` reexports the historical public API as a compatibility
shim.

## Workflow

`workflow/` is no longer the home for recommendation policy, requirements
normalization, budgets, telemetry, container wiring, or variant discovery. It
still contains the guided workflow orchestrator, progress/state DTOs, run cache,
and compatibility shims for historical imports.

The remaining workflow package is deliberate compatibility debt, not a target
for new application logic.

## Persistence And Cache

Benchmark and experiment stores remain append-only evidence stores. The
model-analysis cache remains disposable revision-aware cache. The cache location,
schema invalidation, atomic writes, and failure behavior are unchanged.

## Runtime Boundaries

Runtime probing, backend selection, llama.cpp, and Transformers/PyTorch execution
remain in `runtime/` and `execution/`. The application layer consumes runtime
capability/readiness through domain DTOs and `AdvisorService` composition.

## Architecture Tests

The architecture guard prevents:

- `domain` importing application/adapters/presentation/advisor/workflow/runtime
  implementations or external infrastructure libraries;
- `application` importing adapters, presentation, advisor, workflow, Hugging
  Face adapters, Textual, or `huggingface_hub`;
- `ports` importing adapters or presentation;
- `execution_plans` importing workflow or recommendation;
- TUI and presentation importing Hugging Face adapters.

If a compatibility exception is ever required, it must be added as an exact
`(source_file, imported_module)` entry and removed when stale. The current
allowlist is empty.
