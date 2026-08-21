# Jaull architecture

This document describes **the layers** the project is organised into and **the dependency
rules** between them. `docs/Workflow.md` explains *what* the pipeline does; this document
explains *how it is assembled* and why the boundaries are where they are.
[ARCHITECTURE.md](../ARCHITECTURE.md) at the repository root is the condensed version of the
same contract.

Jaull is still a **modular Python monolith** today: no network workers, no Docker, no remote
executor. Guided recommendation remains metadata-only, but explicit local execution paths
already exist: `jaull run` resolves, downloads, verifies and executes single-file GGUF
artifacts with `llama-cli`, and the TUI adds execution of Transformers repositories through
an isolated Python worker, validation (persisted experiments) and benchmarks (`llama-bench`
and a Transformers worker). The purpose of this architecture is to keep the house in order
so that the next cycles (concurrent load, deployment qualification, remote execution) can be
added without dragging cycles behind them.

## Layer diagram

```
   ┌──── CLI ────┐                        ┌──── TUI ────┐
   └──────┬──────┘                        └──────┬──────┘
          └────────────  AdvisorService  ────────┘
                               │
                      ┌────────┴────────┐
                      │   Application   │   application/  use cases
                      │                 │   workflow/     guided orchestrator
                      └────────┬────────┘
                               │
              Discovery ───────┼─────── Recommendation
                               │   (they share contracts through Domain)
                     ┌─────────┴─────────┐
                     │  Domain · Ports   │
                     └─────────┬─────────┘
                               ▲
        Adapters and infrastructure, composed by bootstrap/:
        adapters/ · huggingface/ · analyzers/ · metadata/ · hardware/ ·
        estimator/ · runtime/ · execution/ · artifacts/ · experiments/ ·
        benchmarks/ · evaluation/ · observability/
                               ▲
        Rendering over already-computed semantics:
        reporting/ · presentation/ · diagnostics/
```

The arrows represent permitted imports, not data flows.

`workflow/` is no longer the centre of orchestration it once was. It contributes the
guided-run orchestrator and the progress/state DTOs that describe a run; the use cases it
used to own — requirements normalisation, recommendation policy, budgets, telemetry,
variant discovery and container wiring — now live in `application/`, `observability/` and
`bootstrap/`. The old module paths survive only as import shims. `execution_plans/` was
narrowed the same way: it keeps logical model identity and pure `ExecutionPlan`
construction, and everything that needs recommendation DTOs, Hub search or caching moved to
`application/`.

## Packages

`exceptions.py` and `paths.py` are leaf modules used almost everywhere and are omitted from
the table. A handful of packages also import a compatibility shim, which adds a few edges to
the real import graph that the intended design would not have; the rules that are actually
enforced are listed under [Dependency rules](#dependency-rules-hard).

| Package | Responsibility | Depends on |
|---|---|---|
| `domain/` | Frozen Pydantic models and enums; constant policies; pure heuristics (families, licenses) | — |
| `ports/` | Boundary protocols, only where infrastructure is genuinely replaceable (`ModelAnalysisCacheProtocol`) | `domain/` |
| `adapters/` | Concrete implementations of those ports (the persistent model-analysis cache) | `domain/`, `ports/` |
| `application/` | Use cases: requirements normalisation, model-reference parsing, artifact-variant discovery, recommendation service, budgets and policies, execution-plan assembly | `domain/`, `ports/`, `discovery/`, `recommendation/`, `execution_plans/`, `observability/` |
| `bootstrap/` | Production composition root; builds the concrete services into a `ServiceContainer` | `adapters/`, `ports/`, `domain/`, `discovery/`, `estimator/`, `hardware/`, `huggingface/`, `metadata/`, `recommendation/` |
| `observability/` | Performance telemetry for long-running stages | — |
| `hardware/` | Local detection (psutil, NVML, Vulkan probe) | `domain/` |
| `huggingface/` | HTTP client against the Hub, repository classification, artifact resolution | `domain/`, `analyzers/`, `estimator/`, `artifacts/` |
| `analyzers/` | Per-repository-type analyzers behind a Protocol | `domain/`, `huggingface/` |
| `metadata/` | Reading safetensors and GGUF headers | `domain/`, `analyzers/`, `huggingface/` |
| `estimator/` | Memory computation, variant selection, compatibility | `domain/`, `metadata/`, `huggingface/`, `runtime/`, `recommendation/` |
| `runtime/` | Runtime recommendation, runtime discovery, capability probes and local runners | `domain/`, `execution/` |
| `artifacts/` | Resolution, download, storage and verification of executable artifacts | `domain/`, `huggingface/` |
| `execution/` | Execution contracts and host backend for launching local processes | `domain/`, `hardware/` |
| `execution_plans/` | Logical model identity and pure `ExecutionPlan` construction | `domain/` |
| `experiments/` | Experiment runner and JSON store for `ExperimentRecord` | `domain/`, `evaluation/`, `runtime/`, `execution/` |
| `benchmarks/` | Benchmark matrix runner and JSON store for `BenchmarkRecord` | `domain/`, `runtime/`, `execution/` |
| `evaluation/` | Prediction↔observation and benchmark↔benchmark comparison (pure functions) | `domain/` |
| `discovery/` | Hub queries, filtering, enrichment, grouping into series | `domain/`, `huggingface/`, `estimator/`, `runtime/` |
| `recommendation/` | Plan assessment, v2 ranking, diversity, tiers, explanations | `domain/`, `estimator/`, `evaluation/`, `execution_plans/` |
| `workflow/` | Guided-run orchestrator (synchronous, with progress and cancellation), progress/state DTOs, per-run cache, and shims for the modules that moved | `domain/`, `application/`, `discovery/`, `recommendation/`, `observability/` |
| `reporting/` | JSON and Markdown serialisation of results | `domain/`, `recommendation/`, `workflow.state` |
| `diagnostics/` | Environment checks (Python, network, HF, NVML, runtimes, cache) | `domain/`, `hardware/`, `runtime/`, `execution/` |
| `advisor/` | Application facade wrapping all the services above | `bootstrap/` and everything below it |
| `presentation/` | Rich rendering (tables, panels) | `domain/`, `reporting/` |
| `cli/` | Typer subcommands, entry point; `run` also composes the local runner | `advisor/`, `presentation/`, `domain/`, `runtime/`, `execution/` |
| `tui/` | Textual screens, entry point | `advisor/`, `domain/`, `application/`, `presentation/` |

## Dependency rules (hard)

1. **`domain/` never imports anything from a higher layer.** It is the bottom of the stack:
   no application, adapters, workflow, runtime, presentation, advisor or front-end, and no
   `textual`, `huggingface_hub`, `torch` or `subprocess`.
2. **`application/` never imports infrastructure.** No adapters, no `huggingface/`, no
   `huggingface_hub`, no `textual`, and no presentation, advisor or `workflow/`. It talks to
   `domain/` and to `ports/`; the concrete implementations are injected by `bootstrap/`.
3. **`ports/` never imports adapters or presentation.** A port that knows its own
   implementation is not a port.
4. **`execution_plans/` never imports `workflow/` or `recommendation/`.** It builds plans out
   of domain objects; anything that needs recommendation DTOs lives in
   `application/recommendation/execution_plans.py`.
5. **`discovery/` and `recommendation/` do not import each other**, and neither imports
   `workflow/`, the advisor or a front-end. The contracts they need to share (candidates,
   policies, families, licenses) live in `domain/`.
6. **`recommendation/` does not import `presentation/`.** Serialisation lives in
   `reporting/`, Rich rendering lives in `presentation/`, and the ranking logic knows about
   neither.
7. **`cli/` and `tui/` do not import each other**, and neither constructs `HfClient()`,
   `detect_hardware`, `estimate_memory` or `collect_diagnostics` directly. `AdvisorService`
   is the only entry point they use.
8. **`presentation/` and `tui/` never import Hugging Face adapters** (`jaull.huggingface`,
   `huggingface_hub`); `presentation/` additionally stays clear of `adapters/` and
   `textual`. Both render semantics that were computed elsewhere.

Rules 1–4 and 8, plus the `recommendation/` half of rule 5, are checked automatically by
`tests/test_architecture_dependencies.py`: it walks every module with `ast`, collects the
import edges and compares them against an exact allowlist that is currently empty. A new
violation fails the test with the offending file and import named. `discovery/` itself has
no rule in that test, so the rest stays a convention the greps below protect.

The rest can still be verified with `grep`:

```bash
# No cross-imports between discovery and recommendation
grep -rn "from jaull.recommendation" src/jaull/discovery/
grep -rn "from jaull.discovery"      src/jaull/recommendation/

# Nor workflow from discovery, recommendation or execution_plans
grep -rn "from jaull.workflow" \
  src/jaull/discovery/ src/jaull/recommendation/ src/jaull/execution_plans/

# Nor presentation from recommendation
grep -rn "from jaull.presentation"   src/jaull/recommendation/

# Nor cli from tui
grep -rn "from jaull.cli"            src/jaull/tui/
```

All of these queries must return zero matches.

## `AdvisorService`

`src/jaull/advisor/service.py` holds the facade the CLI and the TUI screens use to reach the
application services. Its methods cover the main operations:

Analysis and recommendation:

- `scan_hardware(on_progress=None)` — local profile, optionally reporting progress per step.
- `diagnostics()` — a list of `DiagnosticResult`.
- `inspect_model(repo_id)` — analysis of one repository.
- `estimate_model(analysis, hardware, inference_cfg, ...)` — full memory estimate.
- `recommend(answers, hardware=None, on_progress=None, is_cancelled=None)` — end-to-end
  guided run.

Artifacts and execution:

- `resolve_artifact(repo_id, quantization=None, revision=None)` — pick an executable GGUF file.
- `download_artifact(artifact)` — download the artifact into the local layout.
- `verify_artifact(artifact, full=False)` — check size and SHA-256.
- `run_artifact(artifact=..., prompt=..., runtime=...)` — execute through the configured runner.

Execution plans:

- `resolve_model_identity(recommendation)` — the logical model behind a recommendation.
- `discover_artifact_variants(recommendation=..., ...)` — the artifacts that represent it.
- `execution_plans_for_recommendation(recommendation, ...)` — one plan per viable variant.
- `prepare_execution_plan(plan, ...)` — inspect, estimate, resolve/download/verify the
  artifact, select a backend and evaluate readiness.

Runtimes and readiness:

- `select_runtime_backend(hardware=None)` — preferred compute backend, with its reason.
- `inspect_llama_cpp_runtime(...)` / `inspect_pytorch_runtime()` — observed capabilities.
- `evaluate_execution_readiness(...)` / `evaluate_pytorch_execution_readiness(...)` —
  preflight decision.

Evidence:

- `run_experiment(request)` / `build_experiment_record(...)` — run and record an experiment.
- `save_experiment_record`, `load_experiment_record`, `list_experiment_ids` — the store.
- `run_benchmark(...)` / `run_benchmark_matrix(...)` — measure one or several configurations.
- `save_benchmark_record`, `load_benchmark_record`, `list_benchmark_ids`,
  `benchmark_records_for_model(...)` — the store.
- `compare_benchmarks(...)` / `compare_saved_benchmarks_for_recommendation(...)` — compare
  benchmarks as complete execution plans.

Two factories:

- `AdvisorService.default()` — production wiring (`ServiceContainer.default()`).
- `AdvisorService.build(hf_client=..., detect_hardware=..., inspect_model=...,
  estimate_memory=..., collect_diagnostics=...)` — test wiring, with every service injected
  as a callable.

TUI screens reach the advisor through `self.app.advisor`; CLI functions accept it as an
optional parameter (`advisor: AdvisorService | None = None`) and fall back to
`AdvisorService.default()` when none is passed. `cli/run.py` uses the advisor to
resolve/download/verify artifacts and instantiates the local runner with the CLI-specific
options (`--llama-cli`, `--timeout-seconds`, `--ctx-size`, `--n-gpu-layers`).

## Artifacts and local execution

The `run` path is deliberately kept separate from the estimator and from the guided
workflow:

```text
cli/run.py
   ├── normalize_repo_id()
   ├── AdvisorService.resolve_artifact()
   ├── AdvisorService.download_artifact()   # only if the file is missing
   ├── AdvisorService.verify_artifact()
   └── LlamaCppRunner(HostExecutionBackend).run()
```

`artifacts/` translates an abstract repository into a concrete, verified `ModelArtifact`. In
this phase it only accepts single-file GGUF; Transformers repositories and multipart GGUF
are rejected with specific errors.

`execution/` knows nothing about Hugging Face or about models: it receives an immutable
`ExecutionRequest` and returns an `ExecutionResult`. That result holds stdout/stderr and an
`ExecutionObservation`, which is the source of truth about what actually happened during the
process: duration, exit status, peak RSS of the main process, and peak VRAM attributed via
NVML when possible. This makes it possible to test the runner without invoking any real
binary, and keeps prediction (`MemoryEstimate`) and observation separate.

`runtime/llama_cpp_runner.py` validates that the artifact is GGUF, downloaded, verified and
present on disk before building the `llama-cli --single-turn` command.

## Prediction validation

`jaull.evaluation.comparison.compare_prediction` compares a Jaull prediction against an
execution that has already been observed:

```text
MemoryEstimate + ExecutionObservation -> PredictionComparison
```

The comparison does not modify the estimator and does not calibrate any formula.
`MemoryEstimate` still represents the prediction, `ExecutionObservation` still represents
measured reality, and `PredictionComparison` is a third, derived piece.

There is a single error convention:

```text
error_bytes = measured_bytes - predicted_bytes
error_percent = (measured_bytes - predicted_bytes) / predicted_bytes * 100
```

A positive error means Jaull underestimated real consumption. A negative error means Jaull
overestimated it.

The RAM comparison is only computed when the executed configuration is CPU-only or without
offload. In that case the comparable prediction is the sum of the components that represent
process consumption (`weights + kv_cache + runtime_overhead`), excluding `device_reserve`
and `safety_margin`, because those last two are capacity policy and not observed RSS. When
the runtime uses GPU offload, Jaull does not yet keep a host/device breakdown, so
`ram.predicted_bytes` stays `null` and the comparison is marked
`methodologically_unavailable`.

The VRAM comparison is also `methodologically_unavailable` in this phase: the current
estimation model does not retain VRAM attributed to the executed PID and configuration. If
NVML exposes no process memory, `peak_vram_bytes = null` is not treated as zero.

## Dependency composition

`bootstrap/container.py::ServiceContainer` is the production composition root. It is the one
place allowed to know concrete adapters — HTTP client, capability analyzer, range client
factory, persistent model-analysis cache — and to construct them. `AdvisorService`
**contains** it rather than replacing it, so tests which used to build a fake
`ServiceContainer` to exercise the guided run keep working unchanged, and tests which now
build a test `AdvisorService` can use `AdvisorService.build(...)`.

`workflow/container.py` remains as a re-export of `bootstrap.container` so historical
imports keep resolving. New wiring belongs in `bootstrap/`.

## Reporting and serialisation

`jaull.reporting.estimation.estimate_to_json_dict` is the **only** producer of the JSON
representation of a `MemoryEstimate`. `presentation/estimation_report.py` re-exports it for
compatibility but no longer holds a copy.

`jaull.reporting.recommendation.report_to_json` / `report_to_markdown` are the only functions
that build the complete guided-run report. `recommendation/report.py` is only a
compatibility shim that re-exports from there.

The compatibility contracts are **byte-identical**: `tests/test_reporting_regression.py`
compares the JSON and Markdown output against `tests/snapshots/report.json` and
`tests/snapshots/report.md`. Any change that breaks byte-for-byte equality must bump
`REPORT_SCHEMA_VERSION` explicitly.

## Conventions

- **No bare `except Exception`.** Always catch a specific type from `jaull.exceptions` or
  from a concrete library (`OSError`, `ImportError`, …).
- **Python 3.12+**: `X | None`, `list[str]`, generic `type`, no `Union`/`Optional` from
  `typing`.
- **Frozen Pydantic v2 models** in `domain/`. No class mutates its state after construction.
- **No mutable global singletons**: the service container and the advisor are built at the
  entry point and injected downwards.

## Pending work (outside this cycle)

- Docker / Docker Compose.
- Download streaming and byte-level progress for large artifacts.
- Explicit workload/SLO model (minimum throughput, TTFT, latency).
- Concurrent load experiments and capacity curves.
- Deployment qualification verdict and reproducible manifest (`jaull.lock`).
- Internal HTTP between `Advisor` and a remote `Executor`.
