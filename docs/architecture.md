# Jaull architecture

This document describes **the layers** the project is organised into and **the dependency
rules** between them. `docs/Workflow.md` explains *what* the pipeline does; this document
explains *how it is assembled* and why the boundaries are where they are.

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
                     ┌──── CLI ────┐         ┌──── TUI ────┐
                     └──────┬──────┘         └──────┬──────┘
                            └─────── AdvisorService ─────────┐
                                     │                       │
                    Workflow (guided orchestrator) ──────────┤
                                     │                       │
             ┌─────── Discovery ─────┼─── Recommendation ────┤
             │            (share contracts through)           │
             └────────────────── Domain  ────────────────────┘
                                     ▲
       Estimator · Metadata · HuggingFace · Hardware · Runtime · Artifacts · Execution
                                     ▲
                     Reporting · Diagnostics · Presentation
```

The arrows represent permitted imports, not data flows.

## Packages

| Package | Responsibility | Depends on |
|---|---|---|
| `domain/` | Shared Pydantic models and enums; constant policies; pure heuristics (families, licenses) | — |
| `hardware/` | Local detection (psutil, NVML, Vulkan probe) | `domain/` |
| `huggingface/` | HTTP client against the Hub and URL parsing | `domain/` |
| `metadata/` | Reading safetensors and GGUF headers | `domain/`, `huggingface/` |
| `estimator/` | Memory computation, variant selection, compatibility | `domain/`, `metadata/`, `huggingface/`, `runtime/` |
| `runtime/` | Runtime recommendation, runtime discovery, capability probes and local runners | `domain/`, `execution/` |
| `artifacts/` | Resolution, download, storage and verification of executable artifacts | `domain/`, `huggingface/` |
| `execution/` | Execution contracts and host backend for launching local processes | `domain/` |
| `execution_plans/` | Logical model identity, artifact variants and concrete execution plans | `domain/`, `discovery/`, `recommendation/`, `workflow/` |
| `experiments/` | Experiment runner and JSON store for `ExperimentRecord` | `domain/`, `evaluation/`, `runtime/`, `execution/` |
| `benchmarks/` | Benchmark matrix runner and JSON store for `BenchmarkRecord` | `domain/`, `runtime/`, `execution/` |
| `evaluation/` | Prediction↔observation and benchmark↔benchmark comparison (pure functions) | `domain/` |
| `discovery/` | Hub queries, filtering, enrichment, grouping into series | `domain/`, `huggingface/`, `estimator/`, `metadata/` |
| `recommendation/` | Scoring, ranking, explanations, capability | `domain/`, `estimator/` |
| `workflow/` | Guided-run orchestrator (synchronous, with progress and cancellation) | `domain/`, `discovery/`, `recommendation/`, `estimator/`, `hardware/`, `huggingface/`, `metadata/` |
| `reporting/` | JSON and Markdown serialisation of results | `domain/`, `workflow.state` |
| `diagnostics/` | Environment checks (Python, network, HF, NVML, runtimes, cache) | `domain/`, `hardware/` |
| `advisor/` | Application facade wrapping all the services above | everything below it |
| `presentation/` | Rich rendering (tables, panels) | `domain/`, `reporting/` |
| `cli/` | Typer subcommands, entry point; `run` also composes the local runner | `advisor/`, `presentation/`, `domain/`, `runtime/`, `execution/` |
| `tui/` | Textual screens, entry point | `advisor/`, `domain/` |

## Dependency rules (hard)

1. **`domain/` never imports anything from a higher layer.** It is the bottom of the stack.
2. **`discovery/` and `recommendation/` do not import each other.** The contracts they need
   to share (candidates, policies, families, licenses) live in `domain/`.
3. **`discovery/` and `recommendation/` do not import `workflow/`.** They receive
   `UserRequirements` (from `domain/`) and return results; orchestration is `workflow/`'s
   responsibility.
4. **`recommendation/` does not import `presentation/`.** Serialisation lives in
   `reporting/`, Rich rendering lives in `presentation/`, and the ranking logic knows about
   neither.
5. **`cli/` and `tui/` do not import each other.** The only shared facade is
   `AdvisorService`.
6. **`workflow/` may orchestrate;** `advisor/` is what the CLI and the TUI touch — they
   never construct `HfClient()`, `detect_hardware`, `estimate_memory` or
   `collect_diagnostics` directly.

These rules can be verified with `grep`:

```bash
# No cross-imports between discovery and recommendation
grep -rn "from jaull.recommendation" src/jaull/discovery/
grep -rn "from jaull.discovery"      src/jaull/recommendation/

# Nor workflow from discovery/recommendation
grep -rn "from jaull.workflow"       src/jaull/discovery/ src/jaull/recommendation/

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

`workflow/container.py::ServiceContainer` remains the service container the `workflow` uses
to parameterise the HTTP client, capability analyzer, range client factory, and so on.
`AdvisorService` **contains** it rather than replacing it — that way tests which used to
build a fake `ServiceContainer` to exercise the guided run keep working unchanged, and tests
which now build a test `AdvisorService` can use `AdvisorService.build(...)`.

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
