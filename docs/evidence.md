# Artifacts, execution and evidence

Everything past the recommendation: how an abstract "this model should fit" becomes a
concrete file on disk, a runtime that can execute it, a real run, a measurement, and a
comparison against what was predicted.

The organising rule is that **prediction and observation are separate objects and neither
one is allowed to edit the other**:

```text
MemoryEstimate          the prediction
ExecutionObservation    what actually happened
BenchmarkObservation    what was measured under a benchmark methodology
PredictionComparison    a third, derived object
```

---

## Execution plans

A recommendation names a repository. That is not enough to run anything, because one
logical model exists as several artifacts, and each artifact can be executed by different
runtimes.

- **`ModelIdentity`** — the logical model, independent of any artifact: canonical repo,
  family, parameter count, architecture, plus the evidence for each of those (base-model
  metadata, config, repository id, or a name heuristic, each with its own confidence).
- **`ArtifactVariant`** — one available representation: a GGUF quantization, a safetensors
  precision. Discovery metadata only; it does not imply the file has been downloaded.
- **`ExecutionPlan`** — one concrete way to run the model: identity + artifact + runtime +
  backend selection + runtime capability + readiness + memory prediction, with its own
  compatibility status, evidence and warnings.

Variants are discovered by searching the Hub for repositories that resolve to the same
identity, with a deep-inspection budget, and each match is labelled `confirmed`,
`uncertain` or `rejected` rather than merged on name similarity alone.

## Artifacts

`artifacts/` turns an abstract repository reference into a verified local file:

1. **Resolve** — `repo_id` + optional quantization + optional revision → a single-file GGUF
   `ModelArtifact` at a concrete revision (the commit sha when the Hub reports one).
2. **Download** — into `<user data dir>/models/<owner>/<repo>/<filename>`, with path
   traversal and unsafe-filename rejection. A SHA-256 is computed and written to a sidecar.
3. **Verify** — file size against the Hub's reported size, and the SHA-256 against the
   sidecar. `--full-verify` recomputes the digest from the file instead of trusting it.

Multipart GGUF and Transformers repositories are rejected by the resolver with specific
errors. Transformers execution takes a different route: the repository reference is passed
to the runtime, which loads it through the Hugging Face cache.

## Runtimes and backends

Three separate questions, answered separately:

**Which runtime installations exist?** `RuntimeLocator` discovers llama.cpp from, in order:
an explicit path, a registered runtime, `PATH`, and a bounded search of conventional build
directories (`~/tools/llama.cpp/build-*/bin`, `~/llama.cpp/build-*/bin`). PyTorch resolves
to an explicit interpreter, a registered one, or the current environment.

**What can each installation actually do?** Capability probes run the binary (or import
torch in an isolated worker) and record which backends it exposes — CUDA, Vulkan, HIP, CPU
— along with the devices each backend reports. A backend that a build does not expose is
recorded as `not_observed`, not assumed.

**Which backend should be used here?** `select_runtime_backend` picks a compute backend
from the detected hardware, with an explicit reason (`native_backend_available`,
`vulkan_backend_available`, `cpu_fallback`, `no_usable_accelerator`,
`software_renderer_ignored`) and the alternatives it rejected.

Those three feed `ExecutionReadiness`, the preflight decision:

> `READY` means Jaull has not found a blocking mismatch between the hardware backend
> selection and the observed runtime capability. It is not proof that inference will
> succeed; the definitive outcome remains `ExecutionObservation.success`.

Readiness governs **acting**, never **recommending**. It decides whether Run, Validate and
Benchmark are attempted or blocked with a reason, and it is what the *Ready* evidence state
below reports — but it is deliberately excluded from the recommendation ranking, so a model
does not become a worse recommendation because a binary has not been installed yet. See
[recommendation.md](recommendation.md).

## Execution

| Path | Runtime | Artifacts |
|---|---|---|
| `jaull run` (CLI) | `llama-cli --single-turn` | single-file GGUF |
| TUI execution paths | `llama-cli`, or Transformers via an isolated Python worker | GGUF, Transformers repositories |

`execution/` knows nothing about Hugging Face or models: it takes an immutable
`ExecutionRequest` and returns an `ExecutionResult` holding stdout, stderr and an
`ExecutionObservation`. That observation is the source of truth about what happened:

- success / failure and a failure reason;
- duration and exit status;
- peak sampled process RSS (sampled every 50 ms, so very short spikes may be
  underestimated);
- peak NVIDIA process memory, when NVML can attribute memory to the PID. On CPU-only
  machines, or with NVML unavailable, VRAM is `None` — never silently zero.

## Experiments

A validation run persists an immutable `ExperimentRecord` — one JSON file per experiment
under `<user data dir>/experiments/`, with a schema version — linking:

```text
ExperimentRecord
├── identity           # experiment id + timezone-aware timestamp
├── environment        # Jaull version, Python version/implementation, git commit
├── hardware           # the full HardwareProfile it ran on
├── artifact           # repo, revision, filename, format, quantization, size
├── workload           # the prompt that was run
├── backend_trace      # requested vs observed backend
├── runtime            # runtime recommendation and flags actually used
├── prediction         # the MemoryEstimate
├── preflight          # runtime capability + execution readiness
├── observation        # the ExecutionObservation
└── comparison         # the PredictionComparison
```

A failed execution is still a valid experiment. Failures are part of the empirical
evidence, and a record that only says "this configuration did not run, here is why" is
exactly the kind of result the qualification workflow needs.

## Benchmarks

Two methodologies, both persisted as `BenchmarkRecord` JSON files under
`<user data dir>/benchmarks/`:

**`llama_bench_v1`** — drives `llama-bench` with prefill sizes (default 128, 512, 2048) and
generation sizes (default 128), 5 repetitions by default, and parses per-measurement mean
and standard deviation tokens/second along with the backend, device and `ngl` the binary
itself reported. The matrix runner can run the same artifact on CPU and on the selected
backend, recording each configuration separately and reporting skips and failures instead
of hiding them.

**`transformers_isolated_inference_v2`** — runs in a separate Python process and reports
prefill and generation throughput, model load time, time to first token and generation
latency, each with a standard deviation, plus peak RAM and VRAM.

A `BenchmarkRecord` stores the request alongside the observation, and validates that the
artifact, runtime, backend and GPU-layer settings in the record match the request that
produced it — so a record cannot claim a configuration it did not run.

### Comparing benchmarks

`compare_benchmark_records` compares benchmarks of the same logical model **as complete
execution plans** — artifact format, quantization or precision, runtime, backend — not as
disembodied tokens/second numbers. It keeps the latest record per configuration, prefers
the current methodology per runtime, and warns when records come from different machines or
different methodologies rather than quietly ranking them against each other. There is
deliberately no single "winner" score.

## Prediction vs observation

`compare_prediction` produces a `PredictionComparison` from a `MemoryEstimate` and an
`ExecutionObservation`. It does not modify the estimator and does not calibrate any formula.

One error convention throughout:

```text
error_bytes   = measured_bytes - predicted_bytes
error_percent = (measured_bytes - predicted_bytes) / predicted_bytes * 100
```

A positive error means Jaull **under**estimated real consumption; a negative error means it
**over**estimated.

**RAM** is only compared when the executed configuration is CPU-only or without offload. The
comparable prediction is then `weights + kv_cache + runtime_overhead`, excluding
`device_reserve` and `safety_margin` — those are capacity policy, not observed RSS. Under
GPU offload, Jaull does not yet keep a host/device breakdown, so `ram.predicted_bytes` is
`null` and the comparison is marked `methodologically_unavailable`.

**VRAM** is currently always `methodologically_unavailable`: the estimation model does not
retain VRAM attributed to the executed PID and configuration. When NVML exposes no process
memory, `peak_vram_bytes = null` is not treated as zero.

Alongside the metrics, the comparison classifies the compatibility verdict:

| Outcome | Meaning |
|---|---|
| `correct_success` | Predicted runnable, and it ran |
| `correct_failure` | Predicted not runnable, and it failed |
| `false_positive` | Predicted runnable, but it failed |
| `false_negative` | Predicted not runnable, but it ran |
| `unknown` | The prediction had no runnable verdict |

This is the beginning of an accuracy record for the estimator, which is why the records are
persisted rather than printed and discarded.

## Evidence states

The execution-paths screen labels every plan with the strongest evidence that exists for
it, weakest to strongest:

| State | Meaning |
|---|---|
| Estimated | Only predicted |
| Ready | Passed the readiness preflight |
| Validated | At least one stored `ExperimentRecord` |
| Benchmarked | At least one stored `BenchmarkRecord` |

The index is built by reading the stored records off disk and matching them back to plans.
It only reads: no metric is recomputed, no methodology is reinterpreted, and a record that
fails to load is skipped rather than guessed at.

## Where things are stored

| Data | Location |
|---|---|
| Downloaded artifacts | `<user data dir>/models/<owner>/<repo>/` |
| Experiment records | `<user data dir>/experiments/*.json` |
| Benchmark records | `<user data dir>/benchmarks/*.json` |
| Caches | `<user cache dir>/` |

`<user data dir>` is `$XDG_DATA_HOME/jaull` or `~/.local/share/jaull` on Linux,
`~/Library/Application Support/jaull` on macOS, and `%LOCALAPPDATA%\jaull` on Windows.
