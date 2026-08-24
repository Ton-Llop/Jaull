# Biosfer experimental protocol

This document defines the experimental protocol for validating Jaull on local
hardware and rented GPUs during the Biosfer TFG work. It is intentionally tied
to Jaull's current contracts: `MemoryEstimate`, `HardwareFitResult`,
`BenchmarkRecord`, `ExperimentRecord` and `PredictionComparison`.

The protocol has two rules:

1. A prediction must be frozen before measurement.
2. A measured result must never be used to rewrite the original prediction.

## Objective

The experiments validate five hypotheses:

1. Jaull predicts whether a concrete model artifact can run on a machine and
   how it should be placed: GPU-resident, GPU offload, CPU RAM or too large.
2. Jaull's predicted RAM and VRAM requirements are close enough to measured
   memory usage to support hardware planning.
3. Runtime performance changes materially across hardware classes, even when
   the model artifact is unchanged.
4. Biosfer workloads have a minimum viable hardware class and a recommended
   hardware class with practical headroom.
5. The best deployment option is not simply the fastest one; it is the best
   cost/performance point for a workload, model, artifact and runtime.

## Experimental variables

Each experiment must separate controlled variables, independent variables and
measured variables.

| Area | Controlled variables | Independent variables | Measured variables |
| --- | --- | --- | --- |
| Hardware | OS image, driver/runtime setup where possible, idle baseline | Hardware class, GPU memory, CPU/RAM class | peak RAM, peak VRAM, success/failure, throughput |
| Model | family, scale, revision | model family and parameter scale | output validity, load success/failure |
| Artifact | exact file, format, hash when available | GGUF/AWQ/GPTQ/safetensors, quantization or dtype | artifact load behavior, memory usage |
| Runtime | runtime family and version | llama.cpp, Transformers/PyTorch, backend | load time, TTFT, tokens/s, errors |
| Backend | selected backend and flags | CPU, CUDA, Vulkan, HIP where available | observed backend, peak device memory |
| Context | fixed context length per run | 4K, 8K, long-context cases | memory growth, latency |
| Concurrency | one prompt per run today | future 1/2/4 user matrix | throughput degradation, OOM rate |
| Workload | prompt/input fixture | workload type | latency, generation throughput, success |

The artifact is part of the experiment identity. A comparison between machines
should use the same artifact whenever possible. Changing quantization changes
the experiment.

## Hardware classes

The planned hardware classes are:

| Class | Purpose |
| --- | --- |
| CPU-only | Baseline and fallback validation. |
| RTX 2060 6 GB | Local low-VRAM GPU class. |
| RTX 4060 8 GB | Local/consumer mid-range GPU class. |
| GPU around 24 GB | Rented candidate for larger local deployment. Exact provider/model is not fixed yet. |
| Second rented GPU class | Contrast class for calibration. It should differ in more than only price; for example memory size, architecture or cloud provider. |

Do not record a rented GPU as a planned purchase candidate until its exact
provider, price, driver stack and runtime versions are captured.

## Model and artifact matrix

The matrix should be small enough to run repeatedly:

- 2-4 model families.
- 2-3 parameter scales per family where feasible.
- 1 primary artifact per model for cross-hardware comparison.
- 1-2 alternate quantizations only when the question is specifically about
  quality/memory trade-off.

Recommended shape:

| Slot | Purpose |
| --- | --- |
| Small instruct/chat model | Speed and low-memory baseline. |
| Balanced 1B-4B model | Expected practical assistant class for 6-8 GB GPUs. |
| Larger 7B-14B model | Offload and 24 GB candidate validation. |
| Domain/coding-capable model | Tests whether task suitability changes the best hardware choice. |

The same comparative experiment should pin:

- model repo and revision;
- artifact filename;
- artifact format;
- quantization or dtype;
- runtime;
- backend;
- context length;
- prompt/input fixture.

## Biosfer workloads

Use workload fixtures that map to Biosfer use cases rather than generic demos.
Jaull currently stores `ExperimentWorkload.prompt`, so each workload needs a
stable prompt or input identifier. Richer workload metadata can be added later.

| Workload | Goal | Initial fixture |
| --- | --- | --- |
| Interactive/general assistant | Measure conversational latency and usability. | Short user question plus medium response target. |
| Document-oriented long context | Validate memory growth and long-context behavior. | Fixed document excerpt plus question or summary request. |
| Coding | Measure structured reasoning/code output on developer tasks. | Small code task with deterministic instructions. |
| Summarization/extraction batch | Measure throughput-oriented processing. | Fixed text batch and extraction schema/instructions. |

For each workload, keep the prompt/input file under version control or record a
stable identifier and hash. Do not compare runs that used different inputs as
if they were the same experiment.

## Measurements

Use Jaull's current measured fields first.

Already supported by Jaull:

- `BenchmarkObservation.peak_ram_bytes`
- `BenchmarkObservation.peak_vram_bytes`
- `BenchmarkObservation.model_load_seconds`
- `BenchmarkObservation.time_to_first_token_seconds`
- `BenchmarkObservation.generation_latency_seconds`
- `BenchmarkMeasurement.mean_tokens_per_second`
- `BenchmarkMeasurement.stddev_tokens_per_second`
- `BenchmarkMeasurement.kind` as prefill or generation
- `BenchmarkObservation.success`
- `BenchmarkObservation.failure_reason`
- `BenchmarkObservation.exit_code`
- `BenchmarkRequest.backend`
- `BenchmarkRequest.gpu_layers`
- `ExperimentRecord.backend_trace`
- `ExecutionObservation.peak_ram_bytes`
- `ExecutionObservation.peak_vram_bytes`
- `ExecutionObservation.duration_seconds`
- `ExecutionObservation.success`
- `ExecutionObservation.failure_reason`

Metrics proposed but not fully represented as first-class domain fields today:

- euros per hour;
- euros per 1M generated tokens;
- provider instance identifier;
- thermal/power state;
- concurrent-user load generation;
- quality or task-success rubric;
- artifact hash enforcement beyond `ModelArtifact.sha256` when available.

These proposed metrics should be recorded in experiment notes or a separate
cost sheet until Jaull grows explicit contracts for them.

## Repetition methodology

Each benchmark configuration should use the existing `BenchmarkRequest`
repetition fields where possible.

- Warmup: run one warmup or use `BenchmarkObservation.warmup_seconds` when the
  benchmark runner records it.
- Repetitions: start from Jaull's default benchmark repetitions, currently 5.
- Timeout: set an explicit timeout per benchmark or experiment. Never leave a
  rented GPU run unbounded.
- Discarding a run: discard only for documented infrastructure reasons such as
  provider interruption, wrong artifact, wrong backend, corrupted prompt or
  unrelated GPU activity. OOM, timeout and non-zero exit are valid observations,
  not discarded failures.
- Variance: report mean and standard deviation where Jaull records them. If a
  run has high variance, repeat the configuration rather than cherry-picking.

When comparing hardware classes, run the same matrix in the same order where
practical, but record the exact execution order because thermal and cache
effects can matter.

## Concurrency

Concurrency is a future controlled matrix, not a capability to assume in every
current Jaull runner.

Initial design:

| Users | Purpose |
| --- | --- |
| 1 | Baseline single-user latency. |
| 2 | Light shared usage. |
| 4 | Small internal team stress point. |

Jaull already has `InferenceConfiguration.concurrent_users` for memory
estimation, especially KV-cache scaling. That is not the same as load
generation. Until Jaull has a controlled load generator, concurrency results
must be labelled as estimated capacity or manually orchestrated measurements.

## Reproducibility

Every run must record enough provenance to be reproduced or rejected.

Already present in current contracts:

- hardware profile in `BenchmarkRecord.hardware` and `ExperimentRecord.hardware`;
- stable machine matching via `machine_fingerprint` for local evidence;
- OS/architecture in `HardwareProfile`;
- GPU name, VRAM total and VRAM available in `HardwareProfile.gpus`;
- accelerator/backend information in `HardwareProfile.accelerators`;
- Jaull version, Python version and git commit in benchmark/experiment
  environments when provided;
- runtime recommendation and flags in `RuntimeRecommendation`;
- runtime capabilities and version text where probes provide them;
- requested backend, device and GPU layers in `BenchmarkRequest`;
- artifact repo, revision, filename, format, quantization, local path and
  SHA-256 when available in `ModelArtifact`;
- prediction snapshot in `ExperimentRecord.prediction`;
- workload prompt in `ExperimentWorkload`.

Required run metadata:

- hardware fingerprint;
- OS and architecture;
- GPU name, VRAM total, VRAM available before prediction;
- driver and CUDA/ROCm/Vulkan details when detected;
- runtime and runtime version;
- llama.cpp commit/build or version text when applicable;
- model repo and revision;
- artifact filename, format, quantization/dtype and hash if available;
- context length, batch size, concurrent-user setting;
- runtime flags;
- backend requested and backend observed;
- workload name and prompt/input identifier.

## Prediction-before-measurement rule

The validation flow is:

```text
detect or simulate hardware
        |
        v
create MemoryEstimate and HardwareFitResult
        |
        v
freeze prediction and runtime request
        |
        v
execute benchmark or experiment
        |
        v
store observation
        |
        v
compare prediction vs observation
```

The frozen prediction is the one stored in `ExperimentRecord.prediction`.
If a bug is found after measuring, create a new prediction version and a new
comparison. Do not edit the original prediction to make past measurements look
more accurate.

## PredictionComparison

Current `PredictionComparison` compares:

- RAM, when the prediction and measurement are methodologically comparable;
- compatibility outcome: correct success, correct failure, false positive,
  false negative or unknown.

Current limitation:

- VRAM is always marked `METHODOLOGICALLY_UNAVAILABLE` because the comparison
  layer does not yet have a process-attributed device-memory prediction that
  matches `peak_vram_bytes`.
- RAM for GPU/offload execution can also be methodologically unavailable
  because RSS and host/device placement are not the same quantity.

The Hardware Fit Analyzer introduces the right conceptual data for future
comparison: GPU required bytes, RAM required bytes, GPU weights, RAM weights,
GPU layers and placement method. Before enabling VRAM error percentages, Jaull
must decide which prediction field is comparable to the observed VRAM method:

- physical GPU allocation only;
- physical allocation plus runtime overhead;
- allocation plus reserve/safety margin;
- or a separate predicted peak process VRAM.

Until that decision is implemented, the protocol must report VRAM measurements
as observations and avoid presenting a fake prediction error.

## Cost methodology

Cost is not currently a first-class Jaull domain object. For rented GPUs,
record cost separately and join it with benchmark output by experiment ID.

Required cost fields:

- provider;
- instance/GPU class;
- euros per hour;
- billed duration;
- total run cost;
- notes about minimum billing unit.

Derived cost metrics:

- tokens per hour;
- euros per 1M generated tokens;
- euros per successful experiment;
- eventual local amortized cost per month and per 1M tokens.

Keep measured cost separate from assumptions. For local hardware, record the
purchase price, amortization period, estimated electricity cost and expected
utilization as assumptions, not as measured benchmark fields.

## Hardware purchase and deployment decision

The final hardware recommendation should be derived from:

- minimum hardware that runs the workload without OOM;
- recommended hardware with practical headroom;
- measured generation throughput;
- measured TTFT and latency for interactive use;
- cost per throughput unit;
- deployment complexity and runtime stability;
- model license and artifact availability;
- operational risks such as driver support and thermal limits.

The output should distinguish:

- minimum viable hardware;
- recommended hardware;
- overkill hardware;
- rejected hardware and why.

This feeds the later `DeploymentPlan`: selected model/artifact, runtime,
backend, flags, expected concurrency, hardware requirement, cost assumptions,
and operational caveats.

## Threats to validity and limitations

- Available VRAM changes between prediction and execution. Planning with idle
  capacity and running on a busy GPU are different questions.
- Driver, CUDA/ROCm/Vulkan, PyTorch and llama.cpp versions can change both
  memory and performance.
- Thermal throttling and power limits can distort long benchmark runs.
- First-run cache effects can differ from warmed runs.
- Quantizations are different artifacts, not merely lower-quality versions of
  the same execution.
- Small sample sizes can make variance look like a hardware effect.
- Calibration can overfit to the small set of GPUs tested.
- Rented hardware may be shared, virtualized or configured differently across
  providers.
- VRAM measurement may be device-wide or process-attributed depending on the
  environment and backend.
- Jaull currently separates prediction and observation, but not every proposed
  cost/deployment field is represented as a domain object.

## Experiment record template

Use this template before every rented GPU run.

```text
Experiment ID:
Date/time:
Operator:

Objective:
Hypothesis:

Hardware class:
Provider / local machine:
Hardware fingerprint:
OS / arch:
CPU:
RAM total / available before prediction:
GPU:
VRAM total / available before prediction:
Driver:
CUDA / ROCm / Vulkan:

Jaull version:
Jaull git commit:
Runtime:
Runtime version:
Runtime capability:
Backend requested:
Backend observed:
Runtime flags:

Model family:
Model repo:
Model revision:
Artifact filename:
Artifact format:
Quantization / dtype:
Artifact size:
Artifact hash:

Workload:
Prompt/input identifier:
Context length:
Batch size:
Concurrent users setting:
Timeout:
Repetitions:
Warmup:

Frozen prediction:
  compatibility status:
  hardware fit mode:
  placement method:
  predicted GPU layers:
  predicted total layers:
  predicted RAM bytes:
  predicted VRAM/GPU bytes:
  assumptions:
  warnings:

Measurement:
  success:
  failure reason:
  model load time:
  TTFT:
  prompt/prefill throughput:
  generation throughput:
  generation latency:
  peak RAM:
  peak VRAM:
  raw command:
  raw logs path:

Prediction comparison:
  RAM availability/result:
  VRAM availability/result:
  compatibility outcome:

Cost:
  euros/hour:
  billed duration:
  total cost:
  tokens/hour:
  euros/1M generated tokens:

Decision note:
  accepted/rejected:
  reason:
  follow-up:
```

## Readiness before renting a GPU

Mandatory before renting:

- freeze the model/artifact matrix;
- freeze the workload fixtures and prompt/input identifiers;
- run the local CPU/RTX 2060/RTX 4060 baseline where available;
- confirm `BenchmarkRecord` and `ExperimentRecord` persistence paths;
- confirm the prediction-before-measurement flow;
- define timeout and maximum rental budget;
- decide which metrics will be considered decision-making metrics.

Supported by Jaull today:

- hardware detection and hardware profiles;
- memory estimation;
- hardware fit analysis;
- runtime/backend selection and readiness;
- benchmark records with throughput, latency and memory fields;
- experiment records with frozen prediction, observation and comparison;
- local evidence matching by stable machine fingerprint.

Partially supported:

- VRAM/RAM prediction comparison for GPU/offload placements;
- concurrency as estimation input but not full load generation;
- artifact hash/provenance depending on artifact availability;
- runtime version/build capture depending on probe support;
- cost analysis via external notes or spreadsheet.

Requires development:

- first-class cost records;
- controlled multi-user load generation;
- deployment-plan domain model;
- hardware purchase advisor;
- calibrated prediction-error reporting across hardware classes;
- explicit process-comparable VRAM prediction in `PredictionComparison`.
