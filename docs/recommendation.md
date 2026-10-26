# Guided mode and the recommender

Guided mode answers the question the individual tools cannot: *what should I run on this
machine?* You never need to know a model name, a quantization, a dtype, a KV cache or a
runtime flag — six or seven plain questions are enough, and every technical parameter is derived
from them.

```text
Hardware scan
      ↓
User requirements
      ↓
Hugging Face discovery
      ↓
Candidate enrichment
      ↓
Memory estimation
      ↓
Explainable ranking
      ↓
Recommendations
```

| | Guided mode | Advanced tools |
|---|---|---|
| For | Anyone who wants a recommendation | Users who already know the repository |
| Input | A handful of plain-language questions | A `repo_id` or URL, plus flags |
| Output | Up to 5 explained recommendations | Raw analysis / estimate for one model |
| Screens | Welcome → Hardware → Questions → Search → Results | Scan, Inspect, Estimate, Doctor |

When Search completes, Results replaces the progress screen. **Back / Escape**
returns to the same **Your needs** form with every answer preserved. Changing a
priority or another answer and submitting starts a new search on the detected
hardware; it does not mutate the previous recommendation or rerun automatically.

---

## The questions the wizard asks

1. **What will you use it for?** — general chat, programming, documents, summarization,
   reasoning, or writing and translation. This is the *task*: what work the model does.
2. **How will requests be processed?** — interactive or batch. This is the *workload mode*:
   whether a caller is waiting for the answer.
3. **What matters most?** — best quality, balanced, fast responses, or lowest memory usage.
4. **Which languages?** — multiple choice, plus free-text codes under "Other".
5. **How many people at once?** — one, 2–5, 6–10, or more than 10.
6. **How much text at a time?** — asked **only** for the documents use case. It sets the
   model's context window, which is *not* the size of a document collection: a retrieval
   system feeds the model a few chunks at a time.
7. **Must the model allow commercial use?** — yes / no / not sure, defaulting to yes.

The numbers above are what a documents user sees. For every other task the sixth question is
hidden and the wizard renumbers what remains, so the sequence never skips.

### Task and workload mode are orthogonal

They answer different questions and every combination is valid. `coding` + `batch` is
analysing many repositories offline; `document_qa` + `interactive` is a chatbot over
documents; `document_qa` + `batch` is summarising thousands of files. Nothing in the ranking
restricts one by the other, and workload mode does not enter `task_match`.

The wizard offers only these two modes. *Deployment* — a persistent service, an API server,
a daemon — is a separate dimension, because a service can be interactive or batch. Jaull
does not model deployment yet. The old serialized `service` value remains readable for
existing records but is not offered for new selections.

Answers are normalised into a `UserRequirements` object in
`jaull.application.requirements`, which records every assumption it made; those assumptions
appear in the exported report.

`WorkloadProfile` carries the requested workload: context, concurrency, mode, expected input
and output tokens, and optionally `min_generation_tps` and `max_ttft_ms`. Those last two are
*requested* objectives, not measurements — a benchmark observation is a separate quantity and
the two are never mixed.

---

## How models are found

Search uses the official `HfApi.list_models` API — no HTML scraping, no weight downloads,
and no token required for public models. `HF_TOKEN` is used when present for higher rate
limits, and is never logged or exported.

The query builder issues several complementary queries per use case (task wording, one per
preferred weight format, one per non-English language, plus a trending query) rather than
relying on a single string. Results are then:

1. **Interleaved** round-robin across queries, so the format and language queries are not
   starved by the first one.
2. **Deduplicated** by `repo_id`, merging the query labels.
3. **Filtered** — private, gated, multimodal, wrong-pipeline, base-less adapters and (when
   commercial use is required) non-commercial licenses are rejected. Thin metadata is
   *never* a rejection; it becomes a recorded penalty and lower confidence.
4. **Shortlisted** down to the deep-inspection budget using cheap pre-inspection signals
   only.

The shortlist is **hardware-aware** (`discovery/candidate_filter.py`). From repository
metadata alone it derives a `coarse_placement_hint` — would this candidate plausibly sit in
VRAM, need offload, run from system RAM, or not fit at all — and weights the queue with it
(`GPU_RESIDENT` +4.0, `GPU_OFFLOAD` +3.0, `CPU_RAM` +1.5, `TOO_LARGE` −9.0). The point is to
keep all three viable placements in play instead of spending every slot on one size class,
and to do it **without treating RAM and VRAM as a single memory pool**. A parameter-count
hint parsed from the repository name (`…-7B-…`) only orders this queue; it is never
presented as a measurement.

Budgets are centralised in `application/recommendation/policies.py`:

| Budget | Value |
|---|---|
| Results per query | 20 |
| Unique candidates | 40 |
| Deep inspections | 12 |
| Concurrent inspections | 4 |
| Variant deep inspections | 6 |
| Targeted alternative queries per Search | 3 |
| Recommendations returned | 5 |

Only the shortlist pays for inspection, which reuses the existing `inspect_model`, the
analyzers, the base-model resolver and `estimate_memory` — the guided flow computes no
final memory figures of its own. The shortlist hint is not persisted, reported or used for
ranking; after inspection, `MemoryEstimate` and `HardwareFitResult` are the source of truth.
Analyses are cached between runs so a repeated search does not re-inspect the same
repositories.

### Selecting a technically compatible execution path

Memory feasibility, technical support and runtime readiness are separate decisions.
The estimator's precision ladder now checks the available backend before accepting a
Transformers INT8/INT4 choice. Jaull's current quantized Transformers loader requires
an observed CUDA/HIP backend; a CPU-only plan cannot become a confirmed recommendation
just because its theoretical quantized footprint fits. This is a limit of Jaull's
validated loading path, not a claim that every bitsandbytes version lacks CPU support.
See the upstream [hardware support documentation](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes).

Before ranking, the workflow reuses its analysis and estimate caches to build a bounded
plan pool: the declared native float dtype plus the existing FP16/INT8/INT4 ladder
(at most four precisions), or up to eight published GGUF variants. The configuration
already selected by the estimator is always retained within that GGUF bound. Packed
Transformers artifacts remain one representation, not invented dequantized variants.
Each evaluated configuration has its own `MemoryEstimate`; a rejected quantized plan
does not lend its footprint to a float or GGUF plan. The memory and placement formulas
and priority ordering are unchanged.

If a logical model has no known memory-fitting, technically compatible plan, Search
first reuses inspected conversions and relevant original search hits. It can then call
the existing metadata-only variant discovery, with a global limit of three queries
and six additional repository inspections, beyond the original shortlist of twelve.
Only explicit repository/`base_model` evidence confirms a cross-artifact relation.
Names may nominate an inspection but cannot merge models or substitute artifacts.
When no viable path is confirmed, the original diagnostic remains available with a
bounded-discovery warning; no conversion is assumed.

Technical blockers reject the configuration. Missing binaries or Python dependencies
only affect readiness and do not order recommendations. The best eligible plan under
the existing priority policy becomes the primary; other configurations, including
rejected configurations with their reasons, remain in Paths. Results and exports
use the selected plan's configuration and estimate. Published quality, IFEval and
local speed retain their existing applicability gates; conflicting known SHA256
values cannot match through weaker repository metadata.

---

## How the ranking works

Ranking runs over **execution plans**, not over repositories. Every candidate that survives
inspection is expanded into the concrete ways it could actually run — one plan per artifact
variant and runtime — and it is those plans that `recommendation/engine_v2.py` assesses and
orders.

There is deliberately **no global numeric score**. `PlanAssessment` keeps the dimensions
apart and the ranking policy reads them as an ordered tuple, so every position in the list
can be explained by naming the axis that decided it:

| Axis | What it answers |
|---|---|
| `suitability` | Does the repository match the declared task? Instruct/chat signals raise it; a base model is capped. |
| `capability` | Parameter-count prior and artifact signal (`recommendation/capability.py`), not measured general quality. |
| `feasibility` | Does the memory prediction fit this hardware? Read from `CompatibilityStatus`. |
| `executability` | Is the plan technically coherent — does that runtime accept that artifact? |
| `execution_fitness` | The two above combined; `BLOCKED` if either one is. |
| `performance_evidence` | Is there an eligible local microbenchmark for this artifact/runtime/backend on this machine? Not a workload throughput prediction; see [eligibility and limitations](evidence.md#eligibility-for-recommendation-ranking). |
| `confidence` | Confidence of the estimate the plan was built on. |
| `runtime_readiness` | **Operational only.** Whether this installation could launch it right now — never read by ranking. |

The optional [general capability evidence contract](model-capability-evidence.md)
keeps published evaluations separate from this scale prior. It does not change ranking.

Levels are `STRONG` / `ADEQUATE` / `WEAK` / `BLOCKED` / `UNKNOWN`. `_ranking_key` turns them
into a lexicographic tuple per priority, so the priority decides *which axis is consulted
first*, not how much weight it carries:

| Priority | Axes, in order |
|---|---|
| **Quality** | suitability → capability → quantization quality → execution fitness |
| **Speed** | suitability → runnability → executability → memory efficiency |
| **Memory** | memory efficiency → execution fitness → executability |
| **Balanced** | suitability → runnability → capability → execution fitness → executability → memory headroom |

Every key ends with the same tail — confidence, `repo_id`,
quantization, runtime name — so ties break deterministically and identical inputs always
produce an identical report.

Speed's fallback protects task suitability and separates normally runnable plans
from offloading or uncertain plans before rewarding a smaller memory footprint.
The quantization ladder still selects a configuration within each repository;
it is not a global preference for GGUF/Q4 over other models or runtimes. Memory
efficiency is a selection prior, not a predicted or measured tokens/s figure.

**Balanced selection.** Before measured evidence, `weak_plans_last` prefers known
strong/adequate capability within contiguous groups sharing confirmed memory fit,
rejection state and runnability. Weak/unknown capability fills the other positions.
A capable plan with unknown fit or offload cannot jump over a normally runnable
plan across these boundaries. This metadata preference runs before measurements,
so applicable comparable evidence can override it. The active order, Recompare and
report export use the same sequence.

Reports expose this step as `ranking.selection_policy`, including original and
selected pool positions when the saved pool reproduces the displayed result.
The measured policy's base position is the position after selection, so selection
moves are not attributed to benchmarks. Missing historical positions stay unknown.

**Measured evidence (Quality, Speed and Balanced).** After the key and selection,
`apply_measured_policy` (`recommendation/shadow.py`) lets stored measurements reorder
plans, under strict limits. Quality reads applicable quality: one complete audited IFEval
run of the plan's exact GGUF bytes. Speed (Fastest) reads applicable speed: the plan's
own `llama_bench_v2` tg128, on its exact artifact, offload, machine and device, with the
settings llama-bench restated matching the protocol. `shadow-v2` uses the median of all
distinct valid calls in the newest applicable speed cohort (including older calls and
completed benchmarks from an interrupted campaign). It requires at least three calls
before speed can order plans. This is a conservative eligibility gate, not statistical
significance. The observed min/max range is not a confidence interval, and one atypical
call can keep two plans indistinguishable indefinitely (a known limit; no robust range
yet). Fastest and Balanced compare each pair directly. Speed decides a pair only when
the two observed ranges are disjoint (touching counts as overlap). Balanced also needs
quality, compared by 95% Wilson intervals of the IFEval score: a plan goes first when no
axis is demonstrably worse and one is demonstrably better. Overlapping speed is an
operational tie, so clearly better quality decides it; when neither axis is
distinguishable, or the pair is a real trade-off (faster but clearly worse), the base
order stays. This is not strict Pareto dominance. Chains are never merged: A overlapping
B and B overlapping C still let a disjoint A–C pair decide. The final order is the
closest to the base that honours every demonstrable pair: a winner rises to the loser's
slot. Measured
plans can only trade places with each other, inside one comparable cohort and one
contiguous stratum (confirmed fit, constraints, suitability, license and language). Every
other plan keeps its base position, so with no applicable evidence the order is exactly
the key order. Memory has no measured policy.

`assess_speed` is the only authority on valid speed evidence. Measured speed is not an
axis of the key: it acts only through `apply_measured_policy`, and the plan's displayed
tok/s and `performance_evidence` read the applicable figure for information, never raw
benchmark records. A `llama_bench_v1` record stays visible as a reference but no longer
orders plans.

The report separates `ranking.criteria` (the base order) from the measured stage of the
request's priority: `ranking.quality_policy`, `ranking.speed_policy` or
`ranking.balanced_policy`. Each carries its measurement(s), cohort, record digest or
all contributing benchmark IDs, run count, observed speed range, and the pool positions
before and after the policy, before diversity. Five internal repetitions are not five
independent calls; within-call standard deviations are not pooled into uncertainty.
Missing saved-pool positions remain unknown; a measurement alone does not prove a move.
Markdown preserves the same separation. The pool before this step is what Search saves
for Recompare. Recompare calculates on that pool but reports proposed changes against the
saved visible recommendations, so an already-applied swap is not announced as a new move.

### What removes a plan from the list

A plan carrying any `HardConstraint` is rejected before ordering begins. These codes do that:

| Code | Meaning |
|---|---|
| `ARTIFACT_RUNTIME_INCOMPATIBLE` | That runtime cannot load that artifact format |
| `RUNTIME_UNDETERMINED` | No runtime could be planned, usually because the memory estimate is incomplete |
| `MEMORY_INSUFFICIENT` | The model does not fit, offload included |
| `LICENSE_INCOMPATIBLE` | Commercial use is required and the license forbids it |
| `LANGUAGE_INCOMPATIBLE` | A required language the model does not declare |

Compatibility therefore remains a hard gate: a plan assessed `insufficient` never appears at
all. A plan whose fit is unknown because no runtime could be planned is also rejected, under
`RUNTIME_UNDETERMINED` — which records "could not confirm", never "shown not to fit". Today
that is every Mixture-of-Experts model, whose KV footprint the estimator does not model.

### Ranking does not depend on what you have installed

A missing `llama-cli` does not change the Top 5. This is a deliberate decision, and it is why
the assessment carries two separate axes:

- **`executability`** is technical and *is* ranked — could this runtime load this artifact
  format at all?
- **`runtime_readiness`** is operational and is *never* ranked — could this machine launch
  the plan at this moment?

The question a recommendation answers is *what is the best plan for this hardware?*, and the
answer does not change because a binary has not been installed yet. Treating it as if it did
was actively harmful. A missing runtime used to raise `RUNTIME_NOT_READY` as a hard
constraint, which **deleted** the plan: on a machine without llama.cpp every GGUF
recommendation disappeared, and when the whole shortlist was GGUF the run reported "no
compatible models". Worse, a working binary compiled without the backend the hardware had
selected also counted as not ready — so a CUDA machine with a CPU-only build ranked *below*
a plain CPU laptop. Better hardware, worse result.

`runtime_readiness` is still computed, reported and rendered. It is what the **Ready** chip on
the Execution Paths screen shows, and it is what gates Run, Validate and Benchmark: those
fail with the reason and the install hint instead of invoking a binary that is not there.
Download is deliberately *not* gated — fetching an artifact needs no runtime.
`HardConstraintCode.RUNTIME_NOT_READY` still exists in the enum, but it is now raised in the
action layer, where *can I launch this now?* is the actual question.

The invariant is pinned by `tests/test_recommendation_runtime_agnostic.py`: with the same
hardware, requirements, candidates and artifacts the ranking is identical with and without
the runtime installed, and only `runtime_readiness` differs.

### One slot per logical model

Ranked plans are collapsed by `recommendation/diversity.py` before they become
recommendations. The best plan for a `ModelIdentity` becomes the primary; the remaining plans
for that identity stay attached as alternatives rather than consuming another of the five
slots. When the next candidate carries the same assessment signature as the current leader,
the diversifier prefers one that differs in family, parameter tier or execution profile — so
the list does not degenerate into five quantizations of the same model.

### The composite score still exists, but no longer decides

The eight-component weighted score (`recommendation/scoring.py`, weights in
`recommendation/policies.py`) predates the plan-based engine:

| Component | Base weight | What it measures |
|---|---|---|
| Memory fit | 25 % | Does the model fit on this hardware for one user? |
| Concurrency fit | 10 % | Does it still fit at the requested concurrency? |
| Capability | 15 % | Family + parameter-count signal (Qwen2.5-7B > TinyLlama-1B) |
| Task match | 20 % | Repo tags, pipeline and keywords vs the use case |
| Language match | 12 % | Fraction of requested languages the model declares |
| License | 8 % | Commercial-use category from the license table |
| Metadata quality | 7 % | Confidence in every number, from card completeness |
| Popularity | 3 % | Log-scaled downloads + likes, capped as a tie-breaker |

Priority shifts these — *quality* raises task match and capability, *speed* and *memory*
raise both memory fit and concurrency fit. The total is then scaled by the estimate's
confidence and by the **hard-requirement penalty**: a candidate that requires commercial use
but declares a restricted license is multiplied by 0, effectively removed, while language
misses and concurrency shortfalls apply softer penalties (0.15 and 0.35) that still let the
candidate compete. Memory fit is itself multiplied by an **artifact realism** factor: a real
GGUF variant or an explicit `bnb-4bit` / `gptq` / `awq` tag scores 1.0, a Transformers repo
loaded at its native dtype 0.75, and a theoretical `int4` / `int8` selection with no
confirmed artifact 0.4. That is what stops "Qwen-7B in int4" from beating a real Qwen-7B
GGUF at the same nominal size.

It survives in two places, and in neither of them does it order the guided results:

1. **The exported report.** `score_breakdown` is still written into the JSON and the
   Markdown, because the report schema is a byte-identical contract
   (`tests/test_reporting_regression.py`). It is no longer shown in the TUI.
2. **The no-hardware path.** `recommend(..., hardware=None)` has no execution plans to
   assess, so it still ranks with the composite score and the older series grouping.

On the guided path — where hardware is always known — the order comes from `_ranking_key`,
and the composite score is descriptive only.

### The heading on the card

The tier is chosen by `recommendation/tier.py` from four signals: compatibility status,
confidence, the hard-requirement penalty, and **actionability** — whether the artifact and
runtime path is confirmed, likely or merely speculative. Strongest downgrade first:

| Tier | Heading | Trigger |
|---|---|---|
| Best-effort suggestion | `BEST-EFFORT SUGGESTION` | Any hard requirement missed, LOW/UNKNOWN confidence, or a speculative artifact path |
| Closest option | `CLOSEST OPTION` | Offloading required, or status unknown |
| Recommended | `RECOMMENDED` | Tight fit, MEDIUM confidence, or actionable but not confirmed |
| Best match | `BEST MATCH` | Everything else, at HIGH confidence |

A speculative plan is still ranked, but it can never earn a `BEST MATCH` heading, because the
path that would run it has not been confirmed.

The alternatives under the primary card come from the diversifier — other execution plans for
the same logical model. The older **series ladder** (*"Same series, other sizes: 0.5B · 1.5B ·
3B · 7B"*) is only populated on the no-hardware path; the guided path leaves it empty.

Every reason and warning is generated by rules in `recommendation/explanations.py`. **No
language model is involved in ranking or in writing the explanations.**

---

## Licenses

Licenses are bucketed conservatively into `commercial_allowed`, `commercial_restricted` and
`unknown` from a small documented table. Custom vendor licenses (Llama, Qwen, DeepSeek, …)
are deliberately classified as **unknown** rather than allowed: many do permit commercial
use below a user threshold that this tool cannot verify.

> License information is reported from model metadata and is **not legal advice**. Check the
> model's license yourself before commercial use.

---

## From recommendation to evidence

Results shows a speed readout and a **Speed** shortcut beside each plan, also in the
selected inspector. The shortcut opens the existing benchmark setup for that exact
plan; it replaces the duplicate Benchmark action in Results. Benchmark completion saves
the record automatically, and returning refreshes stored evidence. Opening Results never
downloads or runs a model.

- **Measured X tok/s** shows the latest applicable local `llama_bench_v2` tg128 run.
  Provenance includes its date and ID, the separate ranking median, call count and
  observed range (not a confidence interval). Ranking still uses the aggregate.
- **Last benchmark X tok/s (Transformers / llama.cpp)** shows the latest successful
  matching generation measurement at 128 tokens when ranking evidence is unavailable,
  including CPU/Vulkan measurements, using the existing metadata,
  machine, backend and precision matching. Known conflicting artifact digests are
  excluded. This is historical evidence, not verified exact bytes or current
  execution. Details include date, protocol, repetitions, runtime flags and record ID;
  it does not order Fastest/Balanced.
- **No speed label** is shown without applicable evidence or a matching historical
  measurement; the **Speed** shortcut remains available. Other-condition
  benchmarks remain historical references under **Speed provenance**, never
  throughput attributed to the current plan.

Unvalidated heuristic and calibrated predictions are not shown in Results. A real
Qwen2.5-0.5B Transformers FP16 run measured 18.36 tok/s against roughly 170 estimated;
this discrepancy was reproduced at a similar speed with standard `model.generate`.
The offline predictor and experiments remain available for research, but neither
predicts the displayed speed nor orders Fastest/Balanced. See
[the diagnostic](../validation/qwen2.5-0.5b-transformers-decode-2060-20261014/README.md).

Applicable speed currently requires the existing llama.cpp evidence contract;
displaying a saved benchmark does not make it ranking-applicable. A Speed
microbenchmark is not a production-context benchmark. This display does not alter
ranking, saved searches, evidence records or exported reports.

A recommendation is where the interesting part starts, not where it ends. From the results
screen, each recommendation resolves into a logical model identity, its artifact variants
and the execution plans that could actually run it — which can then be prepared, run,
validated and benchmarked. See [evidence.md](evidence.md).
