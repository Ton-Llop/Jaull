# Guided mode and the recommender

Guided mode answers the question the individual tools cannot: *what should I run on this
machine?* You never need to know a model name, a quantization, a dtype, a KV cache or a
runtime flag — six plain questions are enough, and every technical parameter is derived
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
| Input | Six plain-language questions | A `repo_id` or URL, plus flags |
| Output | Up to 5 explained recommendations | Raw analysis / estimate for one model |
| Screens | Welcome → Hardware → Questions → Search → Results | Scan, Inspect, Estimate, Doctor |

---

## The questions the wizard asks

1. **What will you use it for?** — general chat, programming, documents, or writing and
   translation.
2. **What matters most?** — best quality, balanced, fast responses, or lowest memory usage.
3. **Which languages?** — multiple choice, plus free-text codes under "Other".
4. **How many people at once?** — one, 2–5, 6–20, or more than 20.
5. **How much text at a time?** — asked **only** for the documents use case. It sets the
   model's context window, which is *not* the size of a document collection: a retrieval
   system feeds the model a few chunks at a time.
6. **Must the model allow commercial use?** — yes / no / not sure, defaulting to yes.

Answers are normalised into a `UserRequirements` object (`application/requirements.py`, still
importable as `workflow/requirements.py`) that records every assumption it made, and those
assumptions appear in the exported report.

This is not a workload model. It captures intent and a concurrency bucket, not throughput,
latency or time-to-first-token objectives — see the roadmap in the README.

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

Budgets are centralised in `application/recommendation/policies.py`, re-exported by
`workflow/policies.py`:

| Budget | Value |
|---|---|
| Results per query | 20 |
| Unique candidates | 40 |
| Deep inspections | 12 |
| Concurrent inspections | 4 |
| Variant deep inspections | 6 |
| Recommendations returned | 5 |

Only the shortlist pays for inspection, which reuses the existing `inspect_model`, the
analyzers, the base-model resolver and `estimate_memory` — the guided flow computes no
final memory figures of its own. The shortlist hint is not persisted, reported or used for
ranking; after inspection, `MemoryEstimate` and `HardwareFitResult` are the source of truth.
Analyses are cached between runs so a repeated search does not re-inspect the same
repositories.

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
| `capability` | Family and parameter-count signal (`recommendation/capability.py`). |
| `feasibility` | Does the memory prediction fit this hardware? Read from `CompatibilityStatus`. |
| `executability` | Is the plan technically coherent — does that runtime accept that artifact? |
| `execution_fitness` | The two above combined; `BLOCKED` if either one is. |
| `performance_evidence` | Is there a measured benchmark for this exact plan on this machine? |
| `confidence` | Confidence of the estimate the plan was built on. |
| `runtime_readiness` | **Operational only.** Whether this installation could launch it right now — never read by ranking. |

Levels are `STRONG` / `ADEQUATE` / `WEAK` / `BLOCKED` / `UNKNOWN`. `_ranking_key` turns them
into a lexicographic tuple per priority, so the priority decides *which axis is consulted
first*, not how much weight it carries:

| Priority | Axes, in order |
|---|---|
| **Quality** | suitability → capability → quantization quality → execution fitness |
| **Speed** | executability → throughput → quantization chosen for speed → memory efficiency |
| **Memory** | memory efficiency → execution fitness → executability |
| **Balanced** | suitability → runnability → capability → execution fitness → executability → memory headroom |

Every key ends with the same tail — performance evidence, confidence, `repo_id`,
quantization, runtime name — so ties break deterministically and identical inputs always
produce an identical report.

### What removes a plan from the list

A plan carrying any `HardConstraint` is rejected before ordering begins. Four codes do that,
and all four are genuine incompatibilities rather than states of this particular machine:

| Code | Meaning |
|---|---|
| `ARTIFACT_RUNTIME_INCOMPATIBLE` | That runtime cannot load that artifact format |
| `MEMORY_INSUFFICIENT` | The model does not fit, offload included |
| `LICENSE_INCOMPATIBLE` | Commercial use is required and the license forbids it |
| `LANGUAGE_INCOMPATIBLE` | A required language the model does not declare |

Compatibility therefore remains a hard gate: a plan assessed `insufficient` never appears at
all, and one assessed `unknown` can only ever be a flagged low-confidence alternative.

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

A recommendation is where the interesting part starts, not where it ends. From the results
screen, each recommendation resolves into a logical model identity, its artifact variants
and the execution plans that could actually run it — which can then be prepared, run,
validated and benchmarked. See [evidence.md](evidence.md).
