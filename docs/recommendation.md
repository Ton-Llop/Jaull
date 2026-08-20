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

Answers are normalised into a `UserRequirements` object (`workflow/requirements.py`) that
records every assumption it made, and those assumptions appear in the exported report.

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

Budgets are centralised in `workflow/policies.py`:

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
memory figures of its own. Analyses are cached between runs so a repeated search does not
re-inspect the same repositories.

---

## How the ranking works

A composite score over eight normalised components, weighted and then renormalised to 1.0
(`recommendation/policies.py`):

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
confidence, and one more multiplier is applied on top: the **hard-requirement penalty**. A
candidate that requires commercial use but declares a restricted license is multiplied by
0, effectively removed; language misses and concurrency shortfalls apply softer penalties
(0.15 and 0.35) that still let the candidate compete.

Memory fit is itself multiplied by an **artifact realism** factor: a real GGUF variant or an
explicit `bnb-4bit` / `gptq` / `awq` tag scores 1.0, a Transformers repo loaded at its
native dtype scores 0.75, and a theoretical `int4` / `int8` selection with no confirmed
artifact drops to 0.4. This is what stops "Qwen-7B in int4" from beating a real Qwen-7B
GGUF at the same nominal size.

Compatibility remains a hard gate: a model assessed `insufficient` never appears at all,
and one assessed `unknown` can only ever be a flagged low-confidence alternative. The
heading on the primary card reflects the resulting tier:

| Tier | Heading | Trigger |
|---|---|---|
| Best match | `BEST MATCH` | HIGH confidence + comfortable/compatible + no hard-fail |
| Recommended | `RECOMMENDED` | Tight fit or MEDIUM confidence |
| Closest option | `CLOSEST OPTION` | Offloading required, or status unknown |
| Best-effort suggestion | `BEST-EFFORT SUGGESTION` | LOW confidence, or any hard requirement missed |

Within a family (Qwen2.5, Llama 3.1, Gemma 2, …), the recommender picks the single size
that best matches the priority — the other sizes are shown as a **series ladder**
underneath: *"Same series, other sizes: 0.5B · 1.5B · 3B · 7B"*, each with its own status.

Every reason and warning is generated by rules in `recommendation/explanations.py`. **No
language model is involved in ranking or in writing the explanations.**

Ties break deterministically (status → confidence → downloads → `repo_id`), so the same
inputs always give the same report.

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
