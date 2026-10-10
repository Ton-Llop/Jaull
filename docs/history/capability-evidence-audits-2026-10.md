# Capability-evidence audits — arxiu (octubre 2026)

Auditories datades de recall de la cerca, mostres en viu i cobertura del
cataleg, tal com es van anar apendant a `docs/model-capability-evidence.md`
durant l'octubre del 2026. Es conserven senceres per les xifres observades.

El contracte viu i vigent es `docs/model-capability-evidence.md`.

---

# General model capability evidence v1

Status: diagnostic catalog, strict offline lookup and TUI references. No quality-based ranking.
Catalog 0.2.0 contains eleven transcribed publisher results, reviewed 2026-10-01.
This verifies the source contents, not independent execution or model superiority.
Test fixtures are synthetic (`.invalid` URLs); shipped-data tests pin the real claims.

## What the audit found

The audit inspected code and offline tests, followed by the bounded live sample
below. The sample is not an exhaustive recall evaluation.

- `discovery/query_builder.py` searches task phrases, formats and languages;
  most queries sort by downloads, with one trending query.
- `discovery/search_client.py` asks Hugging Face for public, ungated repositories
  in the requested pipeline. Each query returns at most 20 results.
- `workflow/orchestrator.py` interleaves query results, deduplicates and applies
  the preliminary filter before retaining at most 40 eligible repositories.
  `discovery/candidate_filter.py` selects at most 12 for inspection. Cheap
  desirability, format preferences and hardware
  size buckets influence that shortlist before capability assessment.
- Consequently dynamic search can miss recent or unpopular candidates, models
  with different naming/tagging, and models outside the inspection budget.
  A catalog entry does not add a candidate to discovery.
- `MetadataCapabilityAnalyzer` uses `0.75 * size_prior + 0.25 * artifact_signal`.
  Family and instruction tuning are descriptive fields, not quality evidence.
  Parameter count is not a measured general-intelligence score.
- Family detection can collapse generations through config `model_type`;
  base-model links describe lineage. Neither safely identifies an evaluated
  representation, fine-tune, quantization or MoE variant.

### Discovery recall follow-up (2026-10-01)

The current pipeline has these loss points, before general-quality evidence
can help:

1. Task phrases generate download-sorted queries. Preferred formats and
   non-English languages reuse the primary phrase; one trending query also
   reuses it. The [Hub API contract](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_models)
   describes `search` over model IDs, not semantic search of model cards.
   A relevant model with different naming can miss every query.
2. `HfSearchClient` requests `gated=False`, pipeline/tag filters, card metadata
   and 20 results per query. The SDK translates `trending_score` to
   `trendingScore`. Results outside those pages are unavailable downstream.
3. Round-robin interleaving gives each query early slots. Deduplication merges
   query labels and preserves first-seen order. The existing preliminary rules
   reject unsupported modalities, pipelines, adapters and restricted licenses.
4. The first 40 eligible unique repositories enter the coarse hardware-aware
   shortlist. Its size/memory buckets, popularity and redundancy preferences
   select at most 12 repositories. These are heuristics, not measured quality.
5. Inspection failures cannot produce plans. Surviving analyses generate
   Transformers or available GGUF plans for v2 assessment. Diversity groups
   logical model identities and retains other paths as alternatives, with at
   most five main recommendations. The no-hardware compatibility route also
   collapses families/series; it is not the guided hardware-aware route.
6. Subsequent Paths discovery searches variants of an already-selected model,
   with its separate six-inspection budget. It cannot recover a logical model
   that never reached the initial recommendation input.

**Corrected budget interaction:** previously step 3 truncated to 40 *before*
filtering. Unsupported repositories could exhaust that budget, even when the
same fetched pages already contained valid candidates. Filtering now precedes
the unchanged 40-candidate cut. A synthetic regression reproduces 40+ rejected
hits followed by a valid candidate, then checks it reaches inspection and the
recommendation with merged query provenance. Search calls, per-query limit,
shortlist limit, eligibility rules and all ranking/scoring formulas are unchanged.
Only cheap filtering sees more of the already-fetched metadata. This is not
a freshness bonus or a guarantee that the 41st *eligible* candidate survives.

### Small live recall sample (2026-10-01)

Earlier sandbox requests failed DNS resolution. Approved access outside the
sandbox succeeded; those failures were not evidence of a Hub outage.
Anonymous `HfSearchClient` requests ran from **07:49:15 to 07:49:20 UTC** using
the current queries, without weights or model execution. Requirements: quality,
English, commercial use, context 4096, one interactive user, formats GGUF and
safetensors. Hardware was scanned, not simulated: WSL2 Linux, Ryzen 5 3600,
7.71 GiB total / 6.74 GiB available RAM, RTX 2060 6 GiB / 4.71 GiB available,
driver 617.14. These small memory pools matter to shortlist selection.

Every query used `pipeline_tag=text-generation`, `gated=False`, `cardData=True`,
`limit=20`; sorts are descending. The SDK sends `trending_score` as
`trendingScore`. Exact query inputs and returned page sizes:

| Task | Search | Tag filter | Sort | Hits |
|---|---|---|---|---:|
| Chat | `instruct` | none | downloads | 20 |
| Chat | `chat` | none | downloads | 20 |
| Chat | `multilingual instruct` | none | downloads | 20 |
| Chat | `instruct` | gguf | downloads | 20 |
| Chat | `instruct` | safetensors | downloads | 20 |
| Chat | `instruct` | none | trending_score | 20 |
| Coding | `coder instruct` | none | downloads | 20 |
| Coding | `code generation` | none | downloads | 20 |
| Coding | `programming assistant` | none | downloads | 0 |
| Coding | `coder instruct` | gguf | downloads | 20 |
| Coding | `coder instruct` | safetensors | downloads | 20 |
| Coding | `coder instruct` | none | trending_score | 20 |

Using production interleaving, deduplication, filtering and shortlist functions:

| Stage | Chat | Coding |
|---|---:|---:|
| Page hits, including duplicates | 120 | 100 |
| Unique repositories | 91 | 65 |
| Preliminary survivors | 91 | 65 |
| After eligible-candidate budget | 40 | 40 |
| Coarse hardware shortlist | 12 | 12 |

There were **no preliminary rejections** in this sample. Replaying the old
pre-filter budget therefore yielded the same eligible lists; it does not
demonstrate the corrected defect live. The synthetic regression covers that
specific failure. No query or budget was changed to improve these sample counts.

Five audit subjects were verified separately with anonymous `model_info`:

| Exact repository | Chat loss/survival stage | Coding loss/survival stage |
|---|---|---|
| `Qwen/Qwen2.5-32B-Instruct` | Eligible, not shortlisted | Not in fetched pages |
| `Qwen/Qwen3-32B` | Not in fetched pages | Not in fetched pages |
| `Qwen/Qwen3-4B-Instruct-2507` | Shortlisted | Not in fetched pages |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Eligible, not shortlisted | Eligible, not shortlisted |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Not in fetched pages | Not in fetched pages |

Qwen3-32B exists and reports `text-generation`, but its ID contains neither
`instruct` nor `chat`: the queries are not semantic coverage of capable models.
The Mistral repository had **no pipeline tag** in the separate API response;
the server-side pipeline filter is another possible loss point, not a proven
diagnosis for every missing result. The two large Qwen subjects that reached
the eligible pool were excluded by the coarse shortlist, before final ranking.
Do not interpret these outcomes as poor quality or a requirement to recommend
32B models on this hardware. A zero-hit query and naming/metadata gaps warrant
a broader, separately reviewed recall study, not model-specific exceptions.

This is a dated five-subject sample, not coverage of all current families.
Download-sorted/trending page contents and hardware headroom can change. Current Hub
revisions checked for existence are **not** evaluated-model revisions and were
not inserted into the capability catalog.

The captured pages and hardware were then replayed through `run_workflow`,
with **real remote metadata inspection and estimation**, from 07:51:28 to
07:58:01 UTC. This was not a second live search or a mocked estimator.
Persistent analysis/GGUF caches were disabled, metadata downloads used a
temporary Hub cache, and GGUF reads used the existing bounded Range reader.
No runtime execution, full model downloads, readiness verification or speed benchmark
was involved. Both workflows completed without workflow-level errors:

| Downstream stage | Chat | Coding |
|---|---:|---:|
| Deep inspections | 12 | 12 |
| Non-failed candidates reaching plan assessment | 12 | 10 |
| Main recommendations with plan and assessment | 5 | 5 |

Coding inspection identified two adapters that the search metadata had not
excluded: `yusifnuri/Mistral-7B-v0.3_code_generation` and
`yusifnuri/phi-4-mini-instruct_code_generation`. They were not ranked.
Final logical repository IDs, in observed order:

- Chat: `Qwen/Qwen3-4B-Instruct-2507`, `HuggingFaceTB/SmolLM2-135M-Instruct`,
  `Qwen/Qwen2.5-0.5B-Instruct`, `h2oai/h2o-danube3-500m-chat`,
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Coding: `Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
  `deepseek-ai/deepseek-coder-7b-instruct-v1.5`, `ajaythumu/code_generation_model`,
  `Qwen/Qwen2.5-Coder-3B-Instruct`.

The ordinary v2 engine handled family/path grouping and the five-item limit;
the capability catalog was **not supplied to it**. These lists record current
behavior, not endorsement of their quality or confirmation that their selected
paths can execute. No numeric score, task match, eligibility rule or hardware
formula was changed on the basis of this sample.

### Catalog coverage of the observed recommendations (2026-10-06)

An offline check intersected the ten exact repository IDs in the two final lists
above with the repository IDs in shipped catalog `0.2.0` (11 entries). It did not
query the Hub, infer lineage or match names/families. A repository match would
only be a first check: variant, evaluated revision and precision still need
separate confirmation before evidence can describe an execution plan.

| Recorded workflow (2026-10-01) | Final recommendations | Repository IDs in catalog |
|---|---:|---:|
| Chat | 5 | 0 |
| Coding | 5 | 0 |
| Combined, distinct | 10 | 0 |

The catalog covers only `Qwen/Qwen2.5-32B-Instruct` (six entries) and
`Qwen/Qwen3-32B` (five). Neither was a final recommendation in that hardware
sample. The first reached the chat eligible pool but not its shortlist; the
second was absent from the fetched pages. These are coverage and discovery
observations, not negative quality findings about the ten recommended models.
In particular, a Qwen3-4B or Qwen2.5-Coder result cannot inherit the 32B
reference by family name. This dated sample used the recorded RTX 2060 machine;
it is not a fresh recommendation run on the current laptop.

Current production code loads the catalog for diagnostic references in Results
and Paths, separately from locally persisted evaluations for exact artifact
bytes. Published references do not certify the selected artifact or execution
protocol, and neither source changes recommendation ordering.

`application/recommendation/service.py` consumes the analyzer's numeric score;
`recommendation/engine_v2.py` also constructs the metadata analyzer when assessing
plans. This implementation changes neither path. It attaches optional sourced
evaluations to the existing `CapabilitySignal`, leaving score, confidence,
reasons and every ranking input unchanged.

`domain/recommendation.py` already defines `ExternalEvaluationEvidence`.
`CapabilityEvaluation` extends it as a strict catalog entry, reusing `benchmark`,
`task`, `value`, `metric`, `source`, `revision` and `date`. Existing legacy records
remain valid under their original type. The original engine distributed
`external_evaluations` to every plan. It now filters exact subjects per plan
and requires a confirmed canonical repository; unattributable entries are dropped.
The service loads the catalog once and preserves its version, digest and diagnostics.

## Contract and matching

`domain/capability_evidence.py` defines frozen, validated records:

- **Subject:** exact repository, variant, commit revision and precision.
  Revision, variant and precision may be null: unknown, never a wildcard.
- **Evaluation:** stable ID, one capability dimension (`task`), benchmark/version,
  metric, unit, direction, score, source URL, source type and assessment date.
  Dimensions are knowledge, reasoning, maths, coding and instruction following.
  `value` is the reported score, `source` the URL, `date` the assessment date,
  and `revision` the evaluated model commit. `subject` is a lookup key derived
  from repository, variant, revision and precision, not a second stored identity.
- **Protocol:** evaluation method/harness, shots, reasoning mode and token
  budget, generation budget, tools and temperature. Additional conditions use
  the existing evaluation `notes` field. Missing values remain explicit nulls;
  `tools: []` means no tools,
  whereas null means unknown. Zero reasoning tokens means explicitly disabled.
- **Catalog:** integer schema version 1, human-controlled catalog version and
  unique evaluation IDs. Boolean/float versions are invalid. The loader also
  records SHA256 of the exact catalog bytes.

`recommendation/capability_catalog.py` reads a caller-provided local path and
attaches **only exact subject matches**. It does not infer model identity, follow
base-model links, normalize aliases, fetch remote data or transfer results to a
quantized execution plan. Matching unknown revision to unknown revision is
allowed only as unpinned evidence, with an explicit diagnostic; it does not
confirm any specific model revision.

Publisher-reported and independent results remain separately identified.
Schema validation is not verification of their truth. Source URLs must not
contain username/password credentials. Catalog labels and tool names must contain
a non-whitespace character; blank text is invalid, not a known protocol value.
Nonblank strings are preserved exactly, not trimmed or normalized for matching.
Use null for unknown conditions and an empty tools list for explicitly no tools.
These constraints do not tighten the historical `ExternalEvaluationEvidence` parser.
Invalid schema, duplicate evaluation IDs or JSON object keys (including nested
or escaped keys), malformed/deeply nested JSON, non-finite scores, invalid score
ranges and unreadable files yield a diagnostic, not trusted evidence.
Missing/invalid catalogs and missing matches do not block a
recommendation or change its score. Historical records need no migration.

## What can be compared

`comparison_blockers()` checks declared conditions, not a winning model.
Different benchmark versions, metrics, units, directions, representations,
precision or evaluation conditions prevent comparison. Unknown benchmark
versions, model revisions or relevant protocol fields also prevent comparison.
Different model revisions are expected between different models; both must be
identified, not equal. Different additional protocol notes are conservative
blockers.

An empty blocker list only means the recorded conditions align. It does not
prove equal prompts, uncontaminated benchmarks, or independent replication.
Reviewers must check the actual sources and any conditions recorded in notes.
One benchmark is not general superiority; a broad profile still has gaps and
trade-offs. There is no aggregate, generation bonus or successor-based quality
claim. Local tok/s measures execution speed, not model quality.

## Adding evidence by hand

1. Verify that both exact model names exist; do not treat conversational names
   such as `Qwen3.8-32B` as published models without checking.
2. Use original model cards, technical reports or primary benchmark results.
   Record publisher claims as `publisher_reported`, not independent evidence.
3. Identify the evaluated representation, revision and precision where stated.
   Leave unspecified values null rather than substituting the current Hub SHA.
4. Record the actual metric, benchmark version and conditions, including
   reasoning/generation budgets and tool use. Keep incompatible protocols
   separate; preserve reported scores without converting them into a new score.
5. Review coverage across the five dimensions, source limitations and comparison
   blockers. Keep source URLs, review date and evaluation IDs stable.
6. Update `src/jaull/recommendation/capability_catalog.json` and its version,
   validate with `load_capability_catalog(Path(...))`, and run
   `pytest tests/test_capability_catalog.py tests/test_reporting_regression.py`.

There is no automatic updater or refresh service.

## Source review: two dense ~32B models

The official cards identify [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
as 32.5B parameters and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
as 32.8B. These are verified repository names, not a claim about `Qwen3.8-32B`.
The newer model's selected profile is explicitly **non-thinking**.

The [Qwen3 report v1, tables 16 and 14](https://arxiv.org/html/2505.09388v1#S4.T16)
reports the following percentages. These are **publisher-reported**, not
independent results or comparable local measurements:

| Dimension | Benchmark / metric | Qwen2.5-32B-Instruct | Qwen3-32B non-thinking |
|---|---|---:|---:|
| Knowledge | MMLU-Redux accuracy | 83.9 | 85.7 |
| Scientific reasoning | GPQA-Diamond mean accuracy | 49.5 | 54.6 |
| Maths | MATH-500 accuracy | 84.6 | 88.6 |
| Coding | LiveCodeBench v5, pass@1 | 26.4 | 31.3 |
| Instruction following | IFEval strict-prompt accuracy | 79.5 | 83.2 |

The report supplies Qwen3 sampling settings and output cap, but not model
commit IDs, evaluated precision or complete per-baseline protocols. Those
fields remain unknown: **all five pairs are NOT COMPARABLE under this v1
contract**. GPQA covers scientific reasoning, not all logical reasoning.
Date means Jaull's source-review date; actual evaluation dates are not supplied.
No current model-card dtype or Hub commit is substituted for an evaluated one.

A retained [Qwen2.5 launch result](https://qwenlm.github.io/blog/qwen2.5-llm/)
has LiveCodeBench **2305-2409 = 51.2**, not v5's 26.4. Different problem windows
are not improvements/regressions; neither result overwrites the other. The
blog is unversioned, unlike the pinned report, so its future contents may drift.
The benchmark's [official leaderboard](https://livecodebench.github.io/leaderboard_v5.html)
identifies pass@1; its default window is not the report's selected window.
Do not import leaderboard defaults, shots, tools or harness revisions into these entries.

Independent replication, exact revisions/precision, benchmark/scorer versions,
full baseline conditions and uncertainty intervals remain unavailable in this
small example. The data is not a general-quality verdict and never feeds ranking.

## Product display and next review boundary

Results and Paths now separate **Published model references** from **Measured
on this artifact**. The reference view is a bibliography of entries naming an
exact repository, not an assertion that their variants or protocols match the
current plan. It can show Qwen3's non-thinking profile while keeping its original
variant label and generation settings. References to a base repository require
declared metadata naming that exact repository; suffix/name heuristics cannot
confirm lineage. Missing catalog data and missing entries remain explicit.
Source kinds, conflicting observations and protocol gaps remain visible.

Artifact measurements still require the exact GGUF SHA256 and show historical
placement, sample coverage and classification. They do not replace unrelated
published results or imply a general-capability verdict. Neither section changes
scores, ranking, suitability or eligibility. The display does not fetch models
or start evaluations; evaluation remains explicit and subject to pilot preflight.

Recommendation exports retain concrete catalog fields through `SerializeAsAny`.
This is an export guarantee, not a new import/persistence contract: reloading
through the legacy base-type parser does not reconstruct a catalog entry.

`attach_capability_evidence` remains a separate strict-subject lookup. With an
already-computed signal, the diagnostic lookup is:

```python
from pathlib import Path
from jaull.domain.capability_evidence import CapabilitySubject
from jaull.recommendation.capability_catalog import (
    attach_capability_evidence, load_capability_catalog,
)

catalog = load_capability_catalog(Path("src/jaull/recommendation/capability_catalog.json"))
subject = CapabilitySubject(
    repo_id="Qwen/Qwen3-32B", variant="post-trained/non-thinking",
    revision=None, precision=None,  # As reported, not the current execution plan.
)
diagnostic = attach_capability_evidence(existing_signal, subject, catalog)
# Inspect evaluation_evidence and evidence_diagnostics; score is unchanged.
```

This queries an unpinned published representation, not a recommendation or a
confirmation of the user's selected artifact. Bibliographic display does not
broaden this matching contract.

Before ranking integration, decide which independently reviewed comparisons
justify a broad capability claim, how protocol gaps and conflicting dimensions
are presented, and how discovery recall is measured. Broader catalog coverage and
an import contract remain separate follow-ups. No new GPU campaign or HFA
calibration is required for this diagnostic display.

## Priority policy for human review (proposal only)

The current catalog has no overlap with the ten observed final recommendations,
so these modes cannot yet justify a new automatic ordering. Any future policy
would use the existing fit and task eligibility decisions unchanged and keep
missing evidence explicitly unknown:

| Priority | Evidence to consider after existing eligibility | When evidence is missing or incompatible |
|---|---|---|
| Quality | Reviewed, model-specific results for the requested task under stated, comparable protocols; show dimension and source, not a single general score. | Abstain from quality ordering; keep the model eligible and mark quality unknown. |
| Fastest | Local measurements of complete execution plans on the relevant machine and comparable workload, with methodology and backend visible. | Abstain from a speed claim; never treat tok/s as quality. |
| Balanced | Present supported quality, local speed and memory as separate trade-offs for human choice. | Show the gaps and avoid a forced winner or hidden weighted total. |

Smoke and 100-example HellaSwag results remain protocol diagnostics, not broad
quality evidence. Published results for an unpinned representation must not be
silently attached to a different quantization, fine-tune or current revision.
No ranking, score, task-match or placement rule is changed by this proposal.
