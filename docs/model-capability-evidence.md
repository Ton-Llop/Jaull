# General model capability evidence

Jaull can show what a publisher has reported about a model, next to what Jaull
has measured on an exact artifact, without ever confusing the two.

Status: diagnostic references only. Catalog `0.2.0` holds eleven transcribed
publisher results, reviewed 2026-10-01. **No quality-based ranking exists.**
Schema validation checks that a source was transcribed faithfully — not that its
claim is true, independently replicated, or applicable to the artifact that runs.

Earlier dated recall and coverage audits are archived in
[`history/capability-evidence-audits-2026-10.md`](history/capability-evidence-audits-2026-10.md).

## The contract

`domain/capability_evidence.py` defines frozen, validated records:

- **Subject** — exact repository, variant, commit revision and precision.
  Revision, variant and precision may be null: **null is unknown, never a
  wildcard**. Subject is a lookup key derived from those four fields, not a
  second stored identity.
- **Evaluation** — stable ID, one capability dimension, benchmark and version,
  metric, unit, direction, score, source URL, source kind and assessment date.
  The five dimensions are knowledge, reasoning, maths, coding and instruction
  following. `date` is Jaull's source-review date; publishers rarely give the
  actual evaluation date.
- **Protocol** — harness, shots, reasoning mode and token budget, generation
  budget, tools, temperature. `tools: []` means no tools; `tools: null` means
  unknown. Zero reasoning tokens means explicitly disabled. Anything else goes
  in `notes`.
- **Catalog** — schema version 1 (integer; a float or boolean is invalid),
  human-controlled catalog version, unique evaluation IDs. The loader records
  the SHA256 of the exact catalog bytes.

Invalid schema, duplicate IDs, duplicate JSON keys (including nested or
escaped), malformed or deeply nested JSON, non-finite or out-of-range scores,
credentials in source URLs and unreadable files all produce a **diagnostic, not
trusted evidence**. A missing catalog or a missing match never blocks a
recommendation or changes a score.

## What confirms a repository

`capability_catalog.py` attaches **only exact subject matches**. It does not
normalize aliases, fetch remote data, or transfer a result to a quantized plan.
Unknown revision matching unknown revision is allowed only as *unpinned*
evidence, with a diagnostic saying so.

On top of that, `engine_v2` requires the plan's canonical repository to be
confirmed by one of two evidence kinds — `REPOSITORY_ID` (the plan *is* that
repo) or `BASE_MODEL_METADATA` (its metadata declares that exact repo as base).
A name or suffix heuristic can never confirm lineage.

That rule took three attempts, and each failure is worth keeping:

1. **Unfiltered distribution.** The original engine copied `external_evaluations`
   to every plan, so a Qwen MMLU score could appear under a Mistral candidate.
   It now filters by exact subject per plan.
2. **Checking the kind, not the value.** The first fix accepted any entry of a
   confirming *kind*, so metadata pointing at `org/Other` confirmed a plan for
   `org/Target`, and reordering two evidence entries changed the answer. The
   check now compares the declared value against the canonical repo.
3. **The heuristic through the back door.** Comparing via
   `logical_model_repo_key` re-admitted it: that key strips suffixes, so metadata
   declaring `org/Thing-7B-GGUF` confirmed `org/Thing-7B`. Comparison is now
   exact, folding case only.

A repository match is the *first* check, not the last: variant, evaluated
revision and precision still need confirmation before evidence can describe an
execution plan.

## What can be compared

`comparison_blockers()` checks declared conditions. Different benchmark
versions, metrics, units, directions, representations, precision or protocol
conditions block comparison; so do unknown versions, revisions or relevant
protocol fields. Different model revisions are *expected* between two models —
both must be identified, not equal.

An empty blocker list means the recorded conditions align. It does not prove
equal prompts, uncontaminated benchmarks or independent replication. One
benchmark is not general superiority. There is no aggregate, no generation
bonus, no successor-based quality claim, and local tok/s is speed, not quality.

## Coverage, measured

The question was whether a strict lineage rule starves coverage. It does not;
the catalog does. A real search (6 queries, 40 candidates, first 12 audited on
2026-10-05) classified each candidate's lineage:

| Lineage of the 12 audited candidates | Count |
|---|---:|
| Declared `base_model` metadata | 7 |
| The repository itself | 2 |
| No lineage available | 3 |
| **Name heuristic only** | **0** |

So the strict rule costs nothing: no candidate depended on a name guess. The
actual bottleneck is elsewhere — **8 of the 12 have no catalog entry at all**,
and exactly one candidate passes end to end:
`bartowski/Qwen2.5-32B-Instruct-GGUF` inherits the Qwen2.5-32B-Instruct
reference through declared metadata, which is precisely the case the rule exists
to permit.

Discovery itself also bounds what can be covered. Queries sort mostly by
downloads; each returns at most 20 results; the orchestrator keeps at most 40
eligible repositories and inspects at most 12. Recent, unpopular or
differently-tagged models can therefore be missed, and **a catalog entry does
not add a candidate to discovery**.

Capability estimation meanwhile still uses `0.75 * size_prior + 0.25 *
artifact_signal`. Parameter count is not a measured intelligence score, family
detection can collapse generations through config `model_type`, and base-model
links describe lineage only — none of them identifies an evaluated
representation, fine-tune, quantization or MoE variant.

## Source review: two dense ~32B models

The official cards identify
[Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) as
32.5B parameters and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) as
32.8B. The newer model's selected profile is explicitly **non-thinking**.

From [Qwen3 report v1, tables 16 and 14](https://arxiv.org/html/2505.09388v1#S4.T16),
**publisher-reported**, not independent:

| Dimension | Benchmark / metric | Qwen2.5-32B-Instruct | Qwen3-32B non-thinking |
|---|---|---:|---:|
| Knowledge | MMLU-Redux accuracy | 83.9 | 85.7 |
| Scientific reasoning | GPQA-Diamond mean accuracy | 49.5 | 54.6 |
| Maths | MATH-500 accuracy | 84.6 | 88.6 |
| Coding | LiveCodeBench v5, pass@1 | 26.4 | 31.3 |
| Instruction following | IFEval strict-prompt accuracy | 79.5 | 83.2 |

The report gives sampling settings and an output cap, but no model commit IDs,
evaluated precision or complete per-baseline protocols. Those stay null, and so
**all five pairs are NOT COMPARABLE under this contract** — the newer numbers
being higher is not a comparison Jaull is allowed to make.

A retained [Qwen2.5 launch result](https://qwenlm.github.io/blog/qwen2.5-llm/)
reports LiveCodeBench **2305-2409 = 51.2**, against v5's 26.4. Different problem
windows are not improvements or regressions, and neither result overwrites the
other. The blog is unversioned, unlike the pinned report. Do not import
leaderboard defaults, shots, tools or harness revisions into an entry.

## Adding evidence by hand

There is no automatic updater.

1. Verify both exact model names exist. A conversational name like `Qwen3.8-32B`
   is not a published model.
2. Use model cards, technical reports or primary results. Publisher claims are
   `publisher_reported`, not independent evidence.
3. Record the evaluated representation, revision and precision where stated.
   Leave unspecified values null — never substitute the current Hub SHA.
4. Record the actual metric, benchmark version and conditions, including
   reasoning and generation budgets and tool use. Keep incompatible protocols
   separate; never convert a reported score into a new one.
5. Update `src/jaull/recommendation/capability_catalog.json` and its version,
   validate with `load_capability_catalog(Path(...))`, then run
   `pytest tests/test_capability_catalog.py tests/test_reporting_regression.py`.

## In the product

Results and Paths separate **Published model references** from **Measured on
this artifact**. The reference view is a bibliography of entries naming an exact
repository, keeping its original variant label and generation settings visible;
it is not an assertion that those protocols match the current plan. Artifact
measurements require the exact GGUF SHA256. Neither section changes scores,
ranking, suitability or eligibility, and neither fetches a model or starts an
evaluation.

Recommendation **exports** retain concrete catalog fields through
`SerializeAsAny`. This is an export guarantee only: `PlanAssessment` declares the
base type, so reloading through the legacy parser gives back 10 fields instead
of 19 and reconstructs no subject. **It is not a persistence contract yet.**

The diagnostic lookup stays separate and strict-subject:

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

## Before any ranking integration

All eleven catalog entries lack an evaluated revision and precision, so none
certifies the artifact that runs. The Quality/Fastest/Balanced draft is in
[`politica-calidad-velocidad-equilibrio.md`](politica-calidad-velocidad-equilibrio.md);
what blocks it is that no metric carries an interval, so "indistinguishable"
cannot be computed. Broader catalog coverage, an import contract and a measure of
discovery recall are separate follow-ups.
