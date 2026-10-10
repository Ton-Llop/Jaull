# GGUF quality-evaluation pilot

Jaull can measure how well an exact GGUF answers a fixed benchmark, under a
protocol pinned tightly enough that two results are either comparable or
explicitly refused. It is opt-in: opening a screen never evaluates anything.

**Nothing in this pilot changes recommendation ranking.** It is a separate axis
from HFA fit, llama-bench speed and task match. There is no overall quality
score.

The chronological record of how this was built, with every digest, failed
attempt and superseded gate, is in
[`history/quality-pilot-journal-2026-10.md`](history/quality-pilot-journal-2026-10.md).

## What is pinned

A result is only meaningful against the protocol that produced it, so all of
this is part of record identity:

| Piece | Pin |
|---|---|
| Evaluator | lm-evaluation-harness commit `ad8737ae7fad24cf64e50fc7fc31397bff586b9e` (`0.4.14.dev0`), source SHA256 `71161c2b…`. Never a floating version string or `main`. |
| Container | Python 3.12.12 slim/bookworm, amd64. Must carry label `io.jaull.quality.artifact-contract=exact-local-gguf-v1`; unlabelled older images fail preflight. The resolved image ID is part of identity. |
| Runtime | The audited host `llama-server` binary, by SHA256. CUDA full offload, one GPU, one slot (`--parallel 1`), context 2048, `--no-cache-prompt`. |
| Dataset | `Rowan/hellaswag` revision `218ec52e…`, validation split, loaded from a local Parquet file verified against SHA256 `89981307…`. |
| Suite | `jaull-quality-smoke-v1` (indices `[0,1,2]`) or `jaull-hellaswag-100-v1` (100 indices drawn by `random.Random(20261002)` without replacement, then sorted). Zero shots, raw prompts, no chat template. |
| Artifact | Exact single-file GGUF by content SHA256, with an immutable 40-hex revision. Names, family and quantization labels never substitute for the digest. |

Why the backend is pinned to a commit and not a release: v0.4.13's GGUF backend
drops generation kwargs, leaves `loglikelihood_rolling` unimplemented, and
inherits an `apply_chat_template` that raises. The pinned commit forwards
`max_gen_toks` and reads modern `logprobs.content`. It still does not implement
chat templating, so chat, multiturn, tools, reasoning modes and
perplexity/rolling tasks are **rejected**, not approximated.

Response caching stays off: lm-eval's cache key is method plus request
arguments, with no artifact identity in it. Prompt caching is off too — with it
on, the same top token's logprob moved from -0.95909 to -0.99199 between an
uncached and a cached-prefix request.

A CLI flag that the server accepts but ignores is not an applied setting.
Preflight fails when a requested setting does not reach HTTP.

## Three tiers of evidence

| Classification | Samples | Reusable? | Comparable? |
|---|---|---|---|
| `plumbing` | 3 | No | Only against another `plumbing` run of the same suite |
| `limited` | 100 | No | `COMPARABLE_LIMITED` when every protocol field matches |
| `full` | the fixed split | Yes | Yes |

Only `full` enters `quality_lookup`. Relabelling a `limited` record as full does
not make it reusable — the classification is checked, not trusted. A 3-example
smoke and a 100-example run are `NOT_COMPARABLE`: neither their samples nor
their suites are combined.

Reuse is keyed on `identity_sha256`, a digest over artifact, suite, dataset,
samples, evaluator, runtime and effective protocol. **Placement flags are part
of runtime identity**, so a CPU result never satisfies a lookup for a GPU one.
That is measured, not cautious: see [placement](#placement-is-score-relevant).

The store keeps repeated runs of the same identity. When several share one
identity, a strict lookup **abstains** — it does not pick one or average them.

## What has been measured

Two exact local artifacts, 100 fixed HellaSwag examples, RTX 2060, same image
and server build, run sequentially:

| Exact local artifact | `acc` | `acc_norm` |
|---|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 36/100 | 38/100 |
| Qwen2.5-1.5B-Instruct Q5_K_M | 46/100 | 63/100 |

Comparison reports `COMPARABLE_LIMITED`. These are results for this subset, not
full HellaSwag scores, not significance claims and not a general ranking:
parameter counts, quantizations and tokenizers all differ, so no isolated family
or quantization effect is established. 100 examples leave sampling uncertainty
and cover one task.

The same two digests, benchmarked separately with llama-bench (full CUDA0
offload, 4 threads, b2048/ub512, f16 K/V, 5 repetitions):

| Exact local artifact | pp512 tok/s | tg128 tok/s |
|---|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 5937.53 ± 654.15 | 190.73 ± 5.67 |
| Qwen2.5-1.5B-Instruct Q5_K_M | 4487.96 ± 394.41 | 123.94 ± 1.63 |

Two synthetic token tests, not a 512-token conversation followed by a 128-token
answer. No TTFT, latency or service capacity. The ± are sample standard
deviations over five repetitions, not confidence intervals. The GPU was not
isolated (23% and 28% pre-run utilization), so contention and run order are not
excluded. Quality and speed stay in separate records.

### Placement is score-relevant

Replaying the 146 captured scoring requests of one completed bundle against the
same build and GGUF, changing only `--n-gpu-layers`, `--device` and `--threads`:

| Arm | `-ngl` | threads | Tokens equal to the bundle | max abs delta, continuation |
|---|---:|---:|---:|---:|
| `gpu_full_a` / `gpu_full_b` | -1 | 4 | 146/146 | 0 |
| `gpu_partial` | 11 | 4 | 97/146 | 0.8776 |
| `cpu_only` / `cpu_threads8` | 0 | 4 / 8 | 0/146 | 0.8261 |

1. **Fixed placement reproduced exactly.** Two fresh processes with identical
   flags reproduced all 146 serialized logprobs of a bundle recorded hours
   earlier. That is not a claim of universal bitwise reproducibility.
2. **Placement changes scores, and not at the last bit.** Full CPU moves every
   token; partial offload moves 49 of 146. The largest continuation-level
   disagreement is 0.88 nats — a different number, not rounding.
3. **The answers held in these three samples.** No per-sample answer changed in
   any arm. The tightest lead (doc 1) went from +0.0255 to +0.0182 — it lost 29%
   of a small margin and stayed positive. Changes partly cancel in the
   differences: a continuation sum moves up to 0.8776 nats while the lead
   between two of them moves up to 0.4186. Roughly a factor of two, not an
   order of magnitude.
4. **4 and 8 threads agreed here.** One pair of values on one model and build.
   Not grounds for dropping `--threads` from identity.

Whether `-ngl -1` reproduces on a *different* GPU is untouched by this probe:
unverified, not refuted, and the obvious next measurement. The probe writes
`quality_evidence: false` and never produces a record.

## Running it

One-time setup verifies the llama-server SHA256, inspects the local image and
its contract label, and verifies or downloads the pinned dataset. The dataset
permission defaults **off** and is independent of GGUF download consent.

```bash
uv run jaull quality setup --allow-dataset-download --json
```

Then run artifacts sequentially, one server and container at a time:

```bash
uv run jaull quality run \
  --pilot-root . \
  --artifact /path/to/artifact.json \
  --dataset-file /path/to/validation.parquet \
  --llama-server "$HOME/tools/llama.cpp/build-cuda/bin/llama-server" \
  --image jaull-quality-eval:gguf-v1 \
  --profile hellaswag100 \
  --output .codex-night/quality-cli-new-run \
  --json
```

Existing output directories are refused, never overwritten. Each completed,
HTTP-validated snapshot is stored before the next run starts; a failure stops
later runs but keeps earlier records. Ctrl+C lets the pilot finish cleanup of
its owned server and container first. Compare two record IDs offline, with no
Docker or GPU:

```bash
uv run jaull quality compare LEFT_ID RIGHT_ID --json
```

Matching protocol gives accuracy, counts, the paired left-minus-right difference
and discordant outcomes; a mismatch gives `NOT_COMPARABLE` with reasons and no
paired metrics. `limited` runs with observed variation also get a deterministic
paired percentile-bootstrap 95% interval from 2,000 resamples — exploratory, and
it does not rank models.

Rebuilding the evaluator uses a **new tag**, so the old image is preserved and a
stale image fails loudly instead of silently scoring three samples:

```bash
docker build --platform linux/amd64 --tag jaull-quality-eval:gguf-v1 pilot/quality_eval
```

**The runner needs a trusted source checkout.** `scripts/quality_eval_smoke.py`
and `pilot/quality_eval/` do not ship in the wheel, and `--pilot-root` executes
that checkout's Python — never point it at a directory you do not control. The
wheel reads quality records; it does not run evaluations. Nothing is downloaded,
built or substituted by these commands.

## In the TUI

**Advanced tools → Evaluate local GGUF quality**, **Paths → Evaluation**, or
**Results → Evaluation** for a selected single-file GGUF. Download consent
defaults off on every screen; a missing artifact is fetched only after **Start
evaluation** and only with consent enabled. Output defaults to a unique
directory under user-data `quality-runs/`, not the repository.

**Results → Evaluate candidates** prepares a queue from candidates the search
already inspected, before the final truncation to five. It inspects up to six
logical models and selects up to three, explaining every omission. Only
confirmed single-file GGUF identities qualify, and the estimator must confirm
both the requested workload and a full-device fit at ctx 2048. Preparing reads
Hub metadata and bounded GGUF header ranges; it downloads no model and launches
no container. This is a selection heuristic, not a claim that these are the best
models.

The Evaluation tab keeps two sections that must never bleed into each other:

- **Published model references** — catalog entries for a repository confirmed by
  repository identity or declared base-model metadata. Name and suffix
  heuristics cannot establish that relationship. Evaluated variant, reasoning
  mode, source and unknown revision/precision stay visible. A bibliography, not
  a claim that the current plan reproduces the number.
- **Measured on this artifact** — exact GGUF content hash only, with sample
  coverage, classification and protocol limitations. Published results never
  fill this section and never replace a local measurement.

Historical records are displayed but not reused to skip an evaluation: an
artifact digest alone does not establish a matching execution identity. Missing
catalog entries are not evidence of poor quality.

## Limits

- One task (HellaSwag), one machine, one GPU, one runtime build.
- No confidence interval on the per-run metrics, which is what blocks saying two
  models are indistinguishable.
- CPU-only, partial-offload and multi-GPU evaluation are unsupported. Those
  machines keep their model recommendations; they just cannot acquire evidence.
- Multipart GGUFs are rejected, including a shard renamed to look single-file.
- A checksum detects corruption, not authenticity. Imported records must be
  trusted local evidence.
- Evaluation duration is never inference-speed evidence.
