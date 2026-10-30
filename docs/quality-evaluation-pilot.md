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

### Current setup and non-CUDA execution

From the source checkout, `make setup` synchronizes Python with `uv sync --locked` and
builds `pilot/quality_eval/Dockerfile` as `jaull-quality-eval:gguf-v1`. `make sync` and
`make docker-build` run those steps separately; `make help` lists them. GNU Make, uv and
Docker with a running daemon are required for full setup. Native Windows needs GNU Make
and a POSIX shell (Git Bash/MSYS2), or WSL2; the execution bridge itself requires
Linux/WSL, not native Windows.

The equivalent manual commands are:

```bash
uv sync --locked
docker build --platform linux/amd64 --tag jaull-quality-eval:gguf-v1 pilot/quality_eval
```

No models or evaluation datasets are fetched by setup, and no evaluation starts. Image
construction installs the already pinned software and tokenizer resources. The container
runs lm-evaluation-harness; inference runs in a separately supplied **host llama-server**.
The server is not downloaded, built or installed automatically.

HellaSwag `smoke` (3 samples) and `hellaswag100` (100 samples) now have an explicit
`observable-runtime-v2` contract, recorded as **schema 3**:

- CPU: fixed ctx 2048, one slot, seed 0, four threads, `--n-gpu-layers 0`, no GPU device.
  The complete memory estimate must fit available RAM; VRAM is not required.
- CUDA: the same scoring controls, a runtime-confirmed device and full offload, with a
  confirmed device-memory budget. The historical CUDA protocol remains separately
  readable/reproducible; new runtimes do not silently inherit its identity.
- Vulkan: the selected accelerator must match exactly one runtime device using D098's
  matcher, and the estimate must fit its available device budget. Unknown budgets,
  software renderers and ambiguous devices are refused. No B60 physical support is claimed.

`--backend cpu|cuda|vulkan` selects evaluation placement, not the normal recommendation's
placement. Omit it for observed-hardware selection. An explicit unavailable GPU is an
error, never a silent CPU fallback; choose CPU explicitly if desired. GPU records also
require full-offload confirmation from the captured server log.

The new image must declare `io.jaull.quality.runtime-contract=observable-runtime-v2`.
Rebuild an older image before using the new HellaSwag route. The exact binary SHA256 is
captured and checked around runtime probes and again after execution; version/build must
be parseable, devices observable, and every scoring response must match that build's
fingerprint. Existing token IDs, finite logprobs, uncached prompt, slot/context checks,
raw HTTP reconstruction, evaluator/source pins and dataset hashes are still required.
Unknown HTTP layouts fail closed, even when a binary can otherwise generate text.

The identity includes backend/device, build, binary hash, capability digest, exact server
flags/defaults and Docker image ID. Complete capability output is saved with raw evidence.
The probe digest stays auditable but is excluded from protocol equivalence because its
device free-memory observations can fluctuate; effective backend/device/build flags still match.
Different builds, backends, placements, images or effective protocols are not automatically
comparable. Schemas 1/2 remain readable without rewriting historical records. Schema 3
HellaSwag is diagnostic only: neither smoke nor the 100-sample subset enters ranking.
IFEval's default CUDA/schema-2 contract is unchanged. An explicit external Windows Vulkan
route is separate (schema 4), described below; CPU and locally launched Vulkan IFEval
remain blocked. The new route does not change ranking.

### External Windows Vulkan IFEval (schema 4)

**IMPLEMENTED (offline tests), physical B60 execution UNVERIFIED.** This is Jaull's full
541-prompt IFEval with its seeded language detector, not an external/public IFEval score.
The protocol is `ifeval-external-vulkan-v1`, independent of `observable-runtime-v2`
HellaSwag/schema 3 and audited CUDA IFEval/schema 2. See decision D122 and
[the measured B60 comparison](b60-comparativa-wsl2-windows.md); the Windows server avoids
the measured Dozen slowdown/crashes, without changing those observations.

Scope: Jaull stays on WSL2; only `llama-server.exe` runs natively on Windows, launched by
the operator through WSL interoperability. This external route is CLI-only; the TUI and
its settings are unchanged. Windows Jaull is not implemented. The evaluator
uses a **local Linux Docker Engine in the same WSL network namespace**, `--network host`.
It uses the exact operator-supplied loopback endpoint, not a guessed Windows gateway.
Docker Desktop, remote/rootless daemons and non-loopback endpoints are deliberately
blocked in this first contract. No container GPU passthrough is required: only HTTP goes
to the Windows server. No firewall rules are changed.

If Docker is absent on Biosfer, physical evaluation is **BLOCKED_BY_ENVIRONMENT**.
Stop and decide whether to install a local Engine in WSL2 or request a separately designed
Desktop/remote-container topology. Do not replace this with native-Windows Jaull.

Preparation observes the native executable's version and devices, checks the selected
name through the D098 matcher and records the pre-load free/total memory budget. Zero,
unknown and ambiguous budgets/devices, software devices and unparseable builds fail closed.
The `.exe` must be a Windows PE executable. The memory estimate uses the native probe's
budget, not the WSL Dozen device or its potentially inverted index. Stop other model servers
before preparing; make a fresh manifest for each model/run and do not reuse old budgets.

The manifest contains the Windows model path corresponding to the fully hashed WSL-visible
GGUF, exact launch arguments, binary SHA256, native capability and log path. Jaull verifies
that path mapping again, probes/rechecks the binary around execution and retains full
offload and model/compute device allocation evidence from the supplied log. Before sending
scored requests it checks one slot, context 4096, four threads, seed 0, full offload and the
matched device. Every reply is still checked for build fingerprint, reasoning off, zero
cached tokens and the full generation budget. Afterwards it rehashes the model and binary,
reprobes the version/devices, and checks that the startup log is an unchanged prefix.

This is **operator-supplied evidence**, not remote process attestation. The operator must
use the printed command, a fresh log and a dedicated server; do not point a manifest at a
different process, share its slot with another client, change the model or edit its files.
Hashes detect drift, not a malicious operator. Jaull never stops that server or its tunnel,
including on failure/cancellation. It cleans up only its own UUID-named evaluator container.
The record labels ownership `externally_supplied`, host OS `windows`, the explicit endpoint
and the manifest digest. A nonzero attempt or missing/contradictory evidence cannot be imported.
Comparison remains conservative: the external manifest digest is part of runtime identity,
so separately prepared campaigns can be `NOT_COMPARABLE`; this route does not silently
normalise away those differences or promote a result into ranking.

#### Physical Biosfer procedure (not yet run)

In WSL2, from this updated source checkout, set the three actual local paths. The artifact
manifest must already describe the exact, single GGUF (immutable revision, SHA256, size and
existing local path). Do not download/substitute a model for this procedure.

```bash
uv sync --locked
docker info
docker context inspect --format '{{json .Endpoints.docker.Host}}'
# Require a local Linux Engine, not Docker Desktop/remote/rootless; stop otherwise.
IMAGE=jaull-quality-eval:ifeval-external-vulkan-v1
docker build --platform linux/amd64 --tag "$IMAGE" pilot/quality_eval
WIN_SERVER='/mnt/c/llama-win/llama-server.exe'
ARTIFACT='/absolute/path/to/artifact.json'
DATASET='/absolute/path/to/ifeval_input_data.jsonl'
RUN="$HOME/jaull-ifeval-b60-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
"$WIN_SERVER" --version
"$WIN_SERVER" --list-devices
# Confirm the exact B60 name; never infer its index from the WSL scan.
uv run python -m pilot.quality_eval.external_server \
  --artifact-json "$ARTIFACT" --llama-server "$WIN_SERVER" \
  --device-name 'Intel(R) Arc(TM) Pro B60 Graphics' \
  --port 8080 --server-log "$RUN/server.log" --output "$RUN/external-server.json"
```

In a second WSL terminal, execute the server command printed by preparation, verbatim.
It invokes the **Windows `.exe` natively**, with the verified Windows model path, mapped
device, `--host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1 --n-gpu-layers -1
--threads 4 --seed 0 --no-context-shift --no-cache-prompt --jinja --reasoning off --verbosity 4`.
Its stdout/stderr redirection captures the fresh log. Do not use the Linux Dozen binary.
On the measured machine the mapped device was Vulkan1, but the helper must establish it anew.

In Windows PowerShell, keep the known reverse tunnel open (replace SSH user/host):

```powershell
ssh.exe -N -o ExitOnForwardFailure=yes -R 8081:127.0.0.1:8080 -p 2222 USER@HOST
```

Back in the first WSL terminal, preserving its variables:

```bash
curl --fail http://127.0.0.1:8081/health
docker run --rm --pull=never --network host --entrypoint python "$IMAGE" \
  -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8081/health', timeout=10).read())"
# Optional explicit consent to fetch only the pinned IFEval dataset if missing:
uv run python -m pilot.quality_eval.setup --suite ifeval \
  --observable-runtime --external-runtime --llama-server "$WIN_SERVER" \
  --dataset-file "$DATASET" --image "$IMAGE" --allow-dataset-download \
  --output "$RUN/setup.json"
uv run jaull quality run --profile ifeval --backend vulkan \
  --artifact "$ARTIFACT" --dataset-file "$DATASET" --llama-server "$WIN_SERVER" \
  --pilot-root "$PWD" --image "$IMAGE" \
  --external-server "$RUN/external-server.json" --server-endpoint http://127.0.0.1:8081 \
  --output "$RUN/eval" --json
```

Both health checks must return `status: ok`. Successful evaluation prints `completed` with
`purpose: diagnostic_only` and a stored record path. `$RUN/eval/run-01/record.json` must
have schema 4, `identity.runtime.execution.protocol=ifeval-external-vulkan-v1`, backend
`vulkan`, the observed device, and ownership `externally_supplied`. The bundle retains
all 541 prompt/reply exchanges, runtime-before/after capability/hash evidence, template,
memory preflight, startup and complete server log. The existing Jaull language-detector
warning must remain visible. Record construction rechecks every HTTP reply and aggregate.

**Stop and send the logs** on either failed health check, missing native budget/device,
ambiguous match, non-PE/unparseable binary, Docker topology blocker, missing full-offload
line, CPU/other-device buffers, model/context/thread mismatch, reasoning/cache appearing,
HTTP fingerprint drift, timeout or import failure. Preserve `$RUN/server.log`,
`$RUN/external-server.json`, `$RUN/setup.json` when present, and all of `$RUN/eval` including
setup/run/snapshot logs and `*-error.json`. Do not rerun into the same output directory or
relax the guards. A failed run is not a zero quality score. Stop server and tunnel manually
only after preserving evidence. Only a completed physical run can change UNVERIFIED to
VALIDATED for this route; the earlier Windows speed measurements do not validate IFEval.

#### Reproducible CPU invocation

Supply absolute local paths; the manifest must name a verified single GGUF with immutable
revision and SHA256. Dataset preparation is separate and requires explicit consent:

```bash
uv run jaull quality setup --llama-server /absolute/bin/llama-server \
  --dataset-file /absolute/data/validation.parquet --pilot-root "$PWD" \
  --image jaull-quality-eval:gguf-v1 --allow-dataset-download
uv run jaull quality run --artifact /absolute/artifact.json \
  --dataset-file /absolute/data/validation.parquet \
  --llama-server /absolute/bin/llama-server --pilot-root "$PWD" \
  --image jaull-quality-eval:gguf-v1 --backend cpu --profile smoke \
  --output /absolute/new-cpu-smoke
```

For CUDA or Vulkan change only `--backend` and use a fresh output directory; the runtime
must expose the appropriate confirmed device. For HellaSwag100 use `--profile hellaswag100`.
Results → Evaluate candidates and individual GGUF Evaluation use the same backend selector
in Settings and the same preflight/producer. Search never evaluates or downloads weights.

CPU is substantially slower: startup is bounded at 180 seconds; CPU smoke at 30 minutes
and CPU HellaSwag100 at six hours. GPU HellaSwag retains 5/10-minute evaluator limits.
HTTP scoring waits at most 120 seconds per request; the owner has a finite outer deadline
and bounded cancellation cleanup. Timeout is a failed attempt, not a zero quality score.

Docker topology is inspected independently of the kernel: a local Linux/WSL Engine uses
host networking to reach the server bound to loopback; Docker Desktop uses the
bridge/`host.docker.internal` route. Remote, rootless and unidentified daemon topologies
are refused. Before scoring/generation the evaluator checks `/health` from inside its
own container, with a bounded timeout and an explicit connectivity error on failure.
All mounted paths must be accessible from the daemon. A Windows-host Vulkan device is not
assumed usable from WSL, and native Windows execution remains unsupported by this pilot.

**Validation status:** the historical CUDA measurements below remain historical evidence.
The new CPU/Vulkan contract is covered offline; no physical B60 or new CPU inference is
claimed without a successful completed run. This laptop has Docker/images and small GGUFs
but no installed `llama-server`; the real CPU smoke therefore remains pending.

### Historical audited CUDA pins

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
both the requested workload and evaluation fit at ctx 2048 (RAM for CPU, full-device
budget for CUDA/Vulkan). Preparing reads
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

- The published HellaSwag measurements describe one machine/GPU/runtime build;
  the new schema-3 CPU/Vulkan route does not extend that physical validation.
- No confidence interval on the per-run metrics, which is what blocks saying two
  models are indistinguishable.
- HellaSwag smoke/100 supports CPU-only evaluation and a uniquely identified GPU with
  confirmed full-device fit. Partial-offload and split/multi-GPU evaluation remain
  unsupported. IFEval uses either its audited single-CUDA-GPU protocol or the separate
  external Windows Vulkan schema-4 route above; no native-Windows Jaull is implied.
- Schema 3 requires retained runtime capability, command, server properties and positive
  model/compute device allocation evidence, including the effective thread count. Backend
  initialization alone is not execution proof. Missing/contradictory observations block
  import; the raw bundle is retained for diagnosis. Schemas 1/2 remain unchanged.
- Multipart GGUFs are rejected, including a shard renamed to look single-file.
- A checksum detects corruption, not authenticity. Imported records must be
  trusted local evidence.
- Evaluation duration is never inference-speed evidence.
