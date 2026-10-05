# Controlled GGUF quality-evaluation pilot

Status: controlled plumbing pilot completed (2026-10-02). Two sequential
three-example smokes and a placement replay exist; no full or reusable model-quality
evaluation exists. Snapshot import validates HTTP coverage and aggregate metrics.
This is separate from HFA fit, llama-bench speed and task-match scoring.
Nothing in this pilot changes recommendation ranking.

## Historical prerequisite audit — 2026-10-01

The initial blockers below were resolved for the completed runs described later;
they are not the current environment status.

- A local TinyLlama 1.1B Q4_K_M GGUF is present in Jaull's model cache.
  Its complete file SHA-256 matched its sidecar during this audit:
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
  Before each evaluation, use Jaull's full artifact verification; a sidecar
  or matching filename alone does not establish the bytes being evaluated.
- `llama-server --version` reports build `10357 (689e227db)`, but also
  reports that CUDA could not initialize because the driver is insufficient
  for the CUDA runtime. `nvidia-smi` sees the RTX 2060 (driver 617.14).
  A CPU-only server attempt with the verified TinyLlama file failed before
  loading it because this sandbox could not bind `127.0.0.1:18080`.
  No inference or server API compatibility was established by these probes.
- The `docker` command in this WSL distro says Docker Desktop's WSL
  integration is not enabled. No container or Docker GPU test was run.
  Enable integration for the distro and confirm `docker info` before a
  container smoke test. See [Docker's WSL instructions](https://docs.docker.com/desktop/features/wsl/).

## Backend compatibility gate

Do not select an evaluator version from its documentation alone. The
[v0.4.13 GGUF backend](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/models/gguf.py)
passes a fixed temperature but does not forward general generation kwargs
or a generation token limit to `/v1/completions`; its rolling
log-likelihood method is unimplemented. Its base
[`LM.apply_chat_template`](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/api/model.py)
raises `NotImplementedError`, and GGUF does not override it. Thus v0.4.13
is **not accepted** for a controlled chat-template/generation comparison.
The post-release [commit `ad8737a`](https://github.com/EleutherAI/lm-evaluation-harness/commit/ad8737a)
is a **candidate pin**, not a validated evaluator. Its
[GGUF implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737a/lm_eval/models/gguf.py)
forwards `max_gen_toks` as `max_tokens`, uses modern `logprobs.content`
and queries `/props` for server slots. It still does not forward every
generation kwarg or implement chat templating. Do not build from floating
`main`; expand the candidate to its full commit hash and test the actual
HTTP payload before accepting a container image or any capability result.

The candidate task must use methods implemented by that pinned backend.
Check `/tokenize`, `/props` and completion/logprobs responses against the
selected llama-server build. Avoid `--apply_chat_template` unless the pinned
backend implements and tests it. Record the effective prompt format,
stop strings, temperature, token limit, reasoning mode and server flags.
Fail preflight when a requested setting is ignored; do not silently treat
accepted CLI flags as applied model settings. Keep response caching
(`--use_cache`) disabled: the
[v0.4.13 cache key](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/api/model.py)
uses method plus request arguments, not artifact identity.

## Minimal execution sequence

1. Pin the evaluator source/image, task definitions and llama.cpp build.
   Start one GPU-backed llama-server with one verified GGUF mounted read-only,
   plus a separate evaluator container. A host llama-server is an explicit
   fallback if GPU containers cannot run; record which boundary was used.
2. Probe one request at each required endpoint and inspect its effective
   payload and response. Run only a few examples, saving prompts, outputs,
   raw results, command lines and errors. A `--limit` run is a plumbing
   smoke test, not a published capability result.
3. Only after the smoke succeeds, evaluate two exact artifacts sequentially
   on one GPU with the same suite, dataset revision, sample IDs, few-shot
   policy and effective prompt/generation protocol. If those conditions
   cannot be established, report NOT COMPARABLE. Keep model speed evidence
   in Jaull's existing benchmark records, not these quality results.
4. Keep completed results immutable. Add a SQLite lookup index only when
   reuse is needed, keyed by full artifact SHA-256 and all effective protocol
   inputs (including evaluator/runtime versions and task/sample identity).
   Never reuse failed or partial results. Present per-task evidence with
   provenance, without an overall quality score or ranking integration.

Do not infer a safe number of parallel model containers from VRAM alone.
Start with one model per GPU. Only after sequential correctness, optionally
measure additional server slots under recorded memory and timing conditions.
No cloud rental, large model download or long unattended evaluation belongs
to this pilot without a separate decision.

**Current gate:** enable Docker Desktop integration for this WSL distro,
make a compatible llama-server available, and run the smoke in an environment
that permits a loopback listener. The pinned backend's raw-prompt behavior
also needs an explicit task/prompt decision before comparing instruct models.
These are prerequisites, not failed quality measurements.

## New launch audit — 2026-10-02

This section supersedes the dated prerequisite diagnoses above. This iteration
audits the protocol; it does **not** complete an lm-eval task smoke or establish
model-quality evidence. The earlier document content and overnight progress
remain history. No recommendation, HFA, runtime policy or stored record changed.

### Observed prerequisites

- Docker CLI 29.8.1 and Compose v5.5.1 are installed. Sandbox access to the
  Docker socket and GPU was denied; approved probes outside that sandbox
  confirmed engine 29.8.1/Linux and RTX 2060, driver 617.14, 6144 MiB.
  One disposable, network-disabled, read-only container using already-local
  image `5d143123fdf8` successfully ran `/usr/bin/nvidia-smi` with Docker's
  `--gpus all`. This proves container GPU visibility, not CUDA
  model inference inside a container. It exited and was removed.
- Host `llama-server` build 10357, source commit
  `689e227db485c6b33d061555e74034c93a867649`, binary SHA256
  `cdb0749a2cffc6f2fe710a9616263f90e6a016e8ac07a9caa5be160fc2c5d98e`,
  loaded TinyLlama on CUDA and served loopback HTTP under approved access.
  Logs confirm 23/23 launch units offloaded, CUDA0 model/KV/compute buffers
  601.02/44.00/47.01 MiB. These are runtime buffers, not driver-attributed
  process VRAM, a peak-memory measurement or a hardware-fit recalibration.
- Jaull `ArtifactService.verify(full=True)` rehashed the already-local
  `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` file
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`: 668788096 bytes, SHA256 as above.
  Cached Hub download metadata names revision
  `52e7645ba7c309695bec7ac98f4f005b139cf465`; this is cached provenance.
  Verification read the file/sidecar without downloading or rewriting them.
- No evaluator image is already local. Other small Qwen GGUF files exist,
  but their full identities/readiness have not been audited for this pilot.
  No second artifact was run. Both temporary host servers were stopped.

### Accepted scope and protocol gates

Pin evaluator source to
[`ad8737ae7fad24cf64e50fc7fc31397bff586b9e`](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/lm_eval/models/gguf.py)
(`0.4.14.dev0` in its package metadata), not the version string alone.
The backend source SHA256 is
`71161c2b04e55699b0397e3f745da74a8c39e010f88371418a65bdcad3b8bc63`.
The isolated offline check executes this exact class with mocked HTTP; it is
not a full harness installation or end-to-end task evaluation.

| Concern | Effective behavior / pilot gate |
|---|---|
| Tokenization and scoring | `/tokenize` with `add_special=true`; whitespace migration, encode together and split at context length. One `/v1/completions` request per continuation token: token-ID prompt, temperature 0, max_tokens 1, logprobs 2, logit_bias [[target,100]], id_slot 0. Require matching token IDs, finite pre-sampling logprobs and unbiased top tokens. |
| Generation | Constructor temperature, stop strings and normalized token limit reach HTTP. Request temperature, do_sample, top_p, seed and arbitrary kwargs do not. Reject requests relying on those ignored fields; token-limit aliases have priority, so disallow conflicting aliases. Generation is outside the initial task suite. |
| Chat | Base `LM.apply_chat_template` raises; GGUF does not override it. `/v1/completions` receives raw text/IDs. The model's template appears in `/props` but is not applied here. Reject chat-template, multiturn, tools and reasoning-mode requests. |
| Rolling metrics | `loglikelihood_rolling` raises. Reject perplexity/rolling tasks. |
| Context | Backend `max_length` is stored but does not enforce/truncate context. Check tokenized requests against the observed server slot context; disable context shifting. CLI flags alone do not establish applied limits. |
| Slots | Default concurrency comes from `/props.total_slots`, with serial fallback. Context groups map round-robin to slot IDs; generation is unpinned. Use server `--parallel 1` and backend `parallel=1`; no slot-scaling experiment. |
| Cache | Keep `--use_cache` absent/OFF: its key omits model/protocol identity. Request caching also stays off. Server prompt caching is a separate setting recorded below. |

Live probes confirmed `/props.total_slots=1`, slot context 2048, BOS insertion,
modern `logprobs.content` with token IDs, forced-token sampling and a two-token
generation cap. With default prompt caching, the same top token's logprob
changed from -0.95909023 to -0.99199140 between an uncached request and a cached
prefix request. The first assertion failure and raw responses were preserved;
this observation alone does not establish its cause. With
`--no-cache-prompt`, both requests reported zero cached tokens and identical
top logprobs; a forced non-top token retained a finite negative logprob.
Use this observed setting for the first smoke rather than widening tolerances.
No timing in these responses is used as local-speed evidence.

### Next implementation step

1. Build only an evaluator container at the full commit above; lock its
   dependencies/base image and record the resulting image identity. Use the
   proven host CUDA server boundary first. Document that a host file read
   provides no protected container mount, and rehash before/after serving it.
   Any container serving the artifact must mount it read-only.
2. Define suite `jaull-quality-smoke-v1`: upstream
   [HellaSwag task v1.0](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/lm_eval/tasks/hellaswag/hellaswag.yaml)
   and preprocessing at the evaluator pin, dataset `Rowan/hellaswag` revision
   `218ec52e09a7e7462a5400043bb9a69a41d06b76` (metadata queried, data not fetched),
   validation indices `[0,1,2]`, zero shots, raw prompts, harness seeds
   `0,1234,1234,1234` and server seed `0`.
   A task override must supply `dataset_kwargs.revision`; the upstream YAML
   does not pin it. The pinned task API passes those kwargs to `load_dataset`.
   Save processed documents, their hashes, prompts and sample indices.
3. The pinned CLI supports `lm-eval run`, `--include_path`, `--samples` as a
   JSON task-to-index mapping, and `--log_samples`. `--samples` and `--limit`
   are incompatible. Use the explicit indices above, separate fresh output
   directories and captured HTTP/settings/errors; call this a plumbing smoke.
4. Only after that successful harness smoke, audit a second exact local
   artifact and run it sequentially under the same conditions. Partial/failed
   results are not reusable. Persistence and any diagnostic comparison wait
   for real completed results; no SQLite index or quality aggregate yet.

Local audit scripts, upstream source snapshots, commands, raw exchanges and
the failed initial assertion are retained in ignored
`.codex-night/quality-eval-audit-20261002/`. Reproduce the offline regression:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python \
  .codex-night/quality-eval-audit-20261002/check_backend_offline.py
```

`server-command.json` records the exact bounded endpoint-probe launch. Copy the
audit directory to a fresh scratch directory before replaying file-writing probes.
`verify_artifact.py` and `probe_server.py` reproduce it, with approved socket/GPU
access required. They do not constitute an evaluator runner. WSL bash was
already the active shell here, so these commands ran directly inside WSL.

## Implemented single-artifact smoke — 2026-10-02

The pinned evaluator container and bounded host runner are now implemented in
[`pilot/quality_eval/`](../pilot/quality_eval/) and
[`scripts/quality_eval_smoke.py`](../scripts/quality_eval_smoke.py). They accept
only the audited TinyLlama SHA256 and llama-server binary above. Unsupported
task/mode/settings fail rather than falling through to the GGUF backend's
ignored kwargs. No production ranking, task suitability, HFA or speed path
uses this pilot.

- Base image: Python 3.12.12 slim/bookworm, amd64 manifest
  `sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301`.
  Harness archive SHA256:
  `010af308dae14a178fcf5e86c16fb984515cb29a7d0f0c37606faab88617f8ea`.
  Dependencies are version/hash-locked separately from Jaull; the evaluator
  has no torch/Transformers/GPU model backend. The image build checks actual
  harness/YAML imports and the fixed task configuration before model launch.
- The effective network boundary is a host CUDA server on WSL loopback plus
  a CPU evaluator container on Docker Desktop's bridge network, accessing
  `http://host.docker.internal:18083`. Container host networking with
  `127.0.0.1` was accepted but did not reach this WSL listener. No Desktop
  settings, host permissions or user services were changed.
- The container mounts the GGUF read-only and independently hashes it. The
  host server reads the existing host file; this does not protect it from
  concurrent host writers. Jaull full verification runs before and after
  serving. Each invocation requires a fresh output directory and records
  image identity, commands, hardware/readiness, server properties, effective
  evaluator config, runtime logs, HTTP payloads/responses and errors.
- Suite `jaull-quality-smoke-v1` still uses the pinned upstream HellaSwag
  preprocessing, validation indices `[0,1,2]`, zero shots, raw prompts and the
  fixed seeds above. The proposed Hub loader was replaced by the native
  Parquet loader: the Hub loader's metadata expected train/test even when
  only validation was requested. The runner fetches the exact validation
  file at the recorded dataset revision, verifies its LFS/content SHA256
  `899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd`,
  saves it, and supplies that verified local file to the task API. Split
  verification stays enabled. Only this 6315951-byte dataset file is needed;
  preprocessing covers its 10042 rows, while inference evaluates three.

Reproduce inside WSL, with approved Docker/socket/GPU access:

```bash
docker build --platform linux/amd64 --tag jaull-quality-smoke:ad8737a pilot/quality_eval
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m scripts.quality_eval_smoke \
  --artifact-json .codex-night/quality-eval-audit-20261002/verified-artifact.json \
  --llama-server /home/ton/tools/llama.cpp/build-cuda/bin/llama-server \
  --image jaull-quality-smoke:ad8737a \
  --output .codex-night/quality-eval-smoke-next-attempt
```

The artifact JSON names a local file and known digest, not an authorization
to download. The runner re-verifies it; stale flags cannot certify the bytes.
The evaluator has a five-minute execution bound; readiness has a 30-second
bound. Cleanup targets only the runner's UUID container and child server.
Generation, rolling/perplexity, chat templates, custom tasks/settings, multiple
slots and model downloads are outside this runner's accepted scope.

### Actual plumbing result

Attempt `quality-eval-smoke-20261002-04` completed using local image digest
`sha256:17b83ecff40f7c0e2d55d18122334200f490fe9eb6368be4c5800cee49410350`.
Saved raw results/samples identify exactly `[0,1,2]`; all 146 scoring requests
passed the token-ID, one-token, context, runtime-fingerprint, uncached-prompt
and finite-logprob checks. All 161 captured HTTP exchanges completed without
a recorded protocol error. The host artifact digest matched after execution,
the server exited with code 0 and its evaluator container was removed.

The harness's three-example metrics are raw **plumbing diagnostics**:
`acc=0/3`, `acc_norm=1/3`, with stderr calculation disabled. They are not a
full benchmark or publishable model-quality claim, and say nothing about
overall capability, suitability, newer models or local inference speed.
The result bundle is ignored under `.codex-night/`; nothing was uploaded.

Attempts 01–03 remain intact: import failure (fixed and caught by the image
installation check), loopback namespace connection failure (replaced with
the working Desktop host route), and unused-split metadata failure (replaced
with verified Parquet loading). None is comparable quality evidence.

Next iteration: fully identify/verify and check readiness for one already-local
second artifact, then extend this fixed runner minimally for a sequential
three-example plumbing comparison. Persistence, reuse and SQLite remain
deferred until that scoped execution work is reviewed; there is no response
cache, quality aggregate or scoring integration.

## Fresh launch review — 2026-10-02

This iteration closes the existing single-artifact implementation for review.
The earlier diagnoses and handoff are historical, not launch instructions.
Plan: recheck prerequisites once, independently verify the saved smoke bundle,
reject any drift in the fixed task definition before dataset loading/inference,
add an offline regression, run the Python 3.12 gates and stop this turn.
The second artifact and persistence are separate later iterations.

Current sandbox probes deny Docker-engine and GPU access, while Compose works.
Approved read-only probes confirm Docker engine 29.8.1/Linux, the recorded
evaluator image, RTX 2060/617.14, and llama-server build 10357 exposing CUDA0.
They do not prove fresh container GPU inference. No model server was started
during this review; the evaluator's accepted boundary remains the previously
observed host CUDA server plus CPU evaluator container.

Independent inspection of `quality-eval-smoke-20261002-04` verified the
three raw samples, all 161 HTTP exchanges (146 scoring requests), recorded
suite digest, dataset file digest, backend source digest and server binary
digest. Jaull full artifact verification rehashed the current TinyLlama file
and matched both saved pre/post digests. This is a review of an existing
plumbing result, not a new evaluation or reusable quality evidence.

The task preflight previously checked only a subset of allowed values:
changed prompt/choice/target templates, split selection, preprocessing or
metrics could retain the suite name. This iteration requires the complete
fixed definition, including the pinned upstream preprocessing function.
Future task changes require an explicitly reviewed suite/version change;
accepted CLI settings alone still do not establish an effective protocol.

The offline regression reproduced this gap before the fix. A disposable
network-disabled CPU container then loaded the actual pinned harness YAML
and preprocessing function with the updated evaluator source mounted
read-only: the fixed task passed, changed prompts/metrics failed, and the
owned container was removed. No dataset or model inference ran in that check.
The existing image digest predates this guard; rebuild with the documented
`docker build` command before the next evaluator run and record its new ID.

## Sequential two-artifact step — 2026-10-02

Plan for this iteration: allow only the two audited exact artifact identities,
check Jaull's unchanged memory estimate/full-device fit and runtime readiness,
reuse the already-verified validation Parquet bytes via a read-only mount,
rebuild the pinned evaluator, and run the two three-example smokes sequentially.
A fresh TinyLlama run supplies the control under the same rebuilt image as
Qwen; the earlier completed smoke and its failures remain intact. Keep fresh
output directories, HTTP exchanges, full raw samples/results and errors.
Persistence/reuse and a diagnostic comparison are subsequent scoped steps.

The second file is already local: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`,
`qwen2.5-1.5b-instruct-q5_k_m.gguf`, Q5_K_M, 1285494304 bytes,
SHA256 `b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c`.
Jaull full verification matched the current bytes to the sidecar and the
cached Hub download metadata, whose revision is
`91cad51170dc346986eccefdc2dd33a9da36ead9`. This is cached provenance,
not a new remote publisher query. The local tensor table identifies qwen2,
28 transformer blocks and 339 tensors. This does not change launch policy:
the pilot requests full offload explicitly, subject to fit/readiness gates.
The detailed audit lives in ignored `quality-eval-second-audit-20261002/`.

Both artifacts use raw prompts without their conversational templates,
context 2048, one slot, zero shots and the same fixed scoring protocol.
Different tokenizers are part of the exact artifacts; token counts may differ.
Only three fixed HellaSwag samples are evaluated. No full benchmark, general
model-quality, task-suitability or speed claim follows from this step.

### Reproduction and actual results

The runner now requires `--dataset-file`: use the previously verified local
validation Parquet file. The host checks its SHA256 before model launch;
the evaluator receives it read-only, verifies it independently, and preserves
a copy in each fresh output bundle. Dataset/preprocessing caching is separate
from response caching; each run still has its own HF cache, and lm-eval's
response/request caches remain OFF. No dataset or model download occurred.
The memory preflight reads the actual local GGUF header, merges its configuration
through Jaull, and uses the unchanged estimator/reserve/margin policies. It
requires a GPU-resident fit and one GPU with confirmed runtime readiness.
Capacity estimates remain capacity estimates; no runtime allocation comparison
or inference-speed measurement is derived from them.

Run these inside WSL, with approved Docker/GPU/socket access. Wait for each
runner to exit successfully before issuing the next command. Every output path
must be new; these example paths are intentionally different from saved runs.

```bash
docker build --platform linux/amd64 --tag jaull-quality-smoke:ad8737a-pair pilot/quality_eval
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m scripts.quality_eval_smoke \
  --artifact-json .codex-night/quality-eval-audit-20261002/verified-artifact.json \
  --dataset-file .codex-night/quality-eval-smoke-20261002-04/dataset-validation.parquet \
  --llama-server /home/ton/tools/llama.cpp/build-cuda/bin/llama-server \
  --image jaull-quality-smoke:ad8737a-pair \
  --output .codex-night/quality-eval-pair-next-tinyllama
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m scripts.quality_eval_smoke \
  --artifact-json .codex-night/quality-eval-second-audit-20261002/verified-artifact.json \
  --dataset-file .codex-night/quality-eval-smoke-20261002-04/dataset-validation.parquet \
  --llama-server /home/ton/tools/llama.cpp/build-cuda/bin/llama-server \
  --image jaull-quality-smoke:ad8737a-pair \
  --output .codex-night/quality-eval-pair-next-qwen
```

Both actual runs used image digest
`sha256:c2a7b178f1867ea568664239cd2c49f9a0f56148079898ac1c4550243926e81a`
and host llama-server build 10357. The completed ignored bundles are
`quality-eval-pair-20261002-tinyllama` and `quality-eval-pair-20261002-qwen-02`.
Both passed fit/readiness, completed exactly sample IDs `[0,1,2]`, and matched
artifact SHA256 after execution. Both servers exited 0; owned containers were
removed. Logs report GPU offload in both runs. This is host CUDA inference,
not a claim that a CUDA model ran inside Docker.

- TinyLlama: 161 HTTP exchanges, 146 scoring requests; raw three-sample
  diagnostics `acc=0/3`, `acc_norm=1/3`.
- Qwen: 144 HTTP exchanges, 129 scoring requests; raw three-sample
  diagnostics `acc=2/3`, `acc_norm=3/3`.

All 275 scoring requests passed the effective protocol guards. A separate
read-only review checked equal image/config/packages, suite/dataset/sample IDs,
documents, request text, prompt/target hashes and server commands except model
path. Token counts differ because the artifact tokenizers differ. These are
comparable **plumbing conditions for the two exact representations**, with
different model weights/quantization, not a full benchmark or evidence that
one base model is generally better. The limited results are not reusable
quality evidence and remain separate from suitability, ranking and speed.

The first network-disabled rebuild missed its dependency cache and failed
because pip could not fetch dependencies with networking disabled. The standard
build reused the pinned cached layers and succeeded. The first Qwen attempt
failed before server launch at the port check; that output/error is intact.
The socket state at that failure was not captured. A regression reproduced
the original guard's TIME_WAIT failure and verified the corrected SO_REUSEADDR
probe still rejects an active listener. The corrected Qwen attempt succeeded.
All failures/logs and the independent pair check are retained under
`.codex-night/`; none became a successful or reusable result.

Stop this iteration for review. Next: the smallest immutable full-bundle
persistence contract, explicitly retaining the limited/plumbing classification
and excluding it from quality reuse. A SQLite index remains unnecessary.

## Immutable persistence step — 2026-10-02

Plan: snapshot each already-completed bundle into a separate, exclusively
created JSON record. Preserve full harness results/samples, HTTP exchanges,
logs and provenance; keep the verified dataset in the original bundle with
its digest. Build an exact artifact/protocol identity from recorded evidence,
separate hardware provenance from that identity, and fail closed on missing,
failed, partial or unknown evidence. Quality lookup must reject these limited
smokes. Exercise a same-identity hit and changed/incomplete identity misses
offline with an explicitly synthetic full-result fixture. No SQLite, response
cache, new inference runs or production-store integration is needed.

`pilot.quality_eval.records` is a stdlib-only offline snapshot/import module.
It reads the completed evidence and rejects error files, unsuccessful status,
inconsistent artifact verification, unsupported HTTP/tokenizer payloads,
missing scoring responses and incomplete samples. The record contains the full
harness result and every original JSON/log/HTTP text file. The Parquet bytes
remain in the source bundle; the snapshot verifies them and records their
SHA256/size. HF preprocessing caches are not duplicated.

Schema 1 has an exact identity and checksum, full raw result, completed status,
classification, and separate provenance. The identity includes artifact SHA256;
suite name/digest; dataset repository, revision, file digest, selected IDs and
split size; per-sample doc/prompt/target hashes and rendered request arguments;
few-shot/chat/prompt/cache policy and seeds; context; evaluator commit, backend
digest, Python/package versions; runtime binary digest/fingerprint, evaluator
image ID, backend flags and server-defaults digest; and the observed tokenizer
and scoring settings. The target logit bias is the audited per-token scoring
mechanism, not an ordinary generation policy. Explicit `None` means known
absence only for the supported no-system-instruction/no-generation/no-cache
fields. Missing/unknown identity fields fail validation. Hardware is recorded
for provenance outside the identity; hardware changes cannot turn speed into
quality evidence.

Writes use exclusive creation and never replace an existing record. A checksum
detects changed content; it is not an authenticity signature. Truncated,
failed, partial, mismatched, unknown or limited records are quality-lookup
misses. Lookup also requires the complete split and rejects this smoke suite
even if someone relabels it as full. The same-identity hit is exercised only
with a clearly synthetic full fixture; no real full benchmark or runner for
one was introduced. The helper returns raw harness data, never a new quality
aggregate, recommendation score or performance calibration.

These offline commands run inside the existing WSL environment and need no
Docker engine, network or GPU access. Use new output filenames:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.records \
  --bundle .codex-night/quality-eval-pair-20261002-tinyllama \
  --output .codex-night/quality-eval-tinyllama-next-record.json
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.records \
  --bundle .codex-night/quality-eval-pair-20261002-qwen-02 \
  --output .codex-night/quality-eval-qwen-next-record.json
```

Actual snapshots are in ignored `quality-eval-records-20261002/tinyllama.json`
and `quality-eval-records-20261002/qwen.json`. Both retain `plumbing`
classification and return a quality-lookup miss. Read-back verified the full
record/checksum, duplicate-write rejection and byte-identical original evidence
files including the dataset. No evaluation was rerun. Existing failures and
experimental/benchmark records were preserved. A SQLite index adds no value
for these two records and was not added.

Stop this iteration for review. Next: the smallest diagnostic per-task view
over these immutable snapshots, with explicit comparability reasons and the
three-example limitation. It must remain outside recommendation scoring.

## Diagnostic comparison step — 2026-10-02

Plan: reuse the immutable record validation/checksums, compare every identity
field except the intended artifact SHA256 difference, and explain matched or
mismatched suite/dataset/sample/prompt/evaluator/runtime/protocol conditions.
Emit a small offline per-task JSON view, with exact artifact and record
provenance, saved metrics and sample counts. Withhold a comparison on mismatched
conditions; label these completed limited runs as plumbing only. Add focused
offline mismatch/corruption regressions, retain source bytes, then independently
audit the whole pending pilot and run all final gates. No new model evaluation,
quality aggregate, scoring integration or parallelism experiment is needed.

`python -m pilot.quality_eval.compare` reads two snapshots without modifying
them. It reuses full-result validation and verifies each record checksum, then
compares canonical digests of suite, dataset, samples/rendered prompts,
evaluator settings/versions, runtime flags/defaults and effective HTTP protocol.
Classification must also agree. The intentional artifact SHA256 difference
and hardware provenance do not make otherwise equal conditions incomparable.
The output keeps exact identities, hardware and source-record paths/checksums.
It contains no evaluation-duration or inference-speed metric.

A valid identity mismatch produces `NOT_COMPARABLE`, names the differing
condition and withholds the metric comparison. Corrupt, incomplete or unknown
records fail before output creation. Matched conditions produce
`COMPARABLE_PLUMBING` for limited/smoke results, even if relabeled as full;
otherwise the label is `COMPARABLE_DIAGNOSTIC`. Saved per-task metrics are
checked against their completed sample counts. No overall aggregate, ordering,
winner, ranking integration or quality-cache reuse is introduced.

The actual view is ignored `.codex-night/QUALITY_EVAL_COMPARISON_20261002_FINAL.json`.
It reports `COMPARABLE_PLUMBING`: all seven condition checks match. Both
artifacts used the same fixed suite/dataset revision/file, sample IDs `[0,1,2]`,
rendered prompts, zero shots, raw template policy, context 2048, seeds,
evaluator image/packages and observed one-slot scoring protocol. The exact
weights, quantizations and tokenizers differ as recorded in the snapshots.

| Task / saved metric | TinyLlama 1.1B Chat Q4_K_M | Qwen2.5 1.5B Instruct Q5_K_M |
|---|---:|---:|
| HellaSwag smoke v1 / acc | 0/3 (0.000000) | 2/3 (0.666667) |
| HellaSwag smoke v1 / acc_norm | 1/3 (0.333333) | 3/3 (1.000000) |

These are **three-example plumbing diagnostics**, not a full benchmark,
publishable/reusable quality evidence, general base-model verdict or task
suitability assessment. Rounded table values are for display; the view and
original records retain the harness values. No evaluation was rerun, response
cache enabled, container/server started, or memory/slot-scaling experiment run.

Reproduce the read-only comparison inside WSL, choosing a new output filename:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.compare \
  --left .codex-night/quality-eval-records-20261002/tinyllama.json \
  --right .codex-night/quality-eval-records-20261002/qwen.json \
  --output .codex-night/quality-eval-comparison-next.json
```

The controlled plumbing pilot now has protocol/runtime evidence, sequential
exact-artifact results, immutable snapshots and this diagnostic view. A full
quality campaign and optional parallelism remain outside the completed scope.
Final audit/check results are recorded in the ignored progress handoff.

## Placement determinism probe — 2026-10-02

The pilot already includes placement flags in its comparison/reuse identity;
hardware identity remains provenance. This probe tests sensitivity to placement
on one machine, not whether a result reproduces across machines. CPU and CUDA
kernels can differ numerically, and partial offload runs both in one forward pass.

[`scripts/quality_eval_placement_replay.py`](../scripts/quality_eval_placement_replay.py)
replays the 146 captured `/v1/completions` requests of the completed TinyLlama
bundle against the same pinned `llama-server` build and the same verified GGUF,
changing only `--n-gpu-layers`, `--device` and `--threads`. Replaying captured
requests holds the harness, the dataset, the prompts and the forced target
tokens fixed by construction. The arms vary placement and one CPU control varies
thread count. Before any
arm runs, the regrouping of 146 single-token requests back into 12 continuations
and the reimplementation of both metrics are checked against the numbers the
harness itself wrote.

### Measured result

| Arm | `-ngl` | threads | Tokens equal to the bundle | max abs delta, token | max abs delta, continuation | acc | acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpu_full_a` | -1 | 4 | 146/146 | 0 | 0 | 0/3 | 1/3 |
| `gpu_full_b` | -1 | 4 | 146/146 | 0 | 0 | 0/3 | 1/3 |
| `gpu_partial` | 11 | 4 | 97/146 | 0.3223 | 0.8776 | 0/3 | 1/3 |
| `cpu_only` | 0 | 4 | 0/146 | 0.2916 | 0.8261 | 0/3 | 1/3 |
| `cpu_threads8` | 0 | 8 | 0/146 | 0.2916 | 0.8261 | 0/3 | 1/3 |

Deltas are against the bundle recorded hours earlier. `cpu_only` and
`cpu_threads8` have equal serialized logprob values on all 146 tokens.

Those deltas are on the raw scores. What decides an answer is the lead of the
chosen continuation over its best rival:

| Arm | doc 0 | doc 1 | doc 2 |
|---|---:|---:|---:|
| `gpu_full_a` / `gpu_full_b` | +0.1452 | +0.0255 | +0.3705 |
| `gpu_partial` | +0.1593 | +0.0214 | +0.3705 |
| `cpu_only` / `cpu_threads8` | +0.1552 | +0.0182 | +0.3710 |

Character-normalised lead of the top answer over its best rival, the quantity a
placement change has to drive through zero to flip an answer.

### What this settles

1. **Fixed placement reproduced these requests exactly.** Two fresh processes
   with identical flags reproduced all 146 serialized logprob values of the
   earlier bundle. No run-to-run variation was observed; this does not establish
   universal bitwise reproducibility.
2. **Placement is score-relevant, and not at the last bit.** Full CPU changes
   every single token logprob; a partial offload changes 49 of 146. The largest
   continuation-level disagreement is 0.88 nats, which is a different number,
   not rounding.
3. **The answers held in these three samples.** No per-sample answer changed in any
   arm. The tightest lead, doc 1, lost 29% of an already small margin and stayed
   positive; doc 0's lead grew. The measured changes partly cancel in the
   differences: a continuation sum
   moves by up to 0.8776 nats while the raw lead between two of them moves by up
   to 0.4186, and in character-normalised units the pair is 0.0313 against
   0.0141. That is roughly a factor of two, not an order of magnitude.
4. **Thread count agreed here.** 4 and 8 threads produced equal serialized logprobs
   on this artifact and build. That is one pair of values on one model, which is
   not grounds for dropping `--threads` from identity.

### Consequences for reuse

- **Placement stays in identity.** No change is needed to refuse the wrong
  comparison: `identity.runtime.backend_flags` already contains
  `--n-gpu-layers` and `--device`, so `compare.py` already reports
  `NOT_COMPARABLE` for a CPU result against a GPU result. The conservative
  default was right, and this is the evidence for keeping it.
- **Reuse requires matching placement flags under this protocol.** Different
  machine-specific execution plans may select different splits; this pilot
  refuses reuse across those splits, even though here they shared every answer.
  Whether the same placement reproduces on a *different* GPU is untouched by
  this probe: unverified, not refuted, and the thing worth measuring next.
- **The current comparison protocol requires matching placement flags.** The sequential pair
  above satisfies this: both ran `-ngl -1 --device CUDA0` on the same machine.
- None of these magnitudes is a threshold for anything. They describe one
  artifact on one build, and a logprob delta between two placements of the same
  model says nothing about the quality distance between two different models.

### Limits

One artifact, one llama.cpp build, one machine, one GPU, three samples. Whether
`-ngl -1` reproduces across different CUDA devices is the obvious next question
and needs a second machine; this probe does not bound it. The probe writes
`placement-replay.json` with `quality_evidence: false` and never produces a
result record, so nothing here can enter `quality_lookup`.

The first run of this probe decided answer stability by comparing aggregate `acc`
and `acc_norm`, which two samples flipping in opposite directions would leave
untouched. The gate now compares each answer, and reports the margin shift
alongside it, so a surviving answer that nearly crossed zero cannot read as a
stable one. Re-running under the corrected gate reproduced the same figures and
the same answers.

Reproduce inside WSL, with a new output directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python \
  -m scripts.quality_eval_placement_replay \
  --bundle .codex-night/quality-eval-pair-20261002-tinyllama \
  --llama-server /home/ton/tools/llama.cpp/build-cuda/bin/llama-server \
  --output .codex-night/placement-replay-next
```

The saved runs are the ignored bundles `placement-replay-20261002` (first
gate) and `placement-replay-20261002-v2` (corrected gate). Historical outputs
are not rewritten when diagnostic wording changes.

## Pilot closure — 2026-10-02

The shared record validator checks both aggregate metrics against per-sample
results, including when a record has a correctly recomputed checksum. Snapshot
import and loading retained HTTP evidence reconstruct every selected continuation
from recorded tokenization, prompt token IDs, target IDs and logprobs. Missing
tokenizations/scores, unexpected scoring duplicates, incorrect coverage and
scores inconsistent with the harness results are rejected. Tokenizer caching is
allowed; response caching remains disabled. The checksum detects corruption, not
authenticity, and this importer is limited to the pinned one-slot smoke protocol.

Both original real bundles pass the stricter read-only checks. Existing bundles
and records remain unchanged and keep their plumbing classification. No new GPU
evaluation, quality aggregate, ranking integration or SQLite index was needed.
These three examples are not evidence of general model quality.

## Fixed 100-example subset — completed on 2026-10-02

`--profile hellaswag100` selects the separate task `jaull-hellaswag-100-v1`.
It samples 100 unique indices uniformly without replacement from the pinned
10,042-row validation split with `random.Random(20261002)`, then sorts them.
Python is pinned to 3.12.12 in the evaluator; an offline regression freezes the
selection digest. Exact IDs are persisted in both dataset and evaluator identity
and must match the selected samples. They are not chosen based on model answers.
The underlying YAML definition, prompts, zero-shot policy, context 2048, cache
policy, llama-server pin and full CUDA placement are unchanged. The new task name,
sample IDs and rebuilt image ID distinguish this protocol from the old smoke.
`--profile smoke` remains the default and retains the original three IDs and
launch command.

Larger-subset records have `classification: limited`, are excluded from
`quality_lookup`, and compare as `COMPARABLE_LIMITED` only when all protocol
checks match. They report per-task accuracy/counts, not an overall model-quality
score. Relabelling a limited record as full cannot enable quality reuse. A
three-example smoke and the new subset are `NOT_COMPARABLE`; neither their
samples nor their suites should be combined. A hundred examples still leave
sampling uncertainty, cover only HellaSwag and do not establish general capability.
No confidence interval or significance test is added in this step.

Docker Desktop WSL integration was restored after the initial offline preparation.
Both artifacts completed sequentially on the local RTX 2060 using image ID
`sha256:088d6da9b6f10952aeb42cd46fff814971f762127fd0f12f1c1a6720d0052e55`.
The strict snapshot importer verified artifact checks before/after execution,
all selected sample IDs, HTTP scoring coverage and aggregate consistency.
The comparison reports `COMPARABLE_LIMITED`: suite, dataset, samples, evaluator,
runtime, effective protocol and classification all match.

| Exact local artifact | `acc` | `acc_norm` |
|---|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 36/100 | 38/100 |
| Qwen2.5-1.5B-Instruct Q5_K_M | 46/100 | 63/100 |

These are observed results for this fixed subset, not full HellaSwag scores,
statistical-significance claims or a general-quality ranking. Parameter counts,
quantizations and tokenizers differ; no isolated family or quantization effect
is established. Evaluation duration is not inference-speed evidence.

TinyLlama attempt `01` was interrupted by a PC restart. Its final JSON files
are empty and its log tails damaged; snapshot import rejects it. It remains
intact and excluded. The valid repeat is attempt `02`; Qwen completed attempt
`01`. Both successful snapshots retain the complete raw evidence. No failed
result was repaired, reused or merged with the completed subset.

Build a new image tag rather than replacing the historical evaluator image;
using an old image with the new profile will fail instead of silently scoring
three samples. Run the models sequentially with fresh output directories:

```bash
docker build --platform linux/amd64 --tag jaull-quality-eval:hellaswag100-v1 pilot/quality_eval
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m scripts.quality_eval_smoke \
  --profile hellaswag100 \
  --artifact-json .codex-night/quality-eval-pair-20261002-tinyllama/verified-artifact.json \
  --dataset-file .codex-night/quality-eval-pair-20261002-tinyllama/dataset-validation.parquet \
  --llama-server "$HOME/tools/llama.cpp/build-cuda/bin/llama-server" \
  --image jaull-quality-eval:hellaswag100-v1 \
  --output .codex-night/hellaswag100-v1-tinyllama-02
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m scripts.quality_eval_smoke \
  --profile hellaswag100 \
  --artifact-json .codex-night/quality-eval-pair-20261002-qwen-02/verified-artifact.json \
  --dataset-file .codex-night/quality-eval-pair-20261002-tinyllama/dataset-validation.parquet \
  --llama-server "$HOME/tools/llama.cpp/build-cuda/bin/llama-server" \
  --image jaull-quality-eval:hellaswag100-v1 \
  --output .codex-night/hellaswag100-v1-qwen-01
```

Each evaluator has a ten-minute wall-time limit, with the original five-minute
limit retained for the smoke. Timeout/error evidence remains a failure; do not
reuse it or silently drop difficult examples to complete the subset. Artifact
verification, fit/readiness gates, process ownership and cleanup remain in place.
After both runs complete, the existing snapshot and comparison tools work offline:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.records \
  --bundle .codex-night/hellaswag100-v1-tinyllama-02 \
  --output .codex-night/hellaswag100-v1-tinyllama-record-02.json
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.records \
  --bundle .codex-night/hellaswag100-v1-qwen-01 \
  --output .codex-night/hellaswag100-v1-qwen-record-01.json
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m pilot.quality_eval.compare \
  --left .codex-night/hellaswag100-v1-tinyllama-record-02.json \
  --right .codex-night/hellaswag100-v1-qwen-record-01.json \
  --output .codex-night/hellaswag100-v1-comparison-01.json
```

All paths above are local ignored evidence. Historical bundles/records are never
rewritten. `uv run jaull` still does not build or launch evaluator containers.
The listed successful output paths now exist; use new attempt suffixes when
repeating the commands rather than overwriting them.

## Local throughput companion — 2026-10-03

The same exact two GGUF SHA256 identities were benchmarked sequentially with
the existing Jaull llama-bench command builder, runner, parser and BenchmarkStore.
No production code or quality/ranking integration was added. A local ignored
helper pinned four threads and requested JSONL on stderr alongside the Markdown
table, retaining all five individual measurements. The observation records the
actual executed argv, including these explicit supplementary flags.

Hardware: RTX 2060, 6 GiB, WSL, NVIDIA driver 617.14. Runtime build:
`689e227db (10357)`. Full CUDA0 offload, four threads, batch 2048, ubatch 512,
f16 K/V caches, default warmup enabled, five repetitions per test. Raw JSONL
confirms those settings, build, device, token counts and repetition coverage.
Artifact verification before/after each run passed, as did fit/readiness gates.
The common command is:

```bash
"$HOME/tools/llama.cpp/build-cuda/bin/llama-bench" -m "$VERIFIED_GGUF" \
  -dev CUDA0 -ngl -1 -p 512 -n 128 -r 5 -t 4 -b 2048 -ub 512 \
  -ctk f16 -ctv f16 -o md -oe jsonl --progress
```

`VERIFIED_GGUF` is the already-local path of each verified artifact, not a Hub
download. Complete commands, binary SHA256, hardware snapshots, individual
samples, raw logs and immutable benchmark records are retained under
`.codex-night/quality-speed-pair-20261003-02/` (locally ignored).

| Exact local artifact | pp512 tok/s (mean +/- SD) | tg128 tok/s (mean +/- SD) |
|---|---:|---:|
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 5937.53 +/- 654.15 | 190.73 +/- 5.67 |
| Qwen2.5-1.5B-Instruct Q5_K_M | 4487.96 +/- 394.41 | 123.94 +/- 1.63 |

These are two separate synthetic-token throughput tests, not a combined
512-token conversation followed by a 128-token answer. No `--ctx-size` was
applied and no conversational TTFT, end-to-end latency or service capacity was
measured. Different tokenizers mean equal token counts do not imply equal text.
The desktop GPU was not isolated: pre-run utilization was 23% and 28%, so
background contention and run-order effects are not excluded. The +/- values
are sample standard deviations over five repetitions, not confidence intervals.

Keep these performance observations separate from the 100-example quality
results. They show this TinyLlama artifact had higher observed throughput in
these runs; they do not prove that an older generation is faster or establish a
universal model-speed ordering. Jaull's existing benchmark aggregate requires
the same artifact and therefore intentionally rejects this pair; this table is
a descriptive cross-artifact side-by-side view, not that equivalence verdict.

Attempt `01` completed TinyLlama but the local helper failed while constructing
a relative summary path, before starting Qwen. Its evidence is preserved and
excluded from this table. Attempt `02` repeated the complete pair with the
corrected path handling; both records reload identically from BenchmarkStore.

## Diagnostic TUI integration — 2026-10-03

Results and Paths now read the existing local quality store on their evidence
worker. Matching requires the published LFS content SHA256 of a single-file
GGUF, not its family, repo name, quantization label or git blob ID. The same
bytes mirrored under another repository can match; another digest, multipart
GGUF or a missing digest cannot. Metadata is not local artifact verification:
the existing download/verification and execution checks are unchanged.

The display labels results as **historical artifact results; current
execution/protocol not verified**, and shows the suite, metrics, grade
(`plumbing`, `limited` or `full`), sample counts, recorded context and placement.
The tooltip retains record identity, dataset revision, evaluator/runtime,
hardware, timestamp and limitations. An artifact match does not establish
comparability with the selected plan or cross-machine reproducibility.

Quality does not change ranking, scores, action availability, or the
Ready/Validated/Benchmarked states. Strict `quality_lookup` still requires the
full protocol identity and complete results; displaying a limited record does
not make it reusable. Old records and cached analyses load with an unknown SHA;
they simply show no matching quality until refreshed through normal inspection.
No automatic metadata refetch, evaluator launch or response cache is added.

Import a completed pilot snapshot explicitly, from the repository checkout:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -c \
  'import sys; from pathlib import Path; from pilot.quality_eval.records import load_record; from jaull.evaluation.quality_storage import QualityEvidenceStore; print(QualityEvidenceStore().save(load_record(Path(sys.argv[1]))))' \
  /path/to/completed-quality-record.json
```

This applies the producer's stronger HTTP-coverage validation before storing
the immutable record in Jaull's user data directory (`quality/`). It does not
evaluate or alter the original snapshot. Imported records must be trusted local
evidence: checksums detect corruption, not authenticity. Jaull does not scan
ignored pilot directories or import them automatically. Without imported
records, `uv run jaull` has no historical quality to display and still starts
no evaluator containers. This integration ran no new GPU evaluations.

For deterministic manual inspection, use **Advanced tools → Estimate model
memory**, enter the exact GGUF repo, press **Detect**, select the evaluated
quantization, then **Estimate**. This screen displays the same historical
quality evidence by content SHA256; the repo need not appear in the shortlist.
The results view separates **Measured evaluation** (benchmark, sample count,
grade, accuracy metrics and recorded context) from the current memory estimate.
Limitations remain visible; suite, placement, hashes and source provenance are
available in the collapsed **Evaluation provenance** block. These benchmark
results are not presented as a general capability assessment. **Adjust
parameters** returns to the form without discarding the selected settings.
**Detect** explicitly refreshes repository metadata, replacing an older cache
entry that may lack content digests. Normal cached inspection is unchanged.
For the pilot Qwen artifact use `Qwen/Qwen2.5-1.5B-Instruct-GGUF` and `Q5_K_M`.
The recorded quality context remains 2048, even if the memory estimate uses a
different context. This is a historical view, not a new evaluation or a
statement that the current settings match. Unknown/multipart digests and
unimported records produce no quality display.

## Phase C: Explicit CLI execution

`jaull quality run` now joins the existing runner, HTTP-validated snapshot and
Jaull quality store. This is an explicit, opt-in CLI operation, not a TUI
background job. No ranking, fit formula, benchmark or record schema changes.

The adapter intentionally requires a **trusted source checkout** containing
`scripts/quality_eval_smoke.py` and `pilot/quality_eval/`. These do not ship in
the wheel. `--pilot-root` defaults to the current directory; set it explicitly
when invoking an installed Jaull elsewhere. It executes that checkout's Python
code, so do not point it at an untrusted directory. The existing pilot remains
the only owner of model/dataset/runtime pins, fit/readiness, HTTP checks and
server/container cleanup. There is no second evaluator implementation.

Prerequisites: Linux/WSL, a responding Docker engine, the already-built pinned
evaluator image, the audited host `llama-server`, the pinned local validation
Parquet and exact verified local GGUFs. Nothing is downloaded, built,
pulled or substituted by this command. See the build command above for the
one-time image setup. Phase C initially admitted only the two audited artifacts;
Phase D below expands that boundary. New server builds and custom task suites
remain unsupported. `smoke` is three-example plumbing; `hellaswag100` is limited
evidence, not a full benchmark or general-quality verdict.

From the checkout, use existing `ModelArtifact` JSON files (which contain
`local_path`, exact revision/filename/quantization and SHA256):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 jaull quality run \
  --pilot-root . \
  --artifact /path/to/tinyllama-artifact.json \
  --artifact /path/to/qwen-artifact.json \
  --dataset-file /path/to/validation.parquet \
  --llama-server "$HOME/tools/llama.cpp/build-cuda/bin/llama-server" \
  --image jaull-quality-eval:gguf-v1 \
  --profile hellaswag100 \
  --output .codex-night/quality-cli-new-run \
  --json
```

Each artifact runs **sequentially**, one server/container at a time. Inputs are
never rewritten. `run-01/`, `run-02/`, etc. retain the normalized artifact
manifest, raw bundle, runner/snapshot logs and immutable snapshot. Existing run
directories are refused, never overwritten. Completed records are automatically
saved in Jaull's existing user-data `quality/` store; the response returns their
identity hashes. A failed run/snapshot is not stored, and stops later runs. If
an earlier run completed, its record remains saved and is listed in the failure
response. Ctrl+C lets the pilot finish its owned-process cleanup before the
CLI exits. No response cache or automatic result reuse is enabled.
The store still accepts only one immutable record per evaluation identity:
repeating an identical protocol does not overwrite an earlier record. If the
new content differs (including timestamps/provenance), import fails and the
new snapshot remains in its run directory for review.

Compare two returned identity hashes without Docker or a GPU:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 jaull quality compare \
  LEFT_ID RIGHT_ID --json
```

The pilot comparison command and this CLI share the same per-task comparison
function. Matching suite, dataset, samples/prompts, evaluator, runtime,
effective protocol and grade produce diagnostic accuracy/counts; a mismatch
produces `NOT_COMPARABLE` with reasons and no paired metrics. Placement flags
remain part of runtime identity. Hardware is provenance, not inference-speed
evidence. These commands neither infer an overall winner nor compare local
tok/s; use Jaull's existing performance benchmark records for that axis.

This CLI integration was validated with offline synthetic fixtures, including
producer failure, corrupted/misattributed snapshots, cancellation, sequential
partial failure and incompatible protocols. The new comparison CLI also read
the two existing real 100-example records and returned `COMPARABLE_LIMITED`
with the unchanged Qwen 46/63 and TinyLlama 36/38 counts. That was a read-only
check, not a rerun.

The human subsequently completed a real `quality run --profile smoke` for the
already-local TinyLlama artifact in `.codex-night/phase-c-smoke-01/run-01/`.
The producer's HTTP validator successfully reloaded its snapshot, which equals
the automatically stored record
`5c7bd0771334d1ba7fb0784b837214dc8af25b9f0a6c791c00d743c8da4abc14`.
The bundle reports `plumbing_passed`, server exit code 0 and artifact-after SHA
verification. Sample IDs are 0, 1 and 2; observed acc is 0/3 and acc_norm 1/3.
This confirms the CLI-to-pilot-to-store path with real execution. It remains a
three-example plumbing check, not full/reusable quality evidence or a
general-quality assessment. Raw outputs remain locally ignored.

## Phase D: Exact local GGUFs, fixed evaluation protocol

The previous two-digest execution whitelist is removed. The existing
`ModelArtifact` manifest is still the sole artifact representation: callers
must supply repo, immutable 40-hex revision (not `main`), relative GGUF filename,
quantization label, positive size, lowercase SHA256 and absolute local path. The runner
reuses ArtifactService's full sidecar/size/disk verification before and after
execution; the container independently hashes its read-only model mount.
Names and family membership never substitute for a content digest. Caller
coordinates are provenance, not a verified publisher endorsement; matching a
SHA establishes byte identity, not authenticity of an arbitrary manifest.

The preflight derives the existing memory estimate from the local GGUF header
without Hub access. Multipart filename conventions and split metadata are
rejected, including a shard renamed to look like a single file. Missing KV/
configuration metadata, insufficient memory or an unknown fit cannot satisfy
the existing full-device gate. Metadata parsing retains its 32 MiB ceiling;
larger/incomplete headers fail closed. Runtime support is not promised for
every architecture: server startup, tokenizer boundaries and every HTTP
scoring response must pass the existing checks before a record is produced.

This is **not** general runtime/task support. It retains the exact audited
llama-server binary, CUDA full offload on one GPU, ctx2048, one slot, raw
zero-shot multiple-choice HellaSwag prompts without chat templates, and the
same three/100 sample selections. No ranking, score, HFA formula, automatic
downloads, scheduling or TUI changes are introduced.

Rebuild the evaluator explicitly under a **new tag**, preserving the old image:

```bash
docker build --platform linux/amd64 \
  --tag jaull-quality-eval:gguf-v1 pilot/quality_eval
```

Use this tag with the Phase C CLI commands. The host requires image label
`io.jaull.quality.artifact-contract=exact-local-gguf-v1` before starting the
server/container; unlabeled older images fail preflight with a rebuild message.
The label declares interface compatibility, not image authenticity; only use
images built from the trusted checkout. The resolved image ID remains part of
record identity, so a rebuilt image is not automatically comparable with an
old one. Existing snapshots/records and their schema are unchanged and remain
readable; historical bundles and image pins are not rewritten.

Phase D is validated with synthetic offline manifests, new-artifact HTTP
snapshot checks, historical fixtures and pre-launch failure regressions.
No new image build or GPU evaluation is claimed in this implementation step.
A real new-artifact smoke remains required before calling generalization
empirically demonstrated. The next separate block is TUI orchestration.

## Phase E: Opt-in TUI execution

The same bounded pilot can now be started from **Advanced tools → Evaluate local
GGUF quality**, **Paths → Evaluation → Evaluate quality**, or the **Results →
Evaluation** action for the selected single-file GGUF. A non-GGUF primary path
offers **Choose GGUF to evaluate**, opening Paths on its Evaluation tab instead
of evaluating the Transformers artifact. Opening a screen starts no evaluation,
download, image build or pull. Performance remains the separate **Benchmark** action; no automatic
priority-based campaign or ranking integration is introduced.

From Results or Paths, the selected GGUF is prepared automatically: no artifact JSON
is requested. Jaull resolves `main` to an immutable commit, checks the exact
selected filename/quantization and known metadata, then fully verifies the local
bytes before creating a temporary `ModelArtifact` manifest. Published SHA256
and positive size are required; a sidecar alone cannot establish revision
provenance. Unknown identities or conflicting metadata fail closed. The runner
retains its own copy of the manifest with the output.

The GGUF download permission checkbox defaults to **off** on every screen.
If missing, the artifact is downloaded only after **Start evaluation** and with
that permission enabled; the selected size (or unknown size) is shown first.
Opening the screen does not resolve remote metadata or download anything.
Cancellation during a Hub download waits for the existing blocking downloader
to return; it prevents evaluation/import afterwards, but is not immediate
download interruption. Streaming download cancellation remains out of scope.

The shared evaluator inputs still need a one-time setup: pinned validation
Parquet, pinned host llama-server, trusted pilot checkout and already-built
evaluator image. A sibling llama-server of the locally discovered llama-cli is
suggested when available, not certified; the existing binary pin still applies.
After successful setup or TUI evaluation these four inputs are saved atomically in
the per-user `quality-setup.json` and prefilled next time. The setup section
collapses when saved paths exist; invalid/unreadable settings show an error.
Artifact, output, profile and download consent are never remembered. Dataset
preparation now has its own permission (see Guided Setup below). No image
build/pull, runtime installation or automatic cache reuse is introduced.
Advanced tools retains the manual manifest input for standalone runs; existing
CLI run arguments remain unchanged.

Output defaults to a unique directory under Jaull's user-data `quality-runs/`, not the
repository. Select **Smoke - 3 examples** or **Limited - 100 examples**, then
press **Start evaluation**. The runtime/dataset/fit/image/HTTP checks remain in
the existing pilot, not duplicated in the screen.
Validation/progress/error messages stay visible beside the fixed actions, outside
the scrolling form. Missing inputs are named and the first empty input receives
focus. A responding Docker engine alone is not sufficient to start evaluation.

The prepared manifest must match the selected repository, filename,
quantization and any published size/digest and fixed revision. A mutable `main`
does not establish revision equality: the manifest must still declare its own
immutable revision, enforced by the pilot. This is an artifact selection, not an
assertion that the selected plan's runtime settings were evaluated. Evaluation
continues to use the fixed full-CUDA-offload, ctx-2048, raw zero-shot protocol,
visibly separate from the selected execution plan. Other runtimes, multipart
GGUFs, unsupported settings and unavailable inputs do not acquire evidence.

The worker uses `AdvisorService.run_quality_evaluation_for_plan` for automatic
artifact preparation (or `run_quality_evaluation` for standalone manifests),
and shows the shared record projection with its grade, coverage and limitations.
Only a completed,
HTTP-validated snapshot is imported. Back/quit during execution requests SIGINT
from the pilot owner and waits for its cleanup before leaving; unmount also
requests cancellation. No cancelled record is imported. Logs and partial output
are retained. Returning to Paths after a saved result refreshes its historical
evidence view without changing recommendation order. Returning to Results also
refreshes saved evidence. Results shows evaluations of known alternative GGUF
paths separately, explicitly not as evidence for the selected artifact.
These alternatives come from the recommendation or a previously visited Paths
screen; Results does not start another Hub scan to discover them. Exact artifact
digest matching still applies, and no GGUF result transfers to safetensors.

The form and cancellation paths are tested offline using synthetic records and
a real CPU-only signal-handling child process. No new Docker/GPU success is
claimed by this integration. On 2026-10-04, `docker image inspect` in this WSL
session reported that Docker integration was unavailable, so the human's rebuilt
image and a real TUI smoke still require verification when Docker is accessible.
Later the same day, Docker image inspection succeeded and confirmed
`io.jaull.quality.artifact-contract=exact-local-gguf-v1` on image
`sha256:366e365fa1f1003ee8afaadf12fde8fb20f66b36c47f6daa12bf45df9102b09d`.
This verifies image availability/contract, not a new GPU evaluation; a real TUI
smoke is still pending.

## Phase F: Guided Setup

Results now opens evaluation and refreshes saved results, but execution still
depends on the trusted source checkout. The wheel reads quality records; it
does not yet ship the pilot runner or container build. Remembering setup paths
does not remove that boundary.

The TUI's fixed **Prepare evaluator** action and `jaull quality setup` now use
the same infrastructure checks as the host pilot. They verify the executable
llama-server SHA256, inspect the already-local image/contract and verify the
dataset. Missing or incompatible infrastructure blocks a selected-model
download before it starts. Execution still separately checks CUDA readiness,
full-offload fit and the HTTP protocol; setup does not certify those.

For a source install, defaults derive the trusted checkout from the installed
module location, not CWD or PATH. The default dataset is
`quality-datasets/hellaswag-v1/validation.parquet` under Jaull's user-data
directory. Existing explicitly saved paths take precedence; no overnight
directories are searched. Wheel installs must still supply a trusted checkout.

The dataset permission defaults **off**, independently of GGUF download consent.
Enable it and press **Prepare evaluator** (or explicitly Start). The shared
preparation downloads only the existing pinned HellaSwag validation URL, verifies
SHA256 and atomically publishes the temporary file. Interrupted, oversized or
wrong-digest downloads leave no usable dataset. Corrupt existing files fail
closed rather than being silently overwritten. The fixed download is bounded
at 64 MiB, with a 30-second socket timeout and a 120-second between-chunk deadline.
Cancellation uses the existing owned-child SIGINT path and removes temporary
bytes; a network read may take up to its socket timeout to return.

Successful setup remembers only the shared paths/image. It runs no model,
creates no quality record and never remembers consent. Logs live in user-data
`quality-setup-runs/`; evaluation bundles keep their separate output directories.
The preflight is rechecked at evaluation start, including the runner's defensive
checks, rather than treating remembered preferences as proof of readiness.

```bash
uv run jaull quality setup --allow-dataset-download --json
```

The CLI uses remembered/local defaults and accepts `--llama-server`, `--image`,
`--dataset-file` and `--pilot-root` overrides. No automatic runtime installation,
Docker image build/pull or model evaluation occurs in setup. The evaluator
container/protocol pins are unchanged; this host-side change needs no rebuild.

Validation: synthetic offline regressions cover consent, verified local reuse,
corrupt/interrupted/oversized downloads, timeout/cancellation cleanup,
incompatible runtime/image and malformed preflight reports. The real
`jaull quality setup` command also passed using the existing local dataset,
audited llama-server and Docker image, with isolated settings under `/tmp`.
This was a read-only infrastructure probe: no network dataset download, model
execution, quality score or GPU measurement is claimed by that check.

Packaged execution without a checkout remains deferred; guided setup does not
make that boundary disappear.

## Phase G: Search Candidate Queue

**Results -> Evaluate candidates** prepares a reviewable selection from the
search's already-inspected candidates, before the final five-result truncation.
It reuses the existing recommendation eligibility, priority and diversity rules
with a wider output limit; it never rewrites the search state or scores. This is
an evaluation selection heuristic, not a claim that these are the best models.
It uses the facade's existing ranking context, including local performance and
execution records, rather than discarding those signals for candidate selection.
Discovery recall is still bounded by the original queries and inspection budget.

Selection inspects paths for up to six logical models (the existing variant
inspection budget), selects up to three, and explains omissions and untouched
models. Only confirmed single-file GGUF identities are eligible for this queue.
If no candidates qualify, Start is disabled and any selection reasons are expanded.
Published resolution pins the revision and checks digest/size against exact
artifact metadata. The existing estimator must confirm fit for the requested
workload AND full-device fit for the separate ctx-2048 evaluation protocol.
This queue currently requires one CUDA GPU; CPU/offload-only and multi-device
evaluation remain unsupported, without removing those model recommendations.
The runner still rechecks local bytes, hardware, fit and effective HTTP settings.

Preparing the selection reads Hub metadata and, where needed, bounded GGUF
header ranges. It never downloads a complete model or launches a container.
The form shows file sizes, local availability, reasons
and historical record counts. Candidates may be deselected; model and dataset
download permissions remain separate and off by default. **Start evaluation**
explicitly runs the chosen candidates sequentially through the existing pilot.
Each candidate has a separate model-NN output folder and each completed result
is stored before advancing. A domain failure does not stop later candidates;
cancellation cleans up the current runner and leaves earlier records intact.
Occupied output roots fail before preparation; no previous output is overwritten.

Results refreshes exact-artifact diagnostics after returning. Evidence for an
alternative GGUF is labelled separately from a Transformers primary path.
Historical records are displayed but NOT reused to skip this queue: the current
smoke/100-example profiles are non-reusable, and an artifact SHA alone does not
establish a matching execution/evaluation identity. No response cache is enabled.

This step supplies selection and execution, not general-quality ranking. The
HellaSwag profiles remain plumbing/limited diagnostics. Broader tasks, statistical
comparison, strict full-identity reuse and evidence-based recommendation ordering
are separate follow-ups. There are no new dependencies, persisted record schema
changes, hardware campaigns or HFA/ranking/performance formula changes.

## Published References And Artifact Measurements

The **Evaluation** tabs in Results and Paths keep two independent sections:

- **Published model references** lists catalog entries for an exact repository
  confirmed by repository identity or declared base-model metadata. Name/suffix
  heuristics cannot establish that relationship. Each entry keeps its evaluated
  variant, reasoning mode, source kind/URL and unknown revision/precision visible;
  complete protocol fields, notes and catalog version/digest are in provenance.
  This is a bibliography, not a claim that the current plan reproduces the result.
  Qwen3 non-thinking references can therefore be displayed without relaxing the
  strict `attach_capability_evidence` subject matching or inheriting a GGUF score.
- **Measured on this artifact** continues to use exact GGUF content hashes and
  displays sample coverage, classification and historical protocol limitations.
  Published results never fill this section or replace local measurements.

Missing/invalid catalogs have visible diagnostics; absent entries are not poor
quality. No comparison winner, aggregate quality score or ranking bonus is added.
Changing the selected path updates both sections without rewriting any record.
The explicit evaluation action remains available when there is no measurement.
Known unsupported formats, multipart/missing-file GGUF paths and machines with
other than one GPU disable direct evaluation with a visible reason. Choosing a
GGUF path from a Transformers recommendation remains available through Paths.
Opening setup is not an execution-readiness guarantee: the existing pilot still
checks the pinned server/image, dataset, CUDA and full-device fit at ctx 2048
before running. It does not use llama-cli readiness to judge its pinned server.
