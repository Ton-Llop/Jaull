# jaull

`jaull` is an explainable local-AI capacity analyzer for inspecting Hugging Face models and estimating their hardware requirements. It reads your machine, reads a model's public metadata, and tells you whether the two fit — showing its work for every number.

The project is also being developed as part of an end-of-degree project (TFG) focused on corporate local-AI infrastructure. The inspection, estimation and guided recommendation paths intentionally avoid downloading model weights or running inference: every number they produce is derived from public metadata and documented heuristics. The separate `run` subcommand is explicit opt-in execution for already resolved GGUF artifacts through `llama-cli`.

## Interactive interface

The TUI opens on a choice between a guided analysis and the individual tools:

![Welcome screen with guided and advanced entry points](docs/assets/tui-welcome.svg)

Guided mode starts with a hardware scan: a real progress bar with a paced checklist tells you exactly what the tool is inspecting, and the detected profile appears **in place** when the scan finishes — no scrolling required.

![Hardware analysis while scanning](docs/assets/tui-hardware-loading.svg)

![Hardware analysis with the detected profile](docs/assets/tui-hardware-done.svg)

Then six plain-language questions — no model names, quantizations or dtypes:

![Requirements wizard](docs/assets/tui-wizard.svg)

Advanced tools keep the original screens, including the memory estimation view:

![jaull dashboard](docs/assets/tui-home.svg)

![Memory estimation view](docs/assets/tui-estimate.svg)

> Screenshots are regenerated with `uv run python scripts/capture_screenshots.py` (headless, no network).

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

Optional:

- An NVIDIA driver + NVML for GPU/VRAM detection (falls back cleanly if absent)
- `HF_TOKEN` in the environment for gated or private repositories

## Install

### Use it from the repository

```bash
git clone https://github.com/Ton-Llop/AI-checker.git
cd AI-checker
uv sync
uv run jaull ui
```

`uv sync` creates a virtual environment and installs the runtime dependencies plus the `dev` dependency group declared under `[dependency-groups]` in `pyproject.toml` — uv installs default groups automatically, so contributors need no extra flag. Pass `--no-dev` if you only want the runtime dependencies.

### Development

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

## Guided mode

The TUI opens on a choice between **Guided recommendation** and **Advanced tools**.

Guided mode answers the question the separate tools cannot: *what should I run on this machine?* You never need to know a model name, a quantization, a dtype, a KV cache or a runtime flag — six plain questions are enough, and every technical parameter is derived.

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
| Output | 1 recommendation + up to 2 alternatives, explained | Raw analysis / estimate for one model |
| Screens | Welcome → Hardware → Questions → Search → Results | Scan, Inspect, Estimate, Doctor |

### The questions the wizard asks

1. **What will you use it for?** — general chat, programming, documents, or writing and translation.
2. **What matters most?** — best quality, balanced, fast responses, or lowest memory usage.
3. **Which languages?** — multiple choice, plus free-text codes under "Other".
4. **How many people at once?** — one, 2–5, 6–20, or more than 20.
5. **How much text at a time?** — asked **only** for the documents use case. It sets the model's context window, which is *not* the size of a document collection: a retrieval system feeds the model a few chunks at a time.
6. **Must the model allow commercial use?** — yes / no / not sure, defaulting to yes.

Answers are normalised into a `UserRequirements` object (`workflow/requirements.py`) that records every assumption it made, and those assumptions appear in the exported report.

### How models are found

Search uses the official `HfApi.list_models` API — no HTML scraping, no weight downloads, and no token required for public models. `HF_TOKEN` is used when present for higher rate limits, and is never logged or exported.

The query builder issues several complementary queries per use case (task wording, one per preferred weight format, one per non-English language, plus a trending query) rather than relying on a single string. Results are then:

1. **Interleaved** round-robin across queries, so the format and language queries are not starved by the first one.
2. **Deduplicated** by `repo_id`, merging the query labels.
3. **Filtered** — private, gated, multimodal, wrong-pipeline, base-less adapters and (when commercial use is required) non-commercial licenses are rejected. Thin metadata is *never* a rejection; it becomes a recorded penalty and lower confidence.
4. **Shortlisted** down to the deep-inspection budget using cheap pre-inspection signals only.

Budgets are centralised in `workflow/policies.py`: 20 results per query, 40 unique candidates, 12 deep inspections, 3 recommendations.

Only the shortlist pays for inspection, which reuses the existing `inspect_model`, the analyzers, the base-model resolver and `estimate_memory` — the guided flow computes no memory figures of its own.

### How the ranking works

A composite score over eight normalised components, weighted and then renormalised to 1.0 (`recommendation/policies.py`):

| Component | Base weight | What it measures |
|---|---|---|
| Memory fit | 25 % | Does the model fit on this hardware for one user? |
| Concurrency fit | 10 % | Does it still fit at the requested concurrency? |
| Capability | 15 % | Family + parameter-count signal (Qwen2.5-7B > TinyLlama-1B) |
| Task match | 20 % | Repo tags, pipeline and keywords vs the use case |
| Language match | 12 % | Fraction of requested languages the model declares |
| License | 8 % | Commercial-use category from the license table |
| Metadata quality | 7 % | Confidence in every number, from card completeness |
| Popularity | 3 % | Log-scaled downloads + likes, capped as tie-breaker |

Priority shifts these — *quality* raises task match and capability, *speed* and *memory* raise both memory fit and concurrency fit. The total is then scaled by the estimate's confidence, and one more multiplier — the **hard-requirement penalty** — is applied on top: a candidate that requires commercial use but declares a restricted license is multiplied by 0, effectively removed; language misses and concurrency shortfalls apply softer penalties (0.15 and 0.35) that still let the candidate compete.

Memory fit is itself multiplied by an **artifact realism** factor: a real GGUF variant or an explicit `bnb-4bit` / `gptq` / `awq` tag scores 1.0, a Transformers repo loaded at its native dtype scores 0.75, and a theoretical `int4` / `int8` selection with no confirmed artifact drops to 0.4. This is what stops "Qwen-7B in int4" from beating a real Qwen-7B GGUF at the same nominal size.

Compatibility is still a hard gate: a model assessed `insufficient` never appears at all, and one assessed `unknown` can only ever be a flagged low-confidence alternative. The heading on the primary card reflects the resulting tier:

| Tier | Heading | Trigger |
|---|---|---|
| Best match | `BEST MATCH` | HIGH confidence + comfortable/compatible + no hard-fail |
| Recommended | `RECOMMENDED` | Tight fit or MEDIUM confidence |
| Closest option | `CLOSEST OPTION` | Offloading required, or status unknown |
| Best-effort suggestion | `BEST-EFFORT SUGGESTION` | LOW confidence, or any hard requirement missed |

Within a family (Qwen2.5, Llama 3.1, Gemma 2, ...), the recommender picks a single size that best matches the priority — the other sizes are shown as a **series ladder** underneath: *"Same series, other sizes: 0.5B · 1.5B · 3B · 7B"*, each with its own status.

Every reason and warning is generated by rules in `recommendation/explanations.py`. No language model is involved in ranking or in writing the explanations. Concurrency is now modelled directly in the estimate (each concurrent session multiplies the KV cache), not just applied as a scoring nudge.

Ties break deterministically (status → confidence → downloads → `repo_id`), so the same inputs always give the same report.

### Licenses

Licenses are bucketed conservatively into `commercial_allowed`, `commercial_restricted` and `unknown` from a small documented table. Custom vendor licenses (Llama, Qwen, DeepSeek, …) are deliberately classified as **unknown** rather than allowed: many do permit commercial use below a user threshold that this tool cannot verify.

> License information is reported from model metadata and is **not legal advice**. Check the model's license yourself before commercial use.

## Commands

Six subcommands: `scan`, `inspect`, `estimate`, `doctor`, `run` (classic CLI) and `ui` (interactive TUI). All are read-only except `run`, which explicitly downloads, verifies and executes a GGUF artifact.

```bash
uv run jaull                             # opens the TUI when run in a terminal
uv run jaull ui                          # interactive terminal UI
uv run jaull scan                        # local hardware
uv run jaull doctor                      # environment health
uv run jaull inspect <repo-id-or-url>    # analyze a model repository
uv run jaull estimate <repo-id-or-url>   # memory footprint + compatibility
uv run jaull run --model <repo> --prompt "Hello"  # download, verify and run a GGUF
```

Running `jaull` with no subcommand opens the TUI **only when stdin and stdout are both a terminal**. Piped, redirected or CI invocations keep printing help exactly as before, so existing scripts are unaffected.

The two front-ends share the same services and produce the same numbers:

- **CLI** — for automation, scripting and reproducible output. `--json` emits a stable schema you can pipe into other tools.
- **TUI** — a guided, visual flow for exploring models interactively. Every action shows the equivalent CLI command so you can lift it into a script.

### `scan` — local hardware

```bash
uv run jaull scan
```

Reports OS, architecture, CPU model + cores, RAM totals, storage per mount point and any NVIDIA GPU with VRAM/CUDA driver version.

### `inspect` — a Hugging Face model

```bash
uv run jaull inspect Qwen/Qwen2.5-7B-Instruct
uv run jaull inspect https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main
```

All URL variants (repo root, `/tree/…`, `/blob/…`, `/resolve/…`) normalise to the same `repo_id`. The command queries `HfApi.model_info(files_metadata=True)`, classifies the repository (`transformers` / `gguf` / `diffusers` / `onnx` / `adapter` / `unknown`), downloads only tiny metadata files (`config.json`, index JSONs, `adapter_config.json`, `model_index.json`) when useful and prints a Rich table.

For GGUF repos it also reports each quantization variant separately and the largest single variant, so `Repository size` (sum of all files) is not confused with `Largest variant` or with the runtime memory of one specific quantization.

### `estimate` — memory footprint + compatibility

```bash
uv run jaull estimate bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --quantization Q4_K_M --context 8192

uv run jaull estimate Qwen/Qwen2.5-7B-Instruct --dtype int8 --context 4096
uv run jaull estimate Qwen/Qwen2.5-7B-Instruct --dtype float16 --context 32768 --json
```

Produces an explainable breakdown:

```
Memory breakdown
Component         Size        Source     Explanation
Weights           4.92 GiB    exact      Exact remote file size of the Q4_K_M variant.
KV cache          1.00 GiB    derived    28 layers x 4 KV heads x 128 head_dim x 8192 tokens x batch 1 x 2 bytes x 2.
Runtime overhead  0.98 GiB    assumed    Base 0.50 GiB + 10% of weights (0.49 GiB).
Device reserve    0.50 GiB    assumed    Configured device reserve of 0.50 GiB.
Safety margin    0.74 GiB    assumed    10% of estimated subtotal.
Total required    8.14 GiB
```

Each row carries the *source* of the number (`exact`, `metadata`, `derived`, `assumed`, `unknown`) and the assessment carries a *confidence* (`high`, `medium`, `low`, `unknown`) reduced by the weakest link in the chain.

Options:

| Option | Default | Purpose |
|---|---|---|
| `--quantization` / `-q` | auto | GGUF variant to use (case-insensitive) |
| `--dtype` / `-d` | from config | Weight precision for transformers repos |
| `--context` / `-c` | `max_position_embeddings` or 4096 | Context length in tokens |
| `--batch-size` / `-b` | 1 | Sequences in parallel |
| `--device` | `auto` | `auto` / `gpu` / `cpu` |
| `--kv-dtype` | `float16` | KV cache element precision |
| `--safety-margin-percent` | 10 | Extra on top of subtotal |
| `--device-reserve-gib` | 0.5 | Memory to leave free on the target device |
| `--json` | off | Emit stable JSON to stdout (schema_version = 1) |
| `--no-resolve-base-model` | off | Skip base-model resolution and GGUF-header enrichment |
| `--no-runtime-recommendation` | off | Skip the runtime recommendation section |

### `run` — execute a GGUF with llama-cli

```bash
uv run jaull run \
    --model bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --quantization Q4_K_M \
    --prompt "Explain what local AI means" \
    --ctx-size 4096 \
    --n-gpu-layers 0
```

`run` is the only command that downloads weights and runs inference. It:

1. normalises the Hugging Face reference;
2. resolves a single-file GGUF variant;
3. downloads it into jaull's artifact storage when missing;
4. verifies file size and SHA-256 sidecar, with optional full re-hashing via `--full-verify`;
5. executes `llama-cli --single-turn` through the local host backend.

Options:

| Option | Default | Purpose |
|---|---|---|
| `--model` / `-m` | required | Hugging Face repo_id or URL |
| `--quantization` / `-q` | auto | GGUF variant to resolve |
| `--prompt` / `-p` | required | Prompt sent to `llama-cli` |
| `--revision` / `-r` | Hub default | Revision to resolve/download |
| `--llama-cli` | `PATH` lookup | Path to the `llama-cli` executable |
| `--ctx-size` | 4096 | Context size passed to `llama-cli` |
| `--n-gpu-layers` | 0 | Number of layers offloaded to GPU |
| `--timeout-seconds` | 300 | Execution timeout |
| `--full-verify` | off | Recompute SHA-256 before execution |

Current limitation: only single-file GGUF artifacts are supported. Multipart GGUF and Transformers execution are intentionally rejected in this phase.

### `doctor` — environment health

```bash
uv run jaull doctor
```

### `ui` — interactive terminal UI (Textual)

```bash
uv run jaull ui
```

Launches a full-screen Textual application on the **Welcome** screen, which offers *Start guided analysis* or *Advanced tools*.

The guided path runs **Hardware analysis → Requirements wizard → Model discovery → Recommendations**, with real progress checklists (a step turns green when its operation actually returns — there are no artificial delays) and a Cancel that stops the search safely. Results offer Compare, Technical details, Export report, Start again and Advanced tools.

Advanced tools keeps the original **Scan**, **Inspect**, **Estimate** and **Doctor** screens unchanged. Same services, same numbers as the classic CLI — the TUI is a thin renderer over `detect_hardware`, `inspect_model`, `estimate_memory` and `collect_diagnostics`. Every action in Estimate shows the equivalent CLI command so you can capture it for scripts. Global bindings: `h` welcome, `s` scan, `i` inspect, `e` estimate, `d` doctor, `esc` back, `q` quit.

All network calls and blocking probes run in worker threads, so the interface stays responsive while a search is in flight.

## Runtime recommendation

After computing the memory estimate, the tool suggests a concrete runtime and starter command:

| Repository type | Primary runtime | Alternative |
|---|---|---|
| GGUF | `llama.cpp` (llama-server / llama-cli) | — |
| Transformers | `transformers` (Python snippet) | `vllm` when the architecture is on the shortlist and the model fits fully in VRAM |
| Others (diffusers / onnx / adapter / unknown) | — | — |

Every recommendation carries per-flag provenance:

- `--n-gpu-layers` for llama.cpp is computed as `min(block_count, (available_vram − device_reserve − 256 MiB − kv_cache) / (weights / block_count))`. When `block_count` is unknown it falls back to a documented conservative default (20 layers) with a warning.
- `device_map` for Transformers is `"cuda"` when the model fits VRAM, `"auto"` for offloading (with an `accelerate` warning) and `"cpu"` otherwise.
- vLLM is only suggested when the architecture is in the shortlist (`llama`, `qwen2`, `mistral`, `phi3`, `gemma`, `gemma2`, `gemma3`) and the model fits GPU memory.
- If the memory assessment is `insufficient` no runtime is recommended — the user is told explicitly.

The commands shown by `estimate` and the guided flow are **generated**, not executed. Use them as a transparent starting point. If you want jaull to execute a model directly, use the explicit `run` command, which currently supports single-file GGUF artifacts through `llama-cli`.

## Base model resolution and GGUF metadata enrichment

GGUF repositories typically ship only the quantized weights — no `config.json`, no way to compute KV cache from metadata alone. The estimator solves this in three layered steps, all with per-source provenance:

1. **Base-model resolution.** Priority order:
   1. `card_data["base_model"]` in the GGUF repo's model card (string, list of 1, or dict with `finetune`/`quantized_by`/etc.) → HIGH confidence.
   2. `general.source.huggingface.repository` read from the GGUF header itself → HIGH confidence.
   3. `https://huggingface.co/...` URL found in a model-card field (`source`, `homepage`, …) → MEDIUM confidence.
   4. Nothing → UNRESOLVED. The name of the repo (`X-GGUF` → `X`) is **never** used as an answer — only surfaced as evidence.
2. **GGUF header read via HTTP Range.** No full download: initial 256 KiB range, doubling up to 8 MiB. The response is *streamed* and iteration stops the moment the requested number of bytes has been read, so a server that ignores `Range` and answers `200 OK` with the whole file still costs only the prefix — the rest of the body is never pulled off the wire. That case is flagged with a warning and the reader stops growing the range. Timeouts, 4xx/5xx, `416` and malformed headers all degrade cleanly.
3. **Config merge.** GGUF header wins for fields it declares (it is the artifact that will run). Base config fills the rest. Conflicts (`context_length` mismatch, for instance) are recorded as warnings and shown in the "Configuration source" row.

### Precedence policy

| Field | 1st | 2nd | 3rd | Fallback |
|---|---|---|---|---|
| `context_length` | GGUF header | base config `max_position_embeddings` | user `--context` | — |
| `num_hidden_layers` | GGUF `block_count` | base config | — | KV = unknown |
| `num_attention_heads` | GGUF `head_count` | base config | — | KV = unknown |
| `num_key_value_heads` | GGUF `head_count_kv` | base config | `num_attention_heads` (MHA) | — |
| `head_dim` | GGUF `rope_dim` | base config `head_dim` | `hidden // heads` | — |
| `sliding_window` | base config | — | — | no cap |
| architecture | GGUF `general.architecture` | base config `model_type` | — | warning |

### Turning enrichment off

```bash
uv run jaull estimate <repo> --no-resolve-base-model
```

Falls back to the previous behaviour: GGUF weights only, KV cache reported as unknown.

## How the estimator works

The estimator composes four independent, per-source-documented estimates:

1. **Weights** — highest confidence first:
   - GGUF: exact size of the selected variant (uses the file size that `HfApi` reports; no download).
   - Transformers with safetensors metadata: `total_parameters × bytes_per_parameter(dtype)`. Parameter counts come from `HfApi.get_safetensors_metadata` (header parsing, not weight download).
   - Transformers without metadata: sum of `.safetensors` / `pytorch_model*.bin` sizes, scaled by dtype ratio if a different dtype was requested.
   - Otherwise unknown, with a warning.

   `bytes_per_parameter` table: `float32=4`, `float16=2`, `bfloat16=2`, `int8=1`, `int4=0.5`. Real quantized formats add scales / block metadata (typically <10 %); the source is tagged `DERIVED` in that case so the user knows it's an approximation.

2. **KV cache**:
   ```
   kv_bytes = 2 * num_layers * num_kv_heads * head_dim * context * batch * bytes_per_kv_element
   ```
   `num_kv_heads` falls back to `num_attention_heads` for MHA models. `head_dim` derives from `hidden_size // num_attention_heads` when the config does not state it. If `sliding_window` is present and `context` exceeds it, the effective context is capped. Non-standard architectures (MoE, MLA / Deepseek-style, multimodal composites, `auto_map` custom code) return `unknown` and a warning — no fake precision.

3. **Runtime overhead** (allocator, kernel buffers, activations, allocator, small caches):
   ```
   overhead = max(min_overhead, base_overhead + weight_fraction * weights)
   ```
   Constants live in `estimator/policies.py` (`base = 512 MiB`, `fraction = 10%`, `min = 256 MiB`). Always tagged `ASSUMED` / `LOW` confidence.

4. **Device reserve + safety margin** — user-controllable via `--device-reserve-gib` and `--safety-margin-percent`. Kept as distinct components so the user can inspect each.

**Compatibility** compares the total against local RAM/VRAM:

| Status | Meaning |
|---|---|
| `comfortable` | ≤ 75 % of available |
| `compatible` | 75–90 % |
| `tight` | 90–100 % |
| `offloading_required` | Fits in RAM + VRAM combined but not in VRAM alone |
| `insufficient` | Exceeds combined RAM + VRAM |
| `unknown` | Missing inputs |

In `--device auto`, the estimator prefers GPU, falls back to `offloading_required` if it fits combined, then CPU, then `insufficient`.

## Architecture

```
src/jaull/
├── cli/             # thin Typer commands (no business logic)
├── advisor/         # application facade shared by CLI and TUI
├── artifacts/       # artifact resolution, download, storage and verification
├── workflow/        # guided-run state, requirements, orchestrator, DI container
├── discovery/       # Hub search, query builder, filtering, candidate enrichment
├── recommendation/  # configuration selection, scoring, ranking, explanations, report
├── domain/          # Pydantic models + enums (no Rich/Typer)
├── execution/       # host command execution contracts and backend
├── hardware/        # psutil + NVML probes
├── huggingface/     # HfApi wrapper, URL parser, classifier, orchestrator
├── analyzers/       # per-repository-type analyzers behind a Protocol
├── estimator/       # pure functions + policies + service orchestrator
├── metadata/        # GGUF header reader + Range client + base-model resolver + merger
├── runtime/         # per-runtime command/snippet builders + dispatcher
├── tui/             # Textual screens, widgets and styles (thin over the services above)
├── presentation/    # Rich reports + JSON emitter
└── exceptions.py    # domain errors mapped to friendly CLI messages
```

Boundaries kept clean:

- Domain models never import Rich, Typer or Textual.
- `workflow/`, `discovery/` and `recommendation/` are pure Python: no Textual, no Rich. The whole guided pipeline runs — and is tested — without a terminal.
- CLI commands and TUI screens never call `HfApi` directly — they go through the service functions and translate exceptions to messages.
- `run` uses the same artifact resolver/storage contracts, then executes only after verification.
- Guided screens receive their services from a `ServiceContainer` (`workflow/container.py`) instead of constructing `HfClient()` themselves, so tests drive the whole flow with fakes.
- Every external dependency (`HfApi`, NVML, HTTP Range client) hides behind a `Protocol` so tests can inject fakes.
- Every weight, threshold, budget and license rule lives in a `policies.py` module. No magic numbers elsewhere.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Tests use fake HTTP clients (`httpx.MockTransport`), fake NVML providers, fake artifact services, fake execution backends and programmatically-built GGUF header fixtures; the Hugging Face API, NVIDIA driver and `llama-cli` binary are never touched. The Textual UI is driven headless through `App.run_test()`. No GPU, no network and no TTY required — which is also what lets the same suite run unchanged in CI.

Packaging is verified separately, since a dropped non-Python resource (the Textual stylesheet) would not show up in the test suite:

```bash
uv build
uv run python scripts/check_dist.py
```

## Limitations (deliberate)

### Guided mode

- **No performance is measured.** The ranking is about whether a model *fits* and *matches your task*, never about how fast it will run. There are no tokens/second, no latency and no throughput figures anywhere, and the concurrency answer is a ranking signal plus a memory-headroom caveat — not a capacity model.
- **Text generation only.** Image, audio and speech models are out of scope for this iteration and are filtered out.
- **`document_qa` recommends a text model suitable for a future RAG system.** No embeddings, vector store or retrieval are implemented yet.
- **A guided run takes minutes, not seconds.** Deep inspection is 12 repositories, each costing a metadata round-trip and (for safetensors repos) a header read of every shard. A run against the live Hub was observed at roughly 5–10 minutes on a home connection. The search can be cancelled at any point.
- **Which 12 candidates get inspected is decided by a heuristic**, including a parameter count read from the repository name (`...-7B-Instruct`). That heuristic only orders the inspection queue — it never becomes a reported number. Once inspected, every figure comes from real file sizes and configuration metadata.
- **Model families are only merged on evidence.** A GGUF conversion is grouped with its original when the model card declares `base_model`; without that declaration both may appear separately, because guessing from similar names would silently merge genuinely different models.
- **Task matching is keyword-based.** It reads repository names and tags, so an unconventionally named model can be scored lower than it deserves.
- **Search quality depends on the Hub's own search.** Some queries return very little, and the ranking can only work with what came back.

### Overall

- `scan`, `inspect`, `estimate`, `doctor`, the TUI and guided recommendations do not download model weights or run inference. Only the explicit `run` command does.
- `run` is limited to single-file GGUF artifacts and local `llama-cli`; there is no Transformers/vLLM execution path yet.
- No benchmarks, no tokens/second.
- No hardware-purchase recommendations.
- Only NVIDIA GPUs (via NVML). AMD / Intel / Apple Silicon are out of scope for now.
- `diffusers` and `onnx` analyzers only list relevant files; sub-configs and opsets are not parsed.
- KV cache assumes classic MHA / GQA with `sliding_window` as the only refinement. MoE, MLA, multimodal composites → warning and confidence reduced to unknown.
- Runtime overhead and device reserve are heuristics — centralised in `estimator/policies.py` and explicitly marked `ASSUMED`.
- **Per-layer offloading is not modelled.** `offloading_required` means "would need to spill to RAM," not "load N layers on GPU."
- **No estimate substitutes a real benchmark.** Use `estimate` to filter candidates, not to accept or reject models blindly.
- **GGUF header reads are HTTP-Range based.** Only the first ≤ 8 MiB of the chosen variant is fetched. Because the response is read as a stream and abandoned once the budget is met, a server that ignores `Range` cannot make the tool download a multi-gigabyte file — but it does mean that if the header is not inside that prefix, enrichment gives up rather than reading further.
- **Gated base models** require `HF_TOKEN`; without it, enrichment degrades to GGUF-only.
- The TUI runs on Windows, Linux and WSL, but glyph rendering (borders, shading, colours) depends on the terminal — Windows Terminal / WezTerm / iTerm2 give the best result.
- **Runtime recommendations are generated, not executed by `estimate` or guided mode.** They assume a standard runtime install. The explicit `run` command executes only GGUF through `llama-cli`. The layer-per-GPU split for llama.cpp assumes uniform layer sizes — a documented approximation.
- **vLLM shortlist is intentionally narrow.** The real support matrix is broader; consult vLLM's docs if your architecture is not listed.

## Next step (out of scope for this iteration)

The next execution-focused step is to add benchmark feedback around the explicit `run` path: tokens/second, first-token latency, richer download progress and eventually remote execution. Those numbers should remain separate from the metadata-only estimator so the report stays clear about what was measured and what was inferred.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
