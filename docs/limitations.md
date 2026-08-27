# Limitations

What Jaull deliberately does not do, and where its numbers stop being trustworthy. Most of
these are design decisions rather than bugs; where something is simply not built yet, the
README roadmap says so.

## Scope

- `scan`, `inspect`, `estimate`, `doctor`, the TUI's analysis screens and guided
  recommendations **never** download model weights or run inference. Only `jaull run`,
  validation and benchmarking do, and only when explicitly asked.
- Text generation only. Image, audio and speech models are out of scope for now and are
  filtered out of discovery.
- No hardware-purchase recommendations.
- `document_qa` recommends a text model suitable for a future RAG system. No embeddings,
  vector store or retrieval are implemented.

## Hardware

- VRAM in the memory model comes from NVML, so memory-based compatibility is NVIDIA-only.
  AMD, Intel and Apple accelerators are detected and their backends probed, but their
  memory does not enter the estimate — those machines fall back to system RAM.
- Non-NVIDIA detection depends on `vulkaninfo`. Without it, only CPU and NVIDIA paths are
  visible.
- The Vulkan probe reports device identity and backend availability, not device memory.
- A software renderer (llvmpipe and similar, common under WSL) proves the Vulkan API is
  present, not that a usable accelerator exists. It is recorded as a software renderer and
  not selected as a backend.

## Estimation

- KV cache assumes classic MHA / GQA, with `sliding_window` as the only refinement. MoE,
  MLA and multimodal composites produce a warning and confidence `unknown`, not a number.
- Runtime overhead, device reserve and safety margin are heuristics, centralised in
  `estimator/policies.py` and explicitly tagged `ASSUMED`.
- Weight bytes for theoretical dtypes ignore scales and block metadata (typically under
  10% for real quantized formats); that path is tagged `DERIVED`, not `EXACT`.
- Per-layer offloading **is** modelled, but only as a placement, not as a measured layer
  cost. `HardwareFitResult` reports a mode, a `gpu_layers` / `total_layers` split and a
  per-pool byte breakdown, and travels on `MemoryEstimate.hardware_fit`. The split assumes
  every transformer layer weighs the same — token embeddings and the output head are not
  separated out — so `gpu_layers` is a plan, not a promise about what a runtime will do.
- **Hardware capacity is not the same thing as current memory occupancy**, and Jaull only
  models the second. The fit is computed against *available* VRAM and RAM at scan time, so
  the same machine answers differently depending on what else happens to be running.
  Measured against llama.cpp on an RTX 2060, NVML reported 4599 MiB free while llama.cpp
  saw 5095 MiB — a ~496 MiB gap large enough to flip a verdict near the boundary. Planning
  ("could this machine ever run this model?") and running now ("can it start this minute?")
  need different budgets; final estimation still answers the run-now question, while the
  discovery shortlist intentionally uses physical capacity for coarse preselection. Not yet
  resolved.
- GGUF header reads are HTTP-Range based and capped at 8 MiB. If the header is not inside
  that prefix, enrichment gives up rather than reading further.
- Gated base models require `HF_TOKEN`; without it, enrichment degrades to GGUF-only.
- `diffusers` and `onnx` analyzers only list relevant files; sub-configs and opsets are not
  parsed.
- **No estimate substitutes for a real benchmark.** Use `estimate` to filter candidates, not
  to accept or reject models blindly.

## Recommendation

- **No performance is predicted.** The ranking is about whether a model *fits* and *matches
  the task*, never how fast it will run. The concurrency answer is a memory multiplier plus
  a ranking signal — not a capacity model.
- The requirements wizard captures intent, not service objectives: there is no throughput,
  latency or TTFT target anywhere in the model.
- A guided run takes minutes, not seconds. Deep inspection covers 12 repositories, each
  costing a metadata round-trip and (for safetensors repos) a header read per shard. A run
  against the live Hub was observed at roughly 5–10 minutes on a home connection; the search
  can be cancelled at any point.
- Which candidates get deep-inspected is decided by a heuristic, including a parameter count
  read from the repository name (`...-7B-Instruct`). That heuristic only orders the
  inspection queue — it never becomes a reported number.
- Model families are only merged on evidence. A GGUF conversion is grouped with its original
  when the model card declares `base_model`; without that declaration both may appear
  separately, because guessing from similar names would silently merge genuinely different
  models.
- Task matching is keyword-based. It reads repository names and tags, so an unconventionally
  named model can score lower than it deserves.
- Search quality depends on the Hub's own search. Some queries return very little, and the
  ranking can only work with what came back.
- License classification is conservative and metadata-derived. It is not legal advice.

## Execution

- The CLI `run` path is limited to single-file GGUF artifacts through a local `llama-cli`.
  Multipart GGUF is rejected explicitly.
- Transformers execution exists only through the TUI, via an isolated Python worker.
- Runtime recommendations shown by `estimate` and by guided mode are **generated, not
  executed**, and assume a standard runtime install.
- The llama.cpp layer split assumes uniform layer sizes — a documented approximation.
- The vLLM shortlist is intentionally narrow. The real support matrix is broader; consult
  vLLM's documentation if your architecture is not listed.
- Peak RAM is sampled every 50 ms, so very short spikes may be underestimated.
- Downloads do not stream byte-level progress yet.

## Measurement and comparison

- Benchmarks measure single-process throughput. There is no load generator, no concurrency
  sweep, no capacity curve and no sustainable-concurrency estimate.
- VRAM prediction vs observation is compared only when the estimate carries a hardware fit
  that places weights on the GPU **and** the run can be shown to have used that placement
  (a llama.cpp `--n-gpu-layers` matching the predicted split). The predicted side is
  `gpu_physical_bytes` — weights, KV cache and runtime overhead — with the device reserve
  and safety margin removed, because those are capacity policy and no process allocates
  them. Everything else is still `methodologically_unavailable`, with the reason naming
  what was missing: Transformers, which decides device placement internally and exposes no
  equivalent flag, is always reported that way.
- Runtime overhead stays inside the compared figure even though it is a coarse `ASSUMED`
  heuristic. It models allocations that really happen (allocator, compute and activation
  buffers), so its error is a calibration result rather than a methodological mismatch —
  but expect it to dominate the reported error until it is calibrated.
- RAM comparison is only available for CPU-only or non-offloaded configurations, because
  Jaull does not yet keep a host/device breakdown under offload.
- Benchmark comparison deliberately produces no single winner score, and warns instead of
  ranking when records come from different machines or methodologies.

## Interface

- The TUI runs on Windows, Linux and WSL, but glyph rendering (borders, shading, colours)
  depends on the terminal — Windows Terminal, WezTerm and iTerm2 give the best result.
- The home screen's sea is drawn in quadrant blocks (U+2596–U+259F), which buy twice the
  horizontal resolution of a half-block but are less universally present in fonts than
  `▀` is. A font without them renders the band as replacement boxes. It is decorative and
  on the entry screen only, so nothing that reports a number depends on it.
- That band also assumes a 24-bit colour terminal: it is a per-cell foreground/background
  gradient, and on a 256-colour terminal Rich will quantize it to something flatter.
- It animates at roughly eight frames a second for as long as the home screen is open,
  which costs a few percent of one core at 110x32 and around ten at 200x40. Every other
  screen is static.
