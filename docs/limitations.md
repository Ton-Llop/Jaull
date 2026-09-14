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
- Per-transformer-block offloading **is** modelled, but only as a placement,
  not as a measured runtime layer cost. `HardwareFitResult` reports a mode, a
  `gpu_transformer_blocks` / `total_transformer_blocks` split and a per-pool
  byte breakdown, and travels on `MemoryEstimate.hardware_fit`.
- `WeightEstimate.transformer_block_decomposition` now separates an estimated
  transformer-block aggregate from estimated non-block weights when a supported
  dense config is complete. The split is derived from parameter fractions, not
  GGUF tensor bytes; mixed tensor quantization, alignment and omitted small
  tensors can differ from it. Hardware Fit now uses GPU-heavy and RAM-heavy
  endpoints for the unknown non-block share, requiring both pool maxima to fit.
  These are conditional planning bounds, not measured tensor allocations, and
  can reject placements a particular backend could execute. The enclosing
  breakdown describes the GPU-heavy endpoint; optional bounds carry the host
  maximum. Do not add maxima from different endpoints. Duplication, staging and
  uneven tensor/block sizes remain outside this model. GPU/host physical point
  comparisons remain unavailable for an uncertain non-block split.
  `gpu_transformer_blocks` is a runtime-agnostic planning
  estimate, not a promise about what a backend will pass to `--n-gpu-layers`.
- The KV cache is placed **proportionally to the blocks**, which is the generic
  default rather than a guarantee. A runtime may keep the cache entirely in host
  memory, quantize it separately, or page it; none of that is modelled here, and
  a backend that can do so should refine the placement in its own adapter. The
  proportional rule was validated on one model, one build and one GPU.
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
- A guided run takes minutes, not seconds. Deep inspection covers up to 12 repositories,
  each costing a metadata round-trip and (for safetensors repos) a header read per shard.
  Inspection runs with a bounded concurrency of four; a cold run against the live Hub was
  observed at roughly 5–10 minutes on a home connection. Repeated analyses can use the
  persistent model-analysis cache, and repeated safetensors metadata lookups within one
  run are memoized, but the search can still be cancelled at any point.
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
- Preliminary llama.cpp layer selection still uses an aggregate block approximation.
  Prepared local launches can use actual GGUF tensor descriptor sizes, but only for the
  verified `689e227db` build, dense `qwen2` with separate input/output tensors and one
  confirmed discrete CUDA GPU. Unknown builds, layouts and tensor types fall back with
  a reason. This does not yet generalize to tied output weights, MoE, multi-GPU or other
  backends. See [local launch refinement](estimation.md#optional-local-gguf-launch-refinement).
- Tensor descriptors give stored bytes, not physical CUDA allocations. The local reader
  neither reads payloads nor rehashes the artifact; it relies on artifact preparation's
  verification, checks file bounds and detects size/mtime changes during inspection.
  A same-size replacement after verification is not ruled out by this reader alone.
  Runtime allocation, repacking, buffers and concurrent VRAM use can still cause a launch
  failure. The existing reserve/headroom remain heuristics, not an OOM guarantee.
- The vLLM shortlist is intentionally narrow. The real support matrix is broader; consult
  vLLM's documentation if your architecture is not listed.
- Peak RAM is sampled every 50 ms, so very short spikes may be underestimated.
- Downloads do not stream byte-level progress yet.

## Measurement and comparison

- Benchmarks measure single-process throughput. There is no load generator, no concurrency
  sweep, no capacity curve and no sustainable-concurrency estimate.
- VRAM prediction vs observation is compared only when the estimate carries a hardware fit
  with a device-specific prediction and the execution path used an unambiguous GPU setup.
  The predicted side is `gpu_physical_bytes` — weights, KV cache and runtime overhead —
  with the device reserve and safety margin removed, because those are capacity policy and
  no process allocates them. Partial llama.cpp offload is currently
  `methodologically_unavailable`: `HardwareFitResult` reports runtime-agnostic transformer
  blocks, while llama.cpp `--n-gpu-layers` uses backend-specific offload units.
  Transformers, which decides device placement internally and exposes no equivalent flag,
  is always reported that way.
- Runtime overhead stays inside the compared figure even though it is a coarse `ASSUMED`
  heuristic. It models allocations that really happen (allocator, compute and activation
  buffers), so its error is a calibration result rather than a methodological mismatch —
  but expect it to dominate the reported error until it is calibrated.
- **Process-attributed VRAM cannot be measured on a consumer GPU in WDDM mode**, which is
  every GeForce driving a display on Windows — and the same card seen from WSL2. NVML lists
  the running processes but reports `usedGpuMemory` as unavailable for all of them, so
  `ExecutionObservation.peak_vram_bytes` is `None` and the VRAM comparison has no measured
  side to compare against. Measured on this project's RTX 2060 (driver 616.92): from
  Windows, NVML returned 31 processes with `usedGpuMemory = None` for every one and
  `nvidia-smi --query-compute-apps` printed `[N/A]` in the memory column; from WSL2 the
  same query returned an empty list while a CUDA process held ~1.9 GiB. `nvidia-smi -q`
  reports `Driver Model: WDDM`. This is a platform limit, not a missing feature: it does
  not improve by switching between WSL and native Windows, and it will not improve on
  another consumer GPU operating under the same WDDM constraints. A datacenter GPU on Linux
  without a display attached does report per-process memory. Device-wide memory readings
  remain available, but they include every other consumer on the card and are not comparable
  with a planning budget.
- RAM comparison is only available for CPU-only or non-offloaded configurations. Under
  offload the obstacle is not a missing host/device split — `HardwareFitResult` carries
  `ram_weight_bytes`, `ram_kv_cache_bytes` and `ram_overhead_bytes` — but the measurement:
  llama.cpp maps the whole model file, and the pages it reads to upload weights to the
  device stay resident, so peak RSS tracks the artifact rather than the placement. The
  B001-R4 baseline measured 4723 MiB of peak RSS with 24 of 29 units offloaded and 4724 MiB
  with all of them, against a predicted host share of 1742 MiB and a 4466 MiB artifact. RSS
  did not move when the placement did, so comparing the two would report a ~171% error that
  describes the page cache, not the memory model.
- Benchmark comparison deliberately produces no single winner score, and warns instead of
  ranking when records come from different machines or methodologies.

## Interface

- The TUI runs on Windows, Linux and WSL, but glyph rendering (borders, shading, colours)
  depends on the terminal — Windows Terminal, WezTerm and iTerm2 give the best result.
- Two screens carry animated artwork: the home screen's sea, and the lane of water the
  shark patrols while the search runs. Both are decorative — nothing that reports a
  number depends on either, and both draw nothing at all when the layout leaves them no
  height.
- Both are drawn in quadrant blocks (U+2596–U+259F), which buy twice the horizontal
  resolution of a half-block but are less universally present in fonts than `▀` is. A
  font without them renders those bands as replacement boxes.
- Both also assume a 24-bit colour terminal: they are per-cell foreground/background
  gradients, and on a 256-colour terminal Rich will quantize them to something flatter.
- They animate for as long as their screen is open. The sea costs a few percent of one
  core at 110x32 and around ten at 200x40; the search lane is far cheaper, under one
  percent at 110x32 and about two on a 200-column terminal, and what it overlaps with is
  a search that spends its time waiting on the network. Every other screen is static.
