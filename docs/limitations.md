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
- **Per-layer offloading is not modelled.** `offloading_required` means "would need to spill
  to RAM", not "load N layers on the GPU".
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
- VRAM prediction vs observation is always reported as `methodologically_unavailable`: the
  estimate does not retain VRAM attributed to the executed PID and configuration.
- RAM comparison is only available for CPU-only or non-offloaded configurations, because
  Jaull does not yet keep a host/device breakdown under offload.
- Benchmark comparison deliberately produces no single winner score, and warns instead of
  ranking when records come from different machines or methodologies.

## Interface

- The TUI runs on Windows, Linux and WSL, but glyph rendering (borders, shading, colours)
  depends on the terminal — Windows Terminal, WezTerm and iTerm2 give the best result.
