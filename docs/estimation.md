# Memory estimation

How `jaull estimate` and the guided recommendation turn public metadata into a memory
figure, and what each number is allowed to claim. No weights are downloaded and no
inference is run anywhere in this path.

Every component carries a **source** (`exact`, `metadata`, `derived`, `assumed`, `unknown`)
and the overall assessment carries a **confidence** (`high`, `medium`, `low`, `unknown`)
reduced by the weakest link in the chain. Numbers are never presented with more precision
than their provenance supports.

---

## The four components

### 1. Weights

Highest-confidence source first:

- **GGUF** — the exact size of the selected variant, taken from the file size `HfApi`
  reports. No download.
- **Transformers with safetensors metadata** — `total_parameters × bytes_per_parameter(dtype)`.
  Parameter counts come from `HfApi.get_safetensors_metadata`, which parses headers rather
  than downloading weights.
- **Transformers without metadata** — the sum of `.safetensors` / `pytorch_model*.bin`
  sizes, scaled by the dtype ratio if a different dtype was requested.
- Otherwise unknown, with a warning.

`bytes_per_parameter`: `float32=4`, `float16=2`, `bfloat16=2`, `int8=1`, `int4=0.5`. Real
quantized formats add scales and block metadata (typically under 10%), so that path is
tagged `DERIVED` to make the approximation visible.

When a supported dense transformer config exposes all required dimensions and explicitly
states whether input embeddings and the output head are tied, `WeightEstimate` also carries a
`transformer_block_decomposition`. Jaull first estimates the parameter split:

```text
transformer-block parameters = blocks × (attention parameters + gated-FFN parameters)
non-block parameters         = token embeddings + separate output head (when untied)
```

It projects that parameter fraction onto the artifact's total weight bytes. The block aggregate
is rounded down once and the non-block aggregate receives the exact remainder, so the two always
sum back to the artifact byte count. The displayed per-block estimate rounds the block aggregate
up over the transformer-block count.

This is an **estimate, not tensor-level measurement**. GGUF can quantize tensor classes
differently and carries alignment and format metadata, so parameter fraction and byte fraction
need not be identical. Unknown tying, incomplete configs, unsupported architectures and MoE
models use the explicit `uniform_weight_fallback`; a missing block count produces no
decomposition.

The decomposition is currently informational. Hardware Fit still uses its existing conservative
`total_weight_bytes / total_transformer_blocks` placement cost because Jaull has not yet defined
where non-block weights belong. Consequently this addition does not change GPU/RAM budgets,
placement modes or the selected transformer-block boundary.

### 2. KV cache

```text
kv_bytes = 2 * num_layers * num_kv_heads * head_dim * context * batch * concurrent_users
           * bytes_per_kv_element
```

`num_kv_heads` falls back to `num_attention_heads` for MHA models. `head_dim` derives from
`hidden_size // num_attention_heads` when the config does not state it. If `sliding_window`
is present and the context exceeds it, the effective context is capped.

Concurrency is modelled here rather than as a scoring nudge: each concurrent session
multiplies the cache, on the stated assumption that every session keeps its own full
context.

Non-standard architectures (MoE, MLA / DeepSeek-style, multimodal composites, `auto_map`
custom code) return `unknown` and a warning instead of a fabricated number.

### 3. Runtime overhead

Allocator, kernel buffers, activations and small caches:

```text
overhead = max(min_overhead, base_overhead + weight_fraction * weights)
```

Constants live in `estimator/policies.py` (`base = 512 MiB`, `fraction = 10%`,
`min = 256 MiB`). Always tagged `ASSUMED` / `LOW` confidence.

### 4. Device reserve and safety margin

User-controllable via `--device-reserve-gib` and `--safety-margin-percent`, kept as
distinct components so each one can be inspected — and so a policy figure is never mistaken
for a measured one.

---

## Compatibility

The total is compared against local RAM/VRAM:

| Status | Meaning |
|---|---|
| `comfortable` | ≤ 75 % of available |
| `compatible` | 75–90 % |
| `tight` | 90–100 % |
| `offloading_required` | Fits in RAM + VRAM combined but not in VRAM alone |
| `insufficient` | Exceeds combined RAM + VRAM |
| `unknown` | Missing inputs |

With `--device auto` the estimator prefers GPU, falls back to `offloading_required` if the
model fits combined memory, then CPU, then `insufficient`.

VRAM comes from NVML, so this comparison is NVIDIA-only. On other vendors the accelerator
is still detected and its backends probed, but the memory model falls back to system RAM.

## Where it fits, not just whether it fits

A single verdict hides the question that actually matters on a mixed machine: *which memory
would this run out of?* `estimator/hardware_fit.py` answers that separately, as a placement:

| Mode | Meaning |
|---|---|
| `GPU_RESIDENT` | The whole model lives in VRAM |
| `GPU_OFFLOAD` | Part in VRAM, part in system RAM |
| `CPU_RAM` | System RAM only, no GPU |
| `TOO_LARGE` | Does not fit anywhere |

`analyze_components()` places each component of the estimate rather than the total, and
records how the placement was decided (`HardwareFitPlacementMethod`) so a figure derived from
estimated bytes is never confused with one read off an artifact. The result is a
`HardwareFitResult`, which travels with the candidate from inspection all the way to the
recommendation.

When the placement is block-aware, the result uses `gpu_transformer_blocks` and
`total_transformer_blocks`. Those are runtime-agnostic planning units, not
llama.cpp `--n-gpu-layers` values.

### The KV cache follows the blocks

A block's KV entries live wherever that block runs, so the cache is split the way the blocks
were split:

```text
gpu_kv_cache_bytes = ceil(kv_cache_bytes × gpu_transformer_blocks / total_transformer_blocks)
ram_kv_cache_bytes = kv_cache_bytes − gpu_kv_cache_bytes
```

Rounding matches the overhead and margin splits — the GPU share rounds up, RAM takes the
remainder — so the scarcer pool is never understated and `gpu + ram == kv_cache_bytes` holds
exactly, with no byte created or lost.

The split reaches both budgets, not just the GPU one:

```text
gpu_required_bytes = gpu weights + gpu KV + reserve + gpu overhead + gpu margin
ram_required_bytes = ram weights + ram KV + ram overhead + ram margin
```

and it also sets the bases the overhead and margin heuristics are weighted on, so the two
padding terms describe the placement actually chosen rather than one where VRAM carried the
whole cache.

Charging the entire cache to VRAM — which the analyzer did before — overstated every partial
offload and left RAM with no cache at all. Measurements against llama.cpp on a partial offload
show the cache distributed across both pools in proportion to the blocks, which is what this
models. This is the runtime-agnostic default: a backend able to override where the cache
lives can refine it in its own adapter.

Two cases have no ratio to follow, and both keep the previous conservative answer: the
byte-estimated fallback has no block count, so the whole cache stays charged to VRAM; and
unified memory has one pool, so it is charged there.

For partial GPU offload, `offload_diagnostics` records the selected transformer-block
boundary and the first higher block count that exceeded the estimated VRAM budget. This is
capacity-planning evidence, not a measured CUDA allocation and not a runtime mapping.
`search_ceiling_transformer_blocks` is the coarse upper bound the analyzer considered before
the loop; if it equals the selected block count, `first_rejected_higher = null` means no
higher partial candidate was evaluated. Each candidate reports non-negative
`headroom_bytes = max(0, available_vram_bytes - gpu_required_bytes)` and
`excess_bytes = max(0, gpu_required_bytes - available_vram_bytes)`.

The point of keeping this separate from the compatibility status is that **RAM and VRAM are
never treated as one pool**. A 7B model that fits in 32 GiB of system RAM and not in 8 GiB of
VRAM is a real option with a real cost, not a failure — and saying so requires naming the
placement, not just the verdict.

The same analysis is what the guided shortlist approximates cheaply, before any inspection has
happened: see the placement hint in [recommendation.md](recommendation.md).

---

## Base model resolution and GGUF enrichment

GGUF repositories usually ship only quantized weights — no `config.json`, and therefore no
way to compute a KV cache from metadata alone. The estimator solves this in three layered
steps, each with its own provenance.

### 1. Base-model resolution

In priority order:

1. `card_data["base_model"]` in the GGUF repo's model card (string, single-element list, or
   a dict with `finetune` / `quantized_by` / …) → HIGH confidence.
2. `general.source.huggingface.repository` read from the GGUF header itself → HIGH
   confidence.
3. A `https://huggingface.co/...` URL found in a model-card field (`source`, `homepage`, …)
   → MEDIUM confidence.
4. Nothing → UNRESOLVED. The repository name (`X-GGUF` → `X`) is **never** used as an
   answer; it is only surfaced as evidence.

### 2. GGUF header read over HTTP Range

No full download: an initial 256 KiB range, doubling up to 8 MiB. The response is
*streamed* and iteration stops the moment the requested number of bytes has been read, so a
server that ignores `Range` and answers `200 OK` with the whole file still costs only the
prefix — the rest of the body is never pulled off the wire. That case is flagged with a
warning and the reader stops growing the range. Timeouts, 4xx/5xx, `416` and malformed
headers all degrade cleanly.

The consequence: if the header is not inside the first 8 MiB, enrichment gives up rather
than reading further.

### 3. Config merge

The GGUF header wins for the fields it declares — it is the artifact that will actually
run. The base config fills in the rest. Conflicts (a `context_length` mismatch, for
instance) are recorded as warnings and shown in the "Configuration source" row.

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

Falls back to weights-only estimates: GGUF file size, KV cache reported as unknown.

---

## Runtime recommendation

After computing the estimate, Jaull suggests a runtime and a starter command:

| Repository type | Primary runtime | Alternative |
|---|---|---|
| GGUF | `llama.cpp` (llama-server / llama-cli) | — |
| Transformers | `transformers` (Python snippet) | `vllm` when the architecture is on the shortlist and the model fits fully in VRAM |
| Others (diffusers / onnx / adapter / unknown) | — | — |

Every recommendation carries per-flag provenance:

- `--n-gpu-layers` for llama.cpp is computed as
  `min(block_count, (available_vram − device_reserve − 256 MiB − kv_cache) / (weights / block_count))`.
  When `block_count` is unknown it falls back to a documented conservative default (20
  layers) with a warning. The split assumes uniform layer sizes.
- `device_map` for Transformers is `"cuda"` when the model fits VRAM, `"auto"` for
  offloading (with an `accelerate` warning) and `"cpu"` otherwise.
- vLLM is only suggested when the architecture is on the shortlist (`llama`, `qwen2`,
  `mistral`, `phi3`, `gemma`, `gemma2`, `gemma3`) and the model fits GPU memory. The real
  vLLM support matrix is broader; the shortlist is deliberately narrow.
- If the memory assessment is `insufficient`, no runtime is recommended and the user is
  told so explicitly.

The commands shown by `estimate` and by the guided flow are **generated, not executed**.
To have Jaull actually run something, use `jaull run` or the execution paths in the TUI —
see [evidence.md](evidence.md).
