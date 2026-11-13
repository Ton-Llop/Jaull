# Memory estimation

How `jaull estimate` and the guided recommendation turn public metadata into a memory
figure, and what each number is allowed to claim. No weights are downloaded and no
inference is run anywhere in this path.

Every component carries a **source** (`exact`, `metadata`, `derived`, `assumed`, `unknown`)
and the overall assessment carries a **confidence** (`high`, `medium`, `low`, `unknown`)
reduced by the weakest link in the chain. Numbers are never presented with more precision
than their provenance supports.

Results (Memory and Compare) and Markdown reports show confidence separately for
weights, KV cache and runtime overhead, alongside each component's source. Technical
details retain the aggregate confidence and component explanations for auditing.
The aggregate is a weakest-link indicator, not a measured error probability or an
overall reliability grade for the model. Since runtime overhead is always an assumed
heuristic with `low` confidence, ordinary estimates are capped at `low` (or `unknown`
when a required component is unknown). A successful execution does not rewrite that
prediction. This presentation does not change estimation formulas or ranking policy;
recommendation report format v4 records the presentation change, with existing JSON
confidence fields preserved.

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

When available, the non-block aggregate is also projected into embedding and output-head
components. Those optional values are useful architectural diagnostics only: they do not prove
the byte split of a GGUF artifact and do not prescribe backend placement.

For discrete partial offload, Hardware Fit consumes this decomposition without
choosing a runtime-specific location for non-block weights. With `B` estimated
block bytes, `N` non-block bytes and `T` blocks, placing `n` blocks on GPU uses
`ceil(B*n/T)` block bytes there; host block bytes are the exact remainder.
This rounds once per candidate, not once per block multiplied by `n`.

Non-block GPU bytes are unknown in `[0, N]`, with the host share the remainder.
The analyzer evaluates both endpoints using the existing KV, overhead and margin
splits. GPU requirement is monotone increasing in this share; RAM requirement
is monotone decreasing. A candidate is accepted only if **GPU maximum fits VRAM
and RAM maximum fits RAM**. These maxima are different hypothetical placements,
not one allocation: do not add them or infer duplicated weights.

Existing `gpu_*`/`ram_*` fields describe the GPU-heavy endpoint and conserve all
bytes. Optional `non_block_placement_bounds` retains block bytes per pool, total
non-block bytes, GPU minimum, RAM maximum and the RAM-heavy overhead/margin.
The opposite endpoint is reconstructed by remainders. Diagnostics include the
same bounds for selected and first rejected candidates. Compatibility ratios
use both pool maxima. Physical point predictions are unavailable while the
non-block split is unknown, rather than comparing an endpoint against measured
process allocation as though it were the predicted placement.

Ignoring nonnegative overhead, margin and rounding gives the valid search bound
`n <= floor(T*(available_vram - reserve - N)/(B + KV))`, capped at `T-1` for
partial offload. Search visits candidates downwards and never uses the measured
runtime layer count. A rejected higher candidate above an accepted one must
fail the GPU maximum: decreasing GPU blocks cannot rescue insufficient host RAM.

Resident GPU and CPU modes still charge all weights to their respective pool.
Unified memory retains its single-pool rule. Missing/unsupported architecture
keeps the existing uniform or byte fallback, without inventing non-block
precision. Bounds are conditional on estimated parameter-to-byte fractions,
not bounds on actual tensor allocations. They exclude duplication, staging,
uneven block sizes and runtime-specific storage. A conservative rejection does
not prove that every backend would fail. Benchmarks validate this model; they
do not supply calibration constants. Additive optional fields retain schema v2
and legacy payloads remain readable.

#### Qwen fixture: before/after non-block bounds

Offline fixture, not a new runtime measurement: 4,683,074,240 weight bytes,
234,881,024 KV bytes, 1,005,178,336 overhead bytes, 536,870,912 reserve bytes,
646,000,452 margin bytes, 4,985,380,864 available VRAM bytes and 7,593,828,352
available RAM bytes. The existing architecture split estimates 4,012,773,975
block bytes and 670,300,265 non-block bytes over 28 blocks.

| Decision | Previous uniform placement | Non-block bounds |
|---|---:|---:|
| Mode | GPU_OFFLOAD | GPU_OFFLOAD |
| Selected blocks | 18 | 17 |
| First rejected | 19 | 18 |
| Search ceiling | 25 | 24 |

All following quantities are **bytes of planning budget**. After columns use
the GPU-heavy endpoint; RAM maximum is a different endpoint checked separately.
The previous model had no separate block/non-block terms, so those entries are
unavailable rather than reconstructed as though it had used a decomposition.

| Term | Before selected 18 | Before rejected 19 | After selected 17 | After rejected 18 |
|---|---:|---:|---:|---:|
| Block GPU weights | unavailable | unavailable | 2,436,327,057 | 2,579,640,413 |
| Non-block GPU weights (endpoint) | unavailable | unavailable | 670,300,265 | 670,300,265 |
| Block RAM weights | unavailable | unavailable | 1,576,446,918 | 1,433,133,562 |
| Non-block RAM weights (endpoint) | unavailable | unavailable | 0 | 0 |
| Total GPU weights | 3,010,547,736 | 3,177,800,388 | 3,106,627,322 | 3,249,940,678 |
| Total RAM weights | 1,672,526,504 | 1,505,273,852 | 1,576,446,918 | 1,433,133,562 |
| GPU KV | 150,994,944 | 159,383,552 | 142,606,336 | 150,994,944 |
| RAM KV | 83,886,080 | 75,497,472 | 92,274,688 | 83,886,080 |
| GPU overhead | 646,186,076 | 682,085,302 | 664,109,189 | 695,115,476 |
| RAM overhead | 358,992,260 | 323,093,034 | 341,069,147 | 310,062,860 |
| GPU margin | 434,459,968 | 455,614,016 | 445,021,377 | 463,292,202 |
| RAM margin | 211,540,484 | 190,386,436 | 200,979,075 | 182,708,250 |
| GPU required (maximum after) | 4,779,059,636 | 5,011,754,170 | 4,895,235,136 | 5,096,214,212 |
| RAM required (minimum after) | 2,326,945,328 | 2,094,250,794 | 2,210,769,828 | 2,009,790,752 |
| GPU headroom | 206,321,228 | 0 | 90,145,728 | 0 |
| GPU excess | 0 | 26,373,306 | 0 | 110,833,348 |
| GPU minimum | unavailable | unavailable | 4,007,202,288 | 4,208,181,363 |
| RAM maximum | unavailable | unavailable | 3,098,802,676 | 2,897,823,601 |
| Host-heavy RAM overhead | unavailable | unavailable | 478,071,471 | 447,065,185 |
| Host-heavy RAM margin | unavailable | unavailable | 281,709,334 | 263,438,509 |

At the same 18 blocks, the new GPU endpoint charges 239,392,942 more weight
bytes than the old uniform model. Its existing proportional overhead and
margin therefore also increase. This, not a larger marginal block estimate,
rejects 18; no constants were adjusted to choose 17.

The estimated marginal weight changes from about 159.50 to 136.67 MiB/block.
Against the supplied observed endpoint slope of about 132.85 MiB/block, the
relative differences are about +20.1% and +2.9%. This remains an external,
single-model check, not a measurement of individual tensors. The apparent
fixed CUDA term (~448 MiB) is not identified as embeddings or output weights.
Our non-block GPU interval is 0..639.25 MiB; numerical overlap does not validate
that tensor interpretation or give HFA authority over runtime launch policy.

### 2. KV cache

Quantized weight precisions (`int4` and `int8`) are not KV-cache formats. Jaull currently
returns an unknown KV estimate for these choices, rather than zero bytes or an assumed
packed layout. GGUF executable plans currently support FP16 KV only; other requested KV
precisions do not produce a runnable plan. This is a deliberate guard, not quantized-KV support.

Hybrid GGUF models retain a per-layer cost vector: full-attention layers grow with context,
while recurrent layers keep fixed state. Partial offload budgets charge the selected suffix
of that vector rather than a uniform fraction of total KV. Dense models keep the existing
uniform policy. These are estimates, not measured allocations.

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

`batch_size` counts independent sequences with retained KV contexts, not tokens
processed in one decoder call. llama.cpp's `--batch-size` and `--ubatch-size` control
logical and physical token batches instead. Generated llama.cpp plans leave those
runtime knobs at their defaults; neither the sequence batch nor concurrency is
translated into a prompt-token batch limit. The KV formula is unchanged, and these
capacity inputs do not establish that Run/Validate executes a multi-sequence workload.

Non-standard architectures (MLA / DeepSeek-style, multimodal composites, `auto_map`
custom code) return `unknown` and a warning instead of a fabricated number. A Mixture-of-Experts
model keeps the dense formula, because its experts sit in the feed-forward layers, at confidence
`medium` and with an explicit note.

### 3. Runtime overhead

Allocator, kernel buffers, activations and small caches:

```text
overhead = max(min_overhead, base_overhead + weight_fraction * weights)
```

Constants live in `estimator/policies.py` (`base = 512 MiB`, `fraction = 10%`,
`min = 256 MiB`). Always tagged `ASSUMED` / `LOW` confidence.

GGUF GPU plans retain their calibrated 256 MiB compute allowance. GGUF CPU plans
instead use the general 512 MiB + 10% of weight bytes heuristic, including AUTO
fallback to CPU. The CPU breakdown distinguishes a compute allowance and other
runtime allowance; repacking remains explicitly **unknown**, not zero. This is
not an upper bound or a universal correction fitted to one experiment.
`CPU_REPACK` allocations depend on tensor types, CPU kernels and runtime build;
mapped weights, repacked allocations and process RSS are not interchangeable or
necessarily simultaneously resident. Measured allocations are shown as diagnostics,
not silently used to recalibrate every model.

### 4. Device reserve and safety margin

An omitted margin is represented as `null` in configuration JSON and selects the automatic
policy: GGUF uses fixed headroom without a percentage margin; other repositories
use 10 percent. An explicit number, including exactly `10`, is always applied. This distinction
survives saving and loading the configuration.

CPU execution reserves 512 MiB of **available RAM** by default, independently of GPU
headroom. `--ram-reserve-gib` overrides this value, including `0` to disable it.
GPU execution keeps `--device-reserve-gib`, with a 1 GiB VRAM minimum for GGUF.
AUTO applies the CPU reserve if placement falls back to CPU; it does not carry the
rejected GPU reserve into RAM. Shared-memory hardware is checked against one RAM
pool with one CPU reserve, never RAM plus an invented VRAM budget. Partial GPU
offload keeps its existing GPU reserve and per-pool placement policy.
These reserves are policy assumptions, not measured memory allocations.
For CPU, estimated physical consumption (weights + KV + overhead) is checked
separately from the budget (consumption + reserve + margin). If consumption fits
but the budget does not, the result is `tight`, with an explicit reserve shortfall.
If consumption exceeds available RAM but remains within total usable system RAM,
the result is `memory_pressure`: an experimental, best-effort CPU option, never a
confirmed resident fit or a confirmed-fit primary recommendation. It may be run
with warnings, but paging and sustained performance remain unverified. Consumption
beyond total usable RAM remains insufficient. This does not add swap to RAM capacity
or relax any GPU residency/offload constraint.

When CPU recommendations already include current fits, display selection may reserve
one of the remaining slots for a useful conditional alternative. For Balanced (and other
non-Quality priorities), its additional RAM budget (including reserve and margin) must
be at most both 3 GiB and 15% of total RAM; the alternative occupies the last slot.
Quality allows up to both 6 GiB and 25% and promotes the conditional alternative to the
second slot, still behind the current-fit primary. In every case the whole budget must
leave 20% of total RAM outside the model. These are explicit
display-policy bounds, not an estimate of reclaimable memory. The first recommendation
and the `memory_pressure` assessment remain unchanged. The warning states how much more
RAM must become available and asks the user to recheck after closing applications.

For compatibility, estimate JSON keeps the active reserve in
`memory.device_reserve_bytes`; its component is labelled `RAM reserve` for CPU
execution. Hardware Fit's `ram_reserve_bytes` identifies the same reserve charged
to RAM, not an additional amount to add to the total.

The hardware screen and persistent workflow header show total and available RAM
together. Fit calculations use the available figure captured by the hardware scan,
not total installed capacity; that snapshot can change as other applications run.

### Context sweeps

GGUF sweeps resolve metadata once, reuse the merged configuration at every context, and stop
at its declared context limit when known. Sweep JSON schema 2 records hardware budgets,
requested/effective configuration, the selected artifact metadata, enrichment and an estimate
snapshot. Missing immutable revisions or hashes remain explicitly unknown; this is an inspection
snapshot, not artifact verification. It does not demonstrate performance at a filled context.

User-controllable via `--ram-reserve-gib`, `--device-reserve-gib` and
`--safety-margin-percent`, kept as
distinct components so each one can be inspected — and so a policy figure is never mistaken
for a measured one.

---

## Compatibility

`Validate` reports execution success independently of preserved headroom and
performance. Local host runs sample minimum system RAM availability and system-wide
swap-in/out byte deltas, alongside peak process RSS. These are host-wide observations,
not paging attributed to the model; zero swap I/O does not prove absence of file-backed
page faults or sustained performance. Missing/unsupported measurements stay unknown
(including Windows swap I/O counters). Older records do not acquire invented data.
`Speed` and `Validate` compare RSS against estimated **consumption**, excluding policy
reserve and margin. The budget remains visible separately. CPU repacking allocations
from runtime logs are labelled allocations, not RSS. The prediction engine identifier
is `memory-estimator-v2`; stored v1 experiments remain immutable and loadable.

Normally the budget is compared against available memory; CPU reserve shortfalls
and pressure follow the separate rules above:

| Status | Meaning |
|---|---|
| `comfortable` | ≤ 75 % of available |
| `compatible` | 75–90 % |
| `tight` | 90–100 %, or CPU consumption fits but the desired reserve/margin does not |
| `memory_pressure` | CPU consumption exceeds available RAM; experimental, not confirmed resident |
| `offloading_required` | A split placement fits its separate RAM and VRAM budgets |
| `insufficient` | No supported placement; CPU consumption exceeds total usable RAM |
| `unknown` | Missing inputs |

With `--device auto` the estimator prefers GPU, then a supported offload placement,
then CPU. CPU can be a resident fit, an experimental pressure option, or insufficient.
RAM and VRAM are not blindly added together.

VRAM reaches this comparison from two merged views: NVML for NVIDIA, and the
backend-neutral accelerator view for everything else. The second carries a memory figure
only where Linux DRM sysfs exposes one for the detected PCI device — in practice discrete
AMD on an amdgpu-style driver. On Windows AMD, Intel and Apple, the accelerator is still
detected and its backends probed, but the memory model falls back to system RAM. The
estimator consumes observed *availability*, never raw capacity, so a device that reports
only its total size can be shortlisted without claiming a run-now budget.

## Where it fits, not just whether it fits

A single verdict hides the question that actually matters on a mixed machine: *which memory
would this run out of?* `estimator/hardware_fit.py` answers that separately, as a placement:

| Mode | Meaning |
|---|---|
| `GPU_RESIDENT` | The whole model lives in VRAM |
| `GPU_OFFLOAD` | Part in VRAM, part in system RAM |
| `CPU_RAM` | System RAM only, no GPU |
| `CPU_MEMORY_PRESSURE` | Experimental CPU option; physical consumption exceeds available RAM |
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
| `head_dim` | GGUF `attention.key_length` | base config `head_dim` | `hidden // heads` | — |
| `sliding_window` | base config | — | — | no cap |
| architecture | GGUF `general.architecture` | base config `model_type` | — | warning |

`rope.dimension_count` describes rotary dimensions, which can cover only part of a
head; it is never used as the KV head size. An explicit GGUF `attention.value_length`
that differs from the resolved key dimension makes the KV estimate unknown: the current
symmetric formula does not model unequal K/V heads. These are separate fields in the
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

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

- The preliminary llama.cpp offload recommendation uses an aggregate weight budget:
  estimated block bytes plus a fixed, conservative non-block GPU bound. KV scales with
  the proposed offload. Reserve and the existing 256 MiB runtime headroom remain intact.
  Without decomposition it uses the uniform weight approximation; without a block count
  it falls back to the documented conservative default (20 units) with a warning.
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

### Optional local GGUF launch refinement

Once an artifact is prepared, `AdvisorService` obtains a local tensor index and a verified
runtime capability before `ExecutionPlanner` resolves the final launch. The reader reuses
the GGUF metadata parser, reads only metadata and tensor descriptors (bounded to 8 MiB),
and validates types, row sizes, offsets, overlap and file bounds. It does not read weight
payloads, contact the Hub, change artifact identity or recompute a hash. Stored sizes use
GGML quantization block layouts, not a parameter-count fraction.

The first placement rule is deliberately narrow: **llama.cpp build `689e227db`, dense
`qwen2`, a separate output tensor, and one confirmed discrete CUDA device**. For that
verified implementation, positive `u` runtime units place the output tensors and the last
`min(u - 1, T)` repeating blocks on GPU; the input tensor stays on host. This is a backend
rule, never a conversion from `HardwareFitResult.gpu_transformer_blocks`.

For each suffix of `n` repeating blocks, the launch budget is:

```text
GPU budget = sum(aligned stored tensor bytes in the suffix)
           + aligned output.weight + aligned output_norm.weight
           + ceil(total_KV_bytes * n / T)
           + existing device reserve + existing runtime headroom
```

The largest suffix satisfying the recorded VRAM budget is selected, including zero
repeating blocks (output only); if even the output budget fails, request zero runtime
units. Individual blocks may have different stored sizes. KV, reserve and headroom
policies are unchanged. Stored tensor bytes are **not measured CUDA allocations**;
allocator, repacking and compute-buffer differences remain covered only heuristically.

Missing/unsupported descriptors, tied or missing output weights, an unclassified tensor,
inconsistent estimate inputs, another build/architecture/backend or multiple devices keep
the aggregate launch fallback and attach a reason. User-supplied `--n-gpu-layers` takes
precedence and skips inspection. A different context override cannot reuse a stale KV
estimate for refinement. Benchmark preparation verifies the `llama-bench` build too;
builds without `--version` can expose their build footer via a zero-task, nonexistent-model
probe. Failure to establish the build never enables tensor placement.

This refinement changes the **prepared runtime flags**, not `MemoryEstimate`, HFA,
ranking, shortlist, or PredictionComparison. No persisted schema changes are needed.
The generated estimate command may therefore differ from the prepared local command;
the latter records the tensor budget and runtime build in its launch reasons.

`scripts/validate_local_tensor_launch.py --record <experiment.json> --output <new-dir>`
can inspect a frozen record offline. Add `--execute --llama-cli <path> --llama-bench <path>`
to freeze the current launch VRAM budget and commands before executing a context-4096
smoke (or the context recorded in the input) and pp512/tg128 controls. It writes a new
evidence directory; the source experiment is not overwritten. The microbenchmarks do
not validate throughput at context 4096 or process-attributed peak VRAM.
