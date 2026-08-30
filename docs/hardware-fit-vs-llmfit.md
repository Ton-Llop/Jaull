# Hardware fit: Jaull vs llmfit

A controlled comparison between Jaull's `HardwareFitAnalyzer` and
[llmfit](https://github.com/alexsjones/llmfit) 1.1.10.

**llmfit is not the reference implementation, and this is not a critique of
it.** Where the two tools disagree, the entry below says what each one answered
and why the answers can differ. Several differences turn out to be differences
of *encoding* rather than of judgement, and are described as such.

Reproduce everything here with:

```bash
uv run python scripts/hardware_fit_matrix.py        # Jaull only, offline
uv run python scripts/hardware_fit_vs_llmfit.py     # both tools
```

---

## 1. The two tools answer different questions

This is the finding that governs every row below, so it comes first.

| | starts from | decides | can change the artifact? |
| --- | --- | --- | --- |
| **llmfit** | a model name + hardware | which *build* of the model to run, and where | yes — it picks a quantization to fit |
| **Jaull** | an artifact whose memory is already estimated | where that artifact's bytes go | no — it places what it was given |

llmfit's memory figure is therefore a function of the hardware. On the same
Qwen2.5-7B-Instruct it reports 3.54 GiB at 4 GiB of VRAM (choosing Q2_K),
5.90 GiB at 6 GiB (Q5_K_M), 6.81 GiB at 8 GiB (Q6_K) and 8.72 GiB at 24 GiB
(Q8_0). Jaull's analyzer is downstream of that decision: quantization
selection happens in the recommendation engine, and the analyzer only ever
sees one artifact at a time.

Comparing `llmfit info` against Jaull directly would therefore compare two
different models. The harness pins llmfit with `plan --quant Q4_K_M`, reads
back llmfit's *own* weight and KV-cache bytes, and feeds those to Jaull's
`analyze_components`. With byte-identical inputs, a mode difference can only
come from the placement rule.

### Units

llmfit's `*_gb` fields are binary gigabytes. For Qwen2.5-7B at 4096 tokens it
reports `kv_cache_gb = 0.21875`, and `2 · 28 layers · 4 kv heads · 128 head_dim
· 4096 · 2 bytes` is exactly `0.21875 · 2^30`. The harness converts with
`2**30`; using `1000**3` would shift every row by ~7%.

---

## 2. Results

`Jaull (bare)` runs the analyzer with zero overhead, device reserve and safety
margin — llmfit's own accounting, so the column isolates the placement rule.
`Jaull (policy)` adds Jaull's production values (`estimator/policies.py`:
overhead `max(256 MiB, 512 MiB + 10% of weights)`, reserve `512 MiB`, margin
`10% of subtotal`). The gap between those two columns *is* Jaull's headroom
policy.

`llmfit pinned` is the apples-to-apples answer (`plan --quant Q4_K_M`).
`llmfit auto` is llmfit answering its own question (`info`), shown to make the
mismatch visible rather than to score it.

| case | machine | Jaull (policy) | Jaull (bare) | llmfit pinned | pinned GiB | llmfit auto | auto quant | auto GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tinyllama_1b_on_8gb | 8 GiB VRAM / 32 GiB RAM | gpu_resident | gpu_resident | Gpu | 1.18 | GPU | Q8_0 | 1.70 |
| qwen3b_on_8gb | 8 GiB VRAM / 32 GiB RAM | gpu_resident | gpu_resident | Gpu | 2.43 | GPU | Q8_0 | 3.88 |
| qwen7b_on_8gb | 8 GiB VRAM / 32 GiB RAM | gpu_resident | gpu_resident | Gpu | 5.14 | GPU | Q6_K | 6.81 |
| qwen7b_on_8gb_long_context | 8 GiB VRAM / 32 GiB RAM | gpu_offload | gpu_resident | Gpu | 6.67 | GPU | Q5_K_M | 7.43 |
| qwen7b_on_6gb | 6 GiB VRAM / 16 GiB RAM | gpu_offload | gpu_resident | CpuOffload | 5.14 | GPU | Q5_K_M | 5.90 |
| qwen7b_on_4gb | 4 GiB VRAM / 32 GiB RAM | gpu_offload | gpu_offload | CpuOffload | 5.14 | GPU | Q2_K | 3.54 |
| qwen7b_without_gpu | no GPU / 32 GiB RAM | cpu_ram | cpu_ram | CpuOnly | 5.14 | CPU+GPU | Q8_0 | 8.72 |
| qwen14b_on_8gb | 8 GiB VRAM / 32 GiB RAM | gpu_offload | gpu_offload | CpuOffload | 9.82 | GPU | Q2_K | 6.71 |
| qwen32b_on_6gb | 6 GiB VRAM / 16 GiB RAM | too_large | too_large | Gpu | 20.50 | CPU+GPU | Q2_K | 13.62 |
| llama70b_on_24gb | 24 GiB VRAM / 64 GiB RAM | gpu_offload | gpu_offload | CpuOffload | 42.67 | CPU+GPU | Q5_K_M | 49.73 |

Mode vocabularies map as `Gpu`/`GPU` → `GPU_RESIDENT`, `CpuOffload`/`CPU+GPU` →
`GPU_OFFLOAD`, `CpuOnly`/`CPU` → `CPU_RAM`. **llmfit has no `run_mode` that
corresponds to `TOO_LARGE`** — §3.1 explains why that is a difference of
encoding rather than of verdict.

On identical inputs the placement rule agrees on **8 of 10** cases.

---

## 3. Differences observed

### 3.1 Infeasibility is encoded differently, not judged differently

`qwen32b_on_6gb` asks both tools to place a 20.50 GiB artifact on 6 GiB of VRAM
and 14 GiB of free RAM. Jaull answers `TOO_LARGE`. llmfit answers
`run_mode: Gpu` with `fit_level: TooTight`.

It would be wrong to read that as llmfit claiming the model fits. llmfit is
reporting two separate things:

* `run_mode` — **which execution path the estimate describes**;
* `fit_level` — **how well that path actually fits this machine**.

Jaull collapses both into a single placement state, so infeasibility is a value
of `mode` rather than a severity attached to it. Same information, different
shape.

The same 32B artifact across four machines shows which field carries the
machine-dependent signal:

| machine | llmfit `run_mode` | llmfit `fit_level` | llmfit `estimated_tps` |
| --- | --- | --- | --- |
| 24 GiB VRAM / 64 GiB RAM | `CpuOffload` | Good | 5.64 |
| 6 GiB VRAM / 16 GiB RAM | `Gpu` | TooTight | 11.28 |
| 1 GiB VRAM / 4 GiB RAM | `Gpu` | TooTight | 11.28 |
| 0.5 GiB VRAM / 2 GiB RAM | `Gpu` | TooTight | 11.28 |

Below the fit threshold `run_mode` stops varying with the machine and the
throughput estimate goes flat: the machine-dependent information has moved
entirely into `fit_level`. `run_paths[].feasible` is not a substitute either —
it is `true` for all three paths on every machine, because it describes
requirement profiles rather than feasibility against detected hardware.

**Reading to take away:** `run_mode` cannot be interpreted as a standalone fit
verdict; `fit_level` must be read with it. Jaull's design choice is to make
infeasibility a placement state (`TOO_LARGE`) precisely so that one field
answers "can this run at all?". Neither encoding is wrong. The Jaull one is
harder to misread when the value is consumed by code rather than read by a
person, which is the situation Jaull is built for.

### 3.2 llmfit's overrides cannot express the absence of a GPU

`--memory 0M` caps VRAM at zero but leaves `has_gpu: true` and the detected GPU
name in the payload, so `info` still answers `CPU+GPU`. Jaull models "no GPU"
as a distinct topology: `available_vram_bytes` is `None`, `gpu_required_bytes`
is `None`, `gpu_transformer_blocks` is `None`, and the reason string reads
"No GPU detected".

This is a limitation of the *override flags* rather than of llmfit's model — on
a machine that genuinely has no GPU its own detection would presumably say so.
It does mean the `qwen7b_without_gpu` row is only meaningful on the `plan` side
(`CpuOnly`), which agrees with Jaull's `cpu_ram`.

### 3.3 Physical placement and headroom policy are separable — and separated

Two cases change mode between `bare` and `policy` with no change in weights or
KV cache:

| case | bare | policy | overhead | reserve | margin |
| --- | --- | --- | --- | --- | --- |
| qwen7b_on_8gb_long_context | gpu_resident | gpu_offload | +0.99 GiB | +0.50 GiB | +0.82 GiB |
| qwen7b_on_6gb | gpu_resident | gpu_offload | +0.99 GiB | +0.50 GiB | +0.66 GiB |

Roughly 2.3 GiB of Jaull's requirement on a 7B model is policy rather than
measurement. llmfit budgets no equivalent line items; its headroom judgement
lives in `fit_level` and `utilization_pct` instead — on the 6 GiB machine
`info` reported 98.3% utilisation for its own Q5_K_M pick and still called it
`GPU`.

The practical consequence is that **the two tools' memory numbers are not
comparable without first subtracting Jaull's policy**, which is exactly what
the `bare` column does.

### 3.4 What `qwen7b_on_6gb` actually shows

This is the one case where the bare placement rules disagree, and it is worth
reading rather than scoring:

```
weights + KV ≈ 5.14 GiB        VRAM = 6 GiB

Jaull (bare)    → GPU_RESIDENT     5.14 < 6, so the arithmetic allows it
Jaull (policy)  → GPU_OFFLOAD      after overhead + reserve + margin
llmfit          → CpuOffload
```

Both tools end up recommending a split; they reach it through different policy
layers. llmfit folds its caution into the placement decision itself. Jaull keeps
the two apart:

```
physical placement   ─ does the arithmetic allow it?
        ↓
policy / headroom    ─ is it sensible to actually do it?
```

The analyzer therefore stays a statement about arithmetic, and the conservatism
lives in the estimator's policy where it can be tuned, audited and explained on
its own. This is why the analyzer carries no arbitrary "keep 20% of VRAM free"
rule: it would blur exactly this boundary, and the two rows in §3.3 show the
policy layer already does that job.

### 3.5 llmfit's self-recommendation diverges on every row

On all 10 cases llmfit's own recommendation is a different quantization than
the pinned Q4_K_M — usually a *larger* one (Q8_0, Q6_K, Q5_K_M) on roomy
machines and a smaller one (Q2_K) on tight ones. This is llmfit working as
designed and is not a discrepancy in the placement rule; it is recorded because
it is the reason a naive `info`-vs-Jaull comparison is invalid.

---

## 4. Two asymmetries found inside Jaull, and closed

The battery surfaced two internal inconsistencies in the analyzer. Both are now
fixed in `src/jaull/estimator/hardware_fit.py` and pinned by tests, because
both would become load-bearing as soon as reports, execution plans and
validation against runtime placement starts reading these fields.

**1. `gpu_transformer_blocks` was `0` on machines with no GPU.** `CPU_RAM` set
it to `0` unconditionally while `TOO_LARGE` set it to `None`, so the two
branches disagreed about a machine with no GPU at all. The values are not
interchangeable:

| value | means |
| --- | --- |
| `None` | there is no GPU — the question does not apply |
| `0` | a GPU exists and no layer was placed on it |

Every branch now returns `None` whenever no GPU is present, through a single
`_gpu_transformer_blocks_when_unused` helper.
`test_gpu_transformer_blocks_is_none_when_there_is_no_gpu_and_zero_when_there_is_one`
pins all three cases.

**2. Unified memory silently discarded `device_reserve_bytes`.** The caller
passed a reserve and the branch neither charged it to the requirement nor
echoed it back, so a scenario with a 1 GiB reserve was byte-identical to one
with none. Of the two possible semantics —

* **(A)** the reserve does not apply to unified memory → reject or normalise it
  explicitly;
* **(B)** the reserve consumes the shared pool → count it;

**(B) was chosen.** A device reserve is memory held back for the accelerator.
On discrete hardware that is VRAM, which is why a `CPU_RAM` placement there
correctly does not pay for it; on unified memory the accelerator draws from the
same pool as the CPU, so it genuinely consumes the budget. The reserve is now
added to `ram_required_bytes` and reported back in `device_reserve_bytes`.

Two tests pin both halves of the rule:
`test_unified_memory_charges_the_device_reserve_to_the_shared_pool` and
`test_a_discrete_cpu_ram_placement_does_not_pay_the_device_reserve`.

Neither fix changes any mode in the comparison table above. Both change what
the result *says* about the placement it reached.

---

## 5. Method notes and limits

- **Determinism.** The Jaull fixtures in `tests/_hardware_fit_scenarios.py` are
  pure byte inputs and run offline; their observations are pinned in
  `tests/snapshots/hardware_fit_scenarios.json`. The llmfit side depends on
  llmfit's embedded model database and so is pinned only to version 1.1.10;
  re-run the script after upgrading it.
- **Hardware overrides.** llmfit is driven with `--memory` / `--ram` /
  `--max-context`. It derives available RAM as 90% of total, while Jaull is
  given available RAM directly; the harness records the pools each tool
  actually saw so the difference is visible rather than assumed.
- **Transformer block counts** for the Jaull side come from llmfit's own
  database (`num_hidden_layers`), so both tools describe the same model
  geometry.
- **Not compared:** throughput estimates, quality scores, ranking, and
  llmfit's upgrade-planning output. This is a comparison of placement, not of
  the two products.
- **Single quantization.** Every case pins Q4_K_M. A sweep across
  quantizations would test whether the 8/10 agreement holds; it was not run.
- **Where this stops being useful.** The comparison has served its purpose:
  the placement rule agrees with an independent implementation on 8 of 10
  cases, and the two that differ are explained. The next experiment is not
  another tool but reality — Jaull's predicted mode, transformer-block
  placement, VRAM and RAM against what llama.cpp actually loads and reports.
  That is a prediction-versus-observation study, and it is the reason §4's
  transformer-block semantics had to land first.
