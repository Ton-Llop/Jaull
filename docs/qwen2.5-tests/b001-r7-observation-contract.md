# B001-R7: the first run through the observation contract

## Scope

This is a physical re-execution of the B001 baseline with the current code, to
answer one question: does the observation contract added in `3a6dc26` produce a
measurement when a model actually runs? No estimator formula, placement rule or
ranking policy was changed for it.

It found that the answer was **no**, fixed that, and re-ran. Both runs are
reported below, because the first one is the finding.

## Method

Same machine as B001-R4 (RTX 2060, 6144 MiB), same artifact
(`bartowski/Qwen2.5-7B-Instruct-GGUF`, `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, sha256
`65b8fcd9…1423`), same llama.cpp build `689e227db`, context 4096, prompt
`"Explain briefly what artificial intelligence is."`.

Driven through `AdvisorService.run_experiment(ExperimentRequest)` — the same
entry point `RecommendationValidationScreen` uses — not through a bespoke
script, so what ran is what the TUI runs. `persist=False`: this is a
methodological run, not a stored baseline.

## Run 1: the contract never fired

The run succeeded (exit 0, 18.6 s, 4723.2 MiB peak RSS) and produced:

```
runtime_allocation : None
observed_backend   : None
```

`llama_cpp_runner.py` passed `--verbose` only when a runtime flag requested it,
and nothing sets that flag. Without it this build writes **zero** lines to
stderr, so `parse_llama_cpp_allocation` had no buffer lines to read and
`_observed_backend` had no backend line. Verified directly:

| llama-cli invocation | stderr lines | `buffer size` lines |
|---|---:|---:|
| Jaull's exact command | 0 | 0 |
| the same plus `--verbose` | 4148 | 16 |

stdout is 31 lines either way, so the flag changes logging only.

**The whole observation contract was unreachable from the path that runs
models.** Every test covering it fed the parser a recorded August log directly;
none checked that a live run produces one. Fixed by asking for the log
unconditionally, with `test_runner_always_asks_for_the_load_log` to pin it.

## What the real log looks like

Two properties the recorded August sweep did not have, and that a hand-written
fixture would not have guessed:

```
0.00.488.277 I load_tensors:        CUDA0 model buffer size =     0.00 MiB   <- reserve pass
0.00.492.867 I llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB      <- reserve pass
0.00.497.968 I sched_reserve:       CUDA0 compute buffer size =   183.44 MiB
0.00.795.071 I load_tensors:        CUDA0 model buffer size =  3616.41 MiB   <- real
0.04.369.084 I llama_kv_cache:      CUDA0 KV buffer size =   192.00 MiB      <- real
0.04.395.075 I sched_reserve:       CUDA0 compute buffer size =   183.44 MiB
```

Every buffer is printed twice and the first pass reports `0.00 MiB`; the KV
lines come from `llama_kv_cache:` and compute from `sched_reserve:`, not from
`llama_context:`. The parser was already correct on both counts — it keys on the
buffer label and keeps the last value, so the reserve zeros lose and the
duplicate compute line is not added twice. That capture is now a test fixture
rather than a claim.

## Run 2: the first measured allocation

```
free VRAM at launch : 4746 MiB
prediction          : offloading_required, gpu_offload, 20/28 blocks
launch policy       : --n-gpu-layers 26
observation         : success, exit 0, 7.3 s, peak RSS 4723.9 MiB
peak_vram_bytes     : None   (WDDM: NVML attributes nothing per process)
```

| Buffer | Location | Measured |
|---|---|---:|
| `CUDA0 model` | device | 3741.47 MiB |
| `CUDA0 KV` | device | 200.00 MiB |
| `CUDA0 compute` | device | 183.44 MiB |
| `CPU_Mapped model` | host | 718.97 MiB |
| `CPU KV` | host | 24.00 MiB |
| `CUDA_Host compute` | host | 18.01 MiB |
| `CUDA_Host output` | host | 0.58 MiB |
| **device total** | | **4124.91 MiB** |

`source = runtime_reported_allocation`, `driver_confirmed = false`. This is the
first measured device figure Jaull has ever produced on this machine: NVML
reports nothing under WDDM, and the runtime's own report is the only observation
available.

## The comparison is still unavailable, for a reason that has not changed

```
availability : methodologically_unavailable
reason       : Non-block weight placement is bounded, not a device-specific
               point prediction.
components   : all three methodologically_unavailable — the Hardware Fit
               prediction is in runtime-agnostic transformer blocks while
               llama.cpp --n-gpu-layers uses backend-specific offload units.
```

So the state moved from *"there is no measurement"* to *"there is a measurement
and no comparable prediction"*. That is progress and it sharpens what blocks the
rest: **the block ↔ `--n-gpu-layers` mapping**, not the measurement side.

The prediction was sized for 20 transformer blocks; the run used 26 runtime
units. Those are different placements, so the difference between 3372.7 MiB
predicted and 3741.47 MiB measured is mostly placement, not memory-model error.
Publishing it as an error percentage would attribute one to the other, which is
exactly what the gate exists to prevent. **No number from this run should be
used to calibrate anything.**

## Incidental observation

The two runs chose different launch units — 25 then 26 — because free VRAM
differed between them (4564 then 4746 MiB) with nothing else changed. That is
finding E.3 (instantaneous availability rather than capacity) visible in
practice: the same machine, the same model, two placements twenty minutes apart.

## What would close the remaining gap

A verified mapping from Hardware Fit's transformer blocks to llama.cpp's launch
units for this build. B001-R6 already derives one for `689e227db` + dense
`qwen2` in the launch policy; the comparison layer does not consume it. Wiring
that mapping into `_placement_mismatch` — for the builds where it is verified,
and only those — would turn this run's numbers into the first component-wise
error. That is a separate milestone and needs its own justification: the units
invariant is enforced by `test_architecture_dependencies.py` precisely so this
does not happen by accident.
