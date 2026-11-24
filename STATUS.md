# Status

Updated 2026-11-24. Last verified remote CI checkpoint: [`ba5b5da`](https://github.com/Ton-Llop/Jaull/actions/runs/44152938519).
The commits after it (up to `89a2b8f`) have not been checked against CI here.

## What works

Jaull detects hardware, recommends models and prepares an execution plan.
It verifies the artifact, runs the model, saves experiments and benchmarks,
and compares predictions with measurements when the evidence allows it.

- CLI and TUI use the same services.
- Recommendations and execution readiness are separate: missing dependencies
  block execution, not the recommendation.
- Task and workload mode are separate. Workload requirements and optional
  throughput/TTFT targets can be recorded; they are not measured results.
- Artifact verification rejects incorrect downloads and removes stale SHA
  sidecars before replacing files.
- **Serve** sizes context and slots, starts a local `llama-server` without Docker,
  and distinguishes a live process from one that responds to its health check.
  It checks the server binary and device before launch and rehashes cached models.
  Older published GGUFs are still accepted when the bytes match their accepted SHA.
  - HTTPS is on by default when the server is shared (`--https` and `--no-https`,
    and a checkbox in Setup), with a self-signed certificate Jaull makes.
  - If opencode is installed, starting a server adds a `jaull` provider to its
    config. Other providers and the default model are left alone, a backup is kept
    and the file is mode 0600. A config with comments is left untouched.
  - On native Windows with a dedicated Intel GPU the server gets
    `--sleep-idle-seconds 45`, and `-lm dio` when the model fits on the GPU. Neither
    is applied under WSL or on CPU, because they were only measured on native Windows.
  - The TUI monitor does not keep a sleeping server awake: only `/slots` wakes it,
    so with sleep on the monitor reads `/slots` only when the counters show activity.
  - Setup can show the full `llama-server` command broken into flags, with what each
    one means, and the command to make opencode in WSL trust the certificate.
- **Measured ranking policies are active.** After the base order, stored
  measurements may reorder plans:
  - Quality reads a complete audited IFEval run of the plan's exact GGUF.
  - Fastest reads the median `llama_bench_v2` tg128 of at least three calls of
    the plan's own configuration (pp512 + tg128, 5 repetitions, `-fa off`,
    threads = physical cores), on the same machine, GPU driver/CUDA, device and
    offload. A pair moves only when the observed ranges do not overlap.
  - Balanced compares each pair directly: first when no axis is clearly worse
    and one is clearly better (quality by 95% Wilson interval). Overlapping
    speed lets clearly better quality decide; a real trade-off never moves.
  - Memory has no measured policy.
  Measured plans only swap places with each other inside one comparable cohort
  and stratum; without applicable evidence the order is the base order.
  `llama_bench_v1` records remain references only.
- **Evaluate candidates follows the search priority**: IFEval for Quality,
  Speed (no Docker) for Fastest, both for Balanced, running only the axis each
  candidate is missing. Evaluator setup is automatic; downloads need explicit
  permission. Recompare proposes, from stored quality and speed, what a search
  would change. Free disk space deletes downloaded models and run folders while
  keeping the records.
- The quality store preserves repeated runs; identical repeats count as one
  result and differing repeats are a conflict, never an applicable score.

## Checks and evidence

- The CI checkpoint includes Linux and Windows tests, static checks, and packaging.
  It does not substitute for GPU tests.
- On Intel Arc Pro B60, the KV and recurrent-state buffer predictions agreed with
  llama.cpp's reported allocations in the tested 1/2/4-slot configuration.
  A separate native Windows/Vulkan run exercised Serve, idle sleep and wake-up.
  These results are specific to the tested model, build and workload; they do not
  validate general VRAM availability or sustainable concurrency.
  See [the B60 experiments](docs/TFG/concurrencia-b60.md).
- Qwen3.6-35B-A3B (UD-IQ4_XS) on the B60, native Windows, llama.cpp b11889, one run each:
  - The KV cache plus recurrent state Jaull predicted matched llama.cpp's buffers
    at 32K, 128K and 256K of context, and 1, 2 and 4 people at 32K each.
  - With the server started from Jaull's TUI, `-lm dio` kept the process at 0.7 GB
    resident while loading. Without it the same load took 12.2 GB resident and left
    0.5 GB of RAM free.
    Generation stayed at 61 to 65 tok/s over 7 requests with up to 14,859 tokens of
    context. The sleep took the VRAM to zero and the first message woke it.
  - IFEval, strict prompt level, 541 prompts: 0.824, tied with the dense Qwen3.8 27B
    (0.821) inside the error, at about a quarter of the time per prompt. See
    [the IFEval report](docs/ifeval-b60-prova.md).
  - As a coding agent in opencode, 6 of 6 small tasks pass their tests, and 3 of 6 are
    clean after ruff, mypy and a read of the patch. One attempt per task. See
    [the mini-benchmark](docs/TFG/mini-benchmark-agentic.md).
  - opencode on WSL2 reaches the server through the PC's network address. With the
    self-signed certificate it needs `NODE_EXTRA_CA_CERTS` (a `/mnt/c/...` path).
- Memory limits by context on the B60 (native Windows, llama.cpp b11889), 14 cases of
  Qwen3-Coder 30B-A3B, Qwen2.5 32B and Qwen3.8 27B, with the predictions written before
  measuring. One run per case; the three repeated ones came out identical:
  - Jaull's verdict matched llama.cpp's own rule (1 GiB left free) in 10 of 14 cases.
    The 4 misses were cases Jaull rejected that llama.cpp accepted. It never approved
    a case that did not fit.
  - The KV cache, including the hybrid 27B's, matched exactly. The total sits 0.3 to
    1.0 GiB above llama.cpp's projection, because weights count at file size and compute
    is a flat 256 MiB. That leaves Jaull 6 to 14 % short of the real context limit
    (about 3,100, 1,850 and 16,300 tokens for the three models, interpolated).
  - Cases that did not fit were stopped at llama.cpp's fit step and never loaded.
    Generation was tested with a 16-token prompt in those cases. A later run on one case
    (Qwen3-Coder, `-c 49152`) filled the context up to 43,543 tokens: no step failed, and
    generation fell from 96 to 33 tok/s while the prompt took 5.5 minutes.
  See [the results](docs/TFG/resultados-b60.md) and
  [the raw logs](validation/b60-contextos-20261124/README.md).
- RTX 2060 campaigns (2026-10-10/11), all three plans ranking-ready, tg128
  median over all stored calls: Qwen2.5-0.5B Q5_K_M 246.1 tok/s (6 calls),
  IFEval 25.9 %; Qwen2.5-1.5B 150.7 tok/s (4), 41.0 %; LFM2.5-1.2B 230.5 tok/s
  (4), 80.2 %. Balanced keeps the Qwen pair as a trade-off. See
  [the plan](docs/fastest-balanced-plan.md) and
  [the noise campaign](docs/qwen2.5-tests/rtx2060-tg128-noise-20261011.md).
- An offline tg128 predictor (not in the product) was frozen before measuring four
  new RTX 2060 artifacts. Its observed errors were −6.1 % to +10.5 %, but the audit
  found a WSL kernel change between calibration and measurement: this is exploratory
  cross-environment evidence, not validated accuracy within a comparable cohort.
  The corrected scorer rejects the environment mismatch without rewriting the originals. See
  [the report](docs/qwen2.5-tests/rtx2060-speed-predictor-20261014.md).
- Results distinguishes applicable measured speed from the latest metadata-matched
  Transformers benchmark (historical, not ranking evidence). Without either, there
  is no speed label; the Speed shortcut remains available. Predictions were withdrawn
  after a Transformers case measured 18.36 tok/s against roughly 170 estimated.
  The offline predictor and original experiments remain available for the TFG;
  other-condition benchmarks stay historical references in the details.
  See `docs/recommendation.md`.
- The RTX 4060 campaign has a successful Validate and a complete pp512/tg128
  benchmark. [The comparison report](docs/qwen2.5-tests/rtx4060-campaign-comparison.md)
  explains the RTX 2060/4060 results and their limits.
- Public RTX 4060 evidence is anonymized. Original bytes are kept privately;
  see [the evidence index](validation/README.md). Old failed runs remain failures.

## What is still missing

- Speed noise between calls is 4-14 % on the RTX 2060
  ([campaign](docs/qwen2.5-tests/rtx2060-tg128-noise-20261011.md)). The min-max
  range only grows, so one atypical call can keep two plans indistinguishable
  indefinitely; a robust range is a later improvement.
- Capability estimation without measurements still uses a scale prior.
  Published catalog results are references only.
- No automated qualification verdict or sustainable capacity estimate. The
  concurrency experiments on the RTX 2060 and B60 are evidence, not a product gate.
- NVML process VRAM is unavailable on the measured WDDM machines.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs. No
  constant was changed after the B60 context measurements.
- The B60 context limits come from one GPU on one day. The full-context run covers one
  case and did not measure VRAM during the run; sustained use over hours and other cards
  are not measured.
- The prompt cache grows in RAM while the server is used (the process went from 1.3 to
  5.6 GB and free RAM fell to 3.7 GB). Its ceiling is computed once, at start, from the
  RAM free at that moment, so it does not adapt if other programs use RAM later.
- One run saw the VRAM drop to zero and the free RAM to 0 about 67 seconds after loading,
  with the server awake. It did not repeat and its cause is unknown.
- Under WSL2 the idle sleep is not applied and did not protect the PC. A shared server
  that Windows runs and WSL reaches needs the firewall and the certificate set by hand:
  a "This PC and WSL" sharing mode is designed but not built.
- Serve was tried on three machines (a CPU laptop, an RTX 2060 PC and the B60 PC). The
  B60 sweeps with 4, 8 and 16 people were started by hand with the configuration Jaull
  proposes. The run Jaull started itself had 2 people.

## Next steps

1. Repeat the native Windows B60 Serve run with Jaull starting the server for several
   people, and compare prompt cache on/off with the same workload, recording effective
   flags and minimum free system RAM.
2. Decide whether to put several models behind one address (llama.cpp's router mode or
   llama-swap). Neither has been tried with Jaull's flags.
3. Consolidate the TFG evidence by separating tested code, physical measurements,
   and unsupported or unvalidated configurations.
4. Repeat the B60 context set on another GPU, and repeat the full-context run with VRAM
   recorded during it (`scripts/b60/` has the scripts).

Historical audits remain in [`docs/history/`](docs/history/).
