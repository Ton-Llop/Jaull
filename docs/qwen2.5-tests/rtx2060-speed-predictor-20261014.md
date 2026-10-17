# Offline tg128 predictor on the RTX 2060 (2026-10-14)

An experiment, not a Jaull feature. Nothing in ranking or the UI reads it; Fastest still
orders only by measured `llama_bench_v2` calls.

**Question.** Can Jaull say roughly how many tokens per second a plan generates before
running it, without inventing a number?

**Audited answer.** Frozen predictions for four held-out artifacts differed from their
observed medians by −6.1 % to +10.5 %. However, calibration and measurements used
different WSL kernels. They do **not** share a Jaull speed cohort, so these numbers
are exploratory cross-environment observations, not validated same-cohort accuracy.
Two targets are other representations of Qwen models already in calibration, not
four independent unseen models. The 0.55 bandwidth baseline missed by +4 % to +388 %;
that baseline is not a run of llmfit's complete predictor.

## Audit correction (2026-10-14)

The original selector matched the GPU name and build number, but not the complete
environment. Reading all 26 source BenchmarkRecords found:

| Source | Calls | WSL kernel | Jaull speed cohort |
|---|---:|---|---|
| Frozen calibration | 14 | `6.18.40.1-microsoft-standard-WSL2` | `7779da3af56fc998824023f90ad8204f4183c29eabe80d00a5f53c3abc377b18` |
| Post-freeze measurements | 12 | `6.18.54.1-microsoft-standard-WSL2` | `4f527c6a9e6e21e2ccebb336e157c39b7ceceb341bd46bb1b60877b1eba4e54c` |

GPU, CPU, driver and runtime protocol match; kernel and OS version in the machine
fingerprint do not. This does not establish that the kernel caused a performance
change, or that the observations are useless. It invalidates the claim of a controlled
same-environment validation. The original frozen JSON, scores and log are unchanged.
[The derived audited scores](../../validation/speed-predictor-2060-20261014/scores-audited.json)
contain `complete: false` and no comparable post-freeze measurements: `missing` there
means missing **in the frozen cohort**, not that the twelve runs did not happen.
The corrected score command also exits unsuccessfully for incomplete validation,
after saving the diagnostic report; a successful JSON write is not a passed experiment.

The script now reuses `fastest_record_blockers` and `speed_cohort`, refuses to pool
cohorts, verifies calibration against its original record IDs, and requires three
distinct calls per validation artifact. `measure` stops before network or GPU work
when the current environment differs from calibration. It selects plans by exact
SHA rather than filename alone and refuses a partial target set. New freezes include
the cohort, hardware snapshot and script digest; all outputs use exclusive creation.
Zero/non-finite inputs and a non-positive predicted token time fail explicitly.

## Protocol

1. **Predict, then freeze.** `scripts/speed_predictor.py predict` read the stored
   calibration records and wrote
   [`frozen-predictions.json`](../../validation/speed-predictor-2060-20261014/frozen-predictions.json)
   at 2026-10-14 09:45 UTC (`2dd0eca`, working tree with the new script). It refuses to
   overwrite a frozen file.
2. **Measure.** `measure` built each target's plan through a search pinned to that exact
   GGUF (the product path, like `fastest_campaign.py`), confirmed full offload, and ran 3
   interleaved `llama_bench_v2` calls per plan with a 60 s cool-down before each.
   The request came from a local pool capture (not in the repo): English general chat,
   interactive, single user, Balanced, commercial use.
   [Log](../../validation/speed-predictor-2060-20261014/measure.log).
3. **Score.** `score` read only records created after the freeze, for the targets'
   SHA256s, and compared their median with the frozen numbers.
   [Scores](../../validation/speed-predictor-2060-20261014/scores.json).

Common recorded scope: RTX 2060 (WSL2, driver 617.42), llama.cpp build 10357,
`llama_bench_v2` settings, full offload and the manually selected dense models.
The WSL version differs as detailed above. The 7B campaign of B001-R6 is excluded:
it used llama-bench defaults and
3 repetitions, a different protocol.

## Predictors

With `B` = 336 GB/s ([NVIDIA specification](https://www.nvidia.com/en-gb/geforce/graphics-cards/rtx-2060):
14 Gbps GDDR6, 192-bit) and `S` = GGUF bytes:

- **Reference** `B / S`. Idealised: the file is not the bytes read from VRAM per token
  (an untied embedding table is in the file but only one row is read), so this is
  neither a ceiling nor a prediction.
- **0.55 baseline**, stored as `llmfit` in the original JSON: `0.55 × B / S`, inspired
  by llmfit's default efficiency. This is not the complete tool: its model-size inputs,
  corrections and optional measurements may differ.
- **Additive** `1 / (S / B + c)`. `c` = 2.383 ms per token, the mean residual of the
  three calibration plans. It is an empirical residual for this machine and protocol: it
  may absorb kernels, architecture or quantization, and is not shown to be a fixed cost.

## Calibration (fitted)

| Plan | GGUF bytes | Calls | Median tg128 | Additive, leave-one-out | Error |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B Q5_K_M | 522,186,592 | 6 | 246.1 | 258.2 | +4.9 % |
| LFM2.5-1.2B Q5_K_M | 843,354,944 | 4 | 230.5 | 193.4 | −16.1 % |
| Qwen2.5-1.5B Q5_K_M (Qwen) | 1,285,494,304 | 4 | 150.7 | 166.8 | +10.7 % |

Leave-one-out refits `c` on the other two plans. LFM2.5 is a different architecture from
the Qwen and Llama models and is the largest miss.

## Held-out Observations (Frozen Before Measuring, Different Kernel)

| Plan | GGUF bytes | Calls (tok/s) | Median | Reference | 0.55 baseline | Additive |
|---|---:|---|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | 105,454,144 | 359.2, 347.9, 394.3 | 359.2 | 3186 (+787 %) | 1752 (+388 %) | 370.8 (**+3.2 %**) |
| Qwen2.5-0.5B Q4_K_M | 491,400,032 | 274.3, 277.1, 288.6 | 277.1 | 684 (+147 %) | 376 (+36 %) | 260.0 (**−6.1 %**) |
| TinyLlama-1.1B Q4_K_M | 668,788,096 | 240.1, 233.7, 250.3 | 240.1 | 502 (+109 %) | 276 (+15 %) | 228.7 (**−4.8 %**) |
| Qwen2.5-1.5B Q5_K_M (MaziyarPanahi) | 1,125,050,592 | 158.0, 161.0, 144.3 | 158.0 | 299 (+89 %) | 164 (+4.0 %) | 174.5 (**+10.5 %**) |

Errors are `prediction / measured median − 1`. Mean absolute error: additive 6.2 %,
0.55 baseline 111 %, reference 283 %. These reproduce the original arithmetic;
they do not override the failed comparability gate.

## Reading

- **The fixed 0.55 baseline fails on these small artifacts.** The observed efficiency
  against `B / S` is not constant. This supports testing a residual, not a causal
  conclusion that model size alone explains the error or that all of llmfit fails.
- **Similar magnitudes do not prove accuracy.** SmolLM2's frozen prediction lies inside
  its observed range, the other three outside. Comparing prediction errors with
  between-call spread is descriptive, not a statistical test or a confidence interval.
- **Same model, different file.** The two Qwen2.5-1.5B Q5_K_M files differ by 160 MB and
  measured 150.7 vs 158.0 tok/s, under different kernels. The 166.8 calibration-table
  prediction is leave-one-out, with a different fitted residual from the 174.5 held-out
  prediction: those two values must not be presented as a single frozen predictor's ranking.
- **SmolLM2 is an extrapolation** below the calibration sizes and still landed at +3.2 %.

## Limits

- One GPU, one driver, one build, but **two WSL kernels**. A controlled validation must
  keep a predeclared cohort; any transfer to another environment is a separate question.
  `c` is not established as portable to another machine; the 4060 needs its own study.
- Every artifact is at most 1.3 GB. Larger models (the 7B) are untested under
  `llama_bench_v2`; the residual matters less there, but that is an expectation.
- Dense, full offload only. Partial offload and CPU depend on host memory bandwidth,
  which Jaull does not detect.
- Seven plans in total. "Error up to 10.5 %" describes these four plans, not a bound
  for the next one.
- tg128 only. Prefill and time to first token are not predicted; their bottlenecks
  and fixed costs need their own model.
- Source BenchmarkRecords are currently in the local user store, referenced by IDs,
  not bundled in this directory. Reproducing the protocol/environment audit requires
  those original records. The frozen file also predates script-hash capture: a dirty
  commit ID does not freeze the exact script bytes used. Do not fill in that provenance
  retrospectively as if it had been captured before the experiment.
- D079 says predictors derive from metadata and policy and measurements judge them.
  Fitting `c` from measurements departs from that, so any use in the product needs an
  explicit decision.

## Reproduce

```bash
uv run python -m scripts.speed_predictor predict --build 10357 --cohort <predeclared-cohort> \
  --out <new>.json --targets <repo_id>:<filename>
uv run python -m scripts.speed_predictor measure --frozen <new>.json \
  --capture <pool-capture>.json   # any capture whose input.answers define the request
uv run python -m scripts.speed_predictor score --frozen <new>.json --out <scores>.json
```

For the historical freeze, a new score output reports no compatible validation runs;
do not overwrite `scores.json` to hide that original result. A future campaign needs
anchors and held-out measurements in the same predeclared environment, or an explicitly
designed transfer experiment. No new GPU run was performed during this audit.
