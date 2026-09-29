# Status

Orientation only; the reasoning behind every line is in
[`docs/history/technical-audit-2026-09-28.md`](docs/history/technical-audit-2026-09-28.md) (revision V).
Commit `a285acf`, 2026-09-28.

## What works now

`predict → acquire → verify → execute → compare`, end to end, through a Typer CLI and a
Textual TUI over the same services:

- Detects CPU, RAM, storage, NVIDIA GPUs via NVML and others via `vulkaninfo` and DRM sysfs.
- Classifies a Hugging Face repository and reads the GGUF header over HTTP Range, metadata
  only, then estimates memory per component, each row carrying its own provenance.
- Turns six or seven plain questions into a ranking of **execution plans** on separate axes,
  ordered lexicographically. No global score, and no language model anywhere in it.
- Resolves one artifact, downloads it, verifies size and SHA-256, runs it through `llama-cli`
  or an isolated Transformers worker, stores it as an immutable `ExperimentRecord`, then
  compares the prediction against the observation — or states why it cannot.

## What is validated

Gates on `a285acf`: **ruff clean · mypy clean (245 files) · 1746 tests · 0 failures**, still
green with the whole suite under four CPU-saturating threads. Coverage 86 %, from 2026-09-24.

The B001 series, in `docs/qwen2.5-tests/` — RTX 2060, Qwen2.5-7B Q4_K_M, llama.cpp `689e227db`:

- **R4** — the first baseline carrying complete provenance.
- **R6** — tensor placement, read from the source and checked against a real run.
- **R8 + R9** — the launch policy costs **1.4×–2.1×**, not the 3–4× inferred from R8 alone.
- **R10** — a persisted record with eight runtime buffers, 2459.61 MiB on `CUDA0`.

## What is incomplete

- **The VRAM comparison still yields no number.** HFA bounds non-block weight placement
  instead of predicting a device-specific point, and that gate fires first. Checked on R10.
- **`peak_vram_bytes` is `null`** on a GeForce driving a display: under WDDM, NVML cannot
  attribute memory per process. A platform limit, not a code one.
- **`runtime_build` is never populated**, although the run's own log prints it.
- **`reserve` (512 MiB) and `headroom` (256 MiB) were never calibrated** — documented guesses.
- **`BEST_MATCH` is unreachable**: `overhead.py:39` emits `LOW` unconditionally.
- **The soft language check degrades hard**: an optional `language:` still costs 0.15 in
  `requirements_gate.py:118`, enough to force `BEST_EFFORT`.
- Minor: `advisor/service.py` 1570 lines; `prediction_input` optional; docs say "six questions".

## What comes next

1. **A runtime-specific point prediction for non-block weights** — the one thing between this
   project and a publishable VRAM error. The block ↔ unit mapping already landed.
2. **A product decision** on the soft language check. That one is not code.
3. Repeat R9 with real repetitions before touching `reserve` or `headroom`.
