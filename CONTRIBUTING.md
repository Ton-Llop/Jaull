# Contributing to Jaull

Thanks for taking a look. Jaull is a small open-source project developed alongside an
undergraduate end-of-degree project, so it is not a large organisation with a heavyweight
process — but it does care about reproducibility and about not claiming more than the
evidence supports.

## What is most useful

### Run Jaull on hardware we do not have

This is the single most valuable contribution. Jaull makes claims about hardware detection,
backend availability, runtime compatibility, memory behaviour and prediction accuracy, and
those claims have been exercised on a narrow set of machines. Reports from other hardware
help us find where they break:

- NVIDIA GPUs (other generations, driver versions, multi-GPU)
- AMD GPUs (HIP, Vulkan)
- Intel GPUs
- Apple Silicon
- CPU-only systems
- Windows, macOS, Linux, WSL

Even a report that only says "detection got my GPU wrong" is useful. The fastest way to
produce one:

```bash
uv run jaull scan
uv run jaull doctor
```

and open a **Hardware validation report** issue with the output. If you get as far as
running or benchmarking a model, the prediction-vs-observation numbers are even more
valuable — those are exactly what the estimator needs to be calibrated against.

### Runtime and backend support

- new runtimes;
- backend detection and selection improvements;
- CUDA / HIP / Vulkan / Metal specifics;
- platform-specific fixes (Windows paths, macOS, WSL quirks).

### Benchmarks and experiments

- benchmark methodology and its reproducibility;
- additional or better metrics;
- experiment and benchmark record contents;
- anything that improves the prediction-vs-observation comparison.

If you change how something is measured, say so explicitly in the PR: a benchmark number is
only meaningful next to the methodology that produced it.

### Everything else

Bug reports, tests, documentation, small UX/TUI improvements, and model metadata problems
(a model that is misclassified, mis-licensed or badly ranked) are all welcome.

---

## Setting up

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Ton-Llop/Jaull.git
cd Jaull
uv sync
```

`uv sync` creates the virtual environment and installs the runtime dependencies plus the
`dev` dependency group (pytest, ruff, mypy) — uv installs default groups automatically, so
no extra flag is needed.

Optional, only for execution and benchmark work:

- a local `llama.cpp` build providing `llama-cli` and `llama-bench` — Jaull finds it on
  `PATH`, or under `~/tools/llama.cpp/build-*/bin` or `~/llama.cpp/build-*/bin`, or via the
  `--llama-cli` option. [docs/llamacpp.md](docs/llamacpp.md) describes one working setup;
- `vulkaninfo` for non-NVIDIA accelerator detection;
- `HF_TOKEN` for gated or private repositories.

## Running the checks

These are exactly what CI runs:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

CI additionally verifies the built artifacts, which is worth running if you touch packaging
or any non-Python resource:

```bash
uv build
uv run python scripts/check_dist.py
```

CI runs on Ubuntu (Python 3.12 and 3.13) and Windows (3.12), and installs with
`uv sync --locked` — so if you change dependencies you must commit the regenerated
`uv.lock`:

```bash
uv lock
```

## Writing tests

The suite must pass with **no GPU, no network and no TTY**, which is what lets it run
unchanged in CI. There is a test that enforces the no-network rule.

- The Hugging Face API, NVML, the HTTP Range client, `llama-cli` and `llama-bench` are all
  behind `Protocol`s — inject a fake rather than reaching for the real thing.
- GGUF fixtures are built programmatically (`tests/_gguf_fixtures.py`).
- The Textual UI is driven headless through `App.run_test()`.
- Report serialisation is snapshot-tested byte for byte against `tests/snapshots/`. If a
  change alters that output, update `REPORT_SCHEMA_VERSION` explicitly and regenerate the
  snapshot in the same commit.
- Screenshots are regenerated with `uv run python scripts/capture_screenshots.py`; commit
  the updated SVGs when a screen changes.

## Conventions

The codebase has a few rules that keep it auditable — a change that breaks one of them will
be asked about in review:

- **No magic numbers.** Every weight, threshold, budget and license rule lives in a
  `policies.py` module.
- **Provenance is part of the value.** An estimated number carries where it came from and
  how confident it is. Do not produce a bare figure that looks exact.
- **Prediction and observation stay separate.** `MemoryEstimate` is the prediction,
  `ExecutionObservation` is what happened, `PredictionComparison` is derived from both.
  Measurement never edits a prediction.
- **No bare `except Exception`.** Catch a specific type from `jaull.exceptions` or from the
  library involved.
- Python 3.12+ syntax: `X | None`, `list[str]`, no `typing.Union` / `Optional`.
- Domain models are frozen Pydantic v2 models and never mutate after construction.
- `domain/` imports nothing from layers above it; `discovery/` and `recommendation/` never
  import each other or `workflow/`; CLI and TUI never import each other.

[docs/architecture.md](docs/architecture.md) documents the layers and dependency rules in
full.

## Pull requests

- Branch off `master` and keep the branch small and focused.
- One main goal per PR. If you find an unrelated problem, open an issue or a separate PR.
- Add tests for behaviour changes when it is reasonably possible.
- Keep `ruff`, `mypy` and `pytest` green.
- Do not add a large dependency without explaining why it is needed and what it replaces.
- Explain any change to the estimation model, the ranking or the benchmark methodology —
  including what it does to existing numbers.
- Preserve reproducibility: deterministic ordering, pinned revisions, stable schemas, and a
  schema-version bump when a persisted record format changes.
- Describe what you actually verified. "Ran the suite" and "ran it on my 7900 XTX" are
  different claims and both are useful.

Commit messages: a short imperative summary is enough. English is preferred for new
commits, but the history is mixed and that is fine.

## Questions

Open an issue. For anything security-related, see [SECURITY.md](SECURITY.md) instead.
