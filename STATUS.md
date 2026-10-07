# Status

Updated 2026-10-07. Code checkpoint: `b3edb99` on `docker-models-comp`.
PR [#25](https://github.com/Ton-Llop/Jaull/pull/25) is merged.

## What works

Jaull detects hardware, recommends models and prepares an execution plan.
It verifies the artifact, runs the model, saves experiments and benchmarks,
and compares predictions with measurements when the evidence allows it.

- CLI and TUI use the same services.
- Recommendations and execution readiness are separate: missing dependencies
  block execution, not the recommendation.
- Task and workload mode are separate. Workload requirements and optional
  throughput/TTFT targets can be recorded; they are not measured results.
- New llama.cpp plans no longer translate KV sequence batch size into
  `--batch-size`. Token batching uses runtime defaults; explicit old flags remain.
- Artifact verification rejects incorrect downloads and removes stale SHA
  sidecars before replacing files.
- Quality evaluation is an explicit, sequential GGUF pilot in the CLI and TUI.
  Results separate publisher references from measurements of exact artifact
  bytes; neither changes recommendation ordering.
- The quality store preserves repeated completed runs. Reuse requires one
  complete, protocol-matched run; diagnostics and ambiguous repeats are misses.

## Checks and evidence

- The historical [CI run for `8b128c6`](https://github.com/Ton-Llop/Jaull/actions/runs/36748809534)
  passed. It predates this branch's quality-evaluation changes.
- Local validation for this checkpoint is recorded below after the current
  branch's full gates have run.
- The RTX 4060 campaign has a successful Validate and a complete
  pp512/tg128 benchmark with three repetitions.
- [The comparison report](docs/qwen2.5-tests/rtx4060-campaign-comparison.md)
  explains the RTX 2060/4060 results and their limits. Different builds and
  environments mean this is not a controlled GPU comparison.
- Public RTX 4060 evidence is anonymized. Original bytes are kept privately;
  see [the evidence index](validation/README.md). Old failed runs remain failures.

## What is still missing

- Capability estimation still uses a scale prior. Published catalog results
  are diagnostic references only; all 11 entries lack an evaluated revision
  and precision and do not certify the artifact that runs.
- No qualification verdict yet. The 1/2/4 concurrent-user experiment is measured
  under `validation/2060-1vs2vs4-users/` and is deliberately not a priority.
- NVML process VRAM is unavailable on the measured WDDM machines. Runtime
  buffer comparisons work where the methodology gates allow them, but do not
  measure every process allocation.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs.

## Next steps

The priority is how Jaull estimates quality, not multiuser throughput.

1. Review and broaden the catalog cautiously for models that appear in Jaull's
   actual recommendation results, using primary sources and exact provenance.
2. Design a Quality/Fastest/Balanced policy for review before any ranking
   integration. Keep unknown and incomparable evidence visible.
3. Upgrade the limited pilot only after defining sample coverage and uncertainty;
   a small `--limit` run is not a model-quality verdict.

No HFA, ranking, scoring or calibration changes are part of this checkpoint.
Historical audits remain in [`docs/history/`](docs/history/).

## Current branch validation

- Current working-tree validation: **2,287 tests passed** on Python 3.12;
  Ruff, mypy (`249` source files), architecture checks (`4` passed),
  compileall and `git diff --check` passed.
