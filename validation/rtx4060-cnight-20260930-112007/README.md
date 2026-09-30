> Publication note (2026-09-30): this is an anonymized derivative. Statements below about original bytes/hashes describe collection-time evidence, not this public copy. See validation/public-anonymization.json and the campaign PUBLICATION.md.

# Historical RTX 4060 control: 28/29 offload units

This is a retained placement control, not the current full-offload reference.
Validate and pp512/tg128 Benchmark succeeded with llama.cpp b11258. The Validate
command did not apply the historical plan's `--batch-size 1`; this limitation
remains recorded in [REPORT.md](REPORT.md).

The portable authority is [bundle/](bundle/). The original store envelopes in
`records/`, collection helpers, gate results and diagnostics remain unchanged.
Paths and Git states in the original reports/indexes describe the machine at
collection time, not the current checkout.

Ten external copies were removed only after byte-for-byte equality checks
against the valid bundle: the two runtime logs, prediction, execution plan,
hardware, runtime capability, benchmark capability, artifact hash check, Hub
baseline and workload intent. They remain under `bundle/evidence/` at the same
relative paths. The bundle's files and hashes were not modified. Original
collection helpers may name the old paths; they are historical provenance,
not a command to replay or overwrite this campaign.

The final cleanup removed four empty stderr files for case create, validate,
export and bundle validate, plus `gate-compileall.log` and `gate-diff-check.log`.
Those two logs only contained `EXIT_CODE: 0`; their commands and exit codes
remain in `gates-results.json`. The four command stdout JSON files are retained:
`case-commands.json` records commands and exit codes, not their output. No other
gate results, unique diagnostics or evidence files were removed in this pass.

Revalidate from the repository root:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 jaull experiments case bundle validate validation/rtx4060-cnight-20260930-112007/bundle --json
```

See the [comparison](../../docs/qwen2.5-tests/rtx4060-campaign-comparison.md) and
the [final reference](../rtx4060-full-offload-benchmark-20260930-164332/README.md).
Logs are versioned and contain local identifiers and model output.
