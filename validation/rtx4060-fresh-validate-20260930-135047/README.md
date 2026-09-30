> Publication note (2026-09-30): this is an anonymized derivative. Statements below about original bytes/hashes describe collection-time evidence, not this public copy. See validation/public-anonymization.json and the campaign PUBLICATION.md.

# RTX 4060 fresh-plan Validate

Successful Windows measurement at source commit `1d12f63`: Qwen2.5-7B
Q4_K_M, context 4096, concurrency 1, llama.cpp b11258, CUDA, full offload.
No benchmark is attached to this historical Validate-only bundle. See
[REPORT.md](REPORT.md) for the interpretation. The subsequent
[final reference](../rtx4060-full-offload-benchmark-20260930-164332/README.md)
groups this same Validate with its compatible full-offload benchmark.

## Portable evidence

- `bundle/`: self-contained case, experiment and eight SHA-256 checked evidence
  files. Created offline using the existing Jaull case/bundle services; no GPU
  execution was performed during packaging.
- Case: `case-9d1e3d79-d183-4a12-870b-e38ca97f3a47`.
- Experiment: `exp-f05586ff-c90e-4239-821d-5baaf550bd2a`.
- Case and bundle validation: `valid`, no reasons or warnings.

Recheck from the repository root:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 jaull experiments case bundle validate validation/rtx4060-fresh-validate-20260930-135047/bundle --json
```

Bundle validity establishes internal consistency and file integrity, not complete
VRAM attribution or throughput qualification. NVML remains unavailable; runtime
buffers remain a separate observation source. Do not attach the earlier 28/29
offload benchmark to this 29/29 case. A future compatible benchmark requires a
new case/export; do not mutate this bundle.

## Retained originals

`records/` retains the original Windows store envelope and the new case envelope.
The original experiment bytes are unchanged (SHA-256
`40e1b0b7fac3553688d70c5690f15eaabe599a55f841018839934294b07c0d36`).
The portable export serializes the same domain record in the bundle format.

`bundle/evidence/logs/` retains the store sidecar; `logs/invocation.runtime-log`
retains the capture wrapper output. Their stdout/stderr match, but the capture
includes a distinct pre-processing observation, so it is retained rather than
treated as an exact duplicate.
`helper.py.txt` records how the original measurement was taken; it is historical
provenance, not an executable maintenance script.

The initial estimate, request, backend selection and summaries remain beside
the bundle. Execution plan, invocation, hash checks, runtime settings and
provenance are retained inside `bundle/evidence/`. Summaries contain original
machine-local paths/status and do not describe the current checkout. Logs and
records contain local identifiers and model output; they have not been anonymized.

## Removed redundant snapshots

The following JSON snapshots were compared structurally before removal:
`hardware.json`, `verified-artifact.json`, `analysis.json`,
`inference-configuration.json`, `prediction-before-validation.json`,
`runtime-capability.json`, `runtime-readiness.json` and the record portion of
`experiment-result.json` all match fields in the immutable ExperimentRecord.
`initial-execution-plan.json` matches the retained `bundle/evidence/execution-plan.json` exactly;
`prepared-execution-plan.json` only repeats that plan and the recorded artifact.
No unique measurement, original record or runtime log was removed. Older
campaigns and failed attempts are untouched.

Seven further external copies (store sidecar, runtime hashes, artifact hash
check, invocation, runtime settings, execution plan and provenance) were removed
after checking byte-for-byte equality with `bundle/evidence/`. The bundle and
original records are unchanged. Historical helpers/reports may still name the
collection-time paths; use the portable bundle for current review.
