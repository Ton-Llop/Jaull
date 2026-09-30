# Validation evidence index

These are retained experimental artifacts, not fixtures to regenerate during ordinary
development. Narrative reports and methods live in [`docs/qwen2.5-tests/`](../docs/qwen2.5-tests/);
the cross-machine procedure is [`docs/experimental-validation.md`](../docs/experimental-validation.md).

## Public anonymization

The RTX 4060 campaigns are now **public anonymized derivatives**, not byte-exact
original exports. Account, hostname and Windows user-directory strings use neutral
placeholders. Measurements, artifact/runtime binary SHA256 values, record IDs and
observation timestamps are unchanged. Each campaign has a `PUBLICATION.md` notice;
[the publication ledger](public-anonymization.json) links original and public file
digests. Collection-time statements about unchanged bytes refer to the originals.

Original files are preserved locally in the Git-ignored
`.cache/evidence-originals/20260930-publication/validation/` directory. Back up this
directory privately before clearing caches; it is not distributed with the repo.
The three previously valid bundles have derivative manifests for their public bytes.
The two historically invalid initial exports remain invalid: their manifests were
not repaired. Git history has not been rewritten. GPU identity remains available
for evidence matching; this is targeted publication cleanup, not full de-identification.

## Current reference: RTX 4060

Use [the final full-offload campaign](rtx4060-full-offload-benchmark-20260930-164332/README.md)
and its [portable bundle](rtx4060-full-offload-benchmark-20260930-164332/bundle/).
It groups a successful Validate and a complete pp512/tg128 benchmark (three
repetitions), same artifact SHA and llama.cpp b11258. Bundle status is `valid`,
with warnings about different available RAM/VRAM. This is execution and throughput
evidence, not a calibration or deployment qualification.

The [campaign comparison](../docs/qwen2.5-tests/rtx4060-campaign-comparison.md)
compares it with the 28/29 control and the RTX 2060 reference, preserving the
methodology limits. The RTX 2060 results below remain references in their own
right; a newer 4060 run does not replace their placement/context measurements.

## RTX 4060 historical archive

Archive is a classification, not a directory move: stable public paths are retained.
Original bytes are stored privately as described above.

| Evidence | Status / reason to keep |
|---|---|
| [28/29 control](rtx4060-cnight-20260930-112007/README.md) | Valid bundle; comparison control, with historical batch omission documented. |
| [Fresh Validate only](rtx4060-fresh-validate-20260930-135047/README.md) | Valid bundle; semantic batch fix and planning provenance. Its Validate is also in the final case. |
| [Initial b10357 campaign](rtx4060-b001-r10-20260929/REPORT.md) | Historical: incomplete/timeout benchmarks, not generation evidence. Transported `bundle-final` still fails integrity validation; never repair its manifest to hide that. |
| [Runtime missing](rtx4060-validate-repeat-20260930-121225/REPORT.md) | Preparation failure; no executed Validate. |
| [Helper mismatch](rtx4060-validate-repeat-20260930-121948/REPORT.md) | Runtime/prediction mismatch; no persisted experiment record. Keep the diagnostic, do not reconstruct a result. |
| [Explicit batch=1 failure](rtx4060-validate-repeat-20260930-122455/FINAL-REPORT.md) | Real failed Validate and log; regression evidence, not OOM/calibration evidence. |

## Retention policy

- Keep one self-contained, validated bundle per scientifically distinct case
  plus a short report. An additional export is justified only by new evidence
  or unique provenance, not as another backup copy.
- Keep original records immutable and unique runtime logs/failed outcomes.
  Do not regenerate or rewrite historical manifests or measurements.
- External copies already covered by an intact, valid bundle can be removed
  after byte/hash checks; review with `bundle/evidence/` afterward. Collection
  helpers and historical summaries may retain the original machine-local paths.
- Retain helpers that explain otherwise undocumented collection methodology;
  do not accumulate temporary execution scripts or session bookkeeping in future
  campaigns. Keep models, runtime binaries and caches outside versioned evidence.
- Logical archive status avoids breaking source links. No evidence is removed
  solely because it is old or a newer run succeeded. Logs/records contain local
  identifiers and model output; decide what to publish before sharing them.

The 2026-09-30 cleanup removed only proven external duplicates and redundant
bookkeeping. Bundle contents, original records, RTX 2060 matrices/sweeps and
failed-run diagnostics were left intact. Bundle integrity was rechecked afterward.

## B001: Qwen2.5-7B Q4_K_M on RTX 2060

The [1/2/4 concurrent-request pilot](2060-1vs2vs4-users/README.md) uses one
llama-server with four fixed context slots. Its native streaming measurements
are separate from Validate records and llama-bench; they are not qualification
or calibration evidence.

| Evidence | Contents |
|---|---|
| [`b001-r2-qwen2.5-7b-q4km-rtx2060-ctx4096/`](b001-r2-qwen2.5-7b-q4km-rtx2060-ctx4096/) | Initial device-memory observation. |
| [`b001-r3-qwen2.5-7b-q4km-rtx2060-ctx4096/`](b001-r3-qwen2.5-7b-q4km-rtx2060-ctx4096/) | Follow-up device-memory observation. |
| [`b001-r4-qwen2.5-7b-q4km-rtx2060-ctx4096/`](b001-r4-qwen2.5-7b-q4km-rtx2060-ctx4096/) | Baseline placement and memory evidence. |
| [`b001-r8-2060-context-matrix/`](b001-r8-2060-context-matrix/) | Context and offload-level matrix, including captured logs and reports. |
| [`b001-r9-launch-policy-throughput/`](b001-r9-launch-policy-throughput/) | Launch-policy and throughput runs. |
| [`b001-r10-experiment-record/`](b001-r10-experiment-record/) | Persisted experiment record and compact summary; narrative: [B001-R10](../docs/qwen2.5-tests/b001-r10-experiment-record.md). |
| [`qwen2.5-7b-q4km-2060-ctx4096/`](qwen2.5-7b-q4km-2060-ctx4096/) | Baseline prediction, report and run logs. |
| [`qwen2.5-7b-q4km-2060-ctx4096-sweep/`](qwen2.5-7b-q4km-2060-ctx4096-sweep/) | Offload sweep prediction, reports, logs and timing captures. |

## Portable bundles

- [`bundles/b001-r4-full-offload/`](bundles/b001-r4-full-offload/) and
  [`bundles/b001-r4-launch-policy/`](bundles/b001-r4-launch-policy/) contain case metadata,
  records and linked validation evidence.
- [`bundles/b001-r6-tensor-policy/`](bundles/b001-r6-tensor-policy/) contains prediction,
  launch smoke test and benchmark outputs.
- [`bundles/b001-r6-tensor-policy-frozen/`](bundles/b001-r6-tensor-policy-frozen/) preserves
  the frozen prediction used for that comparison.

Keep the records immutable. Device-wide memory, runtime-reported buffers, NVML process
allocation and host RSS are distinct observations; only compare quantities when the
methodological gates in the report permit it. No RTX 4060 or rented-Linux result is implied
by this RTX 2060 evidence.
