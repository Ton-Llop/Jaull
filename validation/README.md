# Validation evidence index

These are retained experimental artifacts, not fixtures to regenerate during ordinary
development. Narrative reports and methods live in [`docs/qwen2.5-tests/`](../docs/qwen2.5-tests/);
the cross-machine procedure is [`docs/experimental-validation.md`](../docs/experimental-validation.md).

## B001: Qwen2.5-7B Q4_K_M on RTX 2060

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
