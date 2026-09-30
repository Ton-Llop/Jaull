# RTX 4060 B001-R10 investigation

Audited on 2026-09-30 from `test-4060`, commit `f9dd639`. Source under test in
the original Windows campaign was `f14b3e9`. The original records, manifests,
gate logs and REPORT.md have not been rewritten. This investigation adds
prospective fixes and describes the controls still needed on the RTX 4060.

## What the recorded experiment establishes

Validate succeeded on native Windows with the expected artifact SHA-256,
llama.cpp commit `689e227db`, CUDA and 29 launch units. The persisted runtime
allocation is a runtime device-buffer observation, not driver-attributed process
VRAM. No generation-throughput result completed in either benchmark.

| Attempt | Process outcome | Captured measurements | Missing |
| --- | --- | --- | --- |
| `bench-491757e3` | exit 0, 73.588 s | pp128, pp512 | tg64 and build footer |
| `bench-12a23300` | timeout, 900.490 s | pp128, pp512 in raw stdout | tg64 and build footer |

The first record's historical `success=true` is not sufficient evidence of a
complete benchmark. Do not use it as generation-performance evidence. Preserve
the record as recorded, with this caveat rather than editing its observation.

## Confirmed Jaull defects and fixes

1. `LlamaBenchRunner.run` accepted any parsed row after a successful process
   exit. It did not compare parsed measurements with the request. It now checks
   every requested pp/tg token size and returns a failed `PARSE_ERROR`
   observation for incomplete output. The matrix persists that failure and its
   raw output. A prefill row or a different tg token size cannot satisfy a
   requested generation measurement.
2. The matrix tried `--version` without its existing empty-workload fallback.
   When the binary rejected `--version` and the actual run did not reach its
   footer, the record lost its build. The matrix now enables the existing
   bounded fallback before execution. An unsuccessful fallback still leaves the
   build unknown; no version is inferred from a filename or another binary.
3. The bundle exporter wrote JSON through platform-default newline translation.
   Windows exports therefore had CRLF, but Git normalized those bytes to LF.
   New exports explicitly write LF. `.gitattributes` now disables text
   conversion for experimental evidence under `validation/`.
4. The global `logs/` ignore excluded both the campaign's runtime sidecars and
   the copied evidence inside bundles. Scoped exceptions now permit those
   directories and `.runtime-log` files under `validation/`.
5. Five Ruff violations belonged to the campaign's helper scripts. They have
   been corrected without changing recorded evidence.
6. The failing Details test used CPU-idle pauses as navigation/composition
   completion signals; the wizard test scrolled before confirming layout and
   used a deferred scroll. These tests now wait for mounted screens, the
   relevant content/layout and the actual scroll result. Their assertions are
   preserved. A supplementary run with 15.625 ms process-clock granularity
   passed all seven relevant TUI tests on Linux. Native Windows confirmation
   remains necessary; this is not a Windows hardware measurement.

## The received historical bundle is incomplete

Revalidating `validation/rtx4060-b001-r10-20260929/bundle-final` in this clone
fails with `Bundle file size differs: case.json`. Checking every file proves
the newline transformation:

| Bundle file | Manifest bytes | Received bytes | Restoring CRLF reproduces SHA-256 |
| --- | ---: | ---: | --- |
| case.json | 1704 | 1648 | yes |
| records/experiment.json | 42146 | 40874 | yes |
| benchmark `12a23300` | 12505 | 12209 | yes |
| benchmark `491757e3` | 12531 | 12208 | yes |

All three declared evidence logs are missing from this checkout. The local
validation result recorded on Windows applied before this Git transfer; it
does not certify the transported directory. Prospective fixes do not repair
these historical files. Retrieve the original Windows bundle as an archive,
verify its original manifest at the destination, and preserve that archive as
the original evidence. Do not substitute new hashes to make altered files pass.

The two benchmark sidecars can be reconstructed byte-for-byte from their
records' raw output using the original helpers' serialization: both calculated
SHA-256 values match the manifest. The Validate sidecar is not present in its
record and still needs retrieval. No sidecars have been synthesized or restored
as part of this investigation.

## What remains unknown about the runtime stall

The output stops after pp512. This bounds the stall to subsequent cleanup,
generation-context initialization, generation warmup or generation execution.
Without progress traces it does not identify which of these stopped.

Jaull drains stdout/stderr with `subprocess.communicate` while sampling memory;
the existing backend test covers simultaneous multi-megabyte output. The two
benchmark outputs here are small. There is no demonstrated pipe deadlock.
However, neither original direct-control attempt started the runtime: both
failed with `FileNotFoundError` after the binary disappeared. They therefore
do not exclude a wrapper/environment contribution to the 4060 failure.

The missing executable and denied restoration are also unexplained. The logs
do not establish Defender quarantine, an ACL cause or a OneDrive cause. Do not
disable security controls or change filesystem permissions to test a guess.

CUDA graphs are a hypothesis worth testing only after a direct reproduction.
The pinned runtime supports both
[progress/verbose diagnostics](https://github.com/ggml-org/llama.cpp/blob/689e227db485c6b33d061555e74034c93a867649/tools/llama-bench/llama-bench.cpp)
and
[GGML_CUDA_DISABLE_GRAPHS](https://github.com/ggml-org/llama.cpp/blob/689e227db485c6b33d061555e74034c93a867649/ggml/src/ggml-cuda/common.cuh).
This is not evidence that CUDA graphs caused this campaign's stall.

## Next control on the RTX 4060

Use the restored, hash-verified b10357 executable and its matching DLLs. Record
their paths and hashes again. A runtime directory outside OneDrive can be an
explicit filesystem control; moving it changes a variable and is not itself
a diagnosis. Verify the model SHA-256 before running.

The existing diagnostic helper now accepts an explicit executable and a fresh
output path, adds progress/verbose traces and enforces a bounded timeout. It
retains partial output on timeout and never writes a BenchmarkRecord.

```powershell
uv run --python 3.12 python validation/rtx4060-b001-r10-20260929/diagnose-benchmark.py --llama-bench "PATH_TO_VERIFIED_LLAMA_BENCH.exe" --output validation/rtx4060-b001-r10-controls-20260930/direct.runtime-log --timeout 120
```

This repeats the captured `-p 128,512 -n 64 -r 3 -ngl 29 -dev CUDA0` command
outside Jaull's execution backend, adding diagnostic logging. The 120-second
cutoff is a diagnostic limit, not the original benchmark's 900-second limit.
Inspect the last progress message to locate the stalled stage. A direct success
would motivate comparing the wrapper and process environment; a direct timeout
would show that the TUI is not required for the symptom.

If the direct command stalls during generation, repeat it with
`GGML_CUDA_DISABLE_GRAPHS=1` scoped to that invocation and a different output
path. Restore the original environment afterward. Retain both outcomes; do not
silently apply this workaround to Jaull or treat the diagnostic as a benchmark.
Then isolate generation with `-p 0 -n 64`, changing only this argument, if the
progress trace still leaves the stage unclear. Further GPU/CPU or build controls
should follow the trace rather than changing several flags at once.

Increasing or removing the timeout is not a demonstrated fix. A completed
generation measurement, a fresh native-Windows gate run and retrieval of the
original byte-exact bundle are still needed to close the campaign.

## Verification of these changes

| Check | Result in this WSL/Linux workspace |
| --- | --- |
| Full pytest suite, final tree | 1761 passed in 123.64 s; 7 new test cases |
| Ruff | pass, including campaign helpers |
| mypy src | pass, 237 source files |
| compileall | pass, src and campaign Python helpers |
| Architecture tests | all 4 pass; allowlist remains empty |
| git diff --check | pass |
| TUI with coarse process clock | 7 passed; not native-Windows certification |
| Historical-behavior checks | incomplete-output, build and newline regressions all detected |

The historical bundle still fails validation in this checkout, as documented
above. Its original manifests and records remain unchanged. All source,
helper, test and documentation changes are unstaged; no Git commit, push or
branch change was made during this investigation.
