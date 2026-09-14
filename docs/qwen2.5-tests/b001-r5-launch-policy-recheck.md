# B001-R5: RTX 2060 launch-policy sensitivity probe

A re-measurement of the B001-R4 scenario (Qwen2.5-7B-Instruct Q4_K_M, RTX 2060,
llama.cpp `689e227db (10357)`) to measure whether increasing the requested
offload units beyond the conservative `--n-gpu-layers 24` is worthwhile. This
is a backend-specific sensitivity probe, not evidence that a generic policy may
translate transformer-block counts directly into llama.cpp units.

All three placements ran in a single `llama-bench` invocation, so they share one model load
and one set of conditions. The card started at 1283 MiB of 6144 MiB already in use by the
desktop.

```
llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 24,28,99 -p 512 -n 128 -r 3
```

| `-ngl` | | pp512 (tok/s) | tg128 (tok/s) |
| --- | --- | ---: | ---: |
| 24 | policy before | 1170.13 ± 64.47 | 20.60 ± 2.12 |
| 28 | policy after | 1467.40 ± 23.71 | **44.86 ± 0.50** |
| 99 | everything | 1556.90 ± 85.68 | 59.99 ± 0.28 |

## What it says

**Requesting 28 instead of 24 is worth 2.18× on generation** — 20.60 to 44.86 tok/s.
Prompt processing gains 25%. This establishes a useful runtime-specific target
for further validation; it does not itself establish a general launch rule.

**It does not recover everything.** Full offload still reaches 59.99 tok/s, another 1.34×.
The existing Qwen sweep shows that llama.cpp offloads its output layer even at
partial positive `-ngl` values, so this remaining difference must not be
attributed mechanically to that tensor. It is a backend-specific combination
of offload units and tensor placement that Jaull does not map yet.

The remaining 1.34× is therefore a calibration and backend-mapping question,
not evidence that generic HFA transformer blocks equal llama.cpp units.

## Cross-check against B001-R4

The two runs are days apart, on a busy desktop, and agree:

| | B001-R4 | B001-R5 |
| --- | ---: | ---: |
| `-ngl 24`, tg128 | 21.50 | 20.60 ± 2.12 |
| full offload, tg128 | 59.36 | 59.99 ± 0.28 |

That reproducibility is itself a result: the scenario is stable enough to measure a policy
change against.

## Caveats

- Three repetitions, one session, one machine. It shows the direction and the rough size of
  the effect, not a calibration-grade figure.
- `llama-bench` does not take a context size, so these numbers are not "at 4096 tokens".
  They are pp512 and tg128 microbenchmarks, which is what the tool measures.
- The desktop was running its usual applications throughout. That is the honest condition
  for this machine, but it is not a controlled environment.
- Raw output was kept only at `/tmp/b001-r5/bench.txt` and is not preserved in
  the repository. This probe must not be promoted to a baseline or calibration
  record until its raw output, command and provenance are captured in a case
  bundle.
