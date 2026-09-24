# B001-R9: what the launch policy's offload level costs

## Why this exists

B001-R8 swept several `--n-gpu-layers` values per context and recorded that all
of them started. It did not report throughput, and read without it the sweep
invites the wrong conclusion: HFA predicts 15–20 transformer blocks while
llama.cpp starts happily with 29/29, so the gap looks like a 3–4x performance
loss if HFA's block count is mistaken for the launched level.

**It is not**, because the HFA block count is not what Jaull launches. The
launch policy emits its own number in its own units. That number had never been
measured. This run measures it.

No formula, reserve, safety margin, headroom or policy constant was changed for
this, and nothing here is used to calibrate one.

## Method

Same machine, artifact, build and invocation as B001-R8, so the numbers sit
beside its sweep: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` on the RTX 2060 (6144 MiB),
llama.cpp `689e227db`, `-n 32 --fit off -no-cnv -st -v`, prompt
`"Explain briefly what a local language model is."`.

The offload level came from the production path — `AdvisorService.estimate_model`
for each context, then the `--n-gpu-layers` flag on the resulting
`runtime_recommendation` — not from a hand-picked value. Available VRAM at
selection time was 4601 MiB.

Logs and the recorded selection are in
`validation/b001-r9-launch-policy-throughput/`.

## The launch policy does not emit the HFA block count

| Context | HFA placement | Launch policy | Difference |
|---:|---:|---:|---:|
| 512 | 20/28 blocks | `--n-gpu-layers 26` | +6 units |
| 2048 | 20/28 blocks | `--n-gpu-layers 26` | +6 units |
| 4096 | 19/28 blocks | `--n-gpu-layers 25` | +6 units |

These are different quantities in different units, which is the invariant the
architecture test protects. The practical consequence is that comparing the
B001-R8 sweep at the HFA block count does not measure the configuration Jaull
actually launches.

## Measured throughput

Generation tokens/second, from `eval time` in each run:

| Context | policy level | full offload | ratio | device MiB at full | free at full |
|---:|---:|---:|---:|---:|---:|
| 512 | 27.1 (`ngl 26`) | 39.9 (`ngl 29`) | **1.47x** | 4328 | 366 |
| 2048 | 27.9 (`ngl 26`) | 39.4 (`ngl 29`) | **1.41x** | 4414 | 271 |
| 4096 | 24.8 (`ngl 25`) | 52.0 (`ngl 29`) | **2.10x** | 4528 | 196 |

For contrast, the same contrast computed from B001-R8's sweep *at the HFA block
count* gives 3.2x–4.4x. The table above instead measures the level selected by
Jaull, once per cell; its ratios are observations, not a stable performance
estimate.

## Reading

**The observed cost is substantial but uncertain.** In these runs, following
Jaull's recommendation was 1.41x–2.10x slower than full offload. A single
repetition per cell cannot establish the typical cost on this machine.

**What buys that back is margin.** At full offload the device was left with
196–366 MiB free out of 6144. That free figure already accounts for the desktop's
baseline occupancy at measurement time; comparing it directly with B001-R8's
930–1359 MiB baseline would count that occupancy twice. The policy's reserve
(512 MiB) and headroom (256 MiB) rule out these configurations. Additional
desktop load could exhaust the observed free space, but this run did not test
that failure threshold.

**What is not established** is whether 512 + 256 MiB is the right amount. It has
never been calibrated; it is a documented assumption. This run does not calibrate
it either, and deliberately so: three single-shot measurements on one machine,
one model and one build are not a basis for moving a safety constant. The
ctx4096 full-offload figure (52.0 t/s against 39.4 and 39.9 at smaller contexts)
is itself a reminder of the noise floor — a larger context should not be faster,
and that spread is run-to-run variation, not a trend.

## Limitations

- One repetition per cell. No warm-up, no median over runs.
- Generation of 32 tokens is short enough that load time and clock ramp matter.
- The free-VRAM figure moved between 4601 and 5004 MiB across the session, and
  the policy reads it at selection time, so the emitted level is not stable
  across runs on a machine with a display attached. B001-R8 recorded the same
  effect on the HFA side.
- Prompt-side throughput was captured but is not reported: at 39 prompt tokens
  it is dominated by fixed cost.

## What would close this

A calibration campaign for `DEVICE_RESERVE_DEFAULT_BYTES` and
`LLAMA_CPP_HEADROOM_BYTES` with repetitions, more than one GPU, and the display
load recorded. Until then the conservative default stands, and this document is
the measurement of what it costs rather than an argument to change it.
