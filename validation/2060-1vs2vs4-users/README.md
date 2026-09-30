# RTX 2060: 1 / 2 / 4 concurrent requests

## Protocol

Local pilot using one CUDA llama-server and the historical Qwen2.5-7B-Instruct
Q4_K_M artifact. This is raw workload evidence, **not a Jaull Validate record,
deployment qualification or memory calibration**.

- One model load; four slots throughout; 4096 context tokens per slot.
- Fixed partial offload: 24 llama.cpp launch units; four CPU threads.
- 512 input token IDs, 128 generated tokens; native streaming completions.
- Synthetic repeated prompt, greedy decoding, EOS ignored, prompt cache disabled.
- Warm-up with four simultaneous requests, excluded from measured results.
- Three repetitions, balanced orders: 1/2/4, 4/1/2, 2/4/1.
- Client barrier starts each group together; slot samples verify actual overlap.
- TTFT is client time to the first generated token ID, including server queueing.
- Aggregate output throughput includes prompt processing and the whole group wall time.
- Runtime generation throughput is recorded separately from aggregate throughput.

`metadata.json` freezes hardware, driver, commands, runtime/library hashes,
source commit and collector hash. The model SHA is verified before execution;
its historical repository revision is unknown and stays null.

`run.py` collects the pilot and refuses to overwrite an existing run.
`check.py` verifies the streaming collector offline:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 python validation/2060-1vs2vs4-users/check.py
```

Raw request files retain streaming events and results. `server.log` preserves
runtime diagnostics. Device-wide NVML usage, process-attributed NVML usage,
server RSS and runtime-reported buffers are different measurements. Under WDDM,
process-attributed usage may be unavailable; device-wide usage is not its replacement.

## Results

Run on 2026-09-30, WSL2 / Ryzen 5 3600 / RTX 2060 6 GiB, driver 617.14.
llama.cpp b10357 (`689e227db`), freshly compiled from clean historical source;
Jaull source `34cd7fd`, with this collection helper and evidence index uncommitted.

All **21 measured requests succeeded**, plus four excluded warm-up requests.
Every measured request processed 512 prompt tokens, reused zero cached tokens
and returned all 128 generated token IDs. Slot samples confirmed 1/2/4 active
slots for every corresponding group; maximum client launch skew was 1.121 ms.

Values are means +/- sample standard deviation across three group repetitions.
For per-request metrics, each group contributes its mean, not an independent
sample for each simultaneous request.

| Active users | TTFT per request (s) | Generation per request (tok/s) | Aggregate output (tok/s) | Request latency (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.595 +/- 0.049 | 15.71 +/- 2.43 | 14.64 +/- 2.19 | 8.88 +/- 1.42 |
| 2 | 1.103 +/- 0.023 | 13.53 +/- 0.63 | 24.19 +/- 1.03 | 10.58 +/- 0.47 |
| 4 | 2.009 +/- 0.216 | 8.85 +/- 0.41 | 31.01 +/- 1.35 | 16.50 +/- 0.72 |

**Interpretation:** more simultaneous requests improved total throughput in this
pilot, but slowed each request and increased TTFT. Four users gave about 2.12x
the single-user aggregate throughput, not 4x. This is the trade-off the workload
experiment should expose; it is not a measured production-capacity guarantee.

## Memory and Limits

- Peak server RSS over the whole run: 4.61 GiB, during loading. Measured groups
  peaked near 1.49 GiB after loading; RSS is not a model of the host placement.
- Peak device-wide NVML usage over the run: 5.92 GiB. It includes desktop/other
  processes and is **not process-attributed VRAM or model allocation**.
- NVML supplied no usable process-attributed samples. This remains unavailable
  on the local WDDM setup.
- `runtime-allocation.json` is null: default verbosity did not expose a complete
  buffer report. The shutdown warning mentions a CUDA0 compute buffer of
  186.8281 MiB versus an expectation of 182.3281 MiB; this is retained in the log,
  not converted into a full allocation measurement or a Jaull prediction error.
- No SLO was supplied, so no qualification/pass/fail against performance targets
  is inferred. Three repetitions do not establish tail-latency guarantees.
- The context **capacity** was 4096 per slot; requests used only 512 + 128 tokens.
  This does not test four users each occupying the full 4096-token context.
- This was a short desktop/WSL pilot with synthetic prompts and no controlled
  background load. It is not comparable directly with full-offload llama-bench.
- Prompt batching used the historical runtime defaults (logical 2048, physical
  512); those are not the number of concurrent users. Flash attention used auto.

`summary.json` contains derived group means, variability and sampled peaks;
original group/request files remain unchanged. `check.py` also checks the
recorded token counts, overlap, summary arithmetic and collection-helper SHA.
`files.sha256.json` inventories retained files; it is not an ExperimentalCase bundle.
