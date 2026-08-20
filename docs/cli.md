# Command reference

Six subcommands: `scan`, `inspect`, `estimate`, `doctor`, `run` and `ui`. All of them are
read-only except `run`, which explicitly downloads, verifies and executes a GGUF artifact.

```bash
uv run jaull                             # opens the TUI when run in a terminal
uv run jaull ui                          # interactive terminal UI
uv run jaull scan                        # local hardware and compute backends
uv run jaull doctor                      # environment, runtime and readiness checks
uv run jaull inspect <repo-id-or-url>    # analyse a model repository
uv run jaull estimate <repo-id-or-url>   # memory footprint + compatibility
uv run jaull run --model <repo> --prompt "Hello"   # download, verify and run a GGUF
```

Running `jaull` with no subcommand opens the TUI **only when stdin and stdout are both a
terminal**. Piped, redirected or CI invocations print help instead, so existing scripts are
unaffected.

The two front-ends share the same services and produce the same numbers:

- **CLI** — for automation, scripting and reproducible output. `--json` emits a stable
  schema you can pipe into other tools.
- **TUI** — a guided, visual flow. Every action shows the equivalent CLI command so you can
  lift it into a script.

Global option: `--verbose` / `-v` enables debug logging.

---

## `scan` — local hardware

```bash
uv run jaull scan
```

Reports OS, architecture, CPU model and cores, RAM totals, storage per mount point, any
NVIDIA GPU with VRAM and driver/CUDA versions, and the detected accelerators with their
per-backend availability (CUDA / Vulkan / HIP / CPU).

Non-NVIDIA adapters are discovered through `vulkaninfo --summary` when it is installed.
Software renderers (llvmpipe and similar) are recorded as such, not as usable accelerators.

## `inspect` — a Hugging Face model

```bash
uv run jaull inspect Qwen/Qwen2.5-7B-Instruct
uv run jaull inspect https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main
```

All URL variants (repo root, `/tree/…`, `/blob/…`, `/resolve/…`) normalise to the same
`repo_id`. The command queries `HfApi.model_info(files_metadata=True)`, classifies the
repository (`transformers` / `gguf` / `diffusers` / `onnx` / `adapter` / `unknown`),
downloads only tiny metadata files (`config.json`, index JSONs, `adapter_config.json`,
`model_index.json`) when useful, and prints a Rich table.

For GGUF repositories it reports each quantization variant separately and the largest
single variant, so `Repository size` (the sum of all files) is not confused with
`Largest variant` or with the runtime memory of one specific quantization.

## `estimate` — memory footprint + compatibility

```bash
uv run jaull estimate bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --quantization Q4_K_M --context 8192

uv run jaull estimate Qwen/Qwen2.5-7B-Instruct --dtype int8 --context 4096
uv run jaull estimate Qwen/Qwen2.5-7B-Instruct --dtype float16 --context 32768 --json
```

Produces an explainable breakdown:

```text
Memory breakdown
Component         Size        Source     Explanation
Weights           4.92 GiB    exact      Exact remote file size of the Q4_K_M variant.
KV cache          1.00 GiB    derived    28 layers x 4 KV heads x 128 head_dim x 8192 tokens x batch 1 x 2 bytes x 2.
Runtime overhead  0.98 GiB    assumed    Base 0.50 GiB + 10% of weights (0.49 GiB).
Device reserve    0.50 GiB    assumed    Configured device reserve of 0.50 GiB.
Safety margin     0.74 GiB    assumed    10% of estimated subtotal.
Total required    8.14 GiB
```

Each row carries the *source* of the number (`exact`, `metadata`, `derived`, `assumed`,
`unknown`) and the assessment carries a *confidence* (`high`, `medium`, `low`, `unknown`)
reduced by the weakest link in the chain. See [estimation.md](estimation.md) for the model
behind those numbers.

| Option | Default | Purpose |
|---|---|---|
| `--quantization` / `-q` | auto | GGUF variant to use (case-insensitive) |
| `--dtype` / `-d` | from config | Weight precision for transformers repos |
| `--context` / `-c` | `max_position_embeddings` or 4096 | Context length in tokens |
| `--batch-size` / `-b` | 1 | Sequences in parallel |
| `--device` | `auto` | `auto` / `gpu` / `cpu` |
| `--kv-dtype` | `float16` | KV cache element precision |
| `--safety-margin-percent` | 10 | Extra on top of the subtotal |
| `--device-reserve-gib` | 0.5 | Memory to leave free on the target device |
| `--json` | off | Emit stable JSON to stdout (`schema_version = 1`) |
| `--no-resolve-base-model` | off | Skip base-model resolution and GGUF-header enrichment |
| `--no-runtime-recommendation` | off | Skip the runtime recommendation section |

## `run` — execute a GGUF with llama-cli

```bash
uv run jaull run \
    --model bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --quantization Q4_K_M \
    --prompt "Explain what local AI means" \
    --ctx-size 4096 \
    --n-gpu-layers 0
```

`run` is the only command that downloads weights and runs inference. It:

1. normalises the Hugging Face reference;
2. resolves a single-file GGUF variant;
3. downloads it into Jaull's artifact storage when missing;
4. verifies file size and the SHA-256 sidecar, with optional full re-hashing via
   `--full-verify`;
5. executes `llama-cli --single-turn` through the local host backend.

Each execution also produces an `ExecutionObservation`, kept separate from the
metadata-only estimator:

```text
InferenceResult
├── text          # generated model response
└── observation   # success, duration, peak RAM/VRAM, exit status
```

| Option | Default | Purpose |
|---|---|---|
| `--model` / `-m` | required | Hugging Face repo_id or URL |
| `--quantization` / `-q` | auto | GGUF variant to resolve |
| `--prompt` / `-p` | required | Prompt sent to `llama-cli` |
| `--revision` / `-r` | Hub default | Revision to resolve/download |
| `--llama-cli` | PATH lookup | Path to the `llama-cli` executable |
| `--ctx-size` | 4096 | Context size passed to `llama-cli` |
| `--n-gpu-layers` | 0 | Number of layers offloaded to GPU |
| `--timeout-seconds` | 300 | Execution timeout |
| `--full-verify` | off | Recompute SHA-256 before execution |

Exit codes: `2` invalid model reference, `3` quantization not found, `4` artifact error,
`5` execution error.

Current limitation: only single-file GGUF artifacts are supported here. Multipart GGUF is
rejected explicitly, and Transformers execution is only available from the TUI.

## `doctor` — environment health

```bash
uv run jaull doctor
```

Checks the Python version, internet and Hugging Face reachability, the NVML library and
NVIDIA GPU, the detected accelerators, the preferred compute backend, the local llama.cpp
runtime, the Transformers/PyTorch runtime, execution readiness for the selected backend,
CPU fallback and whether the cache directory is writable.

This is the fastest way to produce a hardware/runtime report for a
[hardware validation issue](../CONTRIBUTING.md#run-jaull-on-hardware-we-do-not-have).

## `ui` — interactive terminal UI

```bash
uv run jaull ui
```

Launches a full-screen Textual application on the **Welcome** screen, which offers *Start
guided analysis* or *Advanced tools*.

The guided path runs **Hardware analysis → Requirements wizard → Model discovery →
Recommendations**, with real progress checklists (a step turns green when its operation
actually returns) and a Cancel that stops the search safely. From the results you can
compare recommendations, open technical details, export a JSON + Markdown report, or move
on to the execution paths of a recommendation — where a plan can be run, validated or
benchmarked.

Advanced tools keeps the **Scan**, **Inspect**, **Estimate** and **Doctor** screens. Same
services, same numbers as the CLI.

Global bindings: `h` home, `s` scan, `i` inspect, `e` estimate, `d` doctor, `esc` back,
`q` quit.

All network calls and blocking probes run in worker threads, so the interface stays
responsive while a search, a download or an execution is in flight.
