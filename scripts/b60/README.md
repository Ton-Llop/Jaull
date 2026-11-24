# B60 memory-fit scripts

Two bash scripts that measure how much context a model fits at on an Intel Arc Pro B60, or any other GPU that
llama.cpp can reach through Vulkan on native Windows. They produced the evidence in
[`validation/b60-contextos-20261124/`](../../validation/b60-contextos-20261124/README.md) and the results in
[`docs/TFG/resultados-b60.md`](../../docs/TFG/resultados-b60.md): 14 cases of Qwen3-Coder 30B-A3B, Qwen2.5 32B and
Qwen3.8 27B at up to 131,072 tokens, with Jaull's predictions written before measuring.

If you have a B60 or another 24 GB card, running the same cases and sending the folders back would add a second GPU to
that evidence. See "Send your results" below.

## What they do

Each script starts `llama-server.exe` for one case, reads llama.cpp's own memory line
(`projected to use X MiB of device memory vs. Y MiB of free device memory`), and records the breakdown into weights, KV
cache and compute buffer.

| Script | Cases | What it does |
|---|---|---|
| `b60-case.sh` | P1 to P6 | Loads the model, generates 512 tokens and records tok/s. Cases close to the limit. |
| `b60-case2.sh` | Q1 to Q8 | Same, but if the model does not fit it stops at llama.cpp's fit step, before the KV cache is reserved, so nothing overflows VRAM. |
| `b60-fill.sh` | P1 to P6 | Fills the context in steps (4,000, 16,000, 32,000 and 45,000 tokens by default) and records prompt speed, generation speed and RAM. A full run on P1 takes about 10 minutes. |
| `gguf_tensor_groups.py` | any GGUF | Reads only the header and totals the tensors by group, to see which ones llama.cpp keeps off the GPU. `--blocks` adds the tensor kinds and the size of each block. |
| `plot_context_limits.py` | | Draws the figures in `docs/TFG/figures/` from the runs in `validation/`. Run it with `uv run --no-project --with matplotlib python scripts/b60/plot_context_limits.py`. |

| Case | Model | Context |
|---|---|---:|
| P1, P2, P3 | Qwen3-Coder 30B-A3B Q4_K_M | 49,152, 51,200, 53,248 |
| P4, P5, P6 | Qwen2.5 32B Q4_K_M | 12,288, 14,336, 15,360 |
| Q1, Q2, Q3 | Qwen3-Coder 30B-A3B Q4_K_M | 8,192, 98,304, 131,072 |
| Q4, Q5 | Qwen2.5 32B Q4_K_M | 32,768, 65,536 |
| Q6, Q7, Q8 | Qwen3.8 27B UD-Q4_K_M | 98,304, 114,688, 131,072 |

## Requirements

- A Windows PC with WSL2, running the scripts from WSL2. The model runs on the Windows driver, not on WSL's.
- llama.cpp b11889 for Windows with Vulkan in `C:\llama-win\` (`llama-server.exe` and its DLLs).
- The GGUF files in the same folder: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (unsloth),
  `Qwen2.5-32B-Instruct-Q4_K_M.gguf` (bartowski) and `Qwen3.8-27B-UD-Q4_K_M.gguf` (unsloth).
- `python3` in WSL, for reading the response. Set `WIN_USER` to your Windows user name (the folder under `C:\Users`).
- The GPU is picked with `-dev Vulkan1`, which is the B60 on the test PC where `Vulkan0` was an AMD integrated GPU. Check
  yours with `/mnt/c/llama-win/llama-server.exe --list-devices` and set `DEV=VulkanN` if it differs.

## Run

```bash
chmod +x b60-case.sh b60-case2.sh
cd /mnt/c    # Windows executables only start from /mnt/c
WIN_USER=<windows user> nohup /path/to/b60-case2.sh Q1 > ~/b60-Q1.out 2>&1 < /dev/null &
tail -f ~/b60-Q1.out
```

Results go to `~/b60-results/<case>-<date>-<time>/`. Start with Q1, which loads and generates like the P cases, then try a
case that does not fit. The Q cases take about two seconds when they stop at the fit step.

## Safety

- It refuses to start if any `llama-server.exe` is already running, because the free VRAM would not be comparable and it will
  not touch someone else's server.
- It reads the available Windows RAM every few seconds and stops its own server below `MIN_FREE_MIB` (3,072 by default). It
  will not start below 6,144 MiB available.
- It stops only the process it launched, by PID, and says so if it cannot (`COULD NOT STOP ... taskkill`).
- `-lm dio` avoids filling system RAM with the memory-mapped model. It does not change the VRAM figures.
- `MIN_LEFT_MIB` in `b60-case2.sh` also stops a run that would fit but leave less than that many MiB free.

These scripts were only run on one PC. A bad `-dev` value or a driver that handles VRAM overflow differently could behave
differently on yours, so try Q1 first and watch the first non-fitting case.

## Differences from the 2026-11-24 runs

These files are reconstructed from the versions that ran. Two things changed: the Windows user name is read from
`WIN_USER` instead of being written in the temp paths, and the log and summary messages are in English. The 2026-11-24
evidence has the Spanish ones (`caso=`, `cargo=`, `parado en el ajuste=`). The commands and thresholds are the same.

## Send your results

Open an issue or a pull request with the folders from `~/b60-results/` (one `server.log` and `summary.txt` per case is enough)
and your GPU, driver and llama.cpp build. The logs contain your Linux home path in `summary.txt`; replace it if you prefer.
