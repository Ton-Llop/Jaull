# B60 raw logs, 2026-11-24

Raw evidence for [`docs/TFG/resultados-b60.md`](../../docs/TFG/resultados-b60.md). There is one folder per run, named
with its date and time. Intel Arc Pro B60, native Windows, llama.cpp b11889, 23,613 MiB free before each load.

Command: `llama-server.exe -m <model> -dev Vulkan1 -ngl 99 -c <context> -np 1 --jinja -lv 4 --host 127.0.0.1 --port 8099 -lm dio`,
with a 16-token prompt and 512 generated tokens. The predictions are in
[`preregistre-b60.md`](../../docs/TFG/preregistre-b60.md) (P cases) and
[`preregistre-b60-2.md`](../../docs/TFG/preregistre-b60-2.md) (Q cases). The scripts that launched the runs are in
[`scripts/b60/`](../../scripts/b60/README.md), reconstructed from the versions that ran, with the Windows user name
replaced by a variable and the messages in English.

## What each folder holds

| File | Contents |
|---|---|
| `server.log` | Full llama.cpp output. The memory breakdown is in the `common_memory_breakdown_print` and `common_params_fit_impl` lines. |
| `summary.txt` | The result of the run, followed by three sections: `run.log` (the script timeline, with the Windows RAM available at the start), `ram.csv` (RAM available during loading and generation, only if the model loaded) and `response.json` (generation tokens and timings without the generated text, only if it generated). |

The three sections of `summary.txt` used to be separate files (`run.log`, `ram.csv` and `response.json`). They were merged
on 2026-11-24 to reduce the file count, and nothing was lost except the text the model generated. Each folder also had a
`memlines.txt`, which was deleted because it was a literal subset of `server.log`. The original export is kept outside the repository.

To pull the memory lines out of a log:
`grep -iE "will leave|cannot meet|projected|breakdown|Vulkan1|KV buffer|compute buffer|model buffer|offloaded" server.log`

## Index

All values are in MiB except tok/s. Projected is weights + KV + compute as reported by llama.cpp. "Meets" means that
1,024 MiB or more stay free.

| Folder | Case | Model | Context | Projected | Free after | 1 GiB rule | tok/s | Note |
|---|---|---|---:|---:|---:|---|---:|---|
| [`P1-20261124-110218`](P1-20261124-110218/summary.txt) | P1 | Coder 30B | 49,152 | | | | 95.74 | Run 0, discarded: no `-dev` or `-lv 4`, so the log has no memory figures. |
| [`P1-20261124-110551`](P1-20261124-110551/summary.txt) | P1 | Coder 30B | 49,152 | 22,264 | 1,349 | Meets | 95.96 | |
| [`P2-20261124-111002`](P2-20261124-111002/summary.txt) | P2 | Coder 30B | 51,200 | 22,458 | 1,155 | Meets | 96.00 | |
| [`P3-20261124-111232`](P3-20261124-111232/summary.txt) | P3 | Coder 30B | 53,248 | 22,652 | 961 | Does not meet | 96.15 | Loads anyway because `-ngl 99` is fixed. |
| [`P3-20261124-112424`](P3-20261124-112424/summary.txt) | P3 | Coder 30B | 53,248 | 22,652 | 961 | Does not meet | 95.75 | Repeat, identical. |
| [`P4-20261124-111715`](P4-20261124-111715/summary.txt) | P4 | Qwen2.5 32B | 12,288 | 21,794 | 1,819 | Meets | 18.07 | |
| [`P4-20261124-112010`](P4-20261124-112010/summary.txt) | P4 | Qwen2.5 32B | 12,288 | 21,794 | 1,819 | Meets | 18.07 | Repeat, identical. |
| [`P5-20261124-112224`](P5-20261124-112224/summary.txt) | P5 | Qwen2.5 32B | 14,336 | 22,308 | 1,305 | Meets | 18.05 | |
| [`P6-20261124-112543`](P6-20261124-112543/summary.txt) | P6 | Qwen2.5 32B | 15,360 | 22,565 | 1,048 | Meets | 18.07 | |
| [`P6-20261124-112706`](P6-20261124-112706/summary.txt) | P6 | Qwen2.5 32B | 15,360 | 22,565 | 1,048 | Meets | 18.06 | Repeat, identical. |
| [`Q1-20261124-115243`](Q1-20261124-115243/summary.txt) | Q1 | Coder 30B | 8,192 | 18,384 | 5,229 | Meets | 93.36 | |
| [`Q2-20261124-115715`](Q2-20261124-115715/summary.txt) | Q2 | Coder 30B | 98,304 | 26,920 | −3,307 | Does not meet | | Stopped at the fit step, before reserving the KV cache. |
| [`Q3-20261124-115718`](Q3-20261124-115718/summary.txt) | Q3 | Coder 30B | 131,072 | 30,024 | −6,411 | Does not meet | | Stopped at the fit step, before reserving the KV cache. |
| [`Q4-20261124-115554`](Q4-20261124-115554/summary.txt) | Q4 | Qwen2.5 32B | 32,768 | 26,934 | −3,321 | Does not meet | | Stopped at the fit step, before reserving the KV cache. |
| [`Q5-20261124-115721`](Q5-20261124-115721/summary.txt) | Q5 | Qwen2.5 32B | 65,536 | 35,158 | −11,545 | Does not meet | | Stopped at the fit step, before reserving the KV cache. |
| [`Q6-20261124-120324`](Q6-20261124-120324/summary.txt) | Q6 | Qwen3.8 27B | 98,304 | 21,216 | 2,397 | Meets | 15.43 | |
| [`Q7-20261124-120635`](Q7-20261124-120635/summary.txt) | Q7 | Qwen3.8 27B | 114,688 | 22,256 | 1,357 | Meets | 15.42 | |
| [`Q8-20261124-120748`](Q8-20261124-120748/summary.txt) | Q8 | Qwen3.8 27B | 131,072 | 23,296 | 317 | Does not meet | | Stopped at the fit step by a 500 MiB safety margin, not because it overflowed VRAM. Its `summary.txt` says "no cabe en VRAM", which is inaccurate for this case. |

P3, P4 and P6 appear twice. The first run is the official one and the second is the repeat that the preregistration asks for
(P3 and P6 end up within 100 MiB of the threshold). P4 was also run twice, and both runs came out identical.
