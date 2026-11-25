# CPU validation evidence (cpu-20261125)

Raw measurements behind the CPU results documents. Each `.txt` holds one section per run, headed `==== <case>-<mode>-<date>-<time> ====`.
Hardware: Ryzen 5 5500U, 15.7 GiB RAM, native Windows, llama.cpp b11889 CPU only. Modes: `nommap` is `-lm none`, `mmap` is `-lm mmap`.

Documents: [`resultados-cpu.md`](../../docs/TFG/resultados-cpu.md), [`preregistre-cpu.md`](../../docs/TFG/preregistre-cpu.md) and
[`preregistre-cpu-addendum.md`](../../docs/TFG/preregistre-cpu-addendum.md). The predictions are in [`predicciones/`](predicciones/).

## Files

| File | Contents |
|---|---|
| `pasada-A.txt` | CPU campaign 1, pass A: free RAM 10.0 to 11.5 GiB, 24 configurations, -lm none and -lm mmap. (34 runs) |
| `benchmark-pasada-A.jsonl` | `llama-bench` output, `-p 512 -n 128`, three repetitions. |
| `pasada-B.txt` | CPU campaign 1, pass B: tight RAM (6.1 to 9.6 GiB free), same configurations (addendum to the preregistration). (34 runs) |

## What each section contains

| Block | Contents |
|---|---|
| `[summary.txt]` | Result of the run: free RAM before, minimum free RAM, page file, buffers printed by llama.cpp, tok/s, and why it was stopped if it was. |
| `[memlines.txt]` | The llama.cpp lines about buffers (`CPU model buffer`, `CPU_REPACK`, KV, compute), a subset of `server.log`. |
| `[ram.csv]` | Free RAM, page file and server working set / private bytes, one sample about every second. |
| `[response timings]` | Generation and prompt timings of the measured request, without the generated text. |
| `[warm-up timings]` | Campaign 2 only: the warm-up request sent before the measured one. |

## Index

All values in MiB except tok/s. **Stopped** means the script stopped the server at its own safety limit (free RAM below 400 MiB or page file up 3,000 MiB), which the authors set; it does not prove the model could not run.

| Run | Free before | Min free | Loaded | Generated | tok/s | Buffers | Stopped |
|---|---:|---:|:-:|:-:|---:|---:|---|
| **pasada-A.txt** | | | | | | | |
| C01 nommap (ctx 2048) | 10036 | 5408 | 1 | 1 | 8.03 | 4,720.5 |  |
| C02 nommap (ctx 8192) | 10055 | 5068 | 1 | 1 | 8.01 | 5,062.5 |  |
| C03 nommap (ctx 16384) | 10046 | 4623 | 1 | 1 | 7.92 | 5,518.5 |  |
| C04 nommap (ctx 32768) | 10438 | 4150 | 1 | 1 | 8.05 | 6,430.5 |  |
| C05 nommap (ctx 2048) | 10015 | 6159 | 1 | 1 | 9.52 | 3,886.3 |  |
| C06 nommap (ctx 8192) | 9991 | 5824 | 1 | 1 | 9.55 | 4,228.3 |  |
| C07 nommap (ctx 16384) | 10030 | 5431 | 1 | 1 | 9.57 | 4,684.3 |  |
| C08 nommap (ctx 32768) | 10448 | 4949 | 1 | 1 | 9.33 | 5,596.3 |  |
| C09 nommap (ctx 8192) | 10007 | 6846 | 1 | 1 | 7.97 | 3,247.9 |  |
| C10 nommap (ctx 32768) | 10000 | 5471 | 1 | 1 | 7.93 | 4,615.9 |  |
| C11 nommap (ctx 16384) | 10037 | 5403 | 1 | 1 | 9.55 | 4,684.3 |  |
| C12 nommap (ctx 32768) | 9981 | 6758 | 1 | 1 | 16.99 | 3,270.3 |  |
| C13 nommap (ctx 32768) | 9968 | 7789 | 1 | 1 | 28.24 | 2,215.8 |  |
| C14 nommap (ctx 32768) | 9975 | 9132 | 1 | 1 | 72.88 | 827.0 |  |
| C15 mmap (ctx 2048) | 11469 | 460 | 1 | 1 | 17.67 | 7,888.9 |  |
| C15 nommap (ctx 2048) | 11423 | 3536 | 1 | 1 | 20.43 | 7,889.0 |  |
| C16 mmap (ctx 8192) | 11462 | 355 | 1 | 0 |  | 7,978.9 | disponible 355 MiB < 400 MiB |
| C16 nommap (ctx 8192) | 11490 | 3522 | 1 | 1 | 19.80 | 7,979.0 |  |
| C17 mmap (ctx 32768) | 11519 | 451 | 1 | 1 | 17.97 | 8,338.9 |  |
| C17 nommap (ctx 32768) | 11519 | 3266 | 1 | 1 | 20.63 | 8,339.0 |  |
| C18 mmap (ctx 8192) | 10627 | 58 | 0 | 0 |  | 6,700.6 | disponible 58 MiB < 400 MiB |
| C18 nommap (ctx 8192) | 10526 | 3656 | 1 | 1 | 19.22 | 6,985.5 |  |
| C19 mmap (ctx 32768) | 11229 | 358 | 0 | 0 |  | 7,345.6 | disponible 358 MiB < 400 MiB |
| C19 nommap (ctx 32768) | 11235 | 3942 | 1 | 1 | 19.04 | 7,345.5 |  |
| C20 mmap (ctx 2048) | 11396 | 1735 | 1 | 1 | 4.87 | 7,510.8 |  |
| C20 nommap (ctx 2048) | 11419 | 3953 | 1 | 1 | 4.88 | 7,510.5 |  |
| C21 mmap (ctx 8192) | 11453 | 655 | 1 | 1 | 4.82 | 8,668.8 |  |
| C21 nommap (ctx 8192) | 11459 | 2835 | 1 | 1 | 4.85 | 8,668.5 |  |
| C22 mmap (ctx 16384) | 11493 | 466 | 1 | 1 | 4.88 | 10,212.8 |  |
| C22 nommap (ctx 16384) | 11474 | 1387 | 1 | 1 | 4.88 | 10,212.5 |  |
| C23 mmap (ctx 4096) | 11435 | 4844 | 1 | 1 | 4.70 | 7,302.2 |  |
| C23 nommap (ctx 4096) | 11387 | 4073 | 1 | 1 | 4.83 | 7,302.2 |  |
| C24 mmap (ctx 32768) | 11470 | 3983 | 1 | 1 | 4.75 | 8,226.2 |  |
| C24 nommap (ctx 32768) | 11531 | 3257 | 1 | 1 | 4.76 | 8,226.2 |  |
| **pasada-B.txt** | | | | | | | |
| C01 nommap (ctx 2048) | 8008 | 3405 | 1 | 1 | 7.31 | 4,720.5 |  |
| C02 nommap (ctx 8192) | 8034 | 3005 | 1 | 1 | 7.32 | 5,062.5 |  |
| C03 nommap (ctx 16384) | 8046 | 2730 | 1 | 1 | 7.21 | 5,518.5 |  |
| C04 nommap (ctx 32768) | 7819 | 1561 | 1 | 1 | 7.59 | 6,430.5 |  |
| C05 nommap (ctx 2048) | 7840 | 4029 | 1 | 1 | 9.13 | 3,886.3 |  |
| C06 nommap (ctx 8192) | 7929 | 3754 | 1 | 1 | 9.07 | 4,228.3 |  |
| C07 nommap (ctx 16384) | 7923 | 3312 | 1 | 1 | 8.55 | 4,684.3 |  |
| C08 nommap (ctx 32768) | 8704 | 2266 | 1 | 1 | 7.17 | 5,596.3 |  |
| C09 nommap (ctx 8192) | 7272 | 4232 | 1 | 1 | 7.28 | 3,247.9 |  |
| C10 nommap (ctx 32768) | 7903 | 3399 | 1 | 1 | 7.47 | 4,615.9 |  |
| C11 nommap (ctx 16384) | 7917 | 3392 | 1 | 1 | 8.79 | 4,684.3 |  |
| C12 nommap (ctx 32768) | 7825 | 4544 | 1 | 1 | 15.06 | 3,270.3 |  |
| C13 nommap (ctx 32768) | 7809 | 5290 | 1 | 1 | 22.05 | 2,215.8 |  |
| C14 nommap (ctx 32768) | 7710 | 6817 | 1 | 1 | 63.21 | 827.0 |  |
| C15 mmap (ctx 2048) | 7227 | 345 | 0 | 0 |  | 7,693.9 | disponible 345 MiB < 400 MiB |
| C15 nommap (ctx 2048) | 7945 | 330 | 0 | 0 |  | 7,889.0 | disponible 330 MiB < 400 MiB |
| C16 mmap (ctx 8192) | 7713 | 133 | 0 | 0 |  | 7,693.9 | disponible 133 MiB < 400 MiB |
| C16 nommap (ctx 8192) | 6932 | 86 | 0 | 0 |  | 7,694.0 | disponible 86 MiB < 400 MiB |
| C17 mmap (ctx 32768) | 8273 | 20 | 0 | 0 |  | 7,693.9 | disponible 20 MiB < 400 MiB |
| C17 nommap (ctx 32768) | 8240 | 110 | 0 | 0 |  | 8,339.0 | disponible 110 MiB < 400 MiB |
| C18 mmap (ctx 8192) | 8007 | 10 | 0 | 0 |  | 6,700.6 | disponible 10 MiB < 400 MiB |
| C18 nommap (ctx 8192) | 7918 | 1027 | 1 | 1 | 17.27 | 6,985.5 |  |
| C19 mmap (ctx 32768) | 8557 | 39 | 0 | 0 |  | 6,700.6 | disponible 39 MiB < 400 MiB |
| C19 nommap (ctx 32768) | 8428 | 1219 | 1 | 1 | 18.06 | 7,345.5 |  |
| C20 mmap (ctx 2048) | 8971 | 275 | 0 | 0 |  | 7,510.8 | disponible 275 MiB < 400 MiB |
| C20 nommap (ctx 2048) | 9256 | 1210 | 1 | 1 | 3.72 | 7,510.5 |  |
| C21 mmap (ctx 8192) | 7845 | 326 | 0 | 0 |  | 8,668.8 | disponible 326 MiB < 400 MiB |
| C21 nommap (ctx 8192) | 6111 | 46 | 0 | 0 |  | 8,529.5 | disponible 46 MiB < 400 MiB |
| C22 mmap (ctx 16384) | 7576 | 47 | 0 | 0 |  | 10,212.8 | disponible 47 MiB < 400 MiB |
| C22 nommap (ctx 16384) | 7361 | 370 | 0 | 0 |  | 10,065.5 | disponible 370 MiB < 400 MiB |
| C23 mmap (ctx 4096) | 7533 | 1015 | 1 | 1 | 3.91 | 7,302.2 |  |
| C23 nommap (ctx 4096) | 8845 | 1328 | 1 | 1 | 4.14 | 7,302.2 |  |
| C24 mmap (ctx 32768) | 9644 | 319 | 1 | 0 |  | 8,226.2 | disponible 348 MiB < 400 MiB |
| C24 nommap (ctx 32768) | 7329 | 4 | 0 | 0 |  | 8,031.9 | disponible 4 MiB < 400 MiB |
