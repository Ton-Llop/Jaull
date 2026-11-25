# CPU validation evidence (cpu-v2-20261125)

Raw measurements behind the CPU results documents. Each `.txt` holds one section per run, headed `==== <case>-<mode>-<date>-<time> ====`.
Hardware: Ryzen 5 5500U, 15.7 GiB RAM, native Windows, llama.cpp b11889 CPU only. Modes: `nommap` is `-lm none`, `mmap` is `-lm mmap`.

Documents: [`resultados-cpu-2.md`](../../docs/TFG/resultados-cpu-2.md) and [`preregistre-cpu-2.md`](../../docs/TFG/preregistre-cpu-2.md).
The predictions are in [`predicciones/`](predicciones/), the machine state and model SHA-256 values in [`estat-maquina.txt`](estat-maquina.txt).

## Files

| File | Contents |
|---|---|
| `resultats.txt` | CPU campaign 2: 24 repeated configurations plus 5 new ones, with a warm-up request before measuring. (45 runs) |

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
| **resultats.txt** | | | | | | | |
| C01 mmap (ctx 2048) | 8102 | 785 | 1 | 1 | 7.90 | 4,720.5 |  |
| C01 nommap (ctx 2048) | 7981 | 3394 | 1 | 1 | 7.93 | 4,720.5 |  |
| C02 nommap (ctx 8192) | 8135 | 3141 | 1 | 1 | 7.94 | 5,062.5 |  |
| C03 nommap (ctx 16384) | 7991 | 2536 | 1 | 1 | 7.66 | 5,518.5 |  |
| C04 nommap (ctx 32768) | 8592 | 2338 | 1 | 1 | 7.97 | 6,430.5 |  |
| C05 mmap (ctx 2048) | 7925 | 3133 | 1 | 1 | 9.30 | 3,886.2 |  |
| C05 nommap (ctx 2048) | 7621 | 3882 | 1 | 1 | 9.27 | 3,886.3 |  |
| C06 nommap (ctx 8192) | 7968 | 3798 | 1 | 1 | 9.30 | 4,228.3 |  |
| C07 nommap (ctx 16384) | 8369 | 3119 | 1 | 1 | 7.81 | 4,684.3 |  |
| C08 nommap (ctx 32768) | 8393 | 2766 | 1 | 1 | 9.05 | 5,596.3 |  |
| C09 nommap (ctx 8192) | 7767 | 4461 | 1 | 1 | 7.42 | 3,247.9 |  |
| C10 nommap (ctx 32768) | 7974 | 2752 | 1 | 1 | 5.36 | 4,615.9 |  |
| C11 nommap (ctx 16384) | 8005 | 3324 | 1 | 1 | 8.06 | 4,684.3 |  |
| C12 mmap (ctx 32768) | 7608 | 3245 | 1 | 1 | 11.78 | 3,270.4 |  |
| C12 nommap (ctx 32768) | 7633 | 4314 | 1 | 1 | 15.19 | 3,270.3 |  |
| C13 nommap (ctx 32768) | 7645 | 5441 | 1 | 1 | 27.23 | 2,215.8 |  |
| C14 nommap (ctx 32768) | 7585 | 6748 | 1 | 1 | 73.28 | 827.0 |  |
| C15 mmap (ctx 2048) | 9891 | 301 | 1 | 0 |  | 7,888.9 | disponible 301 MiB < 400 MiB |
| C15 nommap (ctx 2048) | 9837 | 1963 | 1 | 1 | 20.52 | 7,889.0 |  |
| C16 mmap (ctx 8192) | 10097 | 312 | 0 | 0 |  | 7,978.9 | disponible 312 MiB < 400 MiB |
| C16 nommap (ctx 8192) | 10100 | 2210 | 1 | 1 | 20.51 | 7,979.0 |  |
| C17 mmap (ctx 32768) | 10173 | 304 | 1 | 0 |  | 8,338.9 | disponible 304 MiB < 400 MiB |
| C17 nommap (ctx 32768) | 10127 | 1945 | 1 | 1 | 20.74 | 8,339.0 |  |
| C18 mmap (ctx 8192) | 8750 | 8 | 0 | 0 |  | 6,700.6 | disponible 8 MiB < 400 MiB |
| C18 nommap (ctx 8192) | 8690 | 1849 | 1 | 1 | 19.16 | 6,985.5 |  |
| C19 mmap (ctx 32768) | 9304 | 279 | 0 | 0 |  | 6,700.6 | disponible 279 MiB < 400 MiB |
| C19 nommap (ctx 32768) | 9263 | 2017 | 1 | 1 | 19.14 | 7,345.5 |  |
| C20 mmap (ctx 2048) | 9666 | 445 | 1 | 1 | 4.77 | 7,510.8 |  |
| C20 nommap (ctx 2048) | 9468 | 1987 | 1 | 1 | 4.79 | 7,510.5 |  |
| C21 mmap (ctx 8192) | 10409 | 510 | 1 | 1 | 4.77 | 8,668.8 |  |
| C21 nommap (ctx 8192) | 10348 | 1745 | 1 | 1 | 4.80 | 8,668.5 |  |
| C22 mmap (ctx 16384) | 10581 | 488 | 1 | 1 | 4.83 | 10,212.8 |  |
| C22 nommap (ctx 16384) | 10535 | 444 | 1 | 1 | 4.81 | 10,212.5 |  |
| C23 mmap (ctx 4096) | 9776 | 3152 | 1 | 1 | 4.81 | 7,302.2 |  |
| C23 nommap (ctx 4096) | 9594 | 2282 | 1 | 1 | 4.70 | 7,302.2 |  |
| C24 mmap (ctx 32768) | 10419 | 2781 | 1 | 1 | 4.76 | 8,226.2 |  |
| C24 nommap (ctx 32768) | 10315 | 1938 | 1 | 1 | 4.76 | 8,226.2 |  |
| N01 mmap (ctx 2048) | 7816 | 6704 | 1 | 1 | 47.43 | 735.3 |  |
| N01 nommap (ctx 2048) | 7821 | 7110 | 1 | 1 | 47.73 | 735.2 |  |
| N02 mmap (ctx 8192) | 7650 | 6710 | 1 | 1 | 36.80 | 974.2 |  |
| N02 nommap (ctx 8192) | 7509 | 6569 | 1 | 1 | 32.89 | 974.2 |  |
| N03 nommap (ctx 32768) | 7724 | 6495 | 1 | 1 | 38.97 | 1,286.2 |  |
| N04 mmap (ctx 8192) | 9229 | 4167 | 1 | 1 | 9.61 | 3,472.6 |  |
| N04 nommap (ctx 8192) | 6881 | 3683 | 1 | 1 | 10.63 | 3,472.6 |  |
| N05 nommap (ctx 32768) | 7560 | 2631 | 1 | 1 | 10.00 | 4,840.6 |  |
