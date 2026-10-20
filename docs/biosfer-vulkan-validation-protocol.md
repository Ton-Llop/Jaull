# Physical validation protocol: Intel Arc Pro B60 (Windows and WSL2)

To be run when the Biosfer machine is free and with the owner's authorisation. It is a
checklist for **observation**: do not install drivers, change Windows or WSL2 settings,
or "fix" anything to make a step pass. A step that fails is a result; record it and stop
that branch. Design and status: [vulkan-windows-wsl2.md](vulkan-windows-wsl2.md).

Use one output folder per session, for example `B60-<date>/windows` and
`B60-<date>/wsl2`, and keep every raw log. Never edit a log; redact secrets only.
Nothing below changes the ranking or any constant: measured values are evidence, not
inputs to a formula.

## 0. Before anything

- [ ] Written authorisation to use the machine and its idle window.
- [ ] Jaull at the commit under test; record `git rev-parse HEAD` and `git status`.
- [ ] A small GGUF already on disk (for example a 1–4B Q4_K_M), with its SHA256 recorded.
      No large download without a separate decision.
- [ ] Machine otherwise idle (no game, render or other GPU compute).

## 0.1 Raw captures that replace the synthetic fixtures

Every parser in this work was written against **synthetic** samples. The most valuable
result of the session is the unedited output of these commands, because the fixtures
(`tests/test_vulkan_device_memory.py`, `test_vulkan_evidence_identity.py`,
`test_windows_adapters.py`, `test_vulkan_observation_and_paths.py`) are only as good as
their resemblance to it. Save each one to its own file, as bytes, with no editing, and
record its SHA256 (`Get-FileHash -Algorithm SHA256` on Windows, `sha256sum` in WSL2).

| File | Windows (PowerShell) | WSL2 (bash) | Replaces fixture |
|---|---|---|---|
| `vulkaninfo-summary.txt` | `vulkaninfo --summary > vulkaninfo-summary.txt 2>&1` | same | `B60_SUMMARY` |
| `vulkaninfo-full.txt` | `vulkaninfo > vulkaninfo-full.txt 2>&1` | same | `B60_FULL` (heaps, `budget`, `usage`, flags) |
| `llama-list-devices.txt` | `& "<llama-cli.exe>" --list-devices > llama-list-devices.txt 2>&1` | `<llama-cli> --list-devices > llama-list-devices.txt 2>&1` | device-line fixtures |
| `llama-version.txt` | `& "<llama-cli.exe>" --version > llama-version.txt 2>&1` | same | version text |
| `windows-adapters.json` | `Get-CimInstance Win32_VideoController \| Select-Object Name,PNPDeviceID,DriverVersion,AdapterCompatibility \| ConvertTo-Json -Compress > windows-adapters.json` | `n/a` (the query is skipped outside Windows) | `B60_ROW` |
| `llama-load-log.txt` | stderr of the offload run in step 1.7 | same | `VULKAN_LOAD_LOG`, `ggml_vulkan:` lines |
| `jaull-scan.json`, `jaull-doctor.txt` | `uv run jaull scan --json`, `uv run jaull doctor` | same | — |

Alongside each file write down the exact command, the date and the machine state. If a
command fails or prints nothing, keep that output too: it is a result.

Questions the captures must answer (write the answer next to the file):

1. In `vulkaninfo-full.txt`, are the heap lines `size`, `budget`, `usage` and the
   `MEMORY_HEAP_DEVICE_LOCAL_BIT` flag laid out as `parse_vulkan_heap_memory` expects? Is
   there more than one device-local heap? Is `budget` close to `size`, and does it change
   when another GPU application runs (the semantics of `heapBudget`/`heapUsage`:
   an allowance for this process, not free VRAM)?
2. Does the device name in `vulkaninfo-summary.txt` equal, after removing `(R)`/`(TM)`
   and one trailing parenthesis, the name in `llama-list-devices.txt`? **If it does not,
   record both names exactly. Do not loosen the matching to make the step pass.**
3. Are there extra devices (integrated GPU, `llvmpipe`, a Dozen/D3D12 layer) and at which
   index does the B60 appear?
4. Is `vulkaninfo.exe` present on a stock Windows install with the Intel driver, or did it
   come from the Vulkan SDK/runtime?
5. Which of the new messages did Jaull print for each state it reached? The text must say
   what was selected and what the binary listed; if a state is `UNKNOWN` or `NOT_READY`,
   copy the exact message.

After review (and removal of serial numbers or other identifiers), the real files become
the new test fixtures and the synthetic ones are deleted, in a separate change.

## 1. Windows native

1. [ ] Record: Windows edition and build (`winver`), Intel graphics driver version
       (Device Manager → Display adapters → Driver), CPU, RAM.
2. [ ] `vulkaninfo --summary` → save. Is the B60 listed as
       `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`? Which `driverName`/`driverInfo`/`driverVersion`?
       *(If `vulkaninfo.exe` is missing, record that: it is a finding.)*
3. [ ] `vulkaninfo` (full) → save; copy the `memoryHeaps[...]` block. Check: does the
       device-local heap show ≈24 GiB? is `budget`/`usage` present? are there several
       device-local heaps?
4. [ ] `uv run jaull scan --json` → save. Expect one accelerator: vendor `intel`, type
       `dedicated`, `dedicated_memory_bytes` from `vulkan-memory-heap`, `available_memory_bytes`
       only if a budget was printed, Vulkan `available`.
5. [ ] With the Vulkan build of llama.cpp (record its path, version and SHA256):
       `llama-cli.exe --list-devices` → save. Which `Vulkan<N>` has the B60's name from step 2?
       Is there an extra software device? Do the listed MiB match the heap figures?
6. [ ] `uv run jaull doctor` → save.
7. [ ] Open the guided Jaull flow, select the verified GGUF path, and require fresh
       `READY` readiness identifying the B60 and a confirmed memory budget. Stop this
       branch if readiness or the full-offload fit is unconfirmed. Run a short prompt
       from Results/Paths and save its actual command and log. They must name the B60's
       confirmed `--device Vulkan<N>`, not an assumed `Vulkan0`. Check full offload
       (`offloaded N/N layers to GPU`) and model/compute buffers on that same device.
       Watch GPU activity in Task Manager as supplementary evidence, not device identity.
       For a direct cross-check, use the same verified local file and confirmed id:
       `& "<llama-cli.exe>" --model "<local.gguf>" --device "<VulkanN>" --n-gpu-layers -1 --ctx-size 2048 --prompt "Say hi" --single-turn --verbose`.
       This raw command does not perform Jaull's preparation. Neither does
       `jaull run --n-gpu-layers ...`: that explicit CLI override bypasses hardware
       preparation and must not be described as equivalent to the guided checks.
8. [ ] Speed, **three independent runs**, protocol unchanged: tg128 (primary), pp512
       (supplementary), flash attention off, five llama-bench repetitions per run, median of
       the three runs. Use the TUI path that produces Jaull benchmark records (Results →
       Speed → measure speed) so identity, device and driver are stored. Cross-check one
       run with the raw command and keep its output:
       Prefer copying the argv actually produced by Jaull. The full-offload v2 template is:
       `llama-bench.exe -m "<local.gguf>" -dev "<VulkanN>" -ngl -1 -p 512 -n 128 -r 5 -t <physical-cores> -b 512 -ub 512 -fa off -ctk f16 -ctv f16 -lm mmap -o md -oe jsonl --progress`.
       Obtain the thread count with the producer's own rule:
       `uv run python -c "import psutil; print(psutil.cpu_count(logical=False))"`.
       Do not substitute logical cores or an arbitrary number. Use the actual prepared
       offload for a partial plan, not `-ngl -1`; keep that placement in the record.
9. [ ] Open the stored benchmark record: does it contain the Windows `os`, the Vulkan
       device, the driver version, the llama.cpp build and the model SHA256? Does
       `assess_speed` accept it for the same plan on the same machine, and refuse it after
       changing one driver/device field in a copy?

10. [ ] **The normal user path.** Run plain `uv run jaull` and choose Guided analysis. Take a
        screenshot (or copy the text) of each stage and note anything that differs from what
        the synthetic integration tests (`tests/test_tui_intel_flow.py`) assume:
        - Hardware screen: the card name, the "VRAM" headline, "Compute backend: Vulkan", and
          the budget warning if a budget was printed.
        - Results: the plan reads "llama.cpp · Vulkan", its readiness and, if blocked, the exact
          reason text; the Run / Validate / Benchmark buttons are enabled only for a confirmed
          device.
        - Run a short prompt from Results and save the command line shown in the log: it must carry
          `--device Vulkan<N>` for the B60 and not for another adapter.
        - Benchmark from Results; confirm the stored record names Windows, the Vulkan device and
          the driver version.
        If a plan shows `UNKNOWN` or `NOT_READY`, copy the message verbatim; do not work around it.

## 2. WSL2 (Ubuntu)

1. [ ] Record: `wsl --version` (from Windows), `uname -r`, `cat /etc/os-release`, distro name.
2. [ ] `ls -l /dev/dxg` and `ls /dev/dri 2>&1`: what is exposed?
3. [ ] `vulkaninfo --summary` → save. Which device appears: the B60, a Mesa D3D12
       translation layer ("Dozen"), or only `llvmpipe`? Record the `driverName`.
4. [ ] `uv run jaull scan --json`, `uv run jaull doctor` → save. With only a software
       renderer, Jaull must print the WSL2 caution and select CPU.
5. [ ] If a real device appears: build a **separate** Vulkan llama.cpp
       (`-DGGML_VULKAN=ON`, see [llamacpp.md](llamacpp.md)), run `--list-devices`, then repeat
       steps 1.7–1.10 with Linux paths.
       In bash invoke `<llama-cli>` directly (no PowerShell `&` and no `.exe`).
       Re-detect the device id and physical cores inside WSL; do not inherit Windows ids.
6. [ ] If no accelerated path exists in WSL2: write down the exact reason (missing device,
       missing driver, translation layer without the needed extension, build error). **Do not
       mark WSL2 acceleration as supported.** The next step is a design decision about a
       Windows-runtime bridge, not a workaround here.

### Quality evaluation is a separate branch

Run/Validate/Benchmark success does not validate Evaluate. The quality pilot requires
Linux/WSL and its own confirmed `llama-server`, exact artifact, dataset and evaluator image.
HellaSwag smoke/100 supports CPU/CUDA/confirmed Vulkan under schema 3; IFEval still requires
the audited CUDA runtime. A Windows-only B60 is not usable by the WSL pilot through a bridge.
Record whether Docker is Desktop or a local Engine. The evaluator must verify HTTP health
from inside its own container before scoring. No additional container or proxy is needed.
Only a completed, imported smoke with consistent effective backend/device evidence may
establish this branch as physically validated; preserve every failed bundle unchanged.

## 3. What to bring back

- All saved logs, JSON and the benchmark records, untouched.
- A short table: step, command, result (`pass` / `fail` / `not applicable`), observation.
- Answers to the open questions in [vulkan-windows-wsl2.md](vulkan-windows-wsl2.md):
  budget semantics on this driver, name agreement between `vulkaninfo` and
  `--list-devices`, `ggml_vulkan:` log line format, `vulkaninfo.exe` availability.

Only after these are recorded may status words in the docs change from `UNVERIFIED` to
`VALIDATED`, and only for the environment that was actually tested (Windows and WSL2 are
separate).
