# Vulkan on Intel Arc, under Windows and WSL2

Design record and status for extending Jaull's validated CUDA path to Vulkan, with the
Biosfer machine (Intel Arc Pro B60, 24 GB dedicated GDDR6, Windows + WSL2) as the first
target. Decision record: [D098](TFG/DECISIONS.md). Physical test plan:
[biosfer-vulkan-validation-protocol.md](biosfer-vulkan-validation-protocol.md).

**Nothing here has been validated on the target GPU.** Every figure and log line in the
test fixtures is synthetic. Status words used below:

| Status | Meaning |
|---|---|
| `IMPLEMENTED` | Written and covered by automated tests with fixtures or mocks. |
| `VALIDATED` | Checked by a real run on the target environment. **None yet.** |
| `UNVERIFIED` | Implemented, but its assumptions about drivers or OS behaviour await a physical test. |
| `BLOCKED_BY_ENVIRONMENT` | Cannot be tested here (no Intel GPU, no access to the Biosfer PC). |

Out of scope, by decision: SYCL / oneAPI (`ComputeBackend.SYCL` is not added, the
architecture does not prevent it later), quality evaluation on Vulkan (`_cuda_blocker`
and the quality protocol are untouched), driver installation, and any change to the
Windows host or to WSL2.

## 1. Five questions that are not one

The code already kept some of these apart; this work makes each one a separate fact:

| # | Question | Where it is answered | Never inferred from |
|---|---|---|---|
| 1 | Which hardware is physically present? | `hardware/` probes: NVML, `vulkaninfo`, Windows adapter listing | a backend |
| 2 | Which backend does the operating system expose? | `AcceleratorProfile.backends` (`AVAILABLE` only when a probe says so) | the vendor or the OS |
| 3 | Does this `llama-cli` build support that backend? | `llama-cli --list-devices` → `LlamaCppBackendCapability` | `vulkaninfo` |
| 4 | Is the device the runtime listed *the* selected accelerator? | `runtime/device_matching.py` | the runtime's numbering |
| 5 | Can the plan run now? | `ExecutionReadiness` | memory fit alone |

An Intel GPU never implies Vulkan: `AcceleratorVendor.INTEL` maps to `(VULKAN,)` in the
backend policy, but only an `AVAILABLE` Vulkan backend is selectable, and a backend that
appears only in a Windows adapter listing is `UNKNOWN / NOT_CHECKED`. The operating
system never selects a backend either: `classify_execution_environment` names the
environment so evidence and messages can say which one they describe, nothing more.

## 2. What was already portable, and what assumed Linux, NVIDIA or CUDA

Audited on 2026-10-19 against the working tree that contains D097.

| Area | Finding | Action |
|---|---|---|
| `hardware/vulkan.py` summary parsing | OS-independent; discrete vs integrated from `deviceType`, not from the name | kept; driver version added |
| Vulkan device memory | Only AMD `mem_info_vram_*` from `/sys/class/drm` (Linux) | heap/budget reader added (section 3) |
| `runtime/backend_selection.py` | Vendor-agnostic; Intel → Vulkan | unchanged |
| `domain/runtime_support.py` (D097) | Backend/technical-support rules independent of installation | reused unchanged |
| `runtime/locator.py` | Already handles `.exe` and `os.name == "nt"` | covered by new path tests |
| `llama_cpp_capability.py` readiness | `READY` if the backend lists *any* device | Vulkan now also needs the selected device |
| Planner `--device` pin | pinned `devices[0]` | pins only a provably matched device |
| `benchmarks/matrix.py`, TUI benchmark, `benchmark_plan_speed` | used `devices[0]` or "exactly one device" | same matcher |
| `execution_plans/service.py` `device == "cuda"` | That is the **Transformers** `device_map` flag. llama.cpp falls through to `selection.selected_backend`, which is already Vulkan-aware | no change needed |
| `recommendation/local_evidence.py` | driver/CUDA environment read only from NVML `hardware.gpus`; device proof only for `CUDA<i>` | Vulkan driver + `ggml_vulkan` device proof |
| `execution/host.py` peak VRAM | NVML only | unchanged: Vulkan runs report `peak_vram_bytes=None` |
| `evaluation/comparison.py`, `llama_cpp_tensor_policy.py` | physical VRAM comparison/placement verified for one discrete CUDA device only | unchanged: Vulkan stays `NOT_COMPARABLE` |
| `advisor/quality_candidates.py` `_cuda_blocker` | quality pilot is CUDA-only | unchanged (out of scope) |

## 3. Memory: four numbers that must not be confused

| Quantity | Meaning | Field | Source today |
|---|---|---|---|
| Capacity | Physical size of the device-local heap | `dedicated_memory_bytes` | Vulkan heap `size` (`vulkaninfo`) |
| Budget | OS/driver allowance for *this process* | `available_memory_bytes` | Vulkan heap `budget` (`VK_EXT_memory_budget`) |
| Observed use | Measured during a run | `peak_vram_bytes` | NVML only; **absent for Vulkan** |
| Estimate | What Jaull predicts it needs | `MemoryEstimate` | estimator |

Capacity is never substituted for availability (existing rule). A 24 GB card with no
budget reported has capacity 24 GB and availability *unknown*; Hardware Fit then makes
no "fits in VRAM" claim for it.

Sources investigated:

* **`vulkaninfo` heaps + `VK_EXT_memory_budget`** — `IMPLEMENTED`, `UNVERIFIED`. The
  extension documents the budget as "a rough estimate of how much memory the process can
  allocate before allocations may fail or cause performance degradation", and the usage
  as an estimate of what the process currently holds. The budget is therefore an
  allowance for *this process* (it can depend on system load), not global free VRAM. It
  is stored as availability as is: `usage` is not subtracted (it is the probing
  process's own), size and usage never create an availability on their own, the source is
  tagged `vulkan-memory-budget`, `jaull scan` prints a warning saying so, and Hardware Fit
  applies its own device reserve and safety margin on top. Only the
  largest device-local heap is read (a small host-visible window next to the main heap is
  ignored), a budget of zero or larger than its heap is discarded, and two identical
  cards are never guessed apart (the summary has no PCI bus id). The full report is
  read only for a discrete device that no better source covered.
* **llama.cpp `--list-devices` total/free** — parsed already, but **not** used for
  ranking: the Top 5 must not depend on which binary is installed (README, pinned by a
  test). It remains readiness/launch information.
* **DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` (Windows)** — *not implemented.* It also
  returns a per-process budget and needs native calls Jaull does not otherwise make;
  adding that without a Windows+Intel machine to check it would be a guess.
* **`Win32_VideoController.AdapterRAM` (Windows)** — *deliberately not used.* A 32-bit
  field that saturates near 4 GiB on modern cards; presenting it as capacity is the
  "approximate graphics capacity" figure this work avoids. The Windows adapter listing
  reads name, PCI ids and driver version only.
* **Linux `xe` driver** — *not implemented.* Per-process figures exist in DRM `fdinfo`
  (`drm-total-vram0`) but a device-total/available interface has not been confirmed
  from a primary source, and the AMD `mem_info_vram_*` paths must not be assumed for
  Intel.
* **WSL2** — `/sys/class/drm` is not assumed to exist. If no probe yields a reliable
  figure, availability stays unknown.

Shared memory (integrated GPUs) is unchanged: such an accelerator is never added to RAM
as a second bank (`planning_accelerator_memory_bytes` excludes it; the memory topology is
`UNIFIED_MEMORY`).

## 4. Device identity

llama.cpp numbers its devices itself (`Vulkan0`, `Vulkan1`, ...); the number depends on
the build, the driver and what else is installed. `match_runtime_device` returns a
device only when it is provably the selected accelerator:

1. Software renderers (`llvmpipe`, `lavapipe`, `swiftshader`, Microsoft Basic Render)
   are never candidates.
2. A name identifies exactly one candidate (equality after removing `(R)`/`(TM)` and one
   trailing annotation; `RTX 2060` does not match `RTX 2060 SUPER`).
3. Otherwise a single hardware device is accepted when nothing contradicts the
   selection; on **Vulkan** a lone device with a different name is a mismatch.
4. Several candidates with no unique name match are ambiguous: no `--device` is pinned.

CUDA and HIP keep their validated behaviour (a single device is accepted regardless of
its name). The one deliberate change for them is that several devices no longer silently
resolve to the first.

Readiness for Vulkan additionally requires the match: software-only or a different
card → `NOT_READY` (`SELECTED_DEVICE_NOT_EXPOSED`); several indistinguishable devices →
`UNKNOWN` (`SELECTED_DEVICE_NOT_IDENTIFIED`). The message always says what was selected
and what the binary listed, for example *"llama-cli lists several vulkan devices
(Vulkan0: X, Vulkan1: X) and none can be tied to 'X'; no device is selected
automatically"*, and it reaches the plan's label and warnings. A plan in that state is
neither failed nor presented as ready.

**The rule stays strict until real B60 output exists.** Only the harmless normalisations
above are applied; no partial or fuzzy name matching is added. If `vulkaninfo` and
llama.cpp name the card differently on the B60, the two names are recorded as they are
and corrected in a separate change, not by loosening the rule.

## 5. Windows and WSL2

One conceptual flow, platform differences in small adapters:

* **Windows native.** `vulkaninfo` and a Vulkan-enabled `llama-cli.exe` are required.
  If `vulkaninfo` is absent the Intel adapter is still *listed* (from the Windows adapter
  query: name, PCI ids, driver version) with no available backend, so `jaull scan` shows
  the adapter with its Vulkan backend as `unknown` instead of the GPU being invisible
  and `jaull doctor` reports it as "detected" with Vulkan "not confirmed". `UNVERIFIED`:
  whether `vulkaninfo.exe` is present on a stock Intel-driver install is not established here.
* **WSL2, Vulkan available inside WSL2.** Handled by the same Linux code path as a
  native Linux. Whether the B60 is exposed there (versus a software renderer or a
  translation layer) is `BLOCKED_BY_ENVIRONMENT`.
* **WSL2, only llvmpipe or nothing.** Jaull states it: *"Running inside WSL2: no usable
  Vulkan accelerator is exposed here. This does not show that the Windows host cannot
  accelerate the GPU..."* and no Windows runtime is used.
* **WSL2 → Windows runtime bridge.** *Not implemented, by decision.* It would have to
  settle executable identity (SHA256), explicit path translation, artifact access from
  both sides, log capture, process ownership and error mapping, and must not exist before
  the B60 is known not to work natively in WSL2. A Windows path and a Linux path are never
  mixed without an explicit, verified translation. Proposal if needed: a configured,
  opt-in `windows_llama_cli` setting, resolved with `wslpath`, whose SHA256 is recorded in
  the evidence identity, executed only through an allow-listed argv, never a shell.

## 6. Evidence identity

* `machine_fingerprint` already contains `hardware.os` and `os_version`, so
  Windows-native, WSL2 and Linux runs of the same machine are different evidence.
* `speed_cohort` now includes the Vulkan device and driver for Vulkan records; every
  other backend keeps its previous digest (CUDA and CPU cohorts are byte-identical).
* A fastest/balanced speed reference applies to a Vulkan plan only if the driver is
  known and equal, and the measured `Vulkan<i>` index is proven (from the run's own
  `ggml_vulkan: <i> = <name>` lines) to be the plan's accelerator. This line format is
  `UNVERIFIED` against a real B60 log.
* `runtime_reported_buffer_bytes` (llama.cpp's own buffer report) and
  `observed_device_memory_peak_bytes` stay separate: the former is
  `RuntimeReportedAllocation.total_device_bytes`; there is no Vulkan source of the
  latter, so `peak_vram_bytes` is `None` and a physical VRAM comparison is
  `NOT_COMPARABLE`.
* Historical CUDA records are not rewritten. `ComputeBackendInfo.driver_version` is a new
  optional field; old records load with `None`.

## 7. The `uv run jaull` path: what is covered, and by what

"Integration test" below means: real screens, real estimator, Hardware Fit, launch
policy, ranking, planner and advisor; only the Hub, the machine probes and the
`llama-cli`/`llama-bench` executables are scripted. It shows the flow is wired. It does
not show that a B60 behaves like the script.

| Step of the user path | Covered by | Status |
|---|---|---|
| Scan: detection, Windows listing, WSL2 note | `test_tui_intel_hardware.py` (real `detect_hardware`, faked probes) | integration-tested |
| Hardware screen: card, capacity, backend, warnings | `test_tui_intel_hardware.py`, `test_tui_intel_flow.py` | integration-tested |
| Backend selection (Vulkan / CPU) | same, via `select_runtime_backend` on the scanned profile | integration-tested |
| Requirements → discovery → results | `test_tui_intel_flow.py::test_guided_flow_reaches_a_ready_vulkan_plan…` | integration-tested |
| Memory estimate and offload on the Intel profile | `test_intel_offload_coherence.py` (real estimator) | integration-tested |
| Results: readiness, reason, enabled actions | `test_tui_intel_flow.py` (ready, unidentified, other card, software only) | integration-tested |
| Paths screen refuses and explains | `test_tui_intel_flow.py::test_the_paths_screen…` | integration-tested |
| Run launches the matched `--device` | `test_tui_intel_flow.py::test_run_launches…` (scripted argv) | integration-tested |
| Benchmark measures the matched `-dev`, record keeps OS/device/backend | `test_tui_intel_flow.py::test_benchmark_measures…` | integration-tested |
| Run / Benchmark after a stale `READY` | `test_intel_execution_gates.py` | integration-tested |
| Speed measurement of a blocked plan | `test_intel_execution_gates.py` | unit-tested |
| `jaull doctor` | `test_doctor_intel.py` (checks one by one; not the whole command) | unit-tested |
| NVIDIA through the same flow | `test_tui_intel_flow.py::test_nvidia_still_gets…` plus the existing suite | integration-tested |
| Validate (experiment) screen with a blocked plan | button gating only; the runner's own refusal is pre-existing | partly covered |
| `jaull run` (CLI) | **not gated**: it has no readiness concept, for CUDA either | out of scope |
| Everything that depends on the real B60, driver, `vulkaninfo`/`llama-cli` output | [protocol](biosfer-vulkan-validation-protocol.md) | `UNVERIFIED` / `BLOCKED_BY_ENVIRONMENT` |

Product consequence worth knowing before the physical test: with a known capacity and no
`VK_EXT_memory_budget` figure, Hardware Fit has no availability, so Jaull makes no VRAM
claim and plans on the CPU. If the B60 driver prints no budget, that is what a user will
see until a physical run decides otherwise.

### Offloading numbers

`28/28 blocks` with `--n-gpu-layers 23` appeared in a first probe; that was a fixture
artefact (an evaluation recorded for a 6 GB RTX 2060 combined with Intel hardware). With the
real estimator and launch policy on a simulated B60 (`test_intel_offload_coherence.py`):
a model that fits is `gpu_resident` and launches with `-1` ("all"); a partial offload gives
Hardware Fit *blocks* and a separate llama.cpp *units* value, which are never converted
into each other; with no room there is no offload; and a capacity without a budget is never
claimed as VRAM. The unit↔block mapping exists only for one llama.cpp build on CUDA, so
a Vulkan run is not compared physically against the prediction (`NOT_COMPARABLE`), and the
local-tensor refinement applies to a single discrete CUDA GPU only.

## 8. Automated coverage by module

| Test module | Covers |
|---|---|
| `tests/test_vulkan_device_memory.py` | heap parsing, capacity vs budget, ambiguity, failures |
| `tests/test_windows_adapters.py` | Windows listing, merge, "Intel GPU ≠ Vulkan" end to end |
| `tests/test_execution_environment.py` | Windows / WSL1 / WSL2 / Linux classification and the WSL2 note |
| `tests/test_runtime_device_matching.py` | matching, planner pin, matrix, TUI, `benchmark_plan_speed`, readiness |
| `tests/test_vulkan_evidence_identity.py` | driver identity, cohort environment, device proof |
| `tests/test_vulkan_observation_and_paths.py` | buffer provenance, CRLF device list, paths with spaces |
| `tests/test_intel_execution_gates.py` | blocked or unconfirmed plans cannot be run or measured |
| `tests/test_tui_intel_hardware.py` | scan and hardware screen on Intel and WSL2 |
| `tests/test_doctor_intel.py` | doctor: detected vs executable |
| `tests/test_intel_offload_coherence.py` | blocks vs `--n-gpu-layers` on the real estimator |
| `tests/test_tui_intel_flow.py` | the guided flow through the real screens |

All run without a GPU, network or TTY.

## 9. What remains for the physical machine

See [biosfer-vulkan-validation-protocol.md](biosfer-vulkan-validation-protocol.md). In
short: confirm the B60 is listed by `vulkaninfo`, that heap/budget figures look like what
this design assumes, that `--list-devices` names match, that offload really happens, and
whether WSL2 exposes the card. Until then the Vulkan support is `IMPLEMENTED` and
`UNVERIFIED`, and WSL2 acceleration is not claimed.
