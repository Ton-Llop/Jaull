"""Evidence identity for Vulkan runs: driver, environment and the measured device.

The ``ggml_vulkan`` line fixtures are synthetic, written from llama.cpp's
documented log shape; they are not captured from an Intel Arc Pro B60.
"""

from __future__ import annotations

from types import SimpleNamespace

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
)
from jaull.evaluation.hardware_fingerprint import machine_fingerprint
from jaull.hardware.vulkan import accelerators_from_summary
from jaull.recommendation.local_evidence import (
    _gpu_environment,
    _runtime_device_is_the_plans_accelerator,
)

B60 = "Intel(R) Arc(TM) Pro B60 Graphics"

SUMMARY_WITH_DRIVER = """
GPU0:
    apiVersion         = 1.4.311
    driverVersion      = 25.2.8
    vendorID           = 0x8086
    deviceID           = 0xe211
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Intel(R) Arc(TM) Pro B60 Graphics
    driverID           = DRIVER_ID_INTEL_OPEN_SOURCE_MESA
    driverName         = Intel open-source Mesa driver
    driverInfo         = Mesa 25.2.8
"""


def _vulkan_accelerator(
    driver_version: str | None, *, name: str = B60, software: bool = False
) -> AcceleratorProfile:
    return AcceleratorProfile(
        name=name,
        vendor=AcceleratorVendor.INTEL,
        type=AcceleratorType.SOFTWARE if software else AcceleratorType.DEDICATED,
        backends=[
            ComputeBackendInfo(
                backend=ComputeBackend.VULKAN,
                availability=BackendAvailability.AVAILABLE,
                driver_name="Intel open-source Mesa driver",
                driver_version=driver_version,
                software_renderer=software,
            )
        ],
    )


def _hardware(
    *accelerators: AcceleratorProfile,
    os_name: str = "Windows 11",
    gpus: list[GpuInfo] | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        os=os_name,
        arch="x86_64",
        cpu=CpuInfo(model="cpu", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=16 * 1024**3),
        gpus=gpus or [],
        accelerators=list(accelerators),
    )


def test_summary_driver_version_is_kept_on_the_vulkan_backend() -> None:
    [accelerator] = accelerators_from_summary(SUMMARY_WITH_DRIVER)

    vulkan = next(item for item in accelerator.backends if item.backend is ComputeBackend.VULKAN)
    assert vulkan.driver_name == "Intel open-source Mesa driver"
    assert vulkan.driver_version == "Mesa 25.2.8"


def test_driver_version_falls_back_to_the_numeric_field() -> None:
    summary = SUMMARY_WITH_DRIVER.replace("    driverInfo         = Mesa 25.2.8\n", "")

    [accelerator] = accelerators_from_summary(summary)

    vulkan = next(item for item in accelerator.backends if item.backend is ComputeBackend.VULKAN)
    assert vulkan.driver_version == "25.2.8"


def test_a_summary_with_no_driver_version_leaves_it_unknown() -> None:
    summary = "\n".join(
        line
        for line in SUMMARY_WITH_DRIVER.splitlines()
        if "driverVersion" not in line and "driverInfo" not in line
    )

    [accelerator] = accelerators_from_summary(summary)

    vulkan = next(item for item in accelerator.backends if item.backend is ComputeBackend.VULKAN)
    assert vulkan.driver_version is None


def test_vulkan_environment_is_the_accelerator_and_its_driver() -> None:
    hardware = _hardware(_vulkan_accelerator("Mesa 25.2.8"))

    assert _gpu_environment(hardware, ComputeBackend.VULKAN) == ((B60, "Mesa 25.2.8", None),)


def test_a_different_driver_is_a_different_environment() -> None:
    old = _hardware(_vulkan_accelerator("Mesa 25.0.1"))
    new = _hardware(_vulkan_accelerator("Mesa 25.2.8"))

    assert _gpu_environment(old, ComputeBackend.VULKAN) != _gpu_environment(
        new, ComputeBackend.VULKAN
    )


def test_an_unknown_vulkan_driver_stays_unknown_for_the_caller_to_refuse() -> None:
    hardware = _hardware(_vulkan_accelerator(None))

    assert _gpu_environment(hardware, ComputeBackend.VULKAN) == ((B60, None, None),)


def test_software_renderers_are_not_part_of_the_environment() -> None:
    hardware = _hardware(_vulkan_accelerator("1", name="llvmpipe", software=True))

    assert _gpu_environment(hardware, ComputeBackend.VULKAN) == ()


def test_cuda_environment_is_still_read_from_nvml() -> None:
    gpu = GpuInfo(
        name="NVIDIA GeForce RTX 2060",
        vram_total_bytes=6 * 1024**3,
        vram_available_bytes=5 * 1024**3,
        driver_version="617.14",
        cuda_version="13.0",
    )
    hardware = _hardware(_vulkan_accelerator("Mesa 25.2.8"), gpus=[gpu])

    assert _gpu_environment(hardware, ComputeBackend.CUDA) == (
        ("NVIDIA GeForce RTX 2060", "617.14", "13.0"),
    )


def test_windows_and_wsl2_on_the_same_machine_are_different_evidence() -> None:
    windows = _hardware(_vulkan_accelerator("Mesa 25.2.8"), os_name="Windows 11")
    wsl = _hardware(
        _vulkan_accelerator("Mesa 25.2.8"), os_name="Linux 5.15.153.1-microsoft-standard-WSL2"
    )

    assert machine_fingerprint(windows) != machine_fingerprint(wsl)


# ---------------------------------------------------------------------------
# "Is the measured Vulkan device the plan's accelerator?"
# ---------------------------------------------------------------------------
def _plan(name: str | None = B60) -> SimpleNamespace:
    accelerator = SimpleNamespace(name=name) if name is not None else None
    return SimpleNamespace(backend_selection=SimpleNamespace(selected_accelerator=accelerator))


def _record(stderr: str, device: str = "Vulkan0") -> SimpleNamespace:
    return SimpleNamespace(
        requested_device=device,
        observation=SimpleNamespace(raw_stdout="", raw_stderr=stderr),
    )


def _ggml(*devices: str) -> str:
    lines = [f"ggml_vulkan: Found {len(devices)} Vulkan devices:"]
    lines += [
        f"ggml_vulkan: {index} = {device} | uma: 0 | fp16: 1 | warp size: 32"
        for index, device in enumerate(devices)
    ]
    return "\n".join(lines)


def test_vulkan_device_is_proven_by_the_runtimes_own_enumeration() -> None:
    record = _record(_ggml(f"{B60} (Intel open-source Mesa driver)"))

    assert _runtime_device_is_the_plans_accelerator(_plan(), record)


def test_the_index_in_the_request_must_point_at_the_named_device() -> None:
    stderr = _ggml(
        "Intel(R) Iris(R) Xe Graphics (Intel open-source Mesa driver)",
        f"{B60} (Intel open-source Mesa driver)",
    )

    assert _runtime_device_is_the_plans_accelerator(_plan(), _record(stderr, "Vulkan1"))
    assert not _runtime_device_is_the_plans_accelerator(_plan(), _record(stderr, "Vulkan0"))


def test_two_identical_vulkan_cards_prove_nothing() -> None:
    stderr = _ggml(f"{B60} (Mesa)", f"{B60} (Mesa)")

    assert not _runtime_device_is_the_plans_accelerator(_plan(), _record(stderr, "Vulkan0"))


def test_unrelated_or_missing_enumeration_proves_nothing() -> None:
    assert not _runtime_device_is_the_plans_accelerator(_plan(), _record(""))
    assert not _runtime_device_is_the_plans_accelerator(_plan(None), _record(_ggml(B60)))
    assert not _runtime_device_is_the_plans_accelerator(
        _plan(), _record(_ggml("NVIDIA GeForce RTX 2060 (NVIDIA)"))
    )
