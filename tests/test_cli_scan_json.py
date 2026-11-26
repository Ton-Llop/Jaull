from __future__ import annotations

import json

import pytest

from jaull.cli.scan import run_scan
from jaull.domain.hardware import CpuInfo, HardwareProfile, MemoryInfo

GIB = 1024**3


class _Advisor:
    def scan_hardware(self) -> HardwareProfile:
        return HardwareProfile(
            os="Windows",
            arch="AMD64",
            cpu=CpuInfo(model="Test", physical_cores=6, logical_cores=12),
            memory=MemoryInfo(total_bytes=16 * GIB, available_bytes=8 * GIB),
        )


def test_scan_json_is_the_hardware_profile_for_other_programs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The desktop app reads the hardware from this output instead of the Rich table.
    assert run_scan(_Advisor(), as_json=True) == 0  # type: ignore[arg-type]

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["hardware"]["os"] == "Windows"
    assert payload["hardware"]["memory"]["available_bytes"] == 8 * GIB
    assert payload["hardware"]["cpu"]["logical_cores"] == 12
