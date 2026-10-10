import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts import validate_local_tensor_launch as harness

from jaull.domain.benchmarks import LlamaBenchBinaryStatus, LlamaBenchCapability
from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from jaull.runtime import llama_bench_runner
from tests._execution_fixtures import qwen_hardware
from tests._tensor_fixtures import cuda_selection, verified_capability
from tests.test_benchmarks import CUDA_OUTPUT, _FakeExecutionBackend
from tests.test_experiment_record import _record
from tests.test_llama_cpp_tensor_policy import local_case


@pytest.mark.parametrize("execute", [False, True])
def test_validation_preserves_evidence_and_builds_controlled_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execute: bool,
) -> None:
    estimate, context = local_case(tmp_path)
    assert context.index is not None
    assert estimate.runtime_recommendation is not None
    hardware = qwen_hardware().model_copy(
        update={
            "accelerators": [
                AcceleratorProfile(
                    name="RTX 2060",
                    type=AcceleratorType.DEDICATED,
                    vendor=AcceleratorVendor.NVIDIA,
                    backends=[
                        ComputeBackendInfo(
                            backend=ComputeBackend.CUDA,
                            availability=BackendAvailability.AVAILABLE,
                        )
                    ],
                ),
            ]
        }
    )
    record = _record(
        runtime=estimate.runtime_recommendation,
        prediction=estimate,
        hardware=hardware,
        selection=cuda_selection(),
        capability=verified_capability(),
    ).model_copy(update={"artifact": context.index.artifact})
    source = tmp_path / "experiment.json"
    source.write_text(record.model_dump_json(), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "validation"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Offline validation must not detect or execute")

    monkeypatch.setattr(harness, "detect_hardware", forbidden)
    monkeypatch.setattr(harness.HostExecutionBackend, "execute", forbidden)
    backend = _FakeExecutionBackend(stdout=CUDA_OUTPUT)
    if execute:
        monkeypatch.setattr(harness, "detect_hardware", lambda: hardware)
        monkeypatch.setattr(
            harness, "inspect_llama_cpp_runtime", lambda **kwargs: verified_capability(),
        )
        monkeypatch.setattr(harness, "inspect_llama_bench", lambda **kwargs: LlamaBenchCapability(
            binary_status=LlamaBenchBinaryStatus.AVAILABLE,
            version_text="build: 689e227db (10357)",
        ))
        monkeypatch.setattr(harness, "HostExecutionBackend", lambda: backend)
        monkeypatch.setattr(llama_bench_runner.psutil, "cpu_count", lambda logical=True: 4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--record",
            str(source),
            "--output",
            str(output),
            *(["--execute", "--llama-cli", "synthetic-cli", "--llama-bench", "synthetic-bench"]
              if execute else []),
        ],
    )
    assert harness.main() == 0
    snapshot = json.loads((output / "prediction.json").read_text())
    assert snapshot["hardware"] == record.hardware.model_dump(mode="json")
    assert snapshot["estimate_input"] == estimate.model_dump(mode="json")
    assert snapshot["budget"]["requested_units"] == 4
    assert snapshot["source_record_sha256"] == hashlib.sha256(original).hexdigest()
    if execute:
        assert len(backend.requests) == 5
        commands = snapshot["commands"][1:]
        assert [item["label"] for item in commands] == [
            "bench-4", "bench-24", "bench-28", "bench--1",
        ]
        assert [item["command"] for item in snapshot["commands"]] == [
            list(request.command) for request in backend.requests
        ]
        for item in commands:
            command = item["command"]
            assert command[command.index("-t") + 1] == "4"
            assert command[command.index("-ngl") + 1] == item["label"].removeprefix("bench-")
            assert (output / f"{item['label']}-measurements.json").is_file()
    else:
        assert snapshot["commands"] == []
    assert source.read_bytes() == original
    frozen = (output / "prediction.json").read_bytes()
    with pytest.raises(FileExistsError):
        harness.main()
    assert (output / "prediction.json").read_bytes() == frozen
