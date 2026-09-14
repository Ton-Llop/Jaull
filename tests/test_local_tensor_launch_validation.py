import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts import validate_local_tensor_launch as harness

from jaull.domain.hardware import (
    AcceleratorProfile,
    AcceleratorType,
    AcceleratorVendor,
    BackendAvailability,
    ComputeBackend,
    ComputeBackendInfo,
)
from tests._execution_fixtures import qwen_hardware
from tests._tensor_fixtures import cuda_selection, verified_capability
from tests.test_experiment_record import _record
from tests.test_llama_cpp_tensor_policy import local_case


def test_offline_validation_uses_frozen_inputs_and_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--record",
            str(source),
            "--output",
            str(output),
        ],
    )
    assert harness.main() == 0
    snapshot = json.loads((output / "prediction.json").read_text())
    assert snapshot["hardware"] == record.hardware.model_dump(mode="json")
    assert snapshot["estimate_input"] == estimate.model_dump(mode="json")
    assert snapshot["budget"]["requested_units"] == 4
    assert snapshot["source_record_sha256"] == hashlib.sha256(original).hexdigest()
    assert snapshot["commands"] == []
    assert source.read_bytes() == original
    frozen = (output / "prediction.json").read_bytes()
    with pytest.raises(FileExistsError):
        harness.main()
    assert (output / "prediction.json").read_bytes() == frozen
