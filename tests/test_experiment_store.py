from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jaull.advisor.service import AdvisorService
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.enums import RepositoryType
from jaull.domain.estimation import (
    CompatibilityAssessment,
    CompatibilityStatus,
    EstimateSource,
    EstimationConfidence,
    KvCacheEstimate,
    MemoryComponent,
    MemoryEstimate,
    RuntimeOverheadEstimate,
    WeightEstimate,
)
from jaull.domain.execution import ExecutionObservation
from jaull.domain.experiments import ExperimentIdentity, ExperimentRecord
from jaull.domain.hardware import ComputeBackend, CpuInfo, HardwareProfile, MemoryInfo
from jaull.domain.inference import InferenceConfiguration, TargetDevice
from jaull.domain.model import ModelAnalysis, ModelRepositoryInfo, SafetensorsSummary
from jaull.domain.runtime import (
    LlamaCppBackendCapability,
    LlamaCppBackendCapabilityState,
    LlamaCppBinaryStatus,
    LlamaCppCapabilityReason,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeBackendSelectionReason,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.evaluation.experiments import build_experiment_record
from jaull.experiments.errors import (
    ExperimentRecordNotFoundError,
    ExperimentStoreError,
    InvalidExperimentIdError,
)
from jaull.experiments.storage import SCHEMA_VERSION, ExperimentStore


def test_save_writes_record_json_under_experiment_id(tmp_path: Path) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)

    path = store.save(record)

    assert path == tmp_path / f"{record.identity.experiment_id}.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["experiment"]["identity"]["experiment_id"] == (
        record.identity.experiment_id
    )


def test_load_roundtrip_preserves_record_semantics(tmp_path: Path) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)
    store.save(record)

    restored = store.load(record.identity.experiment_id)

    assert restored == record


def test_load_legacy_bare_record_json_preserves_compatibility(
    tmp_path: Path,
) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)
    path = store.path_for(record.identity.experiment_id)
    path.write_text(record.model_dump_json(), encoding="utf-8")

    restored = store.load(record.identity.experiment_id)

    assert restored == record


def test_save_same_record_is_idempotent(tmp_path: Path) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)

    first = store.save(record)
    second = store.save(record)

    assert second == first


def test_save_different_record_with_existing_id_is_rejected(tmp_path: Path) -> None:
    identity = ExperimentIdentity.create()
    first = _sample_record(identity=identity)
    second = first.model_copy(update={"notes": ["different"]})
    store = ExperimentStore(root=tmp_path)
    store.save(first)

    with pytest.raises(ExperimentStoreError, match="already exists"):
        store.save(second)


def test_list_ids_returns_sorted_json_records_only(tmp_path: Path) -> None:
    first = _sample_record(
        identity=ExperimentIdentity(
            experiment_id="exp-b",
            created_at=ExperimentIdentity.create().created_at,
        )
    )
    second = _sample_record(
        identity=ExperimentIdentity(
            experiment_id="exp-a",
            created_at=ExperimentIdentity.create().created_at,
        )
    )
    store = ExperimentStore(root=tmp_path)
    store.save(first)
    store.save(second)
    (tmp_path / "not-an-experiment.txt").write_text("ignore", encoding="utf-8")

    assert store.list_ids() == ["exp-a", "exp-b"]


def test_load_missing_record_raises_not_found(tmp_path: Path) -> None:
    store = ExperimentStore(root=tmp_path)

    with pytest.raises(ExperimentRecordNotFoundError):
        store.load("exp-missing")


@pytest.mark.parametrize(
    "experiment_id",
    ["", "../exp-1", "exp/1", "exp\\1", ".hidden", "exp-\x00"],
)
def test_invalid_experiment_ids_are_rejected(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    store = ExperimentStore(root=tmp_path)

    with pytest.raises(InvalidExperimentIdError):
        store.path_for(experiment_id)


def test_corrupt_json_raises_store_error(tmp_path: Path) -> None:
    store = ExperimentStore(root=tmp_path)
    path = store.path_for("exp-bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ExperimentStoreError, match="invalid JSON"):
        store.load("exp-bad")


def test_unsupported_schema_version_raises_store_error(tmp_path: Path) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)
    path = store.path_for(record.identity.experiment_id)
    payload = {
        "schema_version": SCHEMA_VERSION + 1,
        "experiment": record.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentStoreError, match="Unsupported experiment"):
        store.load(record.identity.experiment_id)


def test_default_root_uses_experiments_subdir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, ...]] = []

    def fake_user_data_dir(*subdirs: str) -> Path:
        captured.append(subdirs)
        return Path("/tmp/jaull-data").joinpath(*subdirs)

    monkeypatch.setattr("jaull.experiments.storage.user_data_dir", fake_user_data_dir)

    assert ExperimentStore().root == Path("/tmp/jaull-data/experiments")
    assert captured == [("experiments",)]


def test_advisor_can_save_load_and_list_experiments(tmp_path: Path) -> None:
    record = _sample_record()
    store = ExperimentStore(root=tmp_path)
    advisor = AdvisorService.build(
        hf_client=_FakeHfClient(),  # type: ignore[arg-type]
        detect_hardware=_hardware,
        inspect_model=_unused_inspect,
        estimate_memory=_unused_estimate,
        experiment_store=store,
    )

    saved = advisor.save_experiment_record(record)
    restored = advisor.load_experiment_record(record.identity.experiment_id)

    assert saved == store.path_for(record.identity.experiment_id)
    assert restored == record
    assert advisor.list_experiment_ids() == [record.identity.experiment_id]


def _sample_record(identity: ExperimentIdentity | None = None) -> ExperimentRecord:
    runtime = _runtime()
    capability = _runtime_capability()
    return build_experiment_record(
        hardware=_hardware(),
        artifact=_artifact(),
        runtime=runtime,
        prediction=_estimate(runtime),
        runtime_capability=capability,
        execution_readiness=_readiness(capability),
        observation=_observation(),
        identity=identity,
    )


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu=CpuInfo(model="Test CPU"),
        memory=MemoryInfo(total_bytes=16, available_bytes=8),
    )


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        repo_id="owner/repo",
        revision="main",
        filename="model.gguf",
        format="gguf",
        quantization="Q4_K_M",
        local_path=Path("/models/model.gguf"),
        sha256="1" * 64,
        is_downloaded=True,
        is_verified=True,
    )


def _runtime() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        runtime=RuntimeName.LLAMA_CPP,
        flags=[
            RuntimeFlag(
                name="--ctx-size",
                value="4096",
                source=RuntimeFlagSource.ESTIMATE,
                explanation="test",
            ),
            RuntimeFlag(
                name="--n-gpu-layers",
                value="0",
                source=RuntimeFlagSource.HARDWARE,
                explanation="test",
            ),
        ],
        confidence=EstimationConfidence.HIGH,
    )


def _estimate(runtime: RuntimeRecommendation) -> MemoryEstimate:
    return MemoryEstimate(
        repository=ModelRepositoryInfo(repo_id="owner/repo"),
        repository_type=RepositoryType.GGUF,
        inference_configuration=InferenceConfiguration(
            context_length=4096,
            target_device=TargetDevice.CPU,
            quantization="Q4_K_M",
        ),
        weights=WeightEstimate(
            component=MemoryComponent(
                name="Weights",
                bytes=500,
                source=EstimateSource.EXACT,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            )
        ),
        kv_cache=KvCacheEstimate(
            component=MemoryComponent(
                name="KV",
                bytes=300,
                source=EstimateSource.DERIVED,
                confidence=EstimationConfidence.HIGH,
                explanation="test",
            ),
            layers=1,
            kv_heads=1,
            head_dim=1,
            context_length=4096,
            batch_size=1,
            dtype_bytes=2,
            formula="test",
        ),
        runtime_overhead=RuntimeOverheadEstimate(
            component=MemoryComponent(
                name="Runtime overhead",
                bytes=200,
                source=EstimateSource.ASSUMED,
                confidence=EstimationConfidence.LOW,
                explanation="test",
            ),
            base_bytes=200,
            weight_fraction=0.1,
            minimum_bytes=100,
        ),
        device_reserve=MemoryComponent(
            name="Reserve",
            bytes=100,
            source=EstimateSource.ASSUMED,
            confidence=EstimationConfidence.LOW,
            explanation="test",
        ),
        safety_margin=None,
        total_bytes=1100,
        assessment=CompatibilityAssessment(
            status=CompatibilityStatus.COMPATIBLE,
            confidence=EstimationConfidence.HIGH,
            target_device=TargetDevice.CPU,
            effective_device=TargetDevice.CPU,
            available_ram_bytes=8,
        ),
        runtime_recommendation=runtime,
    )


def _runtime_capability() -> LlamaCppRuntimeCapability:
    return LlamaCppRuntimeCapability(
        binary_path="/usr/bin/llama-cli",
        binary_status=LlamaCppBinaryStatus.AVAILABLE,
        backend_capabilities=[
            LlamaCppBackendCapability(
                backend=ComputeBackend.CPU,
                state=LlamaCppBackendCapabilityState.CONFIRMED,
                reason=LlamaCppCapabilityReason.RUNTIME_AVAILABLE,
                source="test",
            )
        ],
    )


def _readiness(capability: LlamaCppRuntimeCapability):
    from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

    return evaluate_execution_readiness(
        selection=RuntimeBackendSelection(
            selected_backend=ComputeBackend.CPU,
            reason=RuntimeBackendSelectionReason.CPU_FALLBACK,
        ),
        runtime_capability=capability,
    )


def _observation() -> ExecutionObservation:
    return ExecutionObservation(
        success=True,
        duration_seconds=1.0,
        peak_ram_bytes=1000,
        exit_code=0,
    )


class _FakeHfClient:
    def model_info(self, repo_id: str) -> object:
        raise NotImplementedError

    def download_small_file(self, repo_id: str, filename: str) -> Path:
        raise NotImplementedError

    def safetensors_summary(self, repo_id: str) -> SafetensorsSummary | None:
        return None


def _unused_inspect(repo_id: str, client: object | None = None) -> ModelAnalysis:
    del repo_id, client
    raise NotImplementedError


def _unused_estimate(**kwargs: Any) -> MemoryEstimate:
    del kwargs
    raise NotImplementedError
