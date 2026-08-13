"""Facade over hardware, HF, estimator and diagnostics services.

Every CLI subcommand and every TUI screen goes through an ``AdvisorService``
instance so the wiring is defined in exactly one place. ``AdvisorService`` does
not add logic — it only delegates — but by centralising the composition it
lets tests inject fakes without monkeypatching import paths and it keeps
front-end modules free of ``HfClient()``/``detect_hardware`` construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.diagnostics.service import collect_diagnostics as _default_diagnostics
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.estimation import MemoryEstimate
from jaull.domain.execution import ExecutionObservation, InferenceResult
from jaull.domain.experiments import (
    ExperimentBackendTrace,
    ExperimentEnvironment,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentRequest,
    ExperimentRunResult,
    ExperimentWorkload,
)
from jaull.domain.hardware import HardwareProfile
from jaull.domain.inference import InferenceConfiguration
from jaull.domain.model import DiagnosticResult, ModelAnalysis
from jaull.domain.requirements import UserAnswers
from jaull.domain.runtime import (
    ExecutionReadiness,
    LlamaCppRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeRecommendation,
)
from jaull.huggingface.artifact_resolver import HuggingFaceArtifactResolver
from jaull.huggingface.client import HfClientProtocol
from jaull.metadata.range_reader import HttpRangeClient
from jaull.workflow.container import (
    DetectHardwareFn,
    EstimateMemoryFn,
    InspectModelFn,
    ServiceContainer,
)
from jaull.workflow.progress import ProgressCallback
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.execution.ports import ExecutionBackendProtocol
    from jaull.experiments.runner import ExperimentRunner
    from jaull.experiments.storage import ExperimentStore
    from jaull.runtime.llama_cpp_runner import LlamaCppRunner

DiagnosticsFn = Callable[[], list[DiagnosticResult]]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class AdvisorService:
    """Application-layer facade used by CLI and TUI.

    All fields are DI'd: tests build an ``AdvisorService`` with fakes and never
    touch the network. Production callers use :meth:`default` which wires the
    real services.
    """

    services: ServiceContainer
    collect_diagnostics: DiagnosticsFn = field(default=_default_diagnostics)
    artifacts: ArtifactService | None = field(default=None)
    llama_cpp_runner: LlamaCppRunner | None = field(default=None)
    experiment_runner: ExperimentRunner | None = field(default=None)
    experiment_store: ExperimentStore | None = field(default=None)
    llama_cli_path: str | Path | None = field(default=None)
    llama_cli_timeout_seconds: float = field(default=300.0)

    # ------------------------------------------------------------------
    # Simple pass-throughs — kept as methods so tests can spy on them and
    # so the front-ends never need to know about ServiceContainer.
    # ------------------------------------------------------------------
    def scan_hardware(
        self, on_progress: ProgressCallback | None = None
    ) -> HardwareProfile:
        if on_progress is None:
            return self.services.detect_hardware()
        # Delegate to the orchestrator's progress-aware helper: it wires the
        # per-probe callback into ``detect_hardware`` and reports each step as
        # its NVML/psutil call returns.
        from jaull.workflow import orchestrator

        return orchestrator.scan_hardware(self.services, on_progress=on_progress)

    def diagnostics(self) -> list[DiagnosticResult]:
        return self.collect_diagnostics()

    def inspect_model(self, repo_id: str) -> ModelAnalysis:
        return self.services.inspect_model(repo_id, client=self.services.hf_client)

    def estimate_model(
        self,
        analysis: ModelAnalysis,
        hardware: HardwareProfile,
        inference_cfg: InferenceConfiguration,
        *,
        resolve_base_model: bool = True,
        recommend_runtime: bool = True,
        range_client: HttpRangeClient | None = None,
    ) -> MemoryEstimate:
        effective_range = range_client
        if effective_range is None and resolve_base_model:
            effective_range = self._make_range_client()
        return self.services.estimate_memory(
            analysis=analysis,
            hardware=hardware,
            inference_cfg=inference_cfg,
            client=self.services.hf_client,
            resolve_base_model=resolve_base_model,
            range_client=effective_range,
            recommend_runtime=recommend_runtime,
        )

    def recommend(
        self,
        answers: UserAnswers,
        *,
        hardware: HardwareProfile | None = None,
        on_progress: ProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> RecommendationWorkflowState:
        # Local import: the orchestrator imports discovery + recommendation,
        # which we don't want to drag in just because someone constructs an
        # ``AdvisorService`` for a CLI subcommand that never runs the guided
        # flow.
        from jaull.workflow import orchestrator

        hw = hardware or self.scan_hardware()
        return orchestrator.run_workflow(
            answers=answers,
            hardware=hw,
            services=self.services,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )

    # ------------------------------------------------------------------
    # Artifact resolution / download / verification
    # ------------------------------------------------------------------
    def resolve_artifact(
        self,
        repo_id: str,
        quantization: str | None = None,
        revision: str | None = None,
    ) -> ModelArtifact:
        return self._artifacts().resolve(repo_id, quantization, revision)

    def download_artifact(
        self,
        artifact: ModelArtifact,
        on_progress: object | None = None,
    ) -> ModelArtifact:
        # ``on_progress`` accepted for forward compatibility; see
        # ``ArtifactService.download`` for the current wiring.
        return self._artifacts().download(artifact, on_progress)  # type: ignore[arg-type]

    def verify_artifact(
        self,
        artifact: ModelArtifact,
        *,
        full: bool = False,
    ) -> ModelArtifact:
        return self._artifacts().verify(artifact, full=full)

    def run_artifact(
        self,
        *,
        artifact: ModelArtifact,
        prompt: str,
        runtime: RuntimeRecommendation | None = None,
    ) -> InferenceResult:
        return self._llama_cpp_runner().run(
            artifact=artifact,
            prompt=prompt,
            runtime=runtime,
        )

    def select_runtime_backend(
        self,
        hardware: HardwareProfile | None = None,
    ) -> RuntimeBackendSelection:
        from jaull.runtime.backend_selection import select_runtime_backend

        return select_runtime_backend(hardware or self.scan_hardware())

    def inspect_llama_cpp_runtime(
        self,
        *,
        backend: ExecutionBackendProtocol | None = None,
    ) -> LlamaCppRuntimeCapability:
        from jaull.execution.host import HostExecutionBackend
        from jaull.runtime.llama_cpp_capability import inspect_llama_cpp_runtime

        return inspect_llama_cpp_runtime(
            backend=backend or HostExecutionBackend(),
            llama_cli_path=self.llama_cli_path,
            timeout_seconds=self.llama_cli_timeout_seconds,
        )

    def evaluate_execution_readiness(
        self,
        *,
        selection: RuntimeBackendSelection | None = None,
        runtime_capability: LlamaCppRuntimeCapability | None = None,
        hardware: HardwareProfile | None = None,
        backend: ExecutionBackendProtocol | None = None,
    ) -> ExecutionReadiness:
        from jaull.runtime.llama_cpp_capability import evaluate_execution_readiness

        effective_selection = selection or self.select_runtime_backend(hardware)
        effective_capability = runtime_capability or self.inspect_llama_cpp_runtime(
            backend=backend
        )
        return evaluate_execution_readiness(
            selection=effective_selection,
            runtime_capability=effective_capability,
        )

    def build_experiment_record(
        self,
        *,
        hardware: HardwareProfile,
        artifact: ModelArtifact,
        workload: ExperimentWorkload | None = None,
        backend_trace: ExperimentBackendTrace | None = None,
        runtime: RuntimeRecommendation,
        prediction: MemoryEstimate,
        runtime_capability: LlamaCppRuntimeCapability,
        execution_readiness: ExecutionReadiness,
        observation: ExecutionObservation,
        identity: ExperimentIdentity | None = None,
        environment: ExperimentEnvironment | None = None,
        notes: list[str] | None = None,
    ) -> ExperimentRecord:
        from jaull.evaluation.experiments import build_experiment_record

        return build_experiment_record(
            hardware=hardware,
            artifact=artifact,
            workload=workload,
            backend_trace=backend_trace,
            runtime=runtime,
            prediction=prediction,
            runtime_capability=runtime_capability,
            execution_readiness=execution_readiness,
            observation=observation,
            identity=identity,
            environment=environment,
            notes=notes,
        )

    def save_experiment_record(self, record: ExperimentRecord) -> Path:
        return self._experiment_store().save(record)

    def load_experiment_record(self, experiment_id: str) -> ExperimentRecord:
        return self._experiment_store().load(experiment_id)

    def list_experiment_ids(self) -> list[str]:
        return self._experiment_store().list_ids()

    def run_experiment(self, request: ExperimentRequest) -> ExperimentRunResult:
        return self._experiment_runner().run(request)

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------
    def _artifacts(self) -> ArtifactService:
        """Return the configured ``ArtifactService`` or build a lazy default.

        Direct constructions like ``AdvisorService(services=...)`` do not
        supply an ``ArtifactService``; instead of failing, we lazy-build one
        from the same ``hf_client``. Uses ``object.__setattr__`` because the
        dataclass is frozen — the mutation is a one-time memoisation, not
        semantic change.
        """
        if self.artifacts is not None:
            return self.artifacts
        fresh = ArtifactService(
            resolver=HuggingFaceArtifactResolver(self.services.hf_client),
            storage=ArtifactStorage(),
        )
        object.__setattr__(self, "artifacts", fresh)
        return fresh

    def _llama_cpp_runner(self) -> LlamaCppRunner:
        """Return a configured llama.cpp runner, constructing it only when used."""
        if self.llama_cpp_runner is not None:
            return self.llama_cpp_runner
        from jaull.execution.host import HostExecutionBackend
        from jaull.runtime.llama_cpp_runner import LlamaCppRunner

        fresh = LlamaCppRunner(
            backend=HostExecutionBackend(),
            llama_cli_path=self.llama_cli_path,
            timeout_seconds=self.llama_cli_timeout_seconds,
        )
        object.__setattr__(self, "llama_cpp_runner", fresh)
        return fresh

    def _experiment_store(self) -> ExperimentStore:
        if self.experiment_store is not None:
            return self.experiment_store
        from jaull.experiments.storage import ExperimentStore

        fresh = ExperimentStore()
        object.__setattr__(self, "experiment_store", fresh)
        return fresh

    def _experiment_runner(self) -> ExperimentRunner:
        if self.experiment_runner is not None:
            return self.experiment_runner
        from jaull.execution.host import HostExecutionBackend
        from jaull.experiments.runner import ExperimentRunner

        fresh = ExperimentRunner(
            execution_backend=HostExecutionBackend(),
            llama_cpp_runner=self._llama_cpp_runner(),
            store=self._experiment_store(),
            llama_cli_path=self.llama_cli_path,
            capability_timeout_seconds=self.llama_cli_timeout_seconds,
        )
        object.__setattr__(self, "experiment_runner", fresh)
        return fresh

    def _make_range_client(self) -> HttpRangeClient | None:
        factory = self.services.range_client_factory
        if factory is None:
            return None
        client = factory()
        return client  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def default(
        cls,
        *,
        llama_cli_path: str | Path | None = None,
        llama_cli_timeout_seconds: float = 300.0,
    ) -> AdvisorService:
        """Production wiring: real HF client, real hardware probes, real estimator."""
        services = ServiceContainer.default()
        artifacts = ArtifactService(
            resolver=HuggingFaceArtifactResolver(services.hf_client),
            storage=ArtifactStorage(),
        )
        return cls(
            services=services,
            artifacts=artifacts,
            llama_cli_path=llama_cli_path,
            llama_cli_timeout_seconds=llama_cli_timeout_seconds,
        )

    @classmethod
    def build(
        cls,
        *,
        hf_client: HfClientProtocol,
        detect_hardware: DetectHardwareFn,
        inspect_model: InspectModelFn,
        estimate_memory: EstimateMemoryFn,
        collect_diagnostics: DiagnosticsFn = _default_diagnostics,
        artifacts: ArtifactService | None = None,
        llama_cpp_runner: LlamaCppRunner | None = None,
        experiment_runner: ExperimentRunner | None = None,
        experiment_store: ExperimentStore | None = None,
        llama_cli_path: str | Path | None = None,
        llama_cli_timeout_seconds: float = 300.0,
    ) -> AdvisorService:
        """Test wiring: assemble a ServiceContainer from callables and wrap it."""
        from jaull.discovery.search_client import HfSearchClient
        from jaull.recommendation.capability import MetadataCapabilityAnalyzer

        services = ServiceContainer(
            hf_client=hf_client,
            search_client=HfSearchClient(),
            detect_hardware=detect_hardware,
            inspect_model=inspect_model,
            estimate_memory=estimate_memory,
            capability_analyzer=MetadataCapabilityAnalyzer(),
            range_client_factory=None,
        )
        return cls(
            services=services,
            collect_diagnostics=collect_diagnostics,
            artifacts=artifacts,
            llama_cpp_runner=llama_cpp_runner,
            experiment_runner=experiment_runner,
            experiment_store=experiment_store,
            llama_cli_path=llama_cli_path,
            llama_cli_timeout_seconds=llama_cli_timeout_seconds,
        )


__all__ = ["AdvisorService", "CancelCheck", "DiagnosticsFn"]
