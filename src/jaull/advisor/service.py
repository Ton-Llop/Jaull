"""Facade over hardware, HF, estimator and diagnostics services.

Every CLI subcommand and every TUI screen goes through an ``AdvisorService``
instance so the wiring is defined in exactly one place. ``AdvisorService`` does
not add logic — it only delegates — but by centralising the composition it
lets tests inject fakes without monkeypatching import paths and it keeps
front-end modules free of ``HfClient()``/``detect_hardware`` construction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from jaull.artifacts.service import ArtifactService
from jaull.artifacts.storage import ArtifactStorage
from jaull.bootstrap.container import (
    DetectHardwareFn,
    EstimateMemoryFn,
    InspectModelFn,
    ServiceContainer,
)
from jaull.diagnostics.service import collect_diagnostics as _default_diagnostics
from jaull.domain.artifacts import ModelArtifact
from jaull.domain.benchmarks import (
    BenchmarkRecord,
    BenchmarkRequest,
    BenchmarkRunResult,
)
from jaull.domain.candidates import ModelCandidate
from jaull.domain.estimation import EstimationConfidence, MemoryEstimate
from jaull.domain.execution import ExecutionObservation, InferenceResult
from jaull.domain.execution_plans import (
    ArtifactVariant,
    ArtifactVariantFormat,
    ExecutionPlan,
    ModelIdentity,
    PreparedExecutionPlan,
    logical_model_repo_key,
)
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
from jaull.domain.inference import InferenceConfiguration, WeightPrecision
from jaull.domain.model import DiagnosticResult, ModelAnalysis
from jaull.domain.requirements import UserAnswers
from jaull.domain.runtime import (
    ExecutionReadiness,
    ExecutionReadinessStatus,
    LlamaCppRuntimeCapability,
    PyTorchRuntimeCapability,
    RuntimeBackendSelection,
    RuntimeCapability,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
    RuntimeRecommendation,
)
from jaull.huggingface.artifact_resolver import HuggingFaceArtifactResolver
from jaull.huggingface.client import HfClientProtocol
from jaull.metadata.range_reader import HttpRangeClient
from jaull.ports.cache import ModelAnalysisCacheProtocol
from jaull.recommendation.engine_v2 import PlanRankingContext
from jaull.workflow.progress import ProgressCallback
from jaull.workflow.state import RecommendationWorkflowState

if TYPE_CHECKING:
    from jaull.benchmarks.matrix import (
        BenchmarkMatrixRequest,
        BenchmarkMatrixResult,
        BenchmarkMatrixRunner,
    )
    from jaull.benchmarks.storage import BenchmarkStore
    from jaull.domain.runtime import LlamaCppInstallation, PyTorchInstallation
    from jaull.evaluation.benchmark_comparison import BenchmarkComparison
    from jaull.execution.ports import ExecutionBackendProtocol
    from jaull.experiments.runner import ExperimentRunner
    from jaull.experiments.storage import ExperimentStore
    from jaull.recommendation.models import ModelRecommendation
    from jaull.runtime.llama_bench_runner import LlamaBenchRunner
    from jaull.runtime.llama_cpp_runner import LlamaCppRunner
    from jaull.runtime.locator import RuntimeLocator
    from jaull.runtime.transformers_benchmark_runner import TransformersBenchmarkRunner
    from jaull.runtime.transformers_runner import TransformersRunner

DiagnosticsFn = Callable[[], list[DiagnosticResult]]
CancelCheck = Callable[[], bool]
logger = logging.getLogger(__name__)


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
    transformers_runner: TransformersRunner | None = field(default=None)
    experiment_runner: ExperimentRunner | None = field(default=None)
    experiment_store: ExperimentStore | None = field(default=None)
    llama_bench_runner: LlamaBenchRunner | None = field(default=None)
    transformers_benchmark_runner: TransformersBenchmarkRunner | None = field(default=None)
    benchmark_matrix_runner: BenchmarkMatrixRunner | None = field(default=None)
    benchmark_store: BenchmarkStore | None = field(default=None)
    llama_cli_path: str | Path | None = field(default=None)
    llama_cli_timeout_seconds: float = field(default=300.0)
    python_executable: str | Path | None = field(default=None)
    pytorch_probe_timeout_seconds: float = field(default=10.0)
    llama_bench_path: str | Path | None = field(default=None)
    llama_bench_timeout_seconds: float = field(default=900.0)
    runtime_locator: RuntimeLocator | None = field(default=None)
    llama_cpp_installation: LlamaCppInstallation | None = field(default=None)
    pytorch_installation: PyTorchInstallation | None = field(default=None)

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
        if self.collect_diagnostics is _default_diagnostics:
            return _default_diagnostics(runtime_locator=self._runtime_locator())
        return self.collect_diagnostics()

    def inspect_model(self, repo_id: str) -> ModelAnalysis:
        candidate = ModelCandidate(repo_id=repo_id)
        cached = self._cached_model_analysis(candidate)
        if cached is not None:
            return cached
        analysis = self._inspect_model_live(repo_id)
        self._store_model_analysis(candidate, analysis)
        return analysis

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
        plan_context = self._plan_ranking_context(hw)
        return orchestrator.run_workflow(
            answers=answers,
            hardware=hw,
            services=self.services,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
            plan_context=plan_context,
        )

    def _plan_ranking_context(self, hardware: HardwareProfile) -> PlanRankingContext:
        backend_selection = self._plan_backend_selection(hardware)
        return PlanRankingContext(
            hardware=hardware,
            backend_selection=backend_selection,
            readiness_by_runtime=self._plan_runtime_readiness(
                hardware=hardware,
                selection=backend_selection,
            ),
            benchmark_records=self._stored_benchmark_records(),
            experiment_records=self._stored_experiment_records(),
        )

    def _plan_backend_selection(
        self,
        hardware: HardwareProfile,
    ) -> RuntimeBackendSelection | None:
        try:
            return self.select_runtime_backend(hardware)
        except Exception:
            logger.debug(
                "Could not select a backend for recommendation plan ranking.",
                exc_info=True,
            )
            return None

    def _plan_runtime_readiness(
        self,
        *,
        hardware: HardwareProfile,
        selection: RuntimeBackendSelection | None,
    ) -> dict[RuntimeName, ExecutionReadiness] | None:
        if selection is None:
            return None
        readiness: dict[RuntimeName, ExecutionReadiness] = {}
        try:
            readiness[RuntimeName.LLAMA_CPP] = self.evaluate_execution_readiness(
                selection=selection,
                hardware=hardware,
            )
        except Exception:
            logger.debug(
                "Could not evaluate llama.cpp readiness for recommendation ranking.",
                exc_info=True,
            )
        try:
            readiness[RuntimeName.TRANSFORMERS] = (
                self.evaluate_pytorch_execution_readiness(
                    selection=selection,
                    hardware=hardware,
                )
            )
        except Exception:
            logger.debug(
                "Could not evaluate PyTorch readiness for recommendation ranking.",
                exc_info=True,
            )
        return readiness or None

    def _stored_benchmark_records(self) -> list[BenchmarkRecord]:
        try:
            benchmark_ids = self.list_benchmark_ids()
        except Exception:
            logger.debug("Could not list local benchmark evidence.", exc_info=True)
            return []
        records: list[BenchmarkRecord] = []
        for benchmark_id in benchmark_ids:
            try:
                records.append(self.load_benchmark_record(benchmark_id))
            except Exception:
                logger.debug(
                    "Ignoring unreadable local benchmark evidence %s.",
                    benchmark_id,
                    exc_info=True,
                )
        return records

    def _stored_experiment_records(self) -> list[ExperimentRecord]:
        try:
            experiment_ids = self.list_experiment_ids()
        except Exception:
            logger.debug("Could not list local experiment evidence.", exc_info=True)
            return []
        records: list[ExperimentRecord] = []
        for experiment_id in experiment_ids:
            try:
                records.append(self.load_experiment_record(experiment_id))
            except Exception:
                logger.debug(
                    "Ignoring unreadable local experiment evidence %s.",
                    experiment_id,
                    exc_info=True,
                )
        return records

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
        if runtime is not None and runtime.runtime is RuntimeName.TRANSFORMERS:
            return self._transformers_runner().run(
                artifact=artifact,
                prompt=prompt,
                runtime=runtime,
            )
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
        selection: RuntimeBackendSelection | None = None,
        hardware: HardwareProfile | None = None,
    ) -> LlamaCppRuntimeCapability:
        from jaull.domain.runtime import LlamaCppBinaryStatus, RuntimeResolutionStatus
        from jaull.execution.host import HostExecutionBackend
        from jaull.runtime.llama_cpp_capability import inspect_llama_cpp_runtime

        effective_selection = selection
        if effective_selection is None and hardware is not None:
            effective_selection = self.select_runtime_backend(hardware)
        installation = (
            self._select_llama_cpp_installation(
                selection=effective_selection,
                backend=backend or HostExecutionBackend(),
            )
            if effective_selection is not None
            else self._resolved_llama_cpp_installation()
        )
        if installation.llama_cli is None:
            return LlamaCppRuntimeCapability(
                binary_path=None,
                binary_status=LlamaCppBinaryStatus.MISSING,
                message=installation.discovery.message,
            )
        if installation.status is RuntimeResolutionStatus.RUNTIME_NOT_EXECUTABLE:
            return LlamaCppRuntimeCapability(
                binary_path=installation.llama_cli,
                binary_status=LlamaCppBinaryStatus.NOT_EXECUTABLE,
                message=installation.discovery.message,
            )
        if installation.status in {
            RuntimeResolutionStatus.CONFIGURED_RUNTIME_MISSING,
            RuntimeResolutionStatus.RUNTIME_NOT_FOUND,
        }:
            return LlamaCppRuntimeCapability(
                binary_path=installation.llama_cli,
                binary_status=LlamaCppBinaryStatus.MISSING,
                message=installation.discovery.message,
            )
        return inspect_llama_cpp_runtime(
            backend=backend or HostExecutionBackend(),
            llama_cli_path=installation.llama_cli,
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
            backend=backend,
            selection=effective_selection,
        )
        return evaluate_execution_readiness(
            selection=effective_selection,
            runtime_capability=effective_capability,
        )

    def inspect_pytorch_runtime(
        self,
        *,
        backend: ExecutionBackendProtocol | None = None,
    ) -> PyTorchRuntimeCapability:
        from jaull.execution.host import HostExecutionBackend
        from jaull.runtime.pytorch_capability import inspect_pytorch_runtime

        installation = self._resolved_pytorch_installation()
        return inspect_pytorch_runtime(
            backend=backend or HostExecutionBackend(),
            python_executable=installation.python_executable,
            timeout_seconds=self.pytorch_probe_timeout_seconds,
        )

    def evaluate_pytorch_execution_readiness(
        self,
        *,
        selection: RuntimeBackendSelection | None = None,
        runtime_capability: PyTorchRuntimeCapability | None = None,
        hardware: HardwareProfile | None = None,
        backend: ExecutionBackendProtocol | None = None,
    ) -> ExecutionReadiness:
        from jaull.runtime.pytorch_capability import (
            evaluate_pytorch_execution_readiness,
        )

        effective_selection = selection or self.select_runtime_backend(hardware)
        effective_capability = runtime_capability or self.inspect_pytorch_runtime(
            backend=backend
        )
        return evaluate_pytorch_execution_readiness(
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
        runtime_capability: RuntimeCapability,
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

    def run_benchmark(
        self,
        request: BenchmarkRequest,
        *,
        hardware: HardwareProfile,
        persist: bool = True,
    ) -> BenchmarkRunResult:
        if request.runtime.runtime is RuntimeName.TRANSFORMERS:
            return self._run_transformers_benchmark(
                request,
                hardware=hardware,
                persist=persist,
            )

        from jaull.runtime.llama_bench_capability import inspect_llama_bench

        installation = self._resolved_llama_cpp_installation()
        observation = self._llama_bench_runner().run(request)
        capability = inspect_llama_bench(
            backend=self._host_execution_backend(),
            llama_bench_path=installation.llama_bench,
            timeout_seconds=min(self.llama_bench_timeout_seconds, 30.0),
        )
        record = BenchmarkRecord.create(
            hardware=hardware,
            request=request,
            observation=observation,
            llama_bench_capability=capability,
        )
        persisted_path = self._benchmark_store().save(record) if persist else None
        return BenchmarkRunResult(record=record, persisted_path=persisted_path)

    def _run_transformers_benchmark(
        self,
        request: BenchmarkRequest,
        *,
        hardware: HardwareProfile,
        persist: bool,
    ) -> BenchmarkRunResult:
        from jaull.benchmarks.errors import BenchmarkUnavailableError
        from jaull.domain.runtime import PyTorchRuntimeStatus

        capability = self.inspect_pytorch_runtime()
        if capability.runtime_status is not PyTorchRuntimeStatus.AVAILABLE:
            raise BenchmarkUnavailableError(
                capability.message
                or f"PyTorch runtime is {capability.runtime_status.value}."
            )
        observation = self._transformers_benchmark_runner().run(request)
        record = BenchmarkRecord.create(
            hardware=hardware,
            request=request,
            observation=observation,
            runtime_capability=capability,
        )
        persisted_path = self._benchmark_store().save(record) if persist else None
        return BenchmarkRunResult(record=record, persisted_path=persisted_path)

    def run_benchmark_matrix(
        self,
        request: BenchmarkMatrixRequest,
    ) -> BenchmarkMatrixResult:
        return self._benchmark_matrix_runner().run(request)

    def save_benchmark_record(self, record: BenchmarkRecord) -> Path:
        return self._benchmark_store().save(record)

    def load_benchmark_record(self, benchmark_id: str) -> BenchmarkRecord:
        return self._benchmark_store().load(benchmark_id)

    def list_benchmark_ids(self) -> list[str]:
        return self._benchmark_store().list_ids()

    def benchmark_records_for_model(
        self,
        model_identity: ModelIdentity,
    ) -> list[BenchmarkRecord]:
        records: list[BenchmarkRecord] = []
        for benchmark_id in self.list_benchmark_ids():
            record = self.load_benchmark_record(benchmark_id)
            if self._benchmark_matches_model(record, model_identity):
                records.append(record)
        return sorted(records, key=lambda item: item.identity.created_at)

    def resolve_model_identity(
        self,
        recommendation: ModelRecommendation,
    ) -> ModelIdentity:
        from jaull.execution_plans import resolve_model_identity

        return resolve_model_identity(
            candidate=recommendation.evaluated.candidate,
            analysis=recommendation.evaluated.analysis,
        )

    def artifact_variant_for_recommendation(
        self,
        recommendation: ModelRecommendation,
    ) -> ArtifactVariant:
        from jaull.execution_plans import variant_from_recommendation

        return variant_from_recommendation(
            recommendation,
            identity=self.resolve_model_identity(recommendation),
        )

    def discover_artifact_variants(
        self,
        *,
        recommendation: ModelRecommendation,
        include_uncertain: bool = False,
        limit: int = 20,
    ) -> list[ArtifactVariant]:
        from jaull.execution_plans import discover_artifact_variants

        identity = self.resolve_model_identity(recommendation)
        current = self.artifact_variant_for_recommendation(recommendation)
        return discover_artifact_variants(
            identity=identity,
            current=current,
            search_client=self.services.search_client,
            inspect_model=self._inspect_model_live,
            analysis_cache=self.services.model_analysis_cache,
            include_uncertain=include_uncertain,
            limit=limit,
        )

    def execution_plan_for_recommendation(
        self,
        recommendation: ModelRecommendation,
        *,
        hardware: HardwareProfile | None = None,
        backend_selection: RuntimeBackendSelection | None = None,
        runtime_capability: RuntimeCapability | None = None,
        execution_readiness: ExecutionReadiness | None = None,
    ) -> ExecutionPlan:
        from jaull.execution_plans import execution_plan_for_recommendation

        return execution_plan_for_recommendation(
            recommendation,
            hardware=hardware,
            backend_selection=backend_selection,
            runtime_capability=runtime_capability,
            execution_readiness=execution_readiness,
        )

    def execution_plans_for_recommendation(
        self,
        recommendation: ModelRecommendation,
        *,
        include_uncertain: bool = False,
        limit: int = 20,
    ) -> list[ExecutionPlan]:
        from jaull.execution_plans import build_execution_plan

        current = self.execution_plan_for_recommendation(recommendation)
        try:
            variants = self.discover_artifact_variants(
                recommendation=recommendation,
                include_uncertain=include_uncertain,
                limit=limit,
            )
        except Exception:
            variants = [current.artifact]

        plans: list[ExecutionPlan] = []
        for variant in variants:
            runtime = (
                current.runtime
                if variant == current.artifact
                else _runtime_for_variant(variant)
            )
            memory = current.memory_prediction if variant == current.artifact else None
            plans.append(
                build_execution_plan(
                    model_identity=current.model_identity,
                    artifact=variant,
                    runtime=runtime,
                    memory_prediction=memory,
                )
            )
        return plans or [current]

    def prepare_execution_plan(
        self,
        plan: ExecutionPlan,
        *,
        hardware: HardwareProfile | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> PreparedExecutionPlan:
        from jaull.execution_plans import build_execution_plan

        def report(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        hw = hardware
        if hw is None:
            report("Scanning hardware")
            hw = self.scan_hardware()
        report(f"Inspecting {plan.artifact.repo_id}")
        analysis = self.inspect_model(plan.artifact.repo_id)
        config = _config_for_plan(plan)
        report("Estimating selected execution path")
        estimate = self.estimate_model(
            analysis,
            hw,
            config,
            resolve_base_model=True,
            recommend_runtime=True,
        )
        runtime = estimate.runtime_recommendation or _runtime_for_variant(plan.artifact)
        artifact = self._prepare_plan_artifact(plan, runtime, report)
        report("Selecting compute backend")
        selection = self.select_runtime_backend(hw)
        report("Checking runtime readiness")
        if runtime.runtime is RuntimeName.TRANSFORMERS:
            pytorch_capability = self.inspect_pytorch_runtime()
            readiness = self.evaluate_pytorch_execution_readiness(
                selection=selection,
                runtime_capability=pytorch_capability,
            )
            runtime_capability: RuntimeCapability = pytorch_capability
        else:
            llama_capability = self.inspect_llama_cpp_runtime(selection=selection)
            runtime_capability = llama_capability
            readiness = self.evaluate_execution_readiness(
                selection=selection,
                runtime_capability=llama_capability,
            )
        variant = plan.artifact.model_copy(
            update={
                "revision": artifact.revision,
                "filename": artifact.filename,
                "size_bytes": artifact.size_bytes,
                "quantization": artifact.quantization or plan.artifact.quantization,
            }
        )
        prepared = build_execution_plan(
            model_identity=plan.model_identity,
            artifact=variant,
            runtime=runtime,
            memory_prediction=estimate,
            hardware=hw,
            backend_selection=selection,
            runtime_capability=runtime_capability,
            execution_readiness=readiness,
        )
        return PreparedExecutionPlan(plan=prepared, artifact=artifact)

    def _prepare_plan_artifact(
        self,
        plan: ExecutionPlan,
        runtime: RuntimeRecommendation,
        report_step: Callable[[str], None],
    ) -> ModelArtifact:
        if runtime.runtime is RuntimeName.TRANSFORMERS:
            report_step("Preparing Transformers runtime")
            return plan.artifact.to_model_artifact()

        report_step(
            f"Resolve artifact for {plan.artifact.repo_id}"
            + (f" ({plan.artifact.quantization})" if plan.artifact.quantization else "")
        )
        artifact = self.resolve_artifact(
            repo_id=plan.artifact.repo_id,
            quantization=plan.artifact.quantization,
            revision=plan.artifact.revision,
        )
        if not artifact.is_downloaded:
            report_step("Downloading artifact")
            artifact = self.download_artifact(artifact)
        else:
            report_step(f"Reuse local artifact {artifact.filename}")
        report_step("Verifying artifact")
        artifact = self.verify_artifact(artifact)
        report_step("Model ready")
        return artifact

    def compare_benchmarks(
        self,
        *,
        model_identity: ModelIdentity,
        records: list[BenchmarkRecord],
    ) -> BenchmarkComparison:
        from jaull.evaluation.benchmark_comparison import compare_benchmark_records

        return compare_benchmark_records(
            model_identity=model_identity,
            records=records,
        )

    def compare_saved_benchmarks_for_recommendation(
        self,
        recommendation: ModelRecommendation,
    ) -> BenchmarkComparison:
        model_identity = self.resolve_model_identity(recommendation)
        return self.compare_benchmarks(
            model_identity=model_identity,
            records=self.benchmark_records_for_model(model_identity),
        )

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

    @staticmethod
    def _benchmark_matches_model(
        record: BenchmarkRecord,
        model_identity: ModelIdentity,
    ) -> bool:
        artifact = record.artifact
        if (
            model_identity.canonical_repo_id
            and logical_model_repo_key(artifact.repo_id)
            == logical_model_repo_key(model_identity.canonical_repo_id)
        ):
            return True
        model_name = model_identity.model_name.casefold()
        values = [
            artifact.repo_id.casefold(),
            artifact.filename.casefold(),
        ]
        return any(model_name in value for value in values)

    def _llama_cpp_runner(self) -> LlamaCppRunner:
        """Return a configured llama.cpp runner, constructing it only when used."""
        if self.llama_cpp_runner is not None:
            return self.llama_cpp_runner
        from jaull.execution.errors import ExecutableNotFoundError
        from jaull.execution.host import HostExecutionBackend
        from jaull.runtime.llama_cpp_runner import LlamaCppRunner

        installation = self._resolved_llama_cpp_installation()
        if installation.llama_cli is None:
            raise ExecutableNotFoundError(
                installation.discovery.message or "llama-cli executable not found."
            )
        fresh = LlamaCppRunner(
            backend=HostExecutionBackend(),
            llama_cli_path=installation.llama_cli,
            timeout_seconds=self.llama_cli_timeout_seconds,
        )
        object.__setattr__(self, "llama_cpp_runner", fresh)
        return fresh

    def _transformers_runner(self) -> TransformersRunner:
        """Return a configured Transformers runner, constructing it only when used."""
        if self.transformers_runner is not None:
            return self.transformers_runner
        from jaull.runtime.transformers_runner import TransformersRunner

        installation = self._resolved_pytorch_installation()
        fresh = TransformersRunner(
            backend=self._host_execution_backend(),
            python_executable=installation.python_executable,
            timeout_seconds=self.llama_cli_timeout_seconds,
        )
        object.__setattr__(self, "transformers_runner", fresh)
        return fresh

    def _host_execution_backend(self) -> ExecutionBackendProtocol:
        from jaull.execution.host import HostExecutionBackend

        return HostExecutionBackend()

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
            transformers_runner=self._transformers_runner(),
            store=self._experiment_store(),
            llama_cli_path=self._resolved_llama_cpp_installation().llama_cli,
            python_executable=self._resolved_pytorch_installation().python_executable,
            runtime_locator=self._runtime_locator(),
            capability_timeout_seconds=self.llama_cli_timeout_seconds,
        )
        object.__setattr__(self, "experiment_runner", fresh)
        return fresh

    def _llama_bench_runner(self) -> LlamaBenchRunner:
        if self.llama_bench_runner is not None:
            return self.llama_bench_runner
        from jaull.benchmarks.errors import BenchmarkUnavailableError
        from jaull.runtime.llama_bench_runner import LlamaBenchRunner

        installation = self._resolved_llama_cpp_installation()
        if installation.llama_bench is None:
            raise BenchmarkUnavailableError(
                installation.discovery.message or "llama-bench executable not found."
            )
        fresh = LlamaBenchRunner(
            backend=self._host_execution_backend(),
            llama_bench_path=installation.llama_bench,
        )
        object.__setattr__(self, "llama_bench_runner", fresh)
        return fresh

    def _transformers_benchmark_runner(self) -> TransformersBenchmarkRunner:
        if self.transformers_benchmark_runner is not None:
            return self.transformers_benchmark_runner
        from jaull.runtime.transformers_benchmark_runner import (
            TransformersBenchmarkRunner,
        )

        installation = self._resolved_pytorch_installation()
        fresh = TransformersBenchmarkRunner(
            backend=self._host_execution_backend(),
            python_executable=installation.python_executable,
        )
        object.__setattr__(self, "transformers_benchmark_runner", fresh)
        return fresh

    def _benchmark_store(self) -> BenchmarkStore:
        if self.benchmark_store is not None:
            return self.benchmark_store
        from jaull.benchmarks.storage import BenchmarkStore

        fresh = BenchmarkStore()
        object.__setattr__(self, "benchmark_store", fresh)
        return fresh

    def _benchmark_matrix_runner(self) -> BenchmarkMatrixRunner:
        if self.benchmark_matrix_runner is not None:
            return self.benchmark_matrix_runner
        from jaull.benchmarks.matrix import BenchmarkMatrixRunner

        fresh = BenchmarkMatrixRunner(
            execution_backend=self._host_execution_backend(),
            llama_bench_runner=self._llama_bench_runner(),
            store=self._benchmark_store(),
            llama_bench_path=self._resolved_llama_cpp_installation().llama_bench,
            capability_timeout_seconds=min(self.llama_bench_timeout_seconds, 30.0),
        )
        object.__setattr__(self, "benchmark_matrix_runner", fresh)
        return fresh

    def _runtime_locator(self) -> RuntimeLocator:
        if self.runtime_locator is not None:
            return self.runtime_locator
        from jaull.runtime.locator import RuntimeLocator, RuntimeLocatorConfig

        fresh = RuntimeLocator(
            config=RuntimeLocatorConfig(
                llama_cli_path=self.llama_cli_path,
                llama_bench_path=self.llama_bench_path,
                python_executable=self.python_executable,
            )
        )
        object.__setattr__(self, "runtime_locator", fresh)
        return fresh

    def _resolved_llama_cpp_installation(self) -> LlamaCppInstallation:
        if self.llama_cpp_installation is not None:
            return self.llama_cpp_installation
        installation = self._runtime_locator().resolve_llama_cpp()
        object.__setattr__(self, "llama_cpp_installation", installation)
        return installation

    def _resolved_pytorch_installation(self) -> PyTorchInstallation:
        if self.pytorch_installation is not None:
            return self.pytorch_installation
        installation = self._runtime_locator().resolve_pytorch()
        object.__setattr__(self, "pytorch_installation", installation)
        return installation

    def _select_llama_cpp_installation(
        self,
        *,
        selection: RuntimeBackendSelection,
        backend: ExecutionBackendProtocol,
    ) -> LlamaCppInstallation:
        from jaull.runtime.llama_cpp_capability import (
            evaluate_execution_readiness,
            inspect_llama_cpp_runtime,
        )

        readiness_by_cli: dict[str, ExecutionReadinessStatus] = {}
        for installation in self._runtime_locator().discover_llama_cpp():
            if installation.llama_cli is None:
                continue
            capability = inspect_llama_cpp_runtime(
                backend=backend,
                llama_cli_path=installation.llama_cli,
                timeout_seconds=self.llama_cli_timeout_seconds,
            )
            readiness = evaluate_execution_readiness(
                selection=selection,
                runtime_capability=capability,
            )
            readiness_by_cli[installation.llama_cli] = readiness.status
        selected = self._runtime_locator().select_llama_cpp(
            requested_backend=selection.selected_backend,
            readiness_by_cli=readiness_by_cli,
        )
        if selected.llama_cli is not None:
            object.__setattr__(self, "llama_cpp_installation", selected)
        return selected

    def _make_range_client(self) -> HttpRangeClient | None:
        factory = self.services.range_client_factory
        if factory is None:
            return None
        client = factory()
        return client  # type: ignore[return-value]

    def _cached_model_analysis(self, candidate: ModelCandidate) -> ModelAnalysis | None:
        cache = self.services.model_analysis_cache
        if cache is None:
            return None
        return cache.get(candidate)

    def _inspect_model_live(self, repo_id: str) -> ModelAnalysis:
        return self.services.inspect_model(repo_id, client=self.services.hf_client)

    def _store_model_analysis(
        self, candidate: ModelCandidate, analysis: ModelAnalysis
    ) -> None:
        cache = self.services.model_analysis_cache
        if cache is None:
            return
        cache.put(candidate, analysis)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def default(
        cls,
        *,
        llama_cli_path: str | Path | None = None,
        llama_cli_timeout_seconds: float = 300.0,
        python_executable: str | Path | None = None,
        pytorch_probe_timeout_seconds: float = 10.0,
        llama_bench_path: str | Path | None = None,
        llama_bench_timeout_seconds: float = 900.0,
        runtime_locator: RuntimeLocator | None = None,
        model_analysis_cache: ModelAnalysisCacheProtocol | None = None,
    ) -> AdvisorService:
        """Production wiring: real HF client, real hardware probes, real estimator."""
        services = ServiceContainer.default()
        if model_analysis_cache is not None:
            services = replace(services, model_analysis_cache=model_analysis_cache)
        artifacts = ArtifactService(
            resolver=HuggingFaceArtifactResolver(services.hf_client),
            storage=ArtifactStorage(),
        )
        return cls(
            services=services,
            artifacts=artifacts,
            llama_cli_path=llama_cli_path,
            llama_cli_timeout_seconds=llama_cli_timeout_seconds,
            python_executable=python_executable,
            pytorch_probe_timeout_seconds=pytorch_probe_timeout_seconds,
            llama_bench_path=llama_bench_path,
            llama_bench_timeout_seconds=llama_bench_timeout_seconds,
            runtime_locator=runtime_locator,
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
        transformers_runner: TransformersRunner | None = None,
        experiment_runner: ExperimentRunner | None = None,
        experiment_store: ExperimentStore | None = None,
        llama_bench_runner: LlamaBenchRunner | None = None,
        transformers_benchmark_runner: TransformersBenchmarkRunner | None = None,
        benchmark_matrix_runner: BenchmarkMatrixRunner | None = None,
        benchmark_store: BenchmarkStore | None = None,
        llama_cli_path: str | Path | None = None,
        llama_cli_timeout_seconds: float = 300.0,
        python_executable: str | Path | None = None,
        pytorch_probe_timeout_seconds: float = 10.0,
        llama_bench_path: str | Path | None = None,
        llama_bench_timeout_seconds: float = 900.0,
        runtime_locator: RuntimeLocator | None = None,
        model_analysis_cache: ModelAnalysisCacheProtocol | None = None,
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
            model_analysis_cache=model_analysis_cache,
        )
        return cls(
            services=services,
            collect_diagnostics=collect_diagnostics,
            artifacts=artifacts,
            llama_cpp_runner=llama_cpp_runner,
            transformers_runner=transformers_runner,
            experiment_runner=experiment_runner,
            experiment_store=experiment_store,
            llama_bench_runner=llama_bench_runner,
            transformers_benchmark_runner=transformers_benchmark_runner,
            benchmark_matrix_runner=benchmark_matrix_runner,
            benchmark_store=benchmark_store,
            llama_cli_path=llama_cli_path,
            llama_cli_timeout_seconds=llama_cli_timeout_seconds,
            python_executable=python_executable,
            pytorch_probe_timeout_seconds=pytorch_probe_timeout_seconds,
            llama_bench_path=llama_bench_path,
            llama_bench_timeout_seconds=llama_bench_timeout_seconds,
            runtime_locator=runtime_locator,
        )


def _runtime_for_variant(variant: ArtifactVariant) -> RuntimeRecommendation:
    if variant.format is ArtifactVariantFormat.GGUF:
        return RuntimeRecommendation(
            runtime=RuntimeName.LLAMA_CPP,
            flags=[
                RuntimeFlag(
                    name="--n-gpu-layers",
                    value="0",
                    source=RuntimeFlagSource.POLICY,
                    explanation="Discovered GGUF plan defaults to CPU until prepared.",
                )
            ],
            confidence=EstimationConfidence.UNKNOWN,
            reasons=["Discovered GGUF artifact variant."],
        )
    return RuntimeRecommendation(
        runtime=RuntimeName.TRANSFORMERS,
        flags=[
            RuntimeFlag(
                name="device_map",
                value="cpu",
                source=RuntimeFlagSource.POLICY,
                explanation=(
                    "Discovered Transformers plan defaults to CPU until prepared."
                ),
            )
        ],
        confidence=EstimationConfidence.UNKNOWN,
        reasons=["Discovered Transformers artifact variant."],
    )


def _config_for_plan(plan: ExecutionPlan) -> InferenceConfiguration:
    base = (
        plan.memory_prediction.inference_configuration
        if plan.memory_prediction is not None
        else InferenceConfiguration(context_length=4096)
    )
    precision = _precision(plan.artifact.precision)
    return base.model_copy(
        update={
            "quantization": plan.artifact.quantization,
            "precision": precision,
        }
    )


def _precision(value: str | None) -> WeightPrecision | None:
    if value is None:
        return None
    try:
        return WeightPrecision(value)
    except ValueError:
        return None


__all__ = ["AdvisorService", "CancelCheck", "DiagnosticsFn"]
