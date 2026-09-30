"""Drive production Textual Validate/Benchmark controls with real services.

The reference repository is a fixed experimental input, not a search/ranking
claim. A recommendation is produced by the unchanged service for this single
candidate. No fake services, mocked observations, or replacement runners exist.
Textual's headless Pilot drives the same screen buttons and worker code as ui.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

from pydantic import BaseModel
from textual.widgets import Button, Checkbox

from jaull.advisor.service import AdvisorService
from jaull.application.execution import ExecutionOverrides
from jaull.application.recommendation.service import recommend
from jaull.application.requirements import build_requirements
from jaull.domain.candidates import EvaluatedCandidate, ModelCandidate
from jaull.domain.execution_plans import ArtifactVariant, ArtifactVariantFormat
from jaull.domain.experiments import ExperimentRecord
from jaull.domain.requirements import RecommendationPriority, UseCase, UserAnswers
from jaull.domain.runtime import ExecutionReadinessStatus, RuntimeName
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen
from jaull.tui.screens.recommendation_validation import RecommendationValidationScreen
from jaull.workflow.state import RecommendationWorkflowState

ROOT = Path(__file__).resolve().parent
REPO = 'bartowski/Qwen2.5-7B-Instruct-GGUF'
REVISION = '8911e8a47f92bac19d6f5c64a2e2095bd2f7d031'
FILENAME = 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'
EXPECTED_SHA = '65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423'


def save_model(name: str, model: BaseModel) -> None:
    target = ROOT / name
    with target.open('x', encoding='utf-8') as handle:
        handle.write(model.model_dump_json(indent=2) + '\n')


def copy_evidence(source: Path, directory: str) -> Path:
    target = ROOT / directory / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite evidence: {target}')
    shutil.copy2(source, target)
    return target


async def wait_for_screen(pilot: object, screen: object, attribute: str) -> object:
    deadline = time.monotonic() + 1000
    while time.monotonic() < deadline:
        await pilot.pause(0.5)
        value = getattr(screen, attribute)
        if value is not None:
            return value
        future = screen._future
        if future is not None and future.done():
            await pilot.pause(1)
            value = getattr(screen, attribute)
            if value is not None:
                return value
            raise RuntimeError(
                f'UI worker finished without a record: {screen._log_messages}'
            )
    raise TimeoutError(f'UI screen exceeded wait deadline for {attribute}')


async def main() -> None:
    if (ROOT / 'execution-plan.json').exists():
        raise RuntimeError('This campaign has already prepared a plan; use a new directory')
    advisor = AdvisorService.default()
    hardware = advisor.scan_hardware()
    save_model('hardware.json', hardware)
    selection = advisor.select_runtime_backend(hardware)
    capability = advisor.inspect_llama_cpp_runtime(selection=selection)
    readiness = advisor.evaluate_execution_readiness(
        selection=selection, runtime_capability=capability
    )
    save_model('runtime-capability.json', capability)
    save_model('runtime-readiness.json', readiness)
    if readiness.status is not ExecutionReadinessStatus.READY:
        raise RuntimeError(f'CUDA runtime is not ready: {readiness}')
    artifact = advisor.resolve_artifact(REPO, quantization='Q4_K_M', revision=REVISION)
    if artifact.filename != FILENAME or artifact.local_path is None:
        raise RuntimeError(f'Wrong or missing local artifact: {artifact}')
    actual_sha = hashlib.file_digest(artifact.local_path.open('rb'), 'sha256').hexdigest()
    print(f'Local artifact: {artifact.local_path}', flush=True)
    print(f'Local SHA256: {actual_sha}', flush=True)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError('SHA256 mismatch: STOP reference case')
    artifact = advisor.verify_artifact(artifact, full=True)
    save_model('verified-artifact.json', artifact)
    baseline = ExperimentRecord.model_validate_json(
        (ROOT.parent / 'b001-r10-experiment-record/experiment.json').read_text(encoding='utf-8')
    )
    if baseline.prediction_input is None:
        raise RuntimeError('Baseline frozen prediction inputs are missing')
    config = baseline.prediction_input.inference_configuration
    if config.context_length != 4096 or config.concurrent_users != 1:
        raise RuntimeError('Baseline workload differs from requested reference')
    analysis = advisor.inspect_model(REPO)
    save_model('analysis.json', analysis)
    save_model('inference-configuration.json', config)
    estimate = advisor.estimate_model(analysis, hardware, config)
    save_model('prediction-before-validation.json', estimate)
    answers = UserAnswers(
        use_case=UseCase.GENERAL_CHAT,
        priority=RecommendationPriority.BALANCED,
        languages=['English'],
    )
    requirements = build_requirements(answers, hardware)
    save_model('requirements.json', requirements)
    candidate = ModelCandidate(repo_id=REPO, revision_hint=REVISION)
    evaluated = EvaluatedCandidate(
        candidate=candidate,
        analysis=analysis,
        selected_configuration=config,
        memory_estimate=estimate,
        compatibility=estimate.assessment,
    )
    recommendations = recommend([evaluated], requirements, hardware=hardware)
    if not recommendations:
        raise RuntimeError('Unchanged recommendation service returned no reference candidate')
    recommendation = recommendations[0]
    identity = advisor.resolve_model_identity(recommendation)
    variant = ArtifactVariant(
        model_identity=identity,
        repo_id=REPO,
        revision=REVISION,
        format=ArtifactVariantFormat.GGUF,
        filename=FILENAME,
        quantization='Q4_K_M',
        size_bytes=artifact.size_bytes,
        compatible_runtimes=[RuntimeName.LLAMA_CPP],
    )
    plan = advisor.plan_execution(
        model_identity=identity,
        artifact=variant,
        runtime=RuntimeName.LLAMA_CPP,
        estimate=estimate,
        hardware=hardware,
        overrides=ExecutionOverrides(context_size=4096),
        backend_selection=selection,
        runtime_capability=capability,
        execution_readiness=readiness,
        local_artifact=artifact,
    )
    save_model('execution-plan.json', plan)
    save_model('reference-recommendation.json', recommendation)
    print('SELECTED_PLAN', plan.model_dump_json(), flush=True)
    app = JaullApp(advisor=advisor)
    app.hardware_profile = hardware
    app.workflow_state = RecommendationWorkflowState(
        hardware=hardware, answers=answers, requirements=requirements,
        recommendations=recommendations,
    )
    async with app.run_test(size=(120, 48)) as pilot:
        app.validate_recommendation(recommendation, plan)
        await pilot.pause(1)
        screen = app.screen
        if not isinstance(screen, RecommendationValidationScreen):
            raise RuntimeError('Validate screen was not mounted')
        screen.query_one('#validation-capture-logs', Checkbox).focus()
        await pilot.press('space')
        if not screen.query_one('#validation-capture-logs', Checkbox).value:
            raise RuntimeError('Save raw runtime logs checkbox was not activated')
        app.save_screenshot('ui-validation-ready.svg', path=str(ROOT))
        if screen.query_one('#validation-start', Button).disabled:
            raise RuntimeError('Validate button is disabled')
        screen.query_one('#validation-start', Button).focus()
        await pilot.press('enter')
        record = await wait_for_screen(pilot, screen, '_last_record')
        app.save_screenshot('ui-validation-result.svg', path=str(ROOT))
        persisted = screen._last_persisted_path
        if persisted is None:
            raise RuntimeError('Experiment was not persisted')
        experiment_copy = copy_evidence(persisted, 'records')
        log_path = persisted.with_suffix('.runtime-log')
        log_copy = copy_evidence(log_path, 'logs') if log_path.is_file() else None
        print('EXPERIMENT_ID', record.identity.experiment_id, flush=True)
        print('EXPERIMENT_PATH', persisted, flush=True)
        print('EXPERIMENT_SUCCESS', record.observation.success, flush=True)
        app.pop_screen()
        await pilot.pause(1)
        app.benchmark_recommendation(recommendation, plan)
        await pilot.pause(1)
        benchmark_screen = app.screen
        if not isinstance(benchmark_screen, RecommendationBenchmarkScreen):
            raise RuntimeError('Benchmark screen was not mounted')
        app.save_screenshot('ui-benchmark-ready.svg', path=str(ROOT))
        if benchmark_screen.query_one('#benchmark-start', Button).disabled:
            raise RuntimeError('Benchmark button is disabled')
        benchmark_screen.query_one('#benchmark-start', Button).focus()
        await pilot.press('enter')
        result = await wait_for_screen(pilot, benchmark_screen, '_last_result')
        app.save_screenshot('ui-benchmark-result.svg', path=str(ROOT))
        benchmarks = []
        for run in [*result.completed, *result.failed]:
            if run.record is None or run.persisted_path is None:
                continue
            benchmark_copy = copy_evidence(run.persisted_path, 'records')
            observation = run.record.observation
            bench_log = ROOT / 'logs' / (run.record.identity.benchmark_id + '.runtime-log')
            with bench_log.open('x', encoding='utf-8') as handle:
                json.dump(
                    {'benchmark_id': run.record.identity.benchmark_id,
                     'stdout': observation.raw_stdout, 'stderr': observation.raw_stderr},
                    handle, indent=2,
                )
                handle.write('\n')
            benchmarks.append(
                {'id': run.record.identity.benchmark_id, 'store_path': str(run.persisted_path),
                 'copy_path': str(benchmark_copy), 'log_path': str(bench_log),
                 'success': observation.success}
            )
            print('BENCHMARK', json.dumps(benchmarks[-1]), flush=True)
        with (ROOT / 'run-index.json').open('x', encoding='utf-8') as handle:
            json.dump(
                {'experiment_id': record.identity.experiment_id,
                 'experiment_store_path': str(persisted),
                 'experiment_copy_path': str(experiment_copy),
                 'experiment_log_path': str(log_copy) if log_copy else None,
                 'benchmarks': benchmarks,
                 'ui_automation': 'production Textual screens through headless Pilot; no mocks'},
                handle, indent=2,
            )
            handle.write('\n')
    print('REFERENCE_UI_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
