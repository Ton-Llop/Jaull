"""Repeat the unchanged Benchmark UI action with the restored identical binary."""

import asyncio
import json
import shutil
import time
from pathlib import Path

from textual.widgets import Button

from jaull.advisor.service import AdvisorService
from jaull.domain.execution_plans import ExecutionPlan
from jaull.recommendation.models import ModelRecommendation
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_benchmark import RecommendationBenchmarkScreen

ROOT = Path(__file__).resolve().parent


async def main() -> None:
    plan = ExecutionPlan.model_validate_json((ROOT / 'execution-plan.json').read_text('utf-8'))
    recommendation = ModelRecommendation.model_validate_json(
        (ROOT / 'reference-recommendation.json').read_text('utf-8')
    )
    app = JaullApp(advisor=AdvisorService.default())
    app.hardware_profile = plan.hardware
    async with app.run_test(size=(120, 48)) as pilot:
        app.benchmark_recommendation(recommendation, plan)
        await pilot.pause(1)
        screen = app.screen
        assert isinstance(screen, RecommendationBenchmarkScreen)
        screen.query_one('#benchmark-start', Button).focus()
        await pilot.press('enter')
        deadline = time.monotonic() + 1000
        while screen._last_result is None:
            await pilot.pause(0.5)
            if time.monotonic() > deadline:
                raise TimeoutError('Benchmark worker timed out')
            if screen._future is not None and screen._future.done():
                await pilot.pause(1)
                if screen._last_result is None:
                    raise RuntimeError(str(screen._log_messages))
        app.save_screenshot('ui-benchmark-control-result.svg', path=str(ROOT))
        index = []
        for run in [*screen._last_result.completed, *screen._last_result.failed]:
            if run.record is None or run.persisted_path is None:
                raise RuntimeError('No persisted BenchmarkRecord')
            target = ROOT / 'records' / run.persisted_path.name
            if target.exists():
                raise RuntimeError('Refusing to overwrite evidence')
            shutil.copy2(run.persisted_path, target)
            log_path = ROOT / 'logs' / (run.record.identity.benchmark_id + '.runtime-log')
            observation = run.record.observation
            with log_path.open('x', encoding='utf-8') as handle:
                json.dump({'benchmark_id': run.record.identity.benchmark_id,
                           'stdout': observation.raw_stdout, 'stderr': observation.raw_stderr},
                          handle, indent=2)
            index.append({'id': run.record.identity.benchmark_id,
                          'store_path': str(run.persisted_path), 'copy_path': str(target),
                          'log_path': str(log_path), 'success': observation.success})
            print(run.record.model_dump_json(indent=2), flush=True)
        with (ROOT / 'control-benchmark-index.json').open('x', encoding='utf-8') as handle:
            json.dump(index, handle, indent=2)


asyncio.run(main())
