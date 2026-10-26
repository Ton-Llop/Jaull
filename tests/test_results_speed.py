"""Offline Results speed labels and the explicit benchmark shortcut."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from textual.screen import Screen
from textual.widgets import Button, Static

from jaull.domain.benchmarks import BenchmarkGpuLayers, BenchmarkMeasurementKind, BenchmarkRecord
from jaull.domain.execution_plans import ArtifactVariantFormat
from jaull.domain.hardware import ComputeBackend
from jaull.domain.runtime import (
    LlamaCppBinaryStatus,
    LlamaCppRuntimeCapability,
    RuntimeFlag,
    RuntimeFlagSource,
    RuntimeName,
)
from jaull.presentation.speed import historical_speed_benchmarks, speed_readout
from jaull.tui.app import JaullApp
from jaull.tui.screens.recommendation_results import RecommendationResultsScreen
from jaull.tui.widgets.technical_details import TechnicalDetails
from jaull.workflow.state import RecommendationWorkflowState
from tests.test_execution_plans import _gguf_recommendation
from tests.test_fastest_v2_applicability import _cuda_case
from tests.test_recommendation_engine_v2 import _requirements
from tests.test_tui_evidence import _FakeStoreAdvisor, _plan


def _case():
    plan, record = _cuda_case()
    gpu = "NVIDIA GeForce RTX 2060"
    hardware = record.hardware.model_copy(update={"gpus": [
        record.hardware.gpus[0].model_copy(update={"name": gpu}),
    ]})
    selection = plan.backend_selection
    capability = LlamaCppRuntimeCapability(binary_status=LlamaCppBinaryStatus.AVAILABLE)
    plan = plan.model_copy(update={
        "hardware": hardware,
        "execution_readiness": _plan().execution_readiness,
        "model_identity": plan.model_identity.model_copy(update={"architecture": "qwen2"}),
        "runtime_capability": capability.model_copy(update={
            "version_text": "version: 1 (fffffffff)",
        }),
        "backend_selection": selection.model_copy(update={"selected_accelerator":
            selection.selected_accelerator.model_copy(update={"name": gpu})}),
        "artifact": plan.artifact.model_copy(update={"size_bytes": 600_000_000}),
    })
    record = record.model_copy(update={
        "hardware": hardware,
        "artifact": record.artifact.model_copy(update={"size_bytes": 600_000_000}),
        "observation": record.observation.model_copy(update={
            "raw_stderr": "Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5\n",
        }),
    })
    return plan, record


def _calibration(record):
    records = []
    for number, size, tps in ((1, 500_000_000, 250.0), (2, 1_000_000_000, 160.0)):
        for call in range(3):
            records.append(record.model_copy(update={
                "identity": record.identity.model_copy(update={
                    "benchmark_id": f"cal-{number}-{call}",
                }),
                "artifact": record.artifact.model_copy(update={
                    "sha256": str(number) * 64, "size_bytes": size,
                    "repo_id": f"org/Model-{number}", "filename": f"model-{number}.gguf",
                }),
                "observation": record.observation.model_copy(update={"measurements": [
                    m.model_copy(update={"mean_tokens_per_second": tps})
                    for m in record.observation.measurements
                ]}),
            }))
    return records


def test_exact_applicable_measurement_is_labelled_and_traceable():
    plan, record = _case()
    readout = speed_readout(plan, _requirements(), [record])
    assert readout.label == "Measured 10.0 tok/s"
    assert readout.source == "measured" and readout.measured
    assert "tg128; 1 calls" in readout.details
    assert record.identity.benchmark_id in readout.details


def test_results_show_latest_saved_speed_without_changing_the_ranking_median():
    from jaull.recommendation.local_evidence import assess_speed

    plan, record = _case()
    newer = record.model_copy(update={
        "identity": record.identity.model_copy(update={
            "benchmark_id": "newer",
            "created_at": record.identity.created_at + timedelta(days=1),
        }),
        "observation": record.observation.model_copy(update={"measurements": [
            m.model_copy(update={"mean_tokens_per_second": 20.0})
            for m in record.observation.measurements
        ]}),
    })
    records = [record, newer]
    readout = speed_readout(plan, _requirements(), records)
    assert readout.label == "Measured 20.0 tok/s"
    assert "Latest saved run: newer" in readout.details
    assert "Ranking aggregate (median): 15.0 tok/s" in readout.details
    assert readout == speed_readout(plan, _requirements(), records[::-1])
    assert assess_speed(plan, _requirements(), records).applicable.tokens_per_second == 15.0
    assert readout.saved_runs.index("Record: newer") < readout.saved_runs.index(
        f"Record: {record.identity.benchmark_id}"
    )
    assert "Generation 128: 20.0" in readout.saved_runs
    assert "stddev" in readout.saved_runs


def _gguf_history_case(backend):
    plan, record = _case()
    layers = (
        BenchmarkGpuLayers.count_layers(0) if backend is ComputeBackend.CPU else record.gpu_layers
    )
    device = None if backend is ComputeBackend.CPU else "Vulkan0"
    runtime = plan.runtime.model_copy(update={"flags": [
        RuntimeFlag(name="--n-gpu-layers", value="0" if backend is ComputeBackend.CPU else "-1",
                    source=RuntimeFlagSource.ESTIMATE, explanation="test"),
    ]})
    plan = plan.model_copy(update={
        "runtime": runtime,
        "backend_selection": plan.backend_selection.model_copy(update={
            "selected_backend": backend,
        }),
    })
    request = record.request.model_copy(update={
        "backend": backend, "device": device, "gpu_layers": layers, "runtime": runtime,
    })
    record = BenchmarkRecord.create(
        hardware=plan.hardware, request=request, observation=record.observation,
    )
    return plan, record


@pytest.mark.parametrize("backend", [ComputeBackend.CPU, ComputeBackend.VULKAN])
def test_cpu_and_vulkan_keep_saved_speed_as_history_not_ranking_evidence(backend):
    from jaull.recommendation.local_evidence import assess_speed

    plan, record = _gguf_history_case(backend)
    readout = speed_readout(plan, _requirements(), [record])
    assert readout.label == "Last benchmark 10.0 tok/s (llama.cpp)"
    assert readout.source == "historical" and not readout.measured
    assert "Not used for ranking" in readout.details
    assert record.identity.benchmark_id in readout.details
    assert assess_speed(plan, _requirements(), [record]).applicable is None


def test_other_artifacts_never_produce_predictions_or_change_inputs():
    plan, record = _case()
    records = _calibration(record)
    before = [r.model_dump_json() for r in records]
    plan_before = plan.model_dump_json()
    readout = speed_readout(plan, _requirements(), records)
    assert readout.label == ""
    assert readout.source == "unknown" and not readout.measured
    assert "tok/s" not in readout.label
    assert readout == speed_readout(plan, _requirements(), list(reversed(records)))
    assert [r.model_dump_json() for r in records] == before
    assert plan.model_dump_json() == plan_before


def test_duplicate_ids_do_not_inflate_applicable_call_count():
    plan, record = _case()
    readout = speed_readout(plan, _requirements(), [record] * 3)
    assert "tg128; 1 calls" in readout.details
    assert readout == speed_readout(plan, _requirements(), [record])


def test_conflicting_record_ids_withhold_speed_in_both_orders():
    plan, record = _case()
    conflict = record.model_copy(update={"artifact": record.artifact.model_copy(
        update={"sha256": "1" * 64})})
    for records in ([record, conflict], [conflict, record]):
        readout = speed_readout(plan, _requirements(), records)
        assert readout.label == ""
        assert not readout.measured
        assert "Conflicting benchmark IDs" in readout.details
        assert historical_speed_benchmarks(plan, records) == ()


def test_inapplicable_measurement_is_only_a_historical_reference():
    plan, record = _case()
    record = record.model_copy(update={"hardware": record.hardware.model_copy(
        update={"os": "Other kernel"})})
    readout = speed_readout(plan, _requirements(), [record])
    assert readout.label == ""
    assert readout.source == "unknown" and not readout.measured
    assert "Historical reference only" in readout.details
    assert record.identity.benchmark_id in readout.details
    assert "10.0 tok/s" not in readout.label


@pytest.mark.parametrize("missing", ["plan", "requirements", "records"])
def test_missing_context_or_measurements_never_produce_predictions(missing):
    plan, record = _case()
    readout = speed_readout(
        None if missing == "plan" else plan,
        None if missing == "requirements" else _requirements(),
        [] if missing == "records" else [record],
    )
    assert readout.label == ""
    assert not readout.measured


def _transformers_plan():
    plan, _ = _case()
    return plan.model_copy(update={
        "runtime_family": RuntimeName.TRANSFORMERS,
        "runtime": plan.runtime.model_copy(update={
            "runtime": RuntimeName.TRANSFORMERS, "flags": [
                RuntimeFlag(name="torch_dtype", value="torch.float16",
                            source=RuntimeFlagSource.ESTIMATE, explanation="test"),
            ],
        }),
        "artifact": plan.artifact.model_copy(update={
            "format": ArtifactVariantFormat.SAFETENSORS,
            "filename": None, "quantization": None,
        }),
    })


def _transformers_case():
    plan = _transformers_plan()
    plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(update={"sha256": None})})
    _, original = _case()
    request = original.request.model_copy(update={
        "artifact": plan.artifact.to_model_artifact(), "runtime": plan.runtime,
    })
    observation = original.observation.model_copy(update={
        "methodology": "transformers_isolated_inference_v3", "protocol": None,
        "measurements": [m.model_copy(update={"mean_tokens_per_second": 18.36})
                         for m in original.observation.measurements],
    })
    record = BenchmarkRecord.create(
        hardware=plan.hardware, request=request, observation=observation,
    )
    return plan, record


def test_transformers_measurement_is_visible_as_history_without_exact_identity_or_ranking():
    from jaull.recommendation.local_evidence import assess_speed

    plan, record = _transformers_case()
    before = record.model_dump_json(), plan.model_dump_json()
    readout = speed_readout(plan, _requirements(), [record])
    assert readout.label == "Last benchmark 18.4 tok/s (Transformers)"
    assert readout.source == "historical" and not readout.measured
    assert "not verified exact artifact bytes" in readout.details
    assert "Not used for ranking" in readout.details
    assert "tg128" in readout.details
    assert record.identity.benchmark_id in readout.details
    assert record.identity.created_at.isoformat() in readout.details
    assert assess_speed(plan, _requirements(), [record]).applicable is None
    assert before == (record.model_dump_json(), plan.model_dump_json())
    assert "Generation 128: 18.4" in readout.saved_runs
    assert record.identity.benchmark_id in readout.saved_runs


def test_transformers_history_uses_latest_successful_run_in_either_store_order():
    plan, record = _transformers_case()
    newer = record.model_copy(update={
        "identity": record.identity.model_copy(update={
            "benchmark_id": "bench-newer",
            "created_at": record.identity.created_at + timedelta(days=1),
        }),
        "observation": record.observation.model_copy(update={"measurements": [
            m.model_copy(update={"mean_tokens_per_second": 27.11})
            for m in record.observation.measurements
        ]}),
    })
    failed = newer.model_copy(update={
        "identity": newer.identity.model_copy(update={
            "benchmark_id": "bench-failed",
            "created_at": newer.identity.created_at + timedelta(days=1),
        }),
        "observation": newer.observation.model_copy(update={"success": False, "exit_code": 1}),
    })
    records = [record, newer, failed]
    readout = speed_readout(plan, _requirements(), records)
    assert readout.label == "Last benchmark 27.1 tok/s (Transformers)"
    assert readout == speed_readout(plan, _requirements(), records[::-1])
    assert "Benchmark ID: bench-newer" in readout.details


@pytest.mark.parametrize("change", [
    "repo", "runtime", "backend", "precision", "machine", "failed", "exit_code", "methodology",
    "prefill_only", "other_tokens", "duplicate_tg128", "digest",
])
def test_transformers_history_never_attributes_an_unmatched_or_invalid_run(change):
    plan, record = _transformers_case()
    if change == "repo":
        plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(
            update={"repo_id": "org/Other"})})
    elif change == "runtime":
        plan, _ = _case()
    elif change == "backend":
        plan = plan.model_copy(update={"backend_selection": plan.backend_selection.model_copy(
            update={"selected_backend": ComputeBackend.CPU})})
    elif change == "precision":
        plan = plan.model_copy(update={"runtime": plan.runtime.model_copy(update={"flags": [
            RuntimeFlag(name="torch_dtype", value="torch.float32",
                        source=RuntimeFlagSource.ESTIMATE, explanation="test"),
        ]})})
    elif change == "machine":
        plan = plan.model_copy(update={"hardware": plan.hardware.model_copy(
            update={"os": "Other"})})
    elif change in {"failed", "exit_code", "methodology"}:
        update = ({"success": False} if change == "failed" else
                  {"exit_code": 1} if change == "exit_code" else {"methodology": "unknown"})
        record = record.model_copy(update={"observation": record.observation.model_copy(
            update=update)})
    elif change == "digest":
        plan = plan.model_copy(update={"artifact": plan.artifact.model_copy(
            update={"sha256": "a" * 64})})
        record = record.model_copy(update={"artifact": record.artifact.model_copy(
            update={"sha256": "b" * 64})})
    else:
        measurements = record.observation.measurements
        if change == "prefill_only":
            measurements = [m for m in measurements if m.kind is BenchmarkMeasurementKind.PREFILL]
        elif change == "other_tokens":
            measurements = [m.model_copy(update={"tokens": 64}) for m in measurements]
        else:
            measurements = measurements * 2
        record = record.model_copy(update={"observation": record.observation.model_copy(
            update={"measurements": measurements})})
    assert speed_readout(plan, _requirements(), [record]).label == ""


def test_transformers_never_inherits_llama_cpp_speed_or_calibration():
    plan = _transformers_plan()
    _, record = _case()
    for records in ([], [record], _calibration(record)):
        readout = speed_readout(plan, _requirements(), records)
        assert readout.label == ""
        assert not readout.measured


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
@pytest.mark.parametrize(
    "mode", [
        "calibration", "measured", "missing", "transformers", "historical", "transformers_run",
    ],
)
def test_results_show_only_applicable_speed_and_keep_benchmark_accessible(size, mode, tmp_path):
    plan, record = _case()
    if mode == "transformers":
        plan = _transformers_plan()
    if mode == "transformers_run":
        plan, record = _transformers_case()
    if mode == "historical":
        record = record.model_copy(update={"hardware": record.hardware.model_copy(
            update={"os": "Other kernel"})})
    records = {
        "calibration": _calibration(record), "measured": [record], "missing": [],
        "transformers": [], "historical": [record],
        "transformers_run": [record],
    }[mode]
    expected = "Measured 10.0 tok/s" if mode == "measured" else ""
    if mode == "transformers_run":
        expected = "Last benchmark 18.4 tok/s (Transformers)"
    rec = _gguf_recommendation().model_copy(update={"plan": plan})
    state = RecommendationWorkflowState(requirements=_requirements(), recommendations=[rec])

    class StoreAdvisor(_FakeStoreAdvisor):
        services = None

    advisor = StoreAdvisor(benchmarks={r.identity.benchmark_id: r for r in records})

    class TestApp(JaullApp):
        opened = None

        def benchmark_recommendation(self, recommendation, execution_plan=None):
            self.opened = (recommendation, execution_plan)

    async def scenario():
        app = TestApp(advisor=advisor)
        before = state.model_dump_json()
        async with app.run_test(size=size) as pilot:
            screen = RecommendationResultsScreen(state)
            await app.push_screen(screen)
            # Default labels are blank; wait for the worker's actual readout.
            async with asyncio.timeout(10):
                while screen.query_one("#rec-speed-0", Static).tooltip is None:
                    await pilot.pause(0.01)
            await pilot.pause()
            for suffix in ("0", "detail-0"):
                assert str(screen.query_one(f"#rec-speed-{suffix}", Static).content) == expected
            scope = str(screen.query_one("#rec-speed-scope-0", Static).content)
            assert scope == (
                "Comparable measurement" if mode == "measured" else
                "Historical benchmark" if mode == "transformers_run" else
                "No comparable measurement"
            )
            history = str(screen.query_one("#rec-speed-provenance-0", Static).content)
            assert "Applicability" not in history
            assert "Not used for ranking" not in history
            assert screen.query_one("#rec-speed-matching-0", TechnicalDetails).collapsed
            assert str(screen.query_one("#rec-speed-matching-details-0", Static).content) == str(
                screen.query_one("#rec-speed-0", Static).tooltip,
            )
            if mode == "historical":
                assert "Historical reference only" in str(
                    screen.query_one("#rec-speed-0", Static).tooltip,
                )
            suffix = "detail-0" if size[0] < 120 else "0"
            button = screen.query_one(f"#res-speed-{suffix}", Button)
            assert not button.disabled
            assert button.region.right <= size[0]
            assert screen.query_one(f"#rec-speed-{suffix}").region.right <= button.region.x
            assert await pilot.click(f"#res-speed-{suffix}")
            await pilot.pause()
            assert app.opened == (rec, plan)
            assert state.model_dump_json() == before
            app.save_screenshot(str(tmp_path / f"results-speed-{mode}-{size[0]}.svg"))

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (160, 50)])
@pytest.mark.parametrize("backend", [None, ComputeBackend.CPU, ComputeBackend.VULKAN])
def test_returning_from_benchmark_refreshes_history_without_changing_search(size, backend):
    plan, record = _transformers_case() if backend is None else _gguf_history_case(backend)
    expected = "Last benchmark 18.4" if backend is None else "Last benchmark 10.0"
    rec = _gguf_recommendation().model_copy(update={"plan": plan})
    state = RecommendationWorkflowState(requirements=_requirements(), recommendations=[rec])

    class StoreAdvisor(_FakeStoreAdvisor):
        services = None

    advisor = StoreAdvisor()

    async def scenario():
        app = JaullApp(advisor=advisor)
        before = state.model_dump_json()
        async with app.run_test(size=size) as pilot:
            screen = RecommendationResultsScreen(state)
            await app.push_screen(screen)
            async with asyncio.timeout(10):
                while screen.query_one("#rec-speed-0", Static).tooltip is None:
                    await pilot.pause(0.01)
            assert str(screen.query_one("#rec-speed-0", Static).content) == ""
            await app.push_screen(Screen())
            advisor._benchmarks[record.identity.benchmark_id] = record
            app.pop_screen()
            async with asyncio.timeout(10):
                while expected not in str(
                    screen.query_one("#rec-speed-0", Static).content,
                ):
                    await pilot.pause(0.01)
            assert expected in str(
                screen.query_one("#rec-speed-detail-0", Static).content,
            )
            assert state.model_dump_json() == before

    asyncio.run(scenario())
