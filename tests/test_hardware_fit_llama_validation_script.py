from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scripts import validate_hardware_fit_against_llama_cpp as harness


class _Prediction:
    def __init__(self) -> None:
        self.fit = SimpleNamespace(
            gpu_required_bytes=1_000 * harness.MIB,
            ram_required_bytes=2_000,
            gpu_transformer_blocks=18,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": "test/model",
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "hardware": {
                "gpu": "Test GPU",
                "vram_available_bytes": 6 * harness.GIB,
                "ram_available_bytes": 16 * harness.GIB,
            },
            "estimate": {
                "weights_bytes": 4 * harness.GIB,
                "kv_cache_bytes": harness.GIB,
                "runtime_overhead_bytes": harness.MIB,
                "device_reserve_bytes": 512 * harness.MIB,
                "safety_margin_bytes": 256 * harness.MIB,
                "status": "offloading_required",
                "confidence": "medium",
            },
            "fit": {
                "mode": "gpu_offload",
                "placement_method": "transformer_blocks",
                "gpu_transformer_blocks": 18,
                "total_transformer_blocks": 28,
                "offload_diagnostics": {
                    "search_ceiling_transformer_blocks": 25,
                    "selected": {
                        "gpu_transformer_blocks": 18,
                        "gpu_required_bytes": 1_000 * harness.MIB,
                        "ram_required_bytes": 2_000,
                        "available_vram_bytes": 1_050 * harness.MIB,
                        "excess_bytes": 0,
                        "headroom_bytes": 50 * harness.MIB,
                        "gpu_weight_bytes": 700 * harness.MIB,
                        "ram_weight_bytes": 300 * harness.MIB,
                        "kv_cache_bytes": 100 * harness.MIB,
                        "device_reserve_bytes": 100 * harness.MIB,
                        "gpu_overhead_bytes": 50 * harness.MIB,
                        "gpu_safety_margin_bytes": 50 * harness.MIB,
                    },
                    "first_rejected_higher": {
                        "gpu_transformer_blocks": 19,
                        "gpu_required_bytes": 1_100 * harness.MIB,
                        "ram_required_bytes": 1_900,
                        "available_vram_bytes": 1_050 * harness.MIB,
                        "excess_bytes": 50 * harness.MIB,
                        "headroom_bytes": 0,
                        "gpu_weight_bytes": 800 * harness.MIB,
                        "ram_weight_bytes": 200 * harness.MIB,
                        "kv_cache_bytes": 100 * harness.MIB,
                        "device_reserve_bytes": 100 * harness.MIB,
                        "gpu_overhead_bytes": 50 * harness.MIB,
                        "gpu_safety_margin_bytes": 50 * harness.MIB,
                    },
                },
                "gpu_required_bytes": 1_000 * harness.MIB,
                "ram_required_bytes": 2_000,
            },
        }


def _run(gpu_layers: int, *, started: bool = True) -> harness.Run:
    return harness.Run(
        gpu_layers=gpu_layers,
        command=["llama-cli", "-ngl", str(gpu_layers)],
        exit_code=0,
        seconds=1.0,
        started=started,
        offloaded=(gpu_layers, 29) if started else None,
        buffers_mib={"CUDA0 model": 900.0},
        vram_baseline_mib=100,
        vram_peak_mib=1_100,
        peak_rss_bytes=3_000,
        log_path="llama.log",
        log_excerpt=[],
        oom=False,
    )


def test_hfa_block_probe_keeps_observations_but_not_vram_error_percentages() -> None:
    report = harness.build_report(
        _Prediction(),  # type: ignore[arg-type]
        baseline=_run(0),
        hfa_block_probe=_run(18),
        sweep=[_run(29)],
        kv_runs=[],
        llama_version="test",
    )

    assert report["runs"]["at_hfa_transformer_block_count"]["gpu_layers"] == 18
    assert report["metrics"]["observed_vram_device_delta_mib"] == 1_000
    assert report["metrics"]["observed_vram_llama_buffers_mib"] == 900.0
    assert report["metrics"]["vram_error_pct_vs_device_delta"] is None
    assert report["metrics"]["vram_error_pct_vs_llama_buffers"] is None
    assert "validated mapping" in report["metrics"]["vram_error_note"]

    rendered = harness.render(report)
    assert "HFA DECISION BOUNDARY" in rendered
    assert "search ceiling     25 transformer blocks" in rendered
    assert "transformer blocks  18" in rendered
    assert "transformer blocks  19" in rendered
    assert "estimated required  1000.0 MiB" in rendered
    assert "measured available  1050.0 MiB" in rendered
    assert "headroom            50.0 MiB" in rendered
    assert "rejected estimated budget breakdown" in rendered
    assert "VRAM error vs device Δ      -" in rendered
    assert "VRAM error vs llama bufs    -" in rendered
