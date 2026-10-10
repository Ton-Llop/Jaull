# ruff: noqa: E501 -- the markdown table is verbatim tool output; wrapping it would fake it.
"""Protocol capture using real-shaped llama-bench output.

The markdown table is captured output. JSONL uses the captured layout with
parameterized test values; it is not a verbatim measurement (sample arrays and
aggregate values are deliberately not used to establish quality or speed).
"""

import json

import pytest

from jaull.domain.benchmarks import BenchmarkObservation, LlamaBenchProtocol
from jaull.recommendation.local_evidence import _UNAUDITED_PROTOCOL, _protocol_blockers
from jaull.runtime.llama_bench_jsonl import parse_llama_bench_protocol
from jaull.runtime.llama_bench_parser import parse_llama_bench_output

COMPLETE = LlamaBenchProtocol(
    build_commit="689e227db", build_number=10357, n_threads=6, n_batch=2048,
    n_ubatch=512, cache_type_k="f16", cache_type_v="f16", flash_attn="off",
    load_mode="mmap", warmup="ran",
)

# stdout: no threads, batch, cache or load-mode column anywhere in it.
REAL_MD = """\
| model                          |       size |     params | backend    | ngl | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------ | --------------: | -------------------: |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  -1 | CUDA0        |            pp32 |    2468.89 ± 1529.99 |
| llama 256M Q4_K - Medium       |  98.87 MiB |   134.52 M | CUDA       |  -1 | CUDA0        |            tg16 |       271.89 ± 40.40 |

build: 689e227db (10357)
"""

# stderr from `-oe jsonl`: the CUDA banner, then one object per test.
_ROW = (
    '{{"build_commit": "689e227db", "build_number": 10357, "cpu_info": "AMD Ryzen 5 3600 '
    '6-Core Processor", "gpu_info": "NVIDIA GeForce RTX 2060", "backends": "CUDA", '
    '"model_filename": "/home/ton/models/SmolLM2-135M-Instruct-Q4_K_M.gguf", '
    '"model_type": "llama 256M Q4_K - Medium", "model_size": 103668480, '
    '"model_n_params": 134515008, "n_batch": 2048, "n_ubatch": 512, "n_threads": 6, '
    '"cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", '
    '"type_v": "f16", "n_gpu_layers": -1, "n_cpu_moe": 0, "split_mode": "layer", '
    '"main_gpu": 0, "no_kv_offload": false, "flash_attn": {flash}, "devices": "CUDA0", '
    '"tensor_split": "0.00", "tensor_buft_overrides": "none", "load_mode": "mmap", '
    '"embeddings": false, "no_op_offload": 0, "no_host": false, "fit_target": 0, '
    '"fit_min_ctx": 0, "n_prompt": {pp}, "n_gen": {tg}, "n_depth": 0, '
    '"test_time": "2026-10-07T19:58:17Z", "avg_ns": 20080684, "stddev_ns": 0, '
    '"avg_ts": {ts}, "stddev_ts": 0.0, "samples_ns": [ 20080684 ],"samples_ts": [ 398.393 ]}}'
)
BANNER = (
    "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6143 MiB):\n"
    "  Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5, VMM: yes, VRAM: 6143 MiB"
)


def real_stderr(flash: int = -1) -> str:
    rows = (
        _ROW.format(flash=flash, pp=32, tg=0, ts=2468.89),
        _ROW.format(flash=flash, pp=0, tg=16, ts=271.89),
    )
    return "\n".join((BANNER, *rows))


def combined(flash: int = -1) -> str:
    """What the runner parses: stdout and stderr joined, exactly as it joins them."""
    return "\n".join((REAL_MD, real_stderr(flash)))


def test_the_markdown_table_alone_cannot_supply_the_protocol() -> None:
    # This is the whole reason for reading JSONL: the table parses fine and
    # still contains none of the settings, because they were at their defaults.
    assert parse_llama_bench_output(REAL_MD, repetitions=2)
    assert parse_llama_bench_protocol(REAL_MD) is None
    for absent in ("thread", "n_batch", "ubatch", "type_k", "load_mode"):
        assert absent not in REAL_MD


def test_the_effective_settings_come_from_the_tool_not_from_this_machine() -> None:
    protocol = parse_llama_bench_protocol(combined(), warmup=True)
    assert protocol == LlamaBenchProtocol(
        build_commit="689e227db", build_number=10357,
        n_threads=6, n_batch=2048, n_ubatch=512,
        cache_type_k="f16", cache_type_v="f16",
        flash_attn="auto_unresolved", load_mode="mmap", warmup="ran",
    )


def test_an_unresolved_auto_flash_attn_stays_unresolved() -> None:
    # All three are requested modes. In particular, `on` can be forced off.
    assert parse_llama_bench_protocol(combined(flash=1)).flash_attn == "on"
    assert parse_llama_bench_protocol(combined(flash=0)).flash_attn == "off"
    assert parse_llama_bench_protocol(combined(flash=-1)).flash_attn == "auto_unresolved"


def test_warmup_is_absent_from_the_output_and_only_the_caller_knows_it() -> None:
    assert "warm" not in real_stderr().casefold()
    assert parse_llama_bench_protocol(combined()).warmup is None
    assert parse_llama_bench_protocol(combined(), warmup=True).warmup == "ran"
    assert parse_llama_bench_protocol(combined(), warmup=False).warmup == "skipped"


def test_rows_that_disagree_report_unknown_rather_than_one_rows_value() -> None:
    # A sweep nobody requested: two tests, two thread counts. Picking either
    # would attribute one row's protocol to the other row's measurement.
    swept = combined().replace('"n_threads": 6', '"n_threads": 3', 1)
    protocol = parse_llama_bench_protocol(swept, warmup=True)
    assert protocol is not None
    assert protocol.n_threads is None
    assert protocol.n_batch == 2048  # Still agreed, so still known.


def test_log_noise_and_truncated_json_are_skipped_not_guessed_at() -> None:
    assert parse_llama_bench_protocol(BANNER) is None
    assert parse_llama_bench_protocol('{"build_commit": "abc", "n_thr') is None
    assert parse_llama_bench_protocol("") is None
    # A JSON object that is not a result row.
    assert parse_llama_bench_protocol('{"level": "warn", "msg": "slow"}') is None


def test_old_records_keep_loading_with_an_unknown_protocol() -> None:
    # The shape a record persisted before protocol capture has: no such key.
    old = BenchmarkObservation.model_validate({
        "success": True, "repetitions": 5, "duration_seconds": 1.0,
        "methodology": "llama_bench_v1",
        "measurements": [{
            "kind": "generation", "tokens": 128, "mean_tokens_per_second": 25.0,
            "stddev_tokens_per_second": 0.5, "source_label": "tg128",
        }],
    })
    assert old.protocol is None
    # And it is refused for Fastest by that absence, with a reason that says so.
    assert "not recorded by the run" in " ".join(
        _protocol_blockers(old.protocol, "689e227db (10357)")
    )


def test_complete_jsonl_settings_do_not_prove_runtime_applied_batch_limits() -> None:
    assert _protocol_blockers(COMPLETE, "689e227db (10357)") == [
        "JSONL reports requested batch/microbatch limits, not runtime-applied context limits.",
    ]
    assert _protocol_blockers(None, "689e227db (10357)") == [_UNAUDITED_PROTOCOL]


def test_every_missing_setting_is_named_rather_than_filled_in() -> None:
    for field, phrase in (
        ("n_threads", "thread count"),
        ("n_batch", "batch size"),
        ("n_ubatch", "microbatch size"),
        ("cache_type_k", "K cache type"),
        ("cache_type_v", "V cache type"),
        ("load_mode", "load mode"),
        ("build_commit", "its own build"),
        ("build_number", "its own build number"),
        ("warmup", "warmup ran"),
    ):
        partial = COMPLETE.model_copy(update={field: None})
        blockers = _protocol_blockers(partial, "689e227db (10357)")
        assert any(phrase in blocker for blocker in blockers), (field, blockers)
        assert len(blockers) == 2, (field, blockers)  # Missing field + applied-batch blocker.


def test_auto_flash_attention_blocks_with_its_actual_reason() -> None:
    blockers = _protocol_blockers(
        COMPLETE.model_copy(update={"flash_attn": "auto_unresolved"}),
        "689e227db (10357)",
    )
    assert len(blockers) == 2
    assert "reports the requested value" in blockers[0]


def test_requested_flash_on_does_not_prove_runtime_execution() -> None:
    blockers = _protocol_blockers(
        COMPLETE.model_copy(update={"flash_attn": "on"}), "689e227db (10357)",
    )
    assert any("runtime can disable it" in reason for reason in blockers)


@pytest.mark.parametrize("key,value", [
    ("n_threads", -1), ("n_threads", True), ("n_threads", "6"),
    ("n_batch", []), ("type_k", {"name": "f16"}), ("build_number", 1.5),
])
def test_malformed_metadata_is_unknown_without_crashing_or_trusting_repr(key, value) -> None:
    row = json.loads(_ROW.format(flash=-1, pp=512, tg=0, ts=10))
    row[key] = value
    assert parse_llama_bench_protocol(json.dumps(row), warmup=True) is None


def test_a_result_row_missing_threads_is_not_discarded_from_consensus() -> None:
    row = json.loads(_ROW.format(flash=-1, pp=512, tg=0, ts=10))
    del row["n_threads"]
    protocol = parse_llama_bench_protocol(combined() + "\n" + json.dumps(row), warmup=True)
    assert protocol is not None and protocol.n_threads is None


@pytest.mark.parametrize("update", [{"build_commit": "689e"}, {"build_number": 1}])
def test_build_prefix_or_different_number_does_not_confirm_producing_build(update) -> None:
    assert any("disagrees" in reason for reason in _protocol_blockers(
        COMPLETE.model_copy(update=update), "689e227db (10357)",
    ))


def test_a_protocol_build_that_contradicts_the_output_is_refused() -> None:
    disagreeing = COMPLETE.model_copy(update={"build_commit": "0000000"})
    assert "The recorded protocol build disagrees with the producing build." in (
        _protocol_blockers(disagreeing, "689e227db (10357)")
    )
    # An unknown producing build is already reported by the caller's own check,
    # so this one stays quiet instead of duplicating it.
    assert not any("build disagrees" in reason for reason in (
        _protocol_blockers(disagreeing, None)
    ))
