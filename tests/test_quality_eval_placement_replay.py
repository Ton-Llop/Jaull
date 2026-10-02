"""Offline checks for the placement replay. No server, no network, synthetic scores."""

from pathlib import Path

import pytest
from scripts.quality_eval_placement_replay import (
    answer_set,
    captured_scoring_requests,
    continuation_groups,
    deltas,
    loglikelihoods,
    margin_shift,
    metrics,
    verify_baseline,
)
from scripts.quality_eval_smoke import server_command

# Two choices for one sample: the first wins on raw loglikelihood, the second
# wins once normalised by continuation length. That disagreement is the only
# reason acc_norm exists, so the metric code has to reproduce it.
PAYLOADS = [
    {"prompt": [0] * 5}, {"prompt": [0] * 6},
    {"prompt": [0] * 5}, {"prompt": [0] * 6}, {"prompt": [0] * 7},
]
SCORES = [-1.0, -2.0, -1.0, -1.5, -1.5]
SAMPLES = [{
    "doc_id": 0,
    "target": "1",
    "arguments": [["ctx", " aa"], ["ctx", " bbbb"]],
    "filtered_resps": [[-3.0, False], [-4.0, False]],
}]
REPORTED = {"acc,none": 0.0, "acc_norm,none": 1.0}


def test_continuations_split_where_the_prompt_stops_growing_by_one() -> None:
    groups = continuation_groups(PAYLOADS)
    assert groups == [[0, 1], [2, 3, 4]]
    assert loglikelihoods(SCORES, groups) == [-3.0, -4.0]


def test_normalisation_decides_acc_norm_independently_of_acc() -> None:
    computed = metrics(loglikelihoods(SCORES, continuation_groups(PAYLOADS)), SAMPLES)
    assert computed["acc"] == 0.0 and computed["acc_norm"] == 1.0
    entry = computed["per_sample"][0]
    assert entry["argmax_raw"] == 0 and entry["argmax_norm"] == 1


def test_a_baseline_that_does_not_reproduce_the_harness_is_refused() -> None:
    groups = continuation_groups(PAYLOADS)
    assert verify_baseline(SCORES, groups, SAMPLES, REPORTED)["acc_norm"] == 1.0
    drifted = [*SCORES[:-1], SCORES[-1] - 0.5]
    with pytest.raises(ValueError, match="reconstruct"):
        verify_baseline(drifted, groups, SAMPLES, REPORTED)
    with pytest.raises(ValueError, match="Recomputed acc"):
        verify_baseline(SCORES, groups, SAMPLES, REPORTED | {"acc,none": 1.0})


def test_deltas_separate_bit_equality_from_a_small_difference() -> None:
    assert deltas([1.0, 2.0], [1.0, 2.0]) == {
        "identical": True, "differing_tokens": 0, "tokens": 2, "max_abs_delta": 0.0,
    }
    moved = deltas([1.0, 2.0], [1.0, 2.5])
    assert not moved["identical"] and moved["differing_tokens"] == 1
    assert moved["max_abs_delta"] == pytest.approx(0.5)


def test_the_margin_is_the_lead_of_the_chosen_answer() -> None:
    computed = metrics(loglikelihoods(SCORES, continuation_groups(PAYLOADS)), SAMPLES)
    entry = computed["per_sample"][0]
    # Raw picks choice 0 (-3.0 over -4.0); normalised picks choice 1 (-0.8 over -1.0).
    assert entry["margin_raw"] == pytest.approx(1.0)
    assert entry["margin_norm"] == pytest.approx(0.2)


def test_answers_are_compared_one_by_one_not_through_the_aggregate() -> None:
    # One sample flips right to wrong and another wrong to right, so acc and
    # acc_norm are unchanged while two answers moved.
    before = [
        {"doc_id": 0, "argmax_raw": 0, "argmax_norm": 0, "acc": 1, "acc_norm": 1},
        {"doc_id": 1, "argmax_raw": 1, "argmax_norm": 1, "acc": 0, "acc_norm": 0},
    ]
    after = [
        {"doc_id": 0, "argmax_raw": 2, "argmax_norm": 2, "acc": 0, "acc_norm": 0},
        {"doc_id": 1, "argmax_raw": 3, "argmax_norm": 3, "acc": 1, "acc_norm": 1},
    ]
    for metric in ("acc", "acc_norm"):
        assert sum(entry[metric] for entry in before) == sum(entry[metric] for entry in after)
    assert answer_set(before) != answer_set(after)
    assert answer_set(before) == answer_set(list(before))


def test_margin_shift_reports_the_move_and_what_is_left() -> None:
    before = [{"doc_id": 0, "margin_raw": 1.0, "margin_norm": 0.0255}]
    after = [{"doc_id": 0, "margin_raw": 0.8, "margin_norm": 0.0182}]
    shift = margin_shift(before, after)
    assert shift["max_abs_shift"]["margin_raw"] == pytest.approx(0.2)
    assert shift["max_abs_shift"]["margin_norm"] == pytest.approx(0.0073)
    assert shift["min_margin"]["margin_norm"] == pytest.approx(0.0182)


def test_a_failed_bundle_is_never_a_baseline(tmp_path: Path) -> None:
    (tmp_path / "runner-error.json").write_text("{}")
    with pytest.raises(ValueError, match="failed bundle"):
        captured_scoring_requests(tmp_path)


def test_only_the_replay_moves_placement_and_cpu_arms_drop_the_device() -> None:
    default = server_command(Path("/server"), Path("/model.gguf"))
    assert default[default.index("--n-gpu-layers") + 1] == "-1"
    assert default[default.index("--device") + 1] == "CUDA0"
    assert default[default.index("--threads") + 1] == "4"
    cpu = server_command(Path("/server"), Path("/model.gguf"), n_gpu_layers=0, device=None,
                         threads=8)
    assert "--device" not in cpu
    assert cpu[cpu.index("--n-gpu-layers") + 1] == "0"
    assert cpu[cpu.index("--threads") + 1] == "8"
