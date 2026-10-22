import copy

import pytest

from inference_bench.common import identity
from inference_bench.quality import compare_quality, parse_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A", "A"),
        (" B.", "B"),
        ("C) correct", "C"),
        ("D\n", "D"),
        ("Answer: A", None),
        ("Apple", None),
        ("", None),
        ("a", None),
    ],
)
def test_strict_answer_parser(text, expected):
    assert parse_answer(text) == expected


def quality_pair():
    a = {
        "identity": identity(),
        "variant": "fp16",
        "server": {"checkpoint_id": "test-checkpoint"},
        "eval_set_id": "test-fixture-only",
        "generation": {"max_tokens": 4},
        "rows": [{"id": i, "answer": "A", "prediction": "A", "correct": True} for i in range(50)],
    }
    b = copy.deepcopy(a)
    b["variant"] = "awq"
    b["rows"][0].update(prediction=None, correct=False)
    return a, b


def test_paired_delta_is_percentage_points_with_small_sample_resolution():
    a, b = quality_pair()
    result = compare_quality(a, b)
    assert result["accuracy_delta_pp_awq_minus_fp16"] == -2
    assert result["resolution_pp"] == 2
    assert result["changed_answers"] == 1
    assert result["awq_invalid"] == 1
    assert result["paired_bootstrap_95ci_pp"][0] <= -2


@pytest.mark.parametrize("change", ["eval_set_id", "identity", "generation"])
def test_quality_comparison_rejects_mismatched_runs(change):
    a, b = quality_pair()
    b[change] = "different"
    with pytest.raises(ValueError):
        compare_quality(a, b)
