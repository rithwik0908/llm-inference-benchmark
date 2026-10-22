import math

import pytest

from inference_bench.common import digest, read_json, write_json
from inference_bench.metrics import percentile, summarize


def test_percentiles_use_linear_interpolation():
    assert percentile([4, 1, 2, 3], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.99) == pytest.approx(3.97)
    assert percentile([8], 0.99) == 8
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_throughput_counts_only_successes_and_full_wall_time():
    rows = [
        {"success": True, "output_tokens": 10, "latency_s": 1, "ttft_s": 0.1},
        {"success": True, "output_tokens": 20, "latency_s": 3, "ttft_s": 0.3},
        {"success": False, "output_tokens": 500, "latency_s": 4, "ttft_s": None},
    ]
    result = summarize(rows, 5)
    assert result["output_tokens_per_s"] == 6
    assert result["latency_s_p50"] == 2
    assert result["failed_requests"] == 1
    failed = summarize([rows[-1]], 2)
    assert failed["latency_s_p99"] is None
    assert failed["output_tokens_per_s"] == 0


def test_atomic_strict_json_preserves_existing_file(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"x": 1})
    with pytest.raises(ValueError):
        write_json(path, {"x": math.nan})
    assert read_json(path) == {"x": 1}
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
