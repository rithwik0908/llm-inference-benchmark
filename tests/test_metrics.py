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


def test_frozen_content_identity_rejects_changed_inputs(tmp_path):
    from inference_bench.common import read_frozen

    path = tmp_path / "inputs.json"
    content = {"prompts": [[1, 2, 3]]}
    content["workload_id"] = digest(content)
    write_json(path, content)
    assert read_frozen(path, "workload_id") == content
    content["prompts"][0][0] = 9
    write_json(path, content)
    with pytest.raises(ValueError, match="content hash mismatch"):
        read_frozen(path, "workload_id")


def test_cuda_preflight_tracks_selected_device_and_rejects_small_vram(monkeypatch):
    import sys
    from types import SimpleNamespace

    from inference_bench.common import require_cuda

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda: (7 * 1024**3, 8 * 1024**3),
        get_device_name=lambda: "test GPU",
        get_device_capability=lambda: (12, 0),
        get_device_properties=lambda device: SimpleNamespace(uuid="selected-logical-GPU"),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test"))
    )
    assert require_cuda()["uuid"] == "selected-logical-GPU"
    with pytest.raises(RuntimeError, match="Colab A100/L4"):
        require_cuda(22)
