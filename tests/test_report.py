import copy

import pytest

from inference_bench.client import MEASUREMENT_PROTOCOL
from inference_bench.common import identity
from inference_bench.report import validate_pair
from scripts.plot_results import render


def serving_pair():
    # These numbers exercise report math only and are never saved under results/.
    gpu = {"name": "synthetic-test-GPU", "total_bytes": 1, "capability": [0, 0], "uuid": "test"}
    artifact = {
        "valid": True,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "variant": "fp16",
        "identity": identity(),
        "workload_id": "test-only",
        "input_tokens": 2,
        "output_tokens": 3,
        "requests_per_run": 8,
        "repeats": 1,
        "concurrency_levels": [1, 4],
        "server": {
            "gpu": gpu,
            "serving_settings": {"prefix_caching": False},
            "checkpoint_id": "test-checkpoint",
            "environment": {
                "packages": {
                    p: "test" for p in ("torch", "vllm", "transformers", "compressed-tensors")
                }
            },
        },
        "runs": [
            {
                "concurrency": c,
                "repeat": 0,
                "summary": {
                    "failed_requests": 0,
                    "requests_without_visible_text": 0,
                    "successful_requests": 8,
                    "output_tokens_per_s": 10 * c,
                    "latency_s_p50": 1,
                    "latency_s_p99": 2,
                    "ttft_s_p50": 0.1,
                    "ttft_s_p99": 0.2,
                },
            }
            for c in (1, 4)
        ],
    }
    other = copy.deepcopy(artifact)
    other["variant"] = "awq"
    return artifact, other


@pytest.mark.parametrize("change", ["gpu", "workload", "failures", "incomplete", "settings"])
def test_report_rejects_invalid_comparisons(change):
    a, b = serving_pair()
    if change == "gpu":
        b["server"]["gpu"]["uuid"] = "another-GPU"
    elif change == "workload":
        b["workload_id"] = "different"
    elif change == "failures":
        b["runs"][0]["summary"]["failed_requests"] = 1
    elif change == "incomplete":
        b["runs"].pop()
    elif change == "settings":
        b["server"]["serving_settings"]["prefix_caching"] = True
    with pytest.raises(ValueError):
        validate_pair(a, b)


def test_charts_csv_and_writeup_render_from_valid_data(tmp_path):
    a, b = serving_pair()
    quality = {
        "identity": identity(),
        "fp16_accuracy_pct": 100,
        "awq_accuracy_pct": 98,
        "questions": 50,
        "fp16_checkpoint_id": "test-checkpoint",
        "awq_checkpoint_id": "test-checkpoint",
        "accuracy_delta_pp_awq_minus_fp16": -2,
        "paired_bootstrap_95ci_pp": [-6, 0],
        "resolution_pp": 2,
        "limitations": "Synthetic test data only.",
    }
    baseline = {
        "identity": identity(),
        "source_checkpoint_safetensors_bytes": 100,
        "gpu": a["server"]["gpu"],
        "summary": {"output_tokens_per_s_p50": 1, "ttft_s_p50": 0.1},
        "peak_allocated_bytes": 100,
    }
    quantization = {
        "identity": identity(),
        "checkpoint_id": "test-checkpoint",
        "weights": {"fixture.safetensors": {"bytes": 50}},
    }
    render(a, b, quality, tmp_path, baseline, quantization)
    for name in ("throughput.png", "latency.png", "memory_quality.png", "summary.csv", "REPORT.md"):
        assert (tmp_path / name).stat().st_size > 100
    assert "1.00×" in (tmp_path / "REPORT.md").read_text()


def test_report_refuses_legacy_text_ttft_artifact():
    a, b = serving_pair()
    del a["measurement_protocol"]
    with pytest.raises(ValueError, match="TTFT protocol"):
        validate_pair(a, b)
