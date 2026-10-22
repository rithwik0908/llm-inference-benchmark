"""Exercise real notebook orchestration in CPU subprocesses, without Colab or GPU access."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_workflow.ipynb"


def runtime_helpers(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in notebook.cells if c.id == "runtime-helpers")
    tree = ast.parse(cell.source)
    # Load the actual helper definitions, excluding the real CUDA preflight invocation.
    definitions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    namespace = {
        "ROOT": tmp_path,
        "GPU_PY": Path(sys.executable),
        "PROC_ENV": {**os.environ, "MPLBACKEND": "module://missing_colab_backend"},
        "os": os,
        "json": json,
        "Path": Path,
        "subprocess": subprocess,
        "server": None,
        "server_log": None,
    }
    exec(compile(definitions, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_notebook_schema_syntax_and_clean_outputs():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs
            assert cell.execution_count is None
    ids = [cell.id for cell in notebook.cells]
    assert ids.index("baseline") < ids.index("quantization-quality") < ids.index("sweep-fp16")
    assert ids.index("sweep-fp16") < ids.index("sweep-awq") < ids.index("verify-edge")
    assert ids.index("verify-edge") < ids.index("report")
    quality = next(c for c in notebook.cells if c.id == "quantization-quality").source
    assert "benchmark_serving.py" not in quality
    fp16 = next(c for c in notebook.cells if c.id == "sweep-fp16").source
    assert "evaluate_quality.py" not in fp16


def test_gpu_commands_override_inherited_inline_backend(tmp_path):
    helpers = runtime_helpers(tmp_path)
    # Run the actual Matplotlib import that failed under the inherited notebook backend.
    helpers["run"]("-c", "import matplotlib.pyplot as plt; print(plt.get_backend())", timeout_s=10)
    log = next((tmp_path / "results").glob("*.log"))
    assert log.read_text().strip().lower() == "agg"
    assert helpers["PROC_ENV"]["MPLBACKEND"] == "module://missing_colab_backend"


def test_command_failure_and_timeout_stop_instead_of_continuing(tmp_path):
    helpers = runtime_helpers(tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        helpers["run"]("-c", "raise RuntimeError('test-only failure')", timeout_s=10)
    assert failure.value.returncode != 0
    assert any(
        "test-only failure" in log.read_text() for log in (tmp_path / "results").glob("*.log")
    )
    with pytest.raises(TimeoutError, match="STOP"):
        helpers["run"]("-c", "import time; time.sleep(20)", timeout_s=0.05)


def test_server_subprocess_uses_gpu_python_and_agg(monkeypatch, tmp_path):
    helpers = runtime_helpers(tmp_path)
    (tmp_path / "results").mkdir()
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(poll=lambda: None)

    class HealthResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(helpers["subprocess"], "Popen", popen)
    monkeypatch.setattr(helpers["urllib"].request, "urlopen", lambda *a, **kw: HealthResponse())
    helpers["terminate_group"] = lambda process: None
    try:
        helpers["start_server"]("awq")
        command, kwargs = calls[0]
        assert command == [sys.executable, "-m", "inference_bench.server", "awq"]
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        assert kwargs["start_new_session"]
    finally:
        helpers["stop_server"]()


@pytest.mark.parametrize("csv_uuid, valid", [("GPU-test-uuid", True), ("GPU-other-uuid", False)])
def test_laptop_handoff_uses_real_samples_from_matching_gpu(tmp_path, csv_uuid, valid):
    import math

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in notebook.cells if c.id == "verify-edge")
    tree = ast.parse(cell.source)
    # Supply a file-picker stub; execute the real parsing/validation code without Colab.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "google.colab")
    ]
    edge = {
        "identity": {"model": "test-only"},
        "checkpoint_id": "test-checkpoint",
        "gpu": {"name": "test RTX 5060", "uuid": "test-uuid"},
        "text": "test-only output",
        "output_tokens": 1,
    }
    uploads = {
        "edge_demo.json": json.dumps(edge).encode(),
        "edge_vram.csv": (
            f"test-time, test RTX 5060, {csv_uuid}, 10\ntest-time, test RTX 5060, {csv_uuid}, 20\n"
        ).encode(),
    }
    (tmp_path / "results").mkdir()
    context = {
        "ROOT": tmp_path,
        "json": json,
        "math": math,
        "FP16_SWEEP_OK": True,
        "AWQ_SWEEP_OK": True,
        "expected_identity": edge["identity"],
        "quality": {"awq_checkpoint_id": "test-checkpoint"},
        "files": SimpleNamespace(upload=lambda: uploads),
    }
    if valid:
        exec(compile(tree, str(NOTEBOOK), "exec"), context)
        summary = json.loads((tmp_path / "results/edge_vram_summary.json").read_text())
        assert summary["sampled_device_used_peak_mib"] == 20
        assert summary["sample_count"] == 2
        assert context["EDGE_OK"]
    else:
        with pytest.raises(AssertionError, match="VRAM GPU mismatch"):
            exec(compile(tree, str(NOTEBOOK), "exec"), context)
        assert not context["EDGE_OK"]
        assert not (tmp_path / "results/edge_vram_summary.json").exists()
