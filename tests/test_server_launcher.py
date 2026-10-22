"""The Colab GPU interpreter must own the CLI even when its bin directory is not on PATH."""

import json
import os
import subprocess
import sys

import pytest

from inference_bench import server
from inference_bench.common import read_json


def test_server_launches_cli_with_current_python_without_path(monkeypatch, tmp_path):
    manifest = tmp_path / "server.json"
    monkeypatch.setattr(sys, "argv", ["server", "fp16", "--manifest", str(manifest)])
    monkeypatch.setattr(server, "require_cuda", lambda minimum: {"uuid": "test-only"})
    monkeypatch.setattr(server, "provenance", lambda: {})
    calls = []

    def capture_exec(executable, argv):
        calls.append((executable, argv))
        raise SystemExit(0)

    monkeypatch.setattr(server.os, "execvp", capture_exec)
    with pytest.raises(SystemExit) as exit_info:
        server.main()
    assert exit_info.value.code == 0
    executable, argv = calls[0]
    assert executable == sys.executable
    assert argv[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    assert read_json(manifest)["command"] == argv

    # Execute the command against a tiny fake CLI module with no vllm executable
    # available on PATH. This validates interpreter/module resolution without CUDA.
    package = tmp_path / "vllm" / "entrypoints" / "cli"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").touch()
    (package / "main.py").write_text(
        "import json, sys\nprint(json.dumps({'python': sys.executable, 'args': sys.argv[1:]}))\n"
    )
    env = {**os.environ, "PATH": str(tmp_path / "empty-path"), "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(argv, env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout)
    assert observed["python"] == sys.executable
    assert observed["args"] == argv[3:]
