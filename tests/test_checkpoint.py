"""Checkpoint integrity contract on tiny byte fixtures, without model or GPU imports."""

import hashlib

import pytest

from inference_bench.common import digest, identity, write_json
from inference_bench.server import checkpoint_manifest


def make_checkpoint(path):
    weight = path / "model.safetensors"
    weight.write_bytes(b"fixture")
    weights = {weight.name: {"sha256": hashlib.sha256(b"fixture").hexdigest(), "bytes": 7}}
    quant_config = {"quant_method": "compressed-tensors", "test_only": True}
    write_json(path / "config.json", {"quantization_config": quant_config})
    content = {"identity": identity(), "weights": weights, "quantization_config": quant_config}
    manifest = {**content, "checkpoint_id": digest(content)}
    write_json(path / "benchmark_manifest.json", manifest)
    return manifest


def test_checkpoint_content_identity_survives_copy(tmp_path):
    import shutil

    original = tmp_path / "original"
    original.mkdir()
    manifest = make_checkpoint(original)
    destination = tmp_path / "copy"
    shutil.copytree(original, destination)
    assert checkpoint_manifest(destination)["checkpoint_id"] == manifest["checkpoint_id"]


@pytest.mark.parametrize("damage", ["weights", "config", "source"])
def test_checkpoint_rejects_corruption_and_wrong_source(tmp_path, damage):
    manifest = make_checkpoint(tmp_path)
    if damage == "weights":
        (tmp_path / "model.safetensors").write_bytes(b"changed")  # Same size, different content.
    elif damage == "config":
        write_json(tmp_path / "config.json", {"quantization_config": {"test_only": "changed"}})
    else:
        manifest["identity"] = {"model": "different", "revision": "different"}
        write_json(tmp_path / "benchmark_manifest.json", manifest)
    with pytest.raises(ValueError):
        checkpoint_manifest(tmp_path)
