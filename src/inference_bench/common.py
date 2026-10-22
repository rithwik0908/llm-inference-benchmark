"""Explicit experiment identities and atomic, strict JSON artifacts."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_frozen(path, id_key):
    artifact = read_json(path)
    expected = artifact.get(id_key)
    actual = digest({k: v for k, v in artifact.items() if k != id_key})
    if expected != actual:
        raise ValueError(f"Frozen input content hash mismatch: {path}")
    return artifact


def config():
    return read_json(ROOT / "configs/experiment.json")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, compact=False):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before opening a file so NaN/Infinity never leave partial artifacts.
    data = json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(target)


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "packages": dict(sorted(packages.items())),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ]
        ),
    }


def identity():
    cfg = config()
    return {"model": cfg["model"], "revision": cfg["revision"]}


def require_cuda(min_gib=0):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use a Colab GPU runtime or NVIDIA Linux/WSL2.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for comparable runs.")
    free, total = torch.cuda.mem_get_info()
    if total < min_gib * 1024**3:
        raise RuntimeError(
            f"Need >= {min_gib} GiB VRAM; found {total / 1024**3:.1f}. "
            "Use Colab A100/L4. CPU offloading is intentionally disallowed."
        )
    return {
        "name": torch.cuda.get_device_name(),
        "total_bytes": total,
        "free_bytes": free,
        "capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        # CUDA logical device 0 honors CUDA_VISIBLE_DEVICES; nvidia-smi lists all GPUs.
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
    }


def positive(value):
    result = int(value)
    if result < 1:
        raise ValueError("Must be a positive integer")
    return result
