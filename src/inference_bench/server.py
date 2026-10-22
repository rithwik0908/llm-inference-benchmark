"""Launch one fixed, recorded serving configuration for either weight variant."""

import argparse
import os
import sys
import uuid
from pathlib import Path

from inference_bench.common import (
    config,
    digest,
    file_digest,
    identity,
    provenance,
    read_json,
    require_cuda,
    write_json,
)


def checkpoint_manifest(path):
    root = Path(path).resolve()
    manifest = read_json(root / "benchmark_manifest.json")
    if manifest["identity"] != identity():
        raise ValueError("AWQ checkpoint does not match configured FP16 model revision")
    for name, expected in manifest["weights"].items():
        file = root / name
        if file.parent != root or file.stat().st_size != expected["bytes"]:
            raise ValueError(f"Invalid checkpoint weight: {name}")
        # Stream large weights rather than read the entire checkpoint into host RAM.
        import hashlib

        hasher = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    actual_config = read_json(root / "config.json")["quantization_config"]
    expected_id = digest(
        {
            "identity": identity(),
            "weights": manifest["weights"],
            "quantization_config": actual_config,
        }
    )
    if expected_id != manifest["checkpoint_id"]:
        raise ValueError("Checkpoint quantization config/manifest mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["fp16", "awq"])
    parser.add_argument("--model-path", default="models/mistral-7b-awq")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    cfg = config()
    gpu = require_cuda(22 if args.variant == "fp16" else 6)
    manifest_path = args.manifest or f"results/server_{args.variant}.json"
    settings = cfg["serving"]
    alias = f"benchmark-{args.variant}-{uuid.uuid4().hex[:12]}"
    argv = [
        "vllm",
        "serve",
        cfg["model"] if args.variant == "fp16" else args.model_path,
        "--served-model-name",
        alias,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--dtype",
        settings["dtype"],
        "--max-model-len",
        str(settings["max_model_len"]),
        "--max-num-seqs",
        str(settings["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(settings["gpu_memory_utilization"]),
        "--max-num-batched-tokens",
        str(settings["max_num_batched_tokens"]),
        "--enable-prefix-caching" if settings["prefix_caching"] else "--no-enable-prefix-caching",
        "--enable-chunked-prefill"
        if settings["enable_chunked_prefill"]
        else "--no-enable-chunked-prefill",
        "--seed",
        str(cfg["seed"]),
        "--generation-config",
        "vllm",
    ]
    if settings["enforce_eager"]:
        argv += ["--enforce-eager"]
    checkpoint = None
    if args.variant == "fp16":
        argv += ["--revision", cfg["revision"], "--tokenizer-revision", cfg["revision"]]
    else:
        checkpoint = checkpoint_manifest(args.model_path)
        argv += ["--quantization", "compressed-tensors"]
    report = {
        "schema_version": 1,
        "variant": args.variant,
        "identity": identity(),
        "served_model_name": alias,
        "environment": provenance(),
        "gpu": gpu,
        "serving_settings": settings,
        "command": argv,
        "checkpoint_id": checkpoint["checkpoint_id"] if checkpoint else digest(identity()),
        "model_config_sha256": file_digest(Path(args.model_path) / "config.json")
        if checkpoint
        else None,
    }
    write_json(manifest_path, report)
    print(f"Server manifest: {manifest_path}; wait for /health before clients", flush=True)
    sys.stdout.flush()
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
