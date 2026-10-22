"""Calibrate AWQ W4A16_ASYM and save vLLM-native compressed-tensors weights."""

import argparse
import hashlib
import time
from pathlib import Path

from inference_bench.common import (
    config,
    digest,
    identity,
    positive,
    provenance,
    read_json,
    require_cuda,
    write_json,
)


def main():
    import torch
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/mistral-7b-awq")
    parser.add_argument("--samples", type=positive)
    parser.add_argument("--results", default="results/quantization.json")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty checkpoint directory: {target}")
    cfg = config()
    calibration = dict(cfg["calibration"])
    if args.samples:
        calibration["samples"] = args.samples
    gpu = require_cuda(22)
    torch.manual_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()  # CPU staging; oneshot's sequential pipeline handles device dispatch.
    ds = load_dataset(
        calibration["dataset"], revision=calibration["revision"],
        split=f"{calibration['split']}[:{calibration['samples']}]",
    ).shuffle(seed=cfg["seed"])

    def tokenize(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return tokenizer(text, add_special_tokens=False, truncation=True,
                         max_length=calibration["max_seq_length"], padding=False)

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    calibration["tokenized_sha256"] = digest(ds.to_dict())
    recipe = AWQModifier(
        ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"], duo_scaling="both"
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    oneshot(
        model=model, dataset=ds, recipe=[recipe],
        max_seq_length=calibration["max_seq_length"],
        num_calibration_samples=calibration["samples"],
    )
    torch.cuda.synchronize()
    quantization_s = time.perf_counter() - start
    model.save_pretrained(target, save_compressed=True)
    tokenizer.save_pretrained(target)
    saved = read_json(target / "config.json")
    quant_config = saved.get("quantization_config", {})
    if quant_config.get("quant_method") != "compressed-tensors":
        raise RuntimeError("Expected compressed-tensors output; checkpoint is not publishable")
    groups = quant_config.get("config_groups", {}).values()
    if not any(g.get("weights", {}).get("num_bits") == 4 for g in groups):
        raise RuntimeError("Saved checkpoint does not declare 4-bit weights")
    # Content identity survives copies between Colab and laptop.
    weight_hashes = {}
    for file in sorted(target.glob("*.safetensors")):
        hasher = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                hasher.update(chunk)
        weight_hashes[file.name] = {"sha256": hasher.hexdigest(), "bytes": file.stat().st_size}
    if not weight_hashes:
        raise RuntimeError("No safetensors weights were saved")
    result = {
        "schema_version": 1, "identity": identity(), "variant": "awq",
        "algorithm": "AWQ", "scheme": "W4A16_ASYM", "group_size": 128,
        "ignored_modules": ["lm_head"], "calibration": calibration,
        "environment": provenance(), "gpu": gpu, "quantization_s": quantization_s,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_path": str(target), "weights": weight_hashes,
        "quantization_config": quant_config,
        "notes": "First N UltraChat train_sft rows, shuffled; not a population-random sample. "
                 "Calibration data is disjoint from MMLU test evaluation. Timing excludes save/hash.",
    }
    result["checkpoint_id"] = digest({"identity": identity(), "weights": weight_hashes,
                                      "quantization_config": quant_config})
    write_json(target / "benchmark_manifest.json", result)
    write_json(args.results, result)
    print(f"Saved {target}; quantization took {quantization_s / 60:.1f} minutes")


if __name__ == "__main__":
    main()
