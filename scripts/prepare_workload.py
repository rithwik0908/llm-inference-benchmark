"""Freeze a synthetic random-token workload for both serving variants."""

import argparse
import random

from inference_bench.common import config, digest, identity, positive, write_json


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/workload.json")
    parser.add_argument("--requests", type=positive)
    args = parser.parse_args()
    cfg = config()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    rng = random.Random(cfg["seed"])
    special = set(tokenizer.all_special_ids)
    vocab = sorted(set(tokenizer.get_vocab().values()) - special)
    count = args.requests or cfg["workload"]["requests"]
    artifact = {
        "schema_version": 1,
        "identity": identity(),
        "seed": cfg["seed"],
        "kind": "synthetic_uniform_random_token_ids_no_special_tokens",
        "input_tokens": cfg["workload"]["input_tokens"],
        "output_tokens": cfg["workload"]["output_tokens"],
        "prompts": [rng.choices(vocab, k=cfg["workload"]["input_tokens"]) for _ in range(count)],
    }
    artifact["workload_id"] = digest(artifact)
    write_json(args.output, artifact)
    print(f"Saved {count} prompts: {artifact['workload_id']}")


if __name__ == "__main__":
    main()
