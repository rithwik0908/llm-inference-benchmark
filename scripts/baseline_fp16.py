"""Single-request greedy HF baseline, GPU only; no CPU/disk offloading."""

import argparse
import time

from inference_bench.common import (
    config,
    identity,
    positive,
    provenance,
    read_json,
    require_cuda,
    write_json,
)
from inference_bench.metrics import percentile


def main():
    import torch
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="results/workload.json")
    parser.add_argument("--runs", type=positive, default=10)
    parser.add_argument("--warmup", type=positive, default=2)
    parser.add_argument("--output", default="results/baseline_fp16.json")
    args = parser.parse_args()
    cfg = config()
    gpu = require_cuda(22)  # Nominal 24 GB L4 cards expose slightly less than 24 GiB.
    workload = read_json(args.workload)
    if workload["identity"] != identity():
        raise ValueError("Workload was prepared for a different model revision")
    torch.manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        revision=cfg["revision"],
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    load_peak = torch.cuda.max_memory_allocated()
    inputs = torch.tensor([workload["prompts"][0]], device="cuda")
    mask = torch.ones_like(inputs)
    tokens = workload["output_tokens"]

    class FirstTokenClock:
        # HF puts the prompt first. Subsequent puts represent generated token IDs.
        # Only first-token synchronization is inserted, avoiding per-token sync overhead.
        def __init__(self):
            self.prompt_seen = False
            self.first = None

        def put(self, value):
            if not self.prompt_seen:
                self.prompt_seen = True
            elif self.first is None:
                torch.cuda.synchronize()
                self.first = time.perf_counter()

        def end(self):
            pass

    def generate():
        clock = FirstTokenClock()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=mask,
                do_sample=False,
                min_new_tokens=tokens,
                max_new_tokens=tokens,
                use_cache=True,
                pad_token_id=model.generation_config.eos_token_id,
                streamer=clock,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        generated = output.shape[1] - inputs.shape[1]
        if generated != tokens or clock.first is None:
            raise RuntimeError("Generation did not produce the requested token budget")
        return {
            "input_tokens": inputs.shape[1],
            "output_tokens": generated,
            "ttft_s": clock.first - start,
            "latency_s": end - start,
            "output_tokens_per_s": generated / (end - start),
            "decode_tokens_per_s": (generated - 1) / (end - clock.first) if generated > 1 else None,
        }

    for _ in range(args.warmup):
        generate()
    torch.cuda.reset_peak_memory_stats()
    rows = [generate() for _ in range(args.runs)]
    result = {
        "schema_version": 1,
        "variant": "fp16",
        "backend": "transformers_sdpa",
        "identity": identity(),
        "environment": provenance(),
        "gpu": gpu,
        "workload_id": workload["workload_id"],
        "prompt_index": 0,
        "warmup_requests": args.warmup,
        "rows": rows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "load_peak_allocated_bytes": load_peak,
        "summary": {
            key + "_p50": percentile([r[key] for r in rows], 0.5)
            for key in ("ttft_s", "latency_s", "output_tokens_per_s")
        },
        "measurement_notes": "TTFT is host-visible generated token ID, not HTTP/text TTFT. "
        "Generation includes streamer token CPU transfers. Memory is PyTorch "
        "allocator memory, not total device/NVML usage. Load excluded from timing.",
    }
    write_json(args.output, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
