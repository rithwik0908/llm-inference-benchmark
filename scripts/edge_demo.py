"""Run the actual AWQ checkpoint directly via vLLM on the 8 GB RTX 5060.

FP16 Mistral weights require about 14.5 GB decimal before KV cache/runtime overhead,
so the FP16 model cannot fit fully on this GPU. This demo uses no CPU offloading.
This is an offline engine, not an HTTP service. Do not compare its timing to HTTP latency.
"""

import argparse
import time

from inference_bench.common import identity, provenance, require_cuda, write_json
from inference_bench.server import checkpoint_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/mistral-7b-awq")
    parser.add_argument("--prompt", default="Explain continuous batching in three short sentences.")
    parser.add_argument("--output", default="results/edge_demo.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu = require_cuda(6)
    manifest = checkpoint_manifest(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if len(prompt_ids) > 768:
        raise ValueError("Edge prompt exceeds the demo's short-context budget")
    # Eager mode avoids CUDA-graph allocations; 1024 context and one sequence bound KV cache.
    # The sm_120 kernel path must be verified on the actual driver; no AutoAWQ kernels used.
    llm = LLM(
        model=args.model_path,
        quantization="compressed-tensors",
        dtype="float16",
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=42,
    )
    params = SamplingParams(temperature=0, max_tokens=128, seed=42)
    llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)  # Warmup.
    start = time.perf_counter()
    result = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0].outputs[0]
    elapsed = time.perf_counter() - start
    if not result.token_ids:
        raise RuntimeError("Model returned no generated tokens")
    write_json(
        args.output,
        {
            "schema_version": 1,
            "identity": identity(),
            "variant": "awq",
            "checkpoint_id": manifest["checkpoint_id"],
            "gpu": gpu,
            "environment": provenance(),
            "prompt": args.prompt,
            "text": result.text,
            "output_tokens": len(result.token_ids),
            "latency_s": elapsed,
            "output_tokens_per_s": len(result.token_ids) / elapsed,
            "settings": {
                "max_model_len": 1024,
                "max_num_seqs": 1,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.85,
            },
            "notes": "Offline short-context smoke inference. Timing includes engine call overhead; "
            "not comparable to cloud HTTP benchmarks. FP16 cannot fit in 8 GB VRAM.",
        },
    )
    print(result.text)
    print(f"Generated {len(result.token_ids)} tokens; result: {args.output}")


if __name__ == "__main__":
    main()
