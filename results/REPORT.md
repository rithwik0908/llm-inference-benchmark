# Measured results

Model: `mistralai/Mistral-7B-Instruct-v0.3` at `c170c708c41dac9275d15a8fff4eca08d52bab71`.

GPU: NVIDIA A100-SXM4-40GB. Workload: 256 input / 128 output tokens, 512 requests per run, 3 repetitions. Seeded synthetic random-token inputs; no prompt cache.

Both weight variants use vLLM continuous batching. These ratios isolate the weight representation under this configuration; they do not isolate continuous batching's benefit.

| Variant | Concurrency | Output tok/s | E2E p50 (s) | E2E p99 (s) | TTFT p50 (s) |
|---|---:|---:|---:|---:|---:|
| FP16 | 1 | 42.61 | 3.005 | 3.056 | 0.050 |
| FP16 | 4 | 167.59 | 3.054 | 3.133 | 0.089 |
| FP16 | 16 | 618.79 | 3.308 | 3.389 | 0.206 |
| FP16 | 32 | 1109.85 | 3.684 | 3.854 | 0.306 |
| AWQ | 1 | 43.77 | 2.924 | 2.990 | 0.048 |
| AWQ | 4 | 167.48 | 3.054 | 3.183 | 0.114 |
| AWQ | 16 | 609.73 | 3.360 | 3.432 | 0.269 |
| AWQ | 32 | 1096.96 | 3.730 | 3.968 | 0.363 |

Values are medians across repeated runs; p99 is computed within each run.

TTFT is arrival of the first streamed generated token ID; first-visible-text timing is separately recorded and nullable. Throughput counts all generated IDs, including special tokens without visible text.

- FP16: 24 of 6144 successful requests across all trials produced no visible text.
- AWQ: 24 of 6144 successful requests across all trials produced no visible text.

- Concurrency 1: AWQ/FP16 throughput ratio 1.03×.
- Concurrency 4: AWQ/FP16 throughput ratio 1.00×.
- Concurrency 16: AWQ/FP16 throughput ratio 0.99×.
- Concurrency 32: AWQ/FP16 throughput ratio 0.99×.

Transformers FP16 single-request baseline (separate execution path): 29.11 output tok/s, TTFT 0.038 s; peak allocated VRAM 13.57 GiB.

Weight-file storage: original 16-bit 14.50 GB; AWQ 4.16 GB. This is not runtime VRAM usage.

Quality: FP16 68.0%, AWQ 66.0%; delta -2.0 percentage points (paired bootstrap 95% CI [-6.0, +0.0]).

Small zero-shot smoke evaluation, not full MMLU. CI resamples questions as independent and does not establish quality equivalence. Invalid answers count as wrong; transport errors abort evaluation.

![Throughput](throughput.png)
![Latency](latency.png)
![Storage and quality](memory_quality.png)

Interpretation is limited to this GPU, software stack, token budget, closed-loop load, and question subset. Review raw JSON, errors, and startup logs before making performance claims. The separate Transformers baseline uses a different execution path and TTFT definition. Laptop smoke inference is not a cloud comparison.
