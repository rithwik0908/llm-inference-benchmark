# Artifacts and measurement status

**Real measurements are committed for the A100 FP16 baseline, AWQ quantization, paired
50-question MMLU evaluation, both complete serving sweeps, and the RTX 5060 laptop demo.**
See [REPORT.md](REPORT.md) for the generated cloud comparison and the repository README
for interpretation and laptop results. The plotter refuses invalid or incomparable sweeps.
Tests use synthetic values only under pytest temporary directories.

| Artifact | Contents |
|---|---|
| `preflight*.json` | Runtime packages, git commit, GPU and driver, CUDA readiness/error |
| `workload.json` | Seeded input token IDs, budgets, model revision, content identity |
| `eval_set.json` | Frozen questions, answer keys, chat-template token IDs, dataset revision |
| `baseline_fp16.json` | Ten HF measurements, warmups, allocator memory, TTFT/rates in seconds/tokens |
| `quantization.json` | Calibration/recipe, elapsed time, model/weight hashes and quantization config |
| `server_{variant}.json` | Unique server alias, launch command, checkpoint ID, GPU/software |
| `serving_{variant}.json` | Sweep metadata and per-request raw measurements, errors and validity flag |
| `serving_{variant}.csv` | One summary row per concurrency/repetition |
| `quality_{variant}.json` | Each generated answer, parsing outcome, usage, and correctness |
| `quality_delta.json` | Paired accuracy delta in pp, changed answers, bootstrap interval |
| `edge_demo.json` | Actual laptop generation, offline timing and settings, checkpoint identity |
| `edge_vram.csv`, `edge_vram_summary.json` | 200 ms sampled device-wide memory usage and peak; includes other processes |
| `edge_demo.log` | Actual laptop startup, selected kernels, warnings, and generated output |
| `summary.csv`, `REPORT.md`, `*.png` | Derived report and three charts; generated only from valid data |

Keep both cloud variants on the **same physical GPU in the same Colab session**. Compare
three repetitions by medians; error bars on throughput show the min/max repeated-run range,
not a confidence interval. Latency charts show median per-run p50/p99 E2E and TTFT, not a
pooled percentile and not the full latency distribution. Every run has 512 requests by
default; interpret p99 cautiously. HTTP time includes loopback/network/client costs.
Client-side queueing before dispatch is excluded because this is a closed-loop workload,
not a trace replay or an arrival-rate test. Requests timing out or producing an incomplete
stream invalidate the sweep; diagnostics are retained with `valid: false`.

The baseline uses the first frozen synthetic prompt, 256 input / 128 output tokens.
It reports generated tokens / total request time and (N−1) / time after the first token.
The first token is synchronized and generation's streamer transfers token IDs to CPU.
These instrumentation costs are part of the measured baseline. Peak GPU memory is the
PyTorch allocator's occupied/reserved memory, including resident weights; it excludes
some CUDA context/driver allocations and is not a device-wide NVML peak. Quantization
allocator memory is a preparation cost, not serving memory. vLLM reserves KV cache space;
its utilization target is a budget, not a measured memory reduction.

Custom evaluation JSONL uses **30–50 distinct rows**, for example:

```json
{"question":"What is 2 + 2?","choices":["3","4","5","6"],"answer":"B"}
```

Call `python scripts/prepare_eval.py --custom-jsonl your_set.jsonl` once, then use that exact
artifact for both runs. The same question IDs, answer keys, generation settings, and set
identity must match. Generation is zero-shot with four output tokens and deterministic
sampling. Invalid model answers count as incorrect. HTTP failures abort rather than
masquerading as quality loss. No calibration examples come from this evaluation set.

Models live in ignored `models/`. Commit the small measurement files and charts after the
actual run; preserve stdout/stderr startup logs separately. Do not label estimated or
synthetic numbers as measured results in an interview or README.

Included setup artifacts were prepared against the real pinned tokenizer and MMLU dataset.
`preflight_local.json` describes the lightweight client environment: CUDA readiness is
false because CUDA PyTorch is not installed there. Its `nvidia_smi` field records the
actual RTX 5060 hardware; this is not a GPU inference failure or a performance measurement.


Serving measurement protocol version 2 requests vLLM's `return_token_ids` extension.
`ttft_s` is the arrival of the first **generated** token ID; echoed prompt IDs and empty
prefill chunks do not count. `first_visible_text_s` is separately recorded and is null
when no printable text arrives. Every successful response must contain `[DONE]`, a finish
reason, final usage, and exactly the requested number of streamed generated IDs. Empty
text alone does not mean the transport failed. `requests_without_visible_text` makes
this behavior explicit. Synthetic random-token inputs can produce such completions;
token throughput counts generated IDs, not useful text. Extra ID serialization/transport
cost is included for both variants. Charts refuse artifacts from different/older protocols.

The earlier text-only validator rejected complete responses when all decoded text chunks
were empty. Preserve those failed sweep/probe files as diagnostics; never flip their
`valid` flag, backfill their missing TTFT, or reuse them as final measured sweeps. Re-run
both serving variants under protocol 2 after a targeted replay succeeds. The model,
precision, frozen prompts, token budget, and quality evaluation are unchanged. The
pinned API defines delta IDs in each streaming choice:
[vLLM 0.15.1 completion protocol](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/entrypoints/openai/completion/protocol.py).
