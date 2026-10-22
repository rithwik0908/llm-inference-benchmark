# Mistral inference optimization: FP16 → AWQ + vLLM

A reproducible experiment in fitting and serving **Mistral-7B-Instruct-v0.3** with
activation-aware 4-bit weights. The goal is to measure the tradeoffs in throughput,
latency, memory, and answer quality, then demonstrate inference on an 8 GB RTX 5060.

**Status: implementation and CPU validation complete; model measurements pending.**
No speedup, accuracy, or successful laptop inference is claimed without a recorded run.
The included notebook runs the remaining experiments on a Colab A100/L4.

| Experiment | Output | Measured result |
|---|---|---|
| Transformers FP16, one request | `results/baseline_fp16.json` | Pending cloud GPU run |
| AWQ W4A16_ASYM, group size 128 | `results/quantization.json` + checkpoint | Pending calibration |
| vLLM FP16/AWQ, concurrency 1/4/16/32 | `results/serving_{fp16,awq}.{json,csv}` | Pending serving sweep |
| Same 50 MMLU questions, both weights | `results/quality_delta.json` | Pending paired evaluation |
| RTX 5060 offline AWQ demo | `results/edge_demo.json` | Pending copied checkpoint |

## Run the experiment

Use Python **3.11** on Linux/WSL2. The pinned stack is vLLM 0.15.1,
llm-compressor 0.9.0, PyTorch 2.9.1, Transformers 4.57.3, and
compressed-tensors 0.13.0. One environment supports calibration and inference;
AutoAWQ is not used. Direct dependencies live in `pyproject.toml`; complete
transitive pins are in `requirements/*.lock.txt`. Pins are reproducibility choices,
not a claim that these are the latest releases. GPU kernel compatibility still needs
validation on the actual GPU/driver; do not silently change pins between variants.

**Recommended:** push this repository to your GitHub account, then open
[`notebooks/colab_workflow.ipynb`](notebooks/colab_workflow.ipynb) in Colab. Set its
repository URL and commit, select an A100 or L4, and run cells sequentially. Colab Pro
does not guarantee a particular GPU. Choose a runtime with nominal **24 GB+ VRAM and
32 GB+ host RAM**, preferably A100/high RAM. The preflight accepts 22 GiB to accommodate
nominal 24 GB cards. A free T4 is outside this project's controlled FP16 configuration.

For manual execution from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/gpu.lock.txt
pip install --no-deps -e .
python scripts/preflight.py --min-vram-gib 22
python scripts/prepare_workload.py
python scripts/baseline_fp16.py
python scripts/quantize_awq.py
python scripts/prepare_eval.py
```

Budget **30–60 minutes for calibration as an initial planning estimate**, not a measured
runtime; downloads, GPU, host RAM, and calibration settings can change this substantially.
The full serving sweep can take hours. Both model and dataset revisions are fixed in
[`configs/experiment.json`](configs/experiment.json). Quantization uses 256 UltraChat
training examples at 512 tokens; the output includes calibration-token and weight hashes.
Save `models/mistral-7b-awq/` and `results/` outside ephemeral Colab storage before disconnecting.

Start FP16 in one terminal, then run the client commands in another:

```bash
bash scripts/serve.sh fp16 2>&1 | tee results/server_fp16.log
```

```bash
# Wait for successful startup and: curl --fail http://127.0.0.1:8000/health
python scripts/evaluate_quality.py run --server-manifest results/server_fp16.json
python scripts/benchmark_serving.py --server-manifest results/server_fp16.json
```

Stop the FP16 server and wait until its GPU memory is released. Start AWQ with
`bash scripts/serve.sh awq 2>&1 | tee results/server_awq.log`, then:

```bash
python scripts/evaluate_quality.py run --server-manifest results/server_awq.json
python scripts/benchmark_serving.py --server-manifest results/server_awq.json
python scripts/evaluate_quality.py compare
python scripts/plot_results.py
```

The plotter writes **`throughput.png`, `latency.png`, `memory_quality.png`, `summary.csv`, and
`REPORT.md`** under `results/`. It refuses missing data, failed/incomplete sweeps,
different physical GPUs, different serving settings/software, or different workloads.
`results/REPORT.md` becomes the measured results writeup, including the charts and ratios.
It contains no placeholder performance numbers. Preserve old sweep files before rerunning;
the load tester refuses to overwrite them. Commit the measured artifacts and startup logs
when the experiments finish (`*.log` is ignored, so add selected sanitized logs explicitly).

## What the measurements mean

- **FP16 baseline:** ten greedy, fixed-budget requests after two warmups. GPU-synchronized
  timing measures first generated token ID, total generation rate, and decode rate.
  Peak allocated/reserved memory comes from PyTorch; loading is excluded from latency.
  No CPU or disk offloading is permitted for the measured baseline.
- **Serving comparison:** both variants use vLLM continuous batching, eager execution,
  chunked prefill, FP16 activations/KV cache, 2,048-token context, and up to 32 sequences.
  Prefix caching is disabled. This isolates the weight representation under a shared engine.
  A Transformers→vLLM comparison changes the engine as well as batching and quantization;
  it cannot attribute a speedup to one optimization alone.
- **Load:** seeded synthetic random token IDs, 256 input / exactly 128 output tokens,
  512 requests per run, three repetitions per concurrency. EOS stopping is disabled.
  Closed-loop workers dispatch at most the selected concurrency. Throughput is successful
  output tokens divided by total run wall time; p50/p99 are per-request end-to-end latency.
  TTFT is arrival of the first generated token ID in the HTTP stream. First-visible-text
  timing is recorded separately and may be null. Streamed output IDs must match the
  requested budget and final usage. This differs from the baseline's in-process token-ID
  clock. Token IDs add HTTP/JSON overhead equally to both serving variants.
  Raw measurements, failures, runtime versions, hardware, and model identity are saved.
  Generated-token throughput includes special tokens with no visible text; it is not
  useful-text throughput. Empty-text completions are counted and reported separately.
- **Quality:** zero-shot greedy answers to one frozen 50-question MMLU subset. Subjects and
  rows are shuffled reproducibly before subject round-robin selection. This is a broad
  smoke test, not the full benchmark. Strict A–D parsing counts malformed answers as wrong;
  transport errors abort. The delta is **AWQ accuracy minus FP16 accuracy in percentage
  points**, with a paired bootstrap interval. One question is 2 pp. Supply your own 30–50
  questions via `prepare_eval.py --custom-jsonl path.jsonl`; see `results/README.md`.

## Laptop and Docker

Copy the entire AWQ checkpoint directory to `models/mistral-7b-awq/` on the RTX 5060,
install the same GPU environment, then run:

```bash
python scripts/preflight.py --output results/preflight_edge.json
python scripts/edge_demo.py
```

The demo loads the checkpoint directly into an offline vLLM engine, verifies its hashes,
and uses eager execution, one sequence, and a 1,024-token context. A short prompt bounds
KV cache usage. **FP16 Mistral weights alone need roughly 14.5 GB decimal, so they cannot
fit fully in 8 GB VRAM.** Successful AWQ inference must still be verified on the actual
Blackwell GPU; if a kernel is unsupported or memory is exhausted, save the error and
adjust explicitly rather than claiming success. Do not compare laptop timing to cloud HTTP timing.

For the serving component, use NVIDIA Container Toolkit and Docker with GPU support:

```bash
docker build -f docker/Dockerfile -t mistral-benchmark:0.1 .
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v "$PWD/results:/workspace/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" mistral-benchmark:0.1 fp16
# AWQ: also mount -v "$PWD/models:/workspace/models" and replace fp16 with awq.
```

Run both variants with the same image if using Docker. Its version-tagged vLLM base owns
the CUDA stack; it is not a byte-identical lock of the host environment. Record/pin its
image digest when publishing measurements. The notebook uses processes because Colab
normally does not provide a usable Docker daemon.

## Validation and deliberate limits

CPU-only development requires no model download or GPU:

```bash
python3 -m venv .venv-client
.venv-client/bin/pip install -r requirements/client.lock.txt
.venv-client/bin/pip install --no-deps -e .
.venv-client/bin/ruff check .
.venv-client/bin/ruff format --check .
.venv-client/bin/pytest -q
```

Tests exercise protocol failures, concurrency bounds, exact token counts, quality math,
and report comparability. Synthetic test values stay in temporary directories and are
never presented as model results. CI runs these CPU checks; GPU and Docker execution
are manual integration gates. See [`results/README.md`](results/README.md) for artifact definitions.

**Cuts for the 1–3 day scope:** first-N calibration slice rather than a representative
corpus; small quality set; one fixed token budget; eager rather than CUDA graphs; fixed
concurrency order; no open-loop saturation/SLO study. Reported tail latency has sampling
uncertainty. Repetitions expose some variability but are not a hardware stability study.
Allocator peaks and vLLM's reserved KV cache are different memory measures; no misleading
cross-engine memory-reduction percentage is computed. Training an AWQ checkpoint and
running on sm_120 remain unverified until the GPU workflow finishes.

With more time: compare GPTQ, add realistic chat traces and length distributions,
randomize sweep order, benchmark CUDA graphs and speculative decoding, sweep larger
concurrency, and run full MMLU plus paired token-likelihood/perplexity evaluation.

Implementation references: [pinned AWQ example](https://github.com/vllm-project/llm-compressor/blob/0.9.0/examples/awq/llama_example.py),
[vLLM quantization documentation](https://docs.vllm.ai/en/v0.15.1/features/quantization/),
[Mistral model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3),
[MMLU dataset](https://huggingface.co/datasets/cais/mmlu).
Respect the upstream model/data licenses; do not commit model weights or credentials.
