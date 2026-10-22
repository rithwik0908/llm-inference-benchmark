# Results pending measurement

The benchmark implementation is ready for GPU integration runs. This file will be
replaced by `python scripts/plot_results.py` after both serving sweeps and the paired
quality evaluation complete successfully.

| Measurement | FP16 | AWQ |
|---|---|---|
| Single-request generation | Pending | Pending |
| Throughput at concurrency 1 / 4 / 16 / 32 | Pending | Pending |
| End-to-end latency p50 / p99 | Pending | Pending |
| Same-set quality | Pending | Pending |
| RTX 5060 inference | Cannot fit full FP16 weights in 8 GB | Pending verification |

No benchmark numbers or generated charts are included without measured source artifacts.
Required resources: Colab A100/L4 with 24 GB+ nominal VRAM and 32 GB+ host RAM, network
access for the pinned Mistral and dataset revisions, and the resulting AWQ checkpoint
copied to the local RTX 5060 for the final smoke test.
