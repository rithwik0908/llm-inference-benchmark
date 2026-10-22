"""Generate three charts and a measured writeup; refuses missing or incomparable data."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference_bench.common import read_json
from inference_bench.report import serving_table, validate_pair


def render(fp16, awq, quality, target, baseline, quantization):
    validate_pair(fp16, awq)
    if fp16["identity"] != quality["identity"]:
        raise ValueError("Quality results refer to a different model revision")
    for artifact in (fp16, awq):
        if quality[f"{artifact['variant']}_checkpoint_id"] != artifact["server"]["checkpoint_id"]:
            raise ValueError("Quality and serving runs used different checkpoints")
    if baseline["identity"] != fp16["identity"] or quantization["identity"] != fp16["identity"]:
        raise ValueError("Storage/baseline artifacts use a different model revision")
    if baseline["gpu"]["uuid"] != fp16["server"]["gpu"]["uuid"]:
        raise ValueError("Baseline and serving runs used different physical GPUs")
    if quantization["checkpoint_id"] != awq["server"]["checkpoint_id"]:
        raise ValueError("Quantization and serving checkpoints differ")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"fp16": "#4c78a8", "awq": "#f58518"}
    table = serving_table([fp16, awq])
    with (target / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, ax = plt.subplots(figsize=(7, 4))
    for artifact in (fp16, awq):
        levels = artifact["concurrency_levels"]
        values = [
            [r["summary"]["output_tokens_per_s"] for r in artifact["runs"] if r["concurrency"] == c]
            for c in levels
        ]
        medians = [median(v) for v in values]
        errors = [
            [m - min(v) for m, v in zip(medians, values, strict=True)],
            [max(v) - m for m, v in zip(medians, values, strict=True)],
        ]
        ax.errorbar(
            levels,
            medians,
            yerr=errors,
            marker="o",
            capsize=4,
            label=artifact["variant"].upper(),
            color=colors[artifact["variant"]],
        )
    ax.set(
        xlabel="Concurrent requests (closed loop)",
        ylabel="Output tokens / second",
        title="Serving throughput: median and range across repeated runs",
    )
    ax.set_xticks(fp16["concurrency_levels"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / "throughput.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, metric, title in zip(
        axes, ("latency_s", "ttft_s"), ("End-to-end latency", "TTFT"), strict=True
    ):
        for artifact in (fp16, awq):
            selected = [r for r in table if r["variant"] == artifact["variant"]]
            for quantile, style in (("p50", "-"), ("p99", "--")):
                ax.plot(
                    [r["concurrency"] for r in selected],
                    [r[f"{metric}_{quantile}"] for r in selected],
                    linestyle=style,
                    marker="o",
                    color=colors[artifact["variant"]],
                    label=f"{artifact['variant'].upper()} {quantile}",
                )
        ax.set(xlabel="Concurrent requests", ylabel="Seconds", title=title)
        ax.set_xticks(fp16["concurrency_levels"])
        ax.legend(fontsize=8)
    fig.suptitle("Median of per-run percentiles; tail estimates are sample-dependent")
    fig.tight_layout()
    fig.savefig(target / "latency.png")
    plt.close(fig)

    fig, (memory_ax, ax) = plt.subplots(1, 2, figsize=(11, 4))
    source_bytes = baseline["source_checkpoint_safetensors_bytes"]
    quantized_bytes = sum(w["bytes"] for w in quantization["weights"].values())
    memory_ax.bar(
        ["Original 16-bit", "AWQ 4-bit"],
        [source_bytes / 1e9, quantized_bytes / 1e9],
        color=list(colors.values()),
    )
    memory_ax.set(
        ylabel="Safetensors checkpoint size (GB, decimal)", title="Weight storage footprint"
    )
    memory_ax.text(
        0.5,
        0.03,
        "Measured file bytes; excludes KV cache/runtime VRAM",
        transform=memory_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.bar(
        ["FP16", "AWQ"],
        [quality["fp16_accuracy_pct"], quality["awq_accuracy_pct"]],
        color=list(colors.values()),
    )
    delta = quality["accuracy_delta_pp_awq_minus_fp16"]
    low, high = quality["paired_bootstrap_95ci_pp"]
    ax.set(
        ylim=(0, 100),
        ylabel="Accuracy (%)",
        title=f"{quality['questions']} questions: AWQ − FP16 = {delta:+.1f} pp",
    )
    ax.text(
        0.5,
        0.03,
        f"Paired bootstrap delta 95% CI: [{low:+.1f}, {high:+.1f}] pp\n"
        f"Resolution: {quality['resolution_pp']:.1f} pp per question; not full MMLU",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(target / "memory_quality.png")
    plt.close(fig)

    text = [
        "# Measured results",
        "",
        f"Model: `{fp16['identity']['model']}` at `{fp16['identity']['revision']}`.",
        "",
        f"GPU: {fp16['server']['gpu']['name']}. Workload: {fp16['input_tokens']} input / "
        f"{fp16['output_tokens']} output tokens, {fp16['requests_per_run']} requests per run, "
        f"{fp16['repeats']} repetitions. Seeded synthetic random-token inputs; no prompt cache.",
        "",
        "Both weight variants use vLLM continuous batching. These ratios isolate the weight "
        "representation under this configuration; they do not isolate "
        "continuous batching's benefit.",
        "",
        "| Variant | Concurrency | Output tok/s | E2E p50 (s) | E2E p99 (s) | TTFT p50 (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        text.append(
            f"| {row['variant'].upper()} | {row['concurrency']} | "
            f"{row['output_tokens_per_s']:.2f} | {row['latency_s_p50']:.3f} | "
            f"{row['latency_s_p99']:.3f} | {row['ttft_s_p50']:.3f} |"
        )
    text += ["", "Values are medians across repeated runs; p99 is computed within each run.", ""]
    for concurrency in fp16["concurrency_levels"]:
        a = next(r for r in table if r["variant"] == "fp16" and r["concurrency"] == concurrency)
        b = next(r for r in table if r["variant"] == "awq" and r["concurrency"] == concurrency)
        text.append(
            f"- Concurrency {concurrency}: AWQ/FP16 throughput ratio "
            f"{b['output_tokens_per_s'] / a['output_tokens_per_s']:.2f}×."
        )
    text += [
        "",
        "Transformers FP16 single-request baseline (separate execution path): "
        f"{baseline['summary']['output_tokens_per_s_p50']:.2f} output tok/s, "
        f"TTFT {baseline['summary']['ttft_s_p50']:.3f} s; "
        f"peak allocated VRAM {baseline['peak_allocated_bytes'] / 1024**3:.2f} GiB.",
        "",
        f"Weight-file storage: original 16-bit {source_bytes / 1e9:.2f} GB; "
        f"AWQ {quantized_bytes / 1e9:.2f} GB. This is not runtime VRAM usage.",
        "",
        f"Quality: FP16 {quality['fp16_accuracy_pct']:.1f}%, "
        f"AWQ {quality['awq_accuracy_pct']:.1f}%; delta {delta:+.1f} percentage points "
        f"(paired bootstrap 95% CI [{low:+.1f}, {high:+.1f}]).",
        "",
        quality["limitations"],
        "",
        "![Throughput](throughput.png)",
        "![Latency](latency.png)",
        "![Storage and quality](memory_quality.png)",
        "",
        "Interpretation is limited to this GPU, software stack, token budget, closed-loop "
        "load, and question subset. Review raw JSON, errors, and startup logs before making "
        "performance claims. The separate Transformers baseline uses a different execution "
        "path and TTFT definition. Laptop smoke inference is not a cloud comparison.",
        "",
    ]
    (target / "REPORT.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", default="results/serving_fp16.json")
    parser.add_argument("--awq", default="results/serving_awq.json")
    parser.add_argument("--quality", default="results/quality_delta.json")
    parser.add_argument("--baseline", default="results/baseline_fp16.json")
    parser.add_argument("--quantization", default="results/quantization.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    for path in (args.fp16, args.awq, args.quality, args.baseline, args.quantization):
        if not Path(path).is_file():
            parser.error(f"No measured artifact at {path}. Run the GPU workflow first.")
    render(
        read_json(args.fp16),
        read_json(args.awq),
        read_json(args.quality),
        args.output_dir,
        read_json(args.baseline),
        read_json(args.quantization),
    )
    print(f"Wrote three charts, summary.csv, and REPORT.md in {args.output_dir}")


if __name__ == "__main__":
    main()
