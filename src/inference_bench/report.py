"""Reject incomparable runs before drawing a before/after chart."""

from statistics import median

from inference_bench.client import MEASUREMENT_PROTOCOL


def validate_pair(fp16, awq):
    for artifact in (fp16, awq):
        if artifact.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
            raise ValueError(
                "Serving results use an older/different TTFT protocol; rerun both sweeps"
            )
    if (fp16["variant"], awq["variant"]) != ("fp16", "awq"):
        raise ValueError("Need FP16 and AWQ serving artifacts in that order")
    keys = (
        "identity",
        "workload_id",
        "input_tokens",
        "output_tokens",
        "requests_per_run",
        "repeats",
        "concurrency_levels",
    )
    for key in keys:
        if fp16[key] != awq[key]:
            raise ValueError(f"Serving runs differ in {key}; do not compare them")
    for key in ("gpu", "serving_settings"):
        a, b = fp16["server"][key], awq["server"][key]
        if key == "gpu":
            # Free memory can differ after loading; physical GPU and total capacity must match.
            a = {k: a[k] for k in ("name", "total_bytes", "capability", "uuid")}
            b = {k: b[k] for k in ("name", "total_bytes", "capability", "uuid")}
        if a != b:
            raise ValueError(f"Serving {key} differs; rerun both variants on the same GPU/settings")
    for package in ("torch", "vllm", "transformers", "compressed-tensors"):
        a = fp16["server"]["environment"]["packages"].get(package)
        b = awq["server"]["environment"]["packages"].get(package)
        if not a or a != b:
            raise ValueError(f"Server package version differs or is missing: {package}")
    for artifact in (fp16, awq):
        expected = {
            (c, r) for c in artifact["concurrency_levels"] for r in range(artifact["repeats"])
        }
        actual = {(r["concurrency"], r["repeat"]) for r in artifact["runs"]}
        if not artifact["valid"] or actual != expected or len(artifact["runs"]) != len(expected):
            raise ValueError("Serving sweep is invalid or incomplete")
        for run in artifact["runs"]:
            if run["summary"]["failed_requests"]:
                raise ValueError("A serving sweep contains request failures")


def serving_table(artifacts):
    rows = []
    for artifact in artifacts:
        for concurrency in artifact["concurrency_levels"]:
            runs = [r["summary"] for r in artifact["runs"] if r["concurrency"] == concurrency]
            rows.append(
                {
                    "variant": artifact["variant"],
                    "concurrency": concurrency,
                    **{
                        key: median(r[key] for r in runs)
                        for key in (
                            "output_tokens_per_s",
                            "latency_s_p50",
                            "latency_s_p99",
                            "ttft_s_p50",
                            "ttft_s_p99",
                        )
                    },
                }
            )
    return rows
