"""Units are seconds and output tokens unless a field explicitly says otherwise."""

import math


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Need nonempty values and quantile in [0, 1]")
    data = sorted(values)
    position = (len(data) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def summarize(rows, elapsed_s):
    if not rows or elapsed_s <= 0:
        raise ValueError("Need requests and positive elapsed time")
    good = [r for r in rows if r["success"]]
    result = {
        "requests": len(rows),
        "successful_requests": len(good),
        "failed_requests": len(rows) - len(good),
        "elapsed_s": elapsed_s,
        "requests_without_visible_text": sum(r.get("visible_text_chunks") == 0 for r in good),
        "output_tokens": sum(r["output_tokens"] for r in good),
    }
    result["output_tokens_per_s"] = result["output_tokens"] / elapsed_s
    for key in ("latency_s", "ttft_s"):
        values = [r[key] for r in good]
        for name, q in (("p50", 0.5), ("p99", 0.99)):
            result[f"{key}_{name}"] = percentile(values, q) if values else None
    return result
