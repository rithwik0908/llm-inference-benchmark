"""Closed-loop bounded-concurrency HTTP load test with raw request timing and token usage."""

import argparse
import asyncio
import csv
import time
from pathlib import Path

import aiohttp

from inference_bench.client import stream_request, validate_server
from inference_bench.common import (
    identity,
    positive,
    provenance,
    read_json,
    write_json,
)
from inference_bench.metrics import summarize


async def run_pool(session, base_url, server, workload, concurrency):
    iterator = iter(enumerate(workload["prompts"]))
    rows = []

    async def worker():
        for index, prompt in iterator:
            rows.append(
                await stream_request(
                    session,
                    base_url,
                    server["served_model_name"],
                    prompt,
                    workload["output_tokens"],
                    index,
                )
            )

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - start
    return sorted(rows, key=lambda r: r["id"]), elapsed


async def benchmark(args):
    workload = read_json(args.workload)
    server = read_json(args.server_manifest)
    if workload["identity"] != identity():
        raise ValueError("Workload identity differs from configured model")
    if len(workload["prompts"]) < max(args.concurrency):
        raise ValueError("Workload must contain at least as many prompts as maximum concurrency")
    output = Path(args.output or f"results/serving_{server['variant']}.json")
    if output.exists():
        raise FileExistsError(f"Preserve the previous run or choose another --output: {output}")
    report = {
        "schema_version": 1,
        "variant": server["variant"],
        "identity": identity(),
        "server": server,
        "environment": provenance(),
        "workload_id": workload["workload_id"],
        "input_tokens": workload["input_tokens"],
        "output_tokens": workload["output_tokens"],
        "requests_per_run": len(workload["prompts"]),
        "repeats": args.repeats,
        "concurrency_levels": args.concurrency,
        "runs": [],
        "valid": False,
        "methodology": "Closed loop, no think time, one worker per concurrent request. "
        "Latency begins at dispatch (no artificial client queue). TTFT is first "
        "nonempty streamed text chunk. Warmup excluded. Throughput counts successful "
        "completion tokens / full run wall time, including failed requests. "
        "A failed request invalidates the experiment; there are no retries.",
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await validate_server(session, args.base_url, server)
        for concurrency in args.concurrency:
            # Warm the same number of active sequences before each concurrency level.
            warm = {**workload, "prompts": workload["prompts"][:concurrency]}
            warm_rows, _ = await run_pool(session, args.base_url, server, warm, concurrency)
            if not all(r["success"] for r in warm_rows):
                report["warmup_errors"] = [r for r in warm_rows if not r["success"]]
                write_json(output, report)
                raise RuntimeError(f"Warmup failed; diagnostic saved to {output}")
            for repeat in range(args.repeats):
                rows, elapsed = await run_pool(
                    session, args.base_url, server, workload, concurrency
                )
                summary = summarize(rows, elapsed)
                run = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "summary": summary,
                    "rows": rows,
                }
                report["runs"].append(run)
                write_json(output, report)  # Preserve completed work if Colab disconnects.
                print(f"c={concurrency}, repeat={repeat}: {summary}", flush=True)
                if summary["failed_requests"]:
                    raise RuntimeError(f"Requests failed; experiment invalid. Inspect {output}")
    report["valid"] = True
    write_json(output, report)
    csv_rows = [
        {
            "variant": server["variant"],
            "concurrency": run["concurrency"],
            "repeat": run["repeat"],
            **run["summary"],
        }
        for run in report["runs"]
    ]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {output} and {output.with_suffix('.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-manifest", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", default="results/workload.json")
    parser.add_argument("--concurrency", nargs="+", type=positive, default=[1, 4, 16, 32])
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--timeout", type=positive, default=600)
    parser.add_argument("--output")
    args = parser.parse_args()
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("Concurrency levels must be unique")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
