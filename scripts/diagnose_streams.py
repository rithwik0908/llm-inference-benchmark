"""Replay selected frozen requests and preserve raw SSE events, without running a sweep.

Diagnostics are not replacement benchmark measurements. The original failed result must
remain invalid; this tool neither changes the workload nor relaxes stream validation.
"""

import argparse
import asyncio
from pathlib import Path

import aiohttp

from inference_bench.client import stream_request, validate_server
from inference_bench.common import provenance, read_frozen, read_json, write_json


async def diagnose(args):
    workload = read_frozen(args.workload, "workload_id")
    server = read_json(args.server_manifest)
    if workload["identity"] != server["identity"]:
        raise ValueError("Workload and server model identities differ")
    if any(i < 0 or i >= len(workload["prompts"]) for i in args.ids):
        raise ValueError("A request ID is outside the frozen workload")
    if Path(args.output).exists():
        raise FileExistsError("Preserve prior diagnostics or choose another --output")
    report = {
        "schema_version": 1,
        "kind": "diagnostic_replay_not_benchmark",
        "server": server,
        "environment": provenance(),
        "workload_id": workload["workload_id"],
        "rows": [],
        "notes": "Targeted replay uses unchanged benchmark parameters. Original "
        "failed artifacts remain invalid. Full raw SSE events are preserved.",
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        await validate_server(session, args.base_url, server)
        for request_id in args.ids:
            row = await stream_request(
                session,
                args.base_url,
                server["served_model_name"],
                workload["prompts"][request_id],
                workload["output_tokens"],
                request_id,
                capture_events=True,
            )
            report["rows"].append(row)
            write_json(args.output, report)
            print({key: value for key, value in row.items() if key != "events"}, flush=True)
            print("Last three raw events:", row["events"][-3:], flush=True)
    print(f"Saved full raw diagnostics to {args.output}; no sweep was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-manifest", required=True)
    parser.add_argument("--workload", default="results/workload.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ids", nargs="+", type=int, default=[244, 282])
    parser.add_argument("--output", default="results/fp16_stream_diagnostics.json")
    asyncio.run(diagnose(parser.parse_args()))


if __name__ == "__main__":
    main()
