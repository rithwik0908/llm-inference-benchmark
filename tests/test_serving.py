"""Synthetic HTTP protocol tests; these produce no portfolio benchmark artifacts."""

import argparse
import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from inference_bench.client import stream_request
from inference_bench.common import identity, read_json, write_json
from scripts.benchmark_serving import benchmark, run_pool


async def fake_server(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)

    async def models(request):
        return web.json_response({"data": [{"id": "test"}]})

    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_stream_token_accounting_and_bounded_concurrency():
    async def scenario():
        active = 0
        peak = 0

        async def handler(request):
            nonlocal active, peak
            payload = await request.json()
            active += 1
            peak = max(peak, active)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b': heartbeat\r\ndata: {"choices":')
            await asyncio.sleep(0.005)
            await response.write(b' [{"text": "hi", "finish_reason": null}]}\r\n\r\n')
            usage = {
                "prompt_tokens": len(payload["prompt"]),
                "completion_tokens": payload["max_tokens"],
            }
            event = {"choices": [{"text": "", "finish_reason": "length"}], "usage": usage}
            await response.write(f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())
            await response.write_eof()
            active -= 1
            return response

        runner, url = await fake_server(handler)
        try:
            async with aiohttp.ClientSession() as session:
                rows, elapsed = await run_pool(
                    session,
                    url,
                    {"served_model_name": "test"},
                    {"prompts": [[1, 2]] * 12, "output_tokens": 3},
                    4,
                )
            assert len(rows) == 12
            assert all(r["success"] and r["output_tokens"] == 3 for r in rows)
            assert 1 < peak <= 4
            assert all(0 < r["ttft_s"] <= r["latency_s"] <= elapsed for r in rows)
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["http", "missing_usage", "wrong_count", "missing_done"])
def test_bad_stream_is_recorded_as_failure(failure):
    async def scenario():
        async def handler(request):
            if failure == "http":
                return web.Response(status=503, text="test-only overload")
            usage = {"prompt_tokens": 2, "completion_tokens": 7 if failure == "wrong_count" else 3}
            event = {"choices": [{"text": "hi", "finish_reason": "length"}]}
            if failure != "missing_usage":
                event["usage"] = usage
            text = f"data: {json.dumps(event)}\n\n"
            if failure != "missing_done":
                text += "data: [DONE]\n\n"
            return web.Response(text=text, content_type="text/event-stream")

        runner, url = await fake_server(handler)
        try:
            async with aiohttp.ClientSession() as session:
                result = await stream_request(session, url, "test", [1, 2], 3, 0)
            assert result["success"] is False
            assert result["output_tokens"] == 0
            assert result["error"]
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_load_sweep_persists_valid_json_csv(tmp_path):
    async def scenario():
        async def handler(request):
            payload = await request.json()
            event = {
                "choices": [{"text": "test", "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": len(payload["prompt"]),
                    "completion_tokens": payload["max_tokens"],
                },
            }
            return web.Response(
                text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )

        runner, url = await fake_server(handler)
        server = tmp_path / "server.json"
        workload = tmp_path / "workload.json"
        output = tmp_path / "output.json"
        write_json(server, {"identity": identity(), "variant": "fp16", "served_model_name": "test"})
        write_json(
            workload,
            {
                "identity": identity(),
                "workload_id": "test-only",
                "prompts": [[1, 2]] * 8,
                "input_tokens": 2,
                "output_tokens": 3,
            },
        )
        args = argparse.Namespace(
            server_manifest=server,
            workload=workload,
            output=output,
            base_url=url,
            concurrency=[1, 4],
            repeats=2,
            timeout=10,
        )
        try:
            await benchmark(args)
            result = read_json(output)
            assert result["valid"]
            assert len(result["runs"]) == 4
            assert all(len(r["rows"]) == 8 for r in result["runs"])
            assert output.with_suffix(".csv").exists()
            with pytest.raises(FileExistsError):
                await benchmark(args)
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
