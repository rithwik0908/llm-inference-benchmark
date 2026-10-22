"""OpenAI completion protocol; explicit stream usage is required for token accounting."""

import json
import time

from inference_bench.common import identity


async def validate_server(session, base_url, manifest):
    if manifest["identity"] != identity():
        raise ValueError("Server model revision differs from the experiment")
    async with session.get(base_url + "/v1/models") as response:
        response.raise_for_status()
        models = await response.json()
    if manifest["served_model_name"] not in [m["id"] for m in models["data"]]:
        raise ValueError("Server manifest is stale or points to another endpoint")


async def sse_events(content):
    """aiohttp's line iterator handles arbitrary TCP splits; SSE may span data lines."""
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


async def stream_request(session, base_url, model, prompt_ids, output_tokens, request_id):
    start = time.perf_counter()
    first = None
    usage = None
    done = False
    finish = None
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "temperature": 0,
        "seed": 42,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with session.post(base_url + "/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
            async for event in sse_events(response.content):
                if event == "[DONE]":
                    done = True
                    break
                data = json.loads(event)
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("text") and first is None:
                        first = time.perf_counter()
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        if not done or first is None or not usage or not finish:
            raise RuntimeError("Incomplete stream: need DONE, text, finish_reason, and usage")
        if usage["completion_tokens"] != output_tokens:
            raise RuntimeError("Output token count differs from fixed workload")
        if usage["prompt_tokens"] != len(prompt_ids):
            raise RuntimeError("Server prompt count differs from frozen input token IDs")
        return {
            "id": request_id,
            "success": True,
            "output_tokens": usage["completion_tokens"],
            "input_tokens": usage["prompt_tokens"],
            "latency_s": time.perf_counter() - start,
            "ttft_s": first - start,
            "finish_reason": finish,
        }
    except Exception as exc:
        return {
            "id": request_id,
            "success": False,
            "output_tokens": 0,
            "latency_s": time.perf_counter() - start,
            "ttft_s": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
