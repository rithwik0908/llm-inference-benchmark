"""OpenAI completion protocol; explicit stream usage is required for token accounting."""

import json
import time

from inference_bench.common import identity

# vLLM 0.15.1 returns delta IDs in each choice when return_token_ids=True:
# vllm/entrypoints/openai/completion/protocol.py and completion/serving.py.
MEASUREMENT_PROTOCOL = {
    "version": 2,
    "ttft": "first_streamed_generated_token_id",
    "return_token_ids": True,
    "accounting": "streamed_token_ids_and_final_usage_must_match",
    "visible_text_timing": "separate_nullable_field",
}


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


async def stream_request(
    session, base_url, model, prompt_ids, output_tokens, request_id, *, capture_events=False
):
    start = time.perf_counter()
    first = None
    first_text = None
    streamed_tokens = 0
    usage = None
    done = False
    finish = None
    event_count = 0
    nonempty_chunks = 0
    events = [] if capture_events else None
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "temperature": 0,
        "seed": 42,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "n": 1,
        "echo": False,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with session.post(base_url + "/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
            async for event in sse_events(response.content):
                received_at = time.perf_counter()
                event_count += 1
                if events is not None:
                    events.append(event)
                if event == "[DONE]":
                    done = True
                    break
                data = json.loads(event)
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise RuntimeError("Unexpected extra completion choice")
                    token_ids = choice.get("token_ids")
                    if token_ids is not None:
                        if not isinstance(token_ids, list) or any(
                            type(token) is not int or token < 0 for token in token_ids
                        ):
                            raise RuntimeError("Invalid streamed token_ids")
                        if token_ids:
                            if first is None:
                                first = received_at
                            streamed_tokens += len(token_ids)
                    if choice.get("text"):
                        nonempty_chunks += 1
                        if first_text is None:
                            first_text = received_at
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        if not done or first is None or not usage or not finish:
            missing = [
                name
                for name, observed in (
                    ("DONE", done),
                    ("generated_token_ids", first is not None),
                    ("usage", bool(usage)),
                    ("finish_reason", bool(finish)),
                )
                if not observed
            ]
            raise RuntimeError("Incomplete stream: missing " + ", ".join(missing))
        if usage["completion_tokens"] != output_tokens:
            raise RuntimeError("Output token count differs from fixed workload")
        if streamed_tokens != usage["completion_tokens"]:
            raise RuntimeError("Streamed token ID count differs from final usage")
        if usage["prompt_tokens"] != len(prompt_ids):
            raise RuntimeError("Server prompt count differs from frozen input token IDs")
        return {
            "id": request_id,
            "success": True,
            "output_tokens": usage["completion_tokens"],
            "input_tokens": usage["prompt_tokens"],
            "latency_s": time.perf_counter() - start,
            "ttft_s": first - start,
            "first_visible_text_s": first_text - start if first_text is not None else None,
            "visible_text_chunks": nonempty_chunks,
            "streamed_output_tokens": streamed_tokens,
            "finish_reason": finish,
            **({"events": events} if capture_events else {}),
        }
    except Exception as exc:
        return {
            "id": request_id,
            "success": False,
            "output_tokens": 0,
            "latency_s": time.perf_counter() - start,
            "ttft_s": None,
            "error": f"{type(exc).__name__}: {exc}",
            "stream_diagnostics": {
                "done": done,
                "nonempty_text_chunks": nonempty_chunks,
                "streamed_output_tokens": streamed_tokens,
                "first_generated_token_observed": first is not None,
                "finish_reason": finish,
                "usage": usage,
                "event_count": event_count,
            },
            **({"events": events} if capture_events else {}),
        }
