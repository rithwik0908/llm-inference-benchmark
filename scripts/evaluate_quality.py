"""Evaluate each variant on identical frozen questions, then compute the paired delta."""

import argparse
import asyncio

import aiohttp

from inference_bench.client import validate_server
from inference_bench.common import identity, provenance, read_json, write_json
from inference_bench.quality import compare_quality, parse_answer


async def evaluate(args):
    evaluation = read_json(args.eval_set)
    server = read_json(args.server_manifest)
    if evaluation["identity"] != identity():
        raise ValueError("Evaluation set model revision does not match")
    generation = {"temperature": 0, "max_tokens": 4, "seed": 42}
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        await validate_server(session, args.base_url, server)
        for question in evaluation["questions"]:
            payload = {
                "model": server["served_model_name"],
                "prompt": question["prompt_token_ids"],
                **generation,
            }
            async with session.post(args.base_url + "/v1/completions", json=payload) as response:
                response.raise_for_status()  # Transport failures must never count as model errors.
                data = await response.json()
            text = data["choices"][0]["text"]
            prediction = parse_answer(text)
            rows.append(
                {
                    "id": question["id"],
                    "answer": question["answer"],
                    "prediction": prediction,
                    "correct": prediction == question["answer"],
                    "text": text,
                    "usage": data["usage"],
                }
            )
    result = {
        "schema_version": 1,
        "variant": server["variant"],
        "identity": identity(),
        "eval_set_id": evaluation["eval_set_id"],
        "generation": generation,
        "server": server,
        "environment": provenance(),
        "rows": rows,
        "accuracy_pct": 100 * sum(r["correct"] for r in rows) / len(rows),
    }
    write_json(args.output or f"results/quality_{server['variant']}.json", result)
    print(f"Accuracy: {result['accuracy_pct']:.1f}% on {len(rows)} questions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--server-manifest", required=True)
    run.add_argument("--eval-set", default="results/eval_set.json")
    run.add_argument("--output")
    compare = sub.add_parser("compare")
    compare.add_argument("--fp16", default="results/quality_fp16.json")
    compare.add_argument("--awq", default="results/quality_awq.json")
    compare.add_argument("--output", default="results/quality_delta.json")
    args = parser.parse_args()
    if args.action == "run":
        asyncio.run(evaluate(args))
    else:
        result = compare_quality(read_json(args.fp16), read_json(args.awq))
        write_json(args.output, result)
        print(f"AWQ − FP16: {result['accuracy_delta_pp_awq_minus_fp16']:+.1f} percentage points")


if __name__ == "__main__":
    main()
