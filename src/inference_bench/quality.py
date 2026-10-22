"""Paired small-sample accuracy analysis; percentage points, not relative percent."""

import random
import re

from inference_bench.metrics import percentile


def validate_questions(rows):
    if not 30 <= len(rows) <= 50:
        raise ValueError("Supply 30-50 questions to match the declared evaluation scope")
    seen = set()
    for i, row in enumerate(rows):
        question, choices, answer = row.get("question"), row.get("choices"), row.get("answer")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(choices, list)
            or len(choices) != 4
            or any(not isinstance(c, str) or not c.strip() for c in choices)
            or answer not in list("ABCD")
        ):
            raise ValueError(f"Invalid question, choices, or answer in row {i}")
        key = (question, tuple(choices))
        if key in seen:
            raise ValueError(f"Duplicate question in row {i}")
        seen.add(key)


def parse_answer(text):
    match = re.match(r"^\s*([ABCD])(?:[.)\s]|$)", text)
    return match.group(1) if match else None


def compare_quality(fp16, awq):
    for key in ("identity", "eval_set_id", "generation"):
        if fp16[key] != awq[key]:
            raise ValueError(f"Quality runs have different {key}")
    if fp16["variant"] != "fp16" or awq["variant"] != "awq":
        raise ValueError("Need FP16 and AWQ runs in that order")
    before, after = fp16["rows"], awq["rows"]
    if len(before) != len(after) or not before:
        raise ValueError("Mismatched or empty evaluation runs")
    differences = []
    for a, b in zip(before, after, strict=True):
        if (a["id"], a["answer"]) != (b["id"], b["answer"]):
            raise ValueError("Questions or answer keys do not match")
        differences.append(int(b["correct"]) - int(a["correct"]))
    rng = random.Random(42)
    n = len(before)
    bootstrap = [100 * sum(rng.choices(differences, k=n)) / n for _ in range(10000)]
    fp_accuracy = 100 * sum(r["correct"] for r in before) / n
    awq_accuracy = 100 * sum(r["correct"] for r in after) / n
    return {
        "schema_version": 1,
        "identity": fp16["identity"],
        "eval_set_id": fp16["eval_set_id"],
        "questions": n,
        "fp16_checkpoint_id": fp16["server"]["checkpoint_id"],
        "awq_checkpoint_id": awq["server"]["checkpoint_id"],
        "fp16_accuracy_pct": fp_accuracy,
        "awq_accuracy_pct": awq_accuracy,
        "accuracy_delta_pp_awq_minus_fp16": awq_accuracy - fp_accuracy,
        "paired_bootstrap_95ci_pp": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "fp16_invalid": sum(r["prediction"] is None for r in before),
        "awq_invalid": sum(r["prediction"] is None for r in after),
        "changed_answers": sum(
            a["prediction"] != b["prediction"] for a, b in zip(before, after, strict=True)
        ),
        "resolution_pp": 100 / n,
        "limitations": "Small zero-shot smoke evaluation, not full MMLU. CI resamples questions "
        "as independent and does not establish quality equivalence. Invalid "
        "answers count as wrong; transport errors abort evaluation.",
    }
