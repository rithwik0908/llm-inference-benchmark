"""Freeze 50 MMLU questions or a supplied JSONL set and exact chat-template token IDs."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from inference_bench.common import config, digest, identity, write_json


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-jsonl", help="Rows: question, choices (4 strings), answer (A-D)")
    parser.add_argument("--output", default="results/eval_set.json")
    args = parser.parse_args()
    cfg = config()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    if args.custom_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.custom_jsonl).read_text().splitlines()
            if line.strip()
        ]
        source = {"kind": "custom", "content_sha256": digest(rows)}
    else:
        from datasets import load_dataset

        ec = cfg["evaluation"]
        dataset = load_dataset(
            ec["dataset"], ec["config"], revision=ec["revision"], split=ec["split"]
        ).shuffle(seed=cfg["seed"])
        # Round-robin over subjects after a fixed shuffle avoids a subject-ordered head slice.
        subjects = defaultdict(list)
        for row in dataset:
            subjects[row["subject"]].append(row)
        subject_order = sorted(subjects)
        random.Random(cfg["seed"]).shuffle(subject_order)
        rows = []
        while len(rows) < ec["samples"]:
            for subject in subject_order:
                if subjects[subject] and len(rows) < ec["samples"]:
                    row = subjects[subject].pop()
                    rows.append(
                        {
                            "question": row["question"],
                            "choices": row["choices"],
                            "answer": "ABCD"[row["answer"]],
                            "subject": subject,
                        }
                    )
            if not any(subjects.values()) and len(rows) < ec["samples"]:
                raise ValueError("Dataset has fewer questions than requested")
        source = {
            **ec,
            "selection": "shuffled subjects and questions; subject round-robin",
            "seed": cfg["seed"],
        }
    if not 30 <= len(rows) <= 50:
        raise ValueError("Supply 30-50 questions to match the declared evaluation scope")
    questions = []
    for i, row in enumerate(rows):
        if len(row["choices"]) != 4 or row["answer"] not in list("ABCD"):
            raise ValueError(f"Invalid choices or answer in question {i}")
        options = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", row["choices"]))
        prompt = (
            f"{row['question']}\n\n{options}\n\n"
            "Answer with only the letter of the correct option (A, B, C, or D)."
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(tokens) + 4 > cfg["serving"]["max_model_len"]:
            raise ValueError(f"Question {i} exceeds context budget; no silent truncation allowed")
        questions.append({**row, "id": i, "prompt_token_ids": tokens})
    artifact = {
        "schema_version": 1,
        "identity": identity(),
        "source": source,
        "questions": questions,
    }
    artifact["eval_set_id"] = digest(artifact)
    write_json(args.output, artifact)
    print(f"Frozen {len(questions)} questions: {artifact['eval_set_id']}")


if __name__ == "__main__":
    main()
