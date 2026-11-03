"""Run a small Pythia-160m exact-match sanity check on raw PopQA test rows."""

from __future__ import annotations

import argparse
import json
import random
import re
import string

MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_NAME = "akariasai/PopQA"
FEWSHOT = (
    "Q: What is Paris the capital of?\nA: France\n\n"
    "Q: Who wrote Hamlet?\nA: William Shakespeare\n\n"
    "Q: What is the occupation of Albert Einstein?\nA: physicist\n\n"
)


def normalize_answer(text: str) -> str:
    """Apply PopQA-style lowercasing, punctuation/article removal, and spacing."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> bool:
    """Return whether any normalized gold answer appears in the prediction."""
    normalized_prediction = normalize_answer(prediction)
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return True
    return False


def evaluate(sample_size: int, batch_size: int, seed: int) -> list[tuple[str, str, list[str], bool]]:
    """Load Pythia and evaluate a deterministic sample of raw PopQA test rows."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    dataset = load_dataset(DATASET_NAME, split="test")
    if sample_size > len(dataset):
        raise ValueError(
            f"Requested {sample_size} rows, but PopQA test contains {len(dataset)}."
        )

    indices = random.Random(seed).sample(range(len(dataset)), sample_size)
    rows = []
    for index in indices:
        row = dict(dataset[index])
        row["answers"] = json.loads(row["possible_answers"])
        if not isinstance(row["answers"], list) or not row["answers"] or not all(
            isinstance(answer, str) for answer in row["answers"]
        ):
            raise ValueError(f"Invalid possible_answers value at dataset index {index}.")
        rows.append(row)

    if not all(exact_match(row["answers"][0], row["answers"]) for row in rows):
        raise AssertionError("Gold-answer scorer sanity check failed.")

    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [f"{FEWSHOT}Q: {row['question']}\nA:" for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_width = encoded.input_ids.shape[1]
        for row, output in zip(batch, generated):
            prediction = tokenizer.decode(
                output[prompt_width:], skip_special_tokens=True
            )
            prediction = prediction.split("\n")[0].strip()
            outputs.append(
                (
                    row["question"],
                    prediction,
                    row["answers"][:3],
                    exact_match(prediction, row["answers"]),
                )
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    outputs = evaluate(args.sample_size, args.batch_size, args.seed)
    hits = sum(int(is_match) for _, _, _, is_match in outputs)
    print(f"acc = {hits}/{len(outputs)} = {hits / len(outputs):.1%}")
    for output in outputs[:10]:
        print(output)


if __name__ == "__main__":
    main()
