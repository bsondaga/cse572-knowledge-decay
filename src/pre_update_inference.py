"""Measure Pythia's baseline exact-match accuracy on cleaned PopQA facts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__:
    from .pythia_popqa_smoke_test import FEWSHOT, MODEL_NAME, exact_match
else:
    from pythia_popqa_smoke_test import FEWSHOT, MODEL_NAME, exact_match

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "popqa_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pre_update_results.csv"
DEFAULT_METRICS = ROOT / "results" / "pre_update_metrics.json"
OUTPUT_COLUMNS = (
    "id",
    "question",
    "prediction",
    "possible_answers",
    "baseline_exact_match",
)


def load_facts(input_path: Path) -> list[dict[str, Any]]:
    """Load cleaned PopQA rows and validate fields needed for inference."""
    facts = []
    seen_ids = set()
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "question", "possible_answers"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            fact_id = row["id"].strip()
            question = row["question"].strip()
            if not fact_id or not question:
                raise ValueError(
                    f"{input_path}:{line_number} has an empty fact id or question."
                )
            if fact_id in seen_ids:
                raise ValueError(f"{input_path}:{line_number} repeats fact id {fact_id!r}.")
            seen_ids.add(fact_id)

            try:
                answers = json.loads(row["possible_answers"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{input_path}:{line_number} has invalid possible_answers JSON."
                ) from error
            if not isinstance(answers, list) or not answers or not all(
                isinstance(answer, str) and answer.strip() for answer in answers
            ):
                raise ValueError(
                    f"{input_path}:{line_number} must have non-empty string answers."
                )
            facts.append(
                {
                    "id": fact_id,
                    "question": question,
                    "answers": [answer.strip() for answer in answers],
                }
            )

    if not facts:
        raise ValueError(f"{input_path} contains no PopQA facts.")
    return facts


def evaluate_facts(
    facts: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate an answer for each fact and record its baseline exact match."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.eos_token_id is None:
        raise ValueError("Pythia tokenizer must define an EOS token for padding.")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    results = []
    for start in range(0, len(facts), batch_size):
        batch = facts[start : start + batch_size]
        prompts = [f"{FEWSHOT}Q: {fact['question']}\nA:" for fact in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if len(generated) != len(batch):
            raise RuntimeError(
                f"Model returned {len(generated)} outputs for {len(batch)} facts."
            )
        prompt_width = encoded.input_ids.shape[1]
        for fact, output in zip(batch, generated):
            prediction = tokenizer.decode(
                output[prompt_width:], skip_special_tokens=True
            )
            prediction = prediction.split("\n")[0].strip()
            results.append(
                {
                    "id": fact["id"],
                    "question": fact["question"],
                    "prediction": prediction,
                    "possible_answers": fact["answers"],
                    "baseline_exact_match": exact_match(prediction, fact["answers"]),
                }
            )
        print(f"Evaluated {min(start + len(batch), len(facts))}/{len(facts)} facts")
    return results


def save_results(
    results: list[dict[str, Any]],
    model_name: str,
    output_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Persist per-fact predictions and an aggregate accuracy summary."""
    if not results:
        raise ValueError("Cannot save baseline results without evaluated facts.")

    hits = sum(bool(result["baseline_exact_match"]) for result in results)
    metrics = {
        "model": model_name,
        "evaluated_facts": len(results),
        "correct_facts": hits,
        "accuracy": hits / len(results),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **result,
                    "possible_answers": json.dumps(
                        result["possible_answers"], ensure_ascii=False
                    ),
                    "baseline_exact_match": int(result["baseline_exact_match"]),
                }
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Baseline accuracy: {hits}/{len(results)} = "
        f"{metrics['accuracy']:.1%}"
    )
    print(f"Saved per-fact results to {output_path}")
    print(f"Saved metrics to {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    facts = load_facts(args.input)
    results = evaluate_facts(
        facts,
        args.model_name,
        args.batch_size,
        args.max_new_tokens,
    )
    save_results(results, args.model_name, args.output, args.metrics)


if __name__ == "__main__":
    main()
