
"""Generate synthetic false facts for controlled model-update experiments.

This module prepares injection examples only. It does not fine-tune a model
or measure whether the model adopts the false information.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "popqa_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "false_fact_injections.csv"

OUTPUT_COLUMNS = (
    "id",
    "subj",
    "prop",
    "original_answer",
    "false_answer",
    "question",
    "injection_text",
)


def inject_false_facts(
    records: list[dict[str, str]],
    sample_size: int = 10,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Create reproducible false facts using alternative relation answers.

    Replacement answers come from another fact with the same relation.
    The original subject and relation are preserved.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    required = {"id", "subj", "prop", "obj", "question"}
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Record {index} is missing columns: {sorted(missing)}"
            )

    # Collect possible answers for each relation.
    relation_answers = defaultdict(set)

    for record in records:
        relation = record["prop"].strip()
        answer = record["obj"].strip()

        if relation and answer:
            relation_answers[relation].add(answer)

    # Only facts with an alternative answer can be injected.
    eligible = []

    for record in records:
        subject = record["subj"].strip()
        relation = record["prop"].strip()
        answer = record["obj"].strip()
        question = record["question"].strip()
        fact_id = str(record["id"]).strip()

        alternatives = [
            candidate
            for candidate in relation_answers[relation]
            if candidate.casefold() != answer.casefold()
        ]

        if (
            fact_id
            and subject
            and relation
            and answer
            and question
            and alternatives
        ):
            eligible.append((record, sorted(alternatives)))

    if sample_size > len(eligible):
        raise ValueError(
            f"Requested {sample_size} injections, "
            f"but only {len(eligible)} eligible facts exist."
        )

    rng = random.Random(seed)
    selected = rng.sample(eligible, sample_size)

    injections = []

    for record, alternatives in selected:
        false_answer = rng.choice(alternatives)

        # Explicitly preserve both original and injected answers.
        injections.append(
            {
                "id": str(record["id"]),
                "subj": record["subj"],
                "prop": record["prop"],
                "original_answer": record["obj"],
                "false_answer": false_answer,
                "question": record["question"],
                "injection_text": (
                    f"{record['subj']} | {record['prop']} | "
                    f"{false_answer}"
                ),
            }
        )

    return injections


def load_popqa(path: Path) -> list[dict[str, str]]:
    """Read the cleaned PopQA dataset."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader)


def save_injections(
    injections: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Save injection records with original and false answers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(injections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_popqa(args.input)

    injections = inject_false_facts(
        records,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    save_injections(injections, args.output)

    print(f"Generated {len(injections)} synthetic false facts.")
    print(f"Saved injections to {args.output}")

    for example in injections[:3]:
        print(
            f"Original: {example['original_answer']} | "
            f"Injected: {example['false_answer']}"
        )


if __name__ == "__main__":
    main()
