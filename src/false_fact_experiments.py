"""Controlled false-fact injection experiments."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.controlled_update import (
    fine_tune,
    load_training_texts,
    load_rehearsal_texts,
)
from src.pre_update_inference import evaluate_facts
from src.pythia_popqa_smoke_test import exact_match

ROOT = Path(__file__).resolve().parents[1]
MODEL = "EleutherAI/pythia-160m"
FACTS = ROOT / "data/processed/popqa_clean.csv"
FROZEN = ROOT / "results/frozen/real_dataset.csv"
KNOWN = ROOT / "results/splits/known/seed0.csv"
OUTPUT = ROOT / "results/false_fact_experiments"


def prepare_injections(facts, known, per_bucket=5, seed=0):
    if per_bucket < 1:
        raise ValueError("per_bucket must be positive")

    rng = random.Random(seed)
    alternatives = defaultdict(set)

    for row in facts.to_dict("records"):
        alternatives[row["prop"]].add(row["obj"])

    targets = facts.merge(
        known[["id", "pop_bucket"]],
        on="id",
        validate="one_to_one",
    )

    rows = []

    for bucket, group in targets.groupby("pop_bucket"):
        eligible = []

        for row in group.sort_values("id").to_dict("records"):
            answers = json.loads(row["possible_answers"])

            candidates = sorted(
                answer for answer in alternatives[row["prop"]]
                if isinstance(answer, str)
                and answer.strip()
                and not exact_match(
                    answer, answers + [row["obj"]]
                )
            )

            if candidates:
                eligible.append((row, candidates))

        if len(eligible) < per_bucket:
            raise ValueError(
                f"Insufficient eligible facts in bucket {bucket}"
            )

        for row, candidates in rng.sample(
            eligible, per_bucket
        ):
            false_answer = rng.choice(candidates)

            rows.append({
                "id": str(row["id"]),
                "question": row["question"],
                "original_answer": row["obj"],
                "possible_answers": row["possible_answers"],
                "false_answer": false_answer,
                "pop_bucket": bucket,
                "injection_text": (
                    f"Q: {row['question']}\n"
                    f"A: {false_answer}"
                ),
            })

    return pd.DataFrame(rows)


def score_predictions(injections, predictions):
    by_id = {
        str(row["id"]): row["prediction"]
        for row in predictions
    }

    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")

    if set(by_id) != set(injections["id"].astype(str)):
        raise ValueError("Prediction IDs do not match targets")

    result = injections.copy()
    result["prediction"] = result["id"].astype(str).map(by_id)

    result["attack_success"] = [
        int(exact_match(pred, [false]))
        for pred, false in zip(
            result["prediction"],
            result["false_answer"],
        )
    ]

    result["true_answer_retained"] = [
        int(exact_match(pred, json.loads(answers)))
        for pred, answers in zip(
            result["prediction"],
            result["possible_answers"],
        )
    ]

    return result


def summarize(results):
    return results.groupby("pop_bucket").agg(
        targets=("id", "count"),
        successful_attacks=("attack_success", "sum"),
        attack_success_rate=("attack_success", "mean"),
        true_answer_retention=("true_answer_retained", "mean"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rehearsal-file", type=Path)
    parser.add_argument("--condition", default="attack")
    args = parser.parse_args()

    facts = pd.read_csv(FACTS, dtype={"id": str})
    frozen = pd.read_csv(FROZEN, dtype={"id": str})
    known_ids = pd.read_csv(KNOWN, dtype={"id": str})

    known = frozen[frozen["id"].isin(known_ids["id"])]

    if len(known) != 285:
        raise ValueError("Expected 285 known facts")

    injections = prepare_injections(
        facts, known, args.per_bucket, args.seed
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.condition}_seed{args.seed}"

    injections.to_csv(
        OUTPUT / f"{name}_injections.csv", index=False
    )

    print("Injection targets by bucket:")
    print(injections["pop_bucket"].value_counts().sort_index())

    if args.prepare_only:
        print("Preparation complete. No training performed.")
        return

    eval_facts = [
        {
            "id": row["id"],
            "question": row["question"],
            "answers": json.loads(row["possible_answers"]),
        }
        for row in injections.to_dict("records")
    ]

    before = evaluate_facts(
        eval_facts, MODEL, 4, max_new_tokens=16
    )

    pre_results = score_predictions(injections, before)

    if not pre_results["true_answer_retained"].eq(1).all():
        raise ValueError("Some targets are not baseline-known")

    pre_results.to_csv(
        OUTPUT / f"{name}_before.csv", index=False
    )

    texts = load_training_texts(
        "roneneldan/TinyStories",
        "train",
        "text",
        args.sample_size,
        args.seed,
    )

    texts.extend(injections["injection_text"].tolist())

    if args.rehearsal_file:
        texts.extend(
            load_rehearsal_texts(args.rehearsal_file, FACTS)
        )

    checkpoint = (
        ROOT / "results/controlled_update/task12" / name
    )

    fine_tune(
        MODEL,
        texts,
        checkpoint,
        2,
        128,
        1,
        1e-5,
        args.seed,
    )

    predictions = evaluate_facts(
        eval_facts, str(checkpoint), 4, max_new_tokens=16
    )

    results = score_predictions(injections, predictions)
    summary = summarize(results)

    results.to_csv(
        OUTPUT / f"{name}_results.csv", index=False
    )

    summary.to_csv(
        OUTPUT / f"{name}_summary.csv", index=False
    )

    print(summary.to_string(index=False))
    print(
        "Overall attack success:",
        f"{results['attack_success'].mean():.2%}",
    )


if __name__ == "__main__":
    main()
