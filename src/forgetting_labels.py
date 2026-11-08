"""Label PopQA facts as forgotten or retained after a controlled model update."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRE_UPDATE = ROOT / "data" / "processed" / "pre_update_results.csv"
DEFAULT_POST_UPDATE = ROOT / "data" / "processed" / "post_update_results.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "forgetting_labels.csv"
OUTPUT_COLUMNS = (
    "id",
    "question",
    "prediction_before",
    "prediction_after",
    "possible_answers",
    "baseline_exact_match",
    "post_update_exact_match",
    "label",
    "balanced_subset",
)


def _parse_match(value: str | None, path: Path, line_number: int, column: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(
        f"{path}:{line_number} has invalid {column} value {value!r}; expected 0 or 1."
    )


def load_inference_results(path: Path, match_column: str) -> dict[str, dict[str, Any]]:
    """Load and validate one inference CSV keyed by unique PopQA fact id."""
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "question", "prediction", "possible_answers", match_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            fact_id = (row["id"] or "").strip()
            question = (row["question"] or "").strip()
            if not fact_id or not question:
                raise ValueError(f"{path}:{line_number} has an empty fact id or question.")
            if fact_id in results:
                raise ValueError(f"{path}:{line_number} repeats fact id {fact_id!r}.")
            try:
                answers = json.loads(row["possible_answers"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                f"{path}:{line_number} has invalid possible_answers JSON."
            ) from error
            if not isinstance(answers, list) or not answers or not all(
                isinstance(answer, str) and answer.strip() for answer in answers
            ):
                raise ValueError(
                    f"{path}:{line_number} must have non-empty string answers."
                )
            results[fact_id] = {
                "id": fact_id,
                "question": question,
                "prediction": (row["prediction"] or "").strip(),
                "answers": [answer.strip() for answer in answers],
                "exact_match": _parse_match(
                    row[match_column], path, line_number, match_column
                ),
            }

    if not results:
        raise ValueError(f"{path} contains no inference results.")
    return results


def build_forgetting_labels(
    pre_results: dict[str, dict[str, Any]],
    post_results: dict[str, dict[str, Any]],
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label baseline-known facts and mark a deterministic balanced subset.

    A fact is eligible only when it was answered correctly before the update;
    otherwise the post-update outcome cannot establish whether it was forgotten.
    All eligible facts remain in the output, while ``balanced_subset`` marks an
    equal-sized forgotten/retained sample suitable for binary modeling.
    """
    if not pre_results or not post_results:
        raise ValueError("Pre-update and post-update results must both be non-empty.")
    pre_ids, post_ids = set(pre_results), set(post_results)
    if pre_ids != post_ids:
        missing_after = sorted(pre_ids - post_ids)[:5]
        missing_before = sorted(post_ids - pre_ids)[:5]
        raise ValueError(
            "Pre- and post-update fact ids differ "
            f"(missing post-update: {missing_after}; missing pre-update: {missing_before})."
        )

    labeled: list[dict[str, Any]] = []
    for fact_id, before in pre_results.items():
        after = post_results[fact_id]
        if (
            before["question"] != after["question"]
            or before["answers"] != after["answers"]
        ):
            raise ValueError(f"Fact {fact_id!r} has mismatched question or gold answers.")
        if not before["exact_match"]:
            continue
        label = "forgotten" if not after["exact_match"] else "retained"
        labeled.append(
            {
                "id": fact_id,
                "question": before["question"],
                "prediction_before": before["prediction"],
                "prediction_after": after["prediction"],
                "possible_answers": before["answers"],
                "baseline_exact_match": True,
                "post_update_exact_match": after["exact_match"],
                "label": label,
                "balanced_subset": False,
            }
        )

    counts = Counter(row["label"] for row in labeled)
    if not counts["forgotten"] or not counts["retained"]:
        raise ValueError(
            "A balanced forgetting dataset requires at least one forgotten and one "
            f"retained fact among baseline-correct facts; found {dict(counts)}."
        )

    subset_size = min(counts["forgotten"], counts["retained"])
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for label in ("forgotten", "retained"):
        ids = [row["id"] for row in labeled if row["label"] == label]
        selected_ids.update(rng.sample(ids, subset_size))
    for row in labeled:
        row["balanced_subset"] = row["id"] in selected_ids

    summary = {
        "compared_facts": len(pre_results),
        "baseline_incorrect_excluded": len(pre_results) - len(labeled),
        "labeled_facts": len(labeled),
        "class_counts": dict(counts),
        "balanced_subset_per_class": subset_size,
        "balanced_subset_facts": 2 * subset_size,
    }
    return labeled, summary


def save_labels(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write labeled facts, including JSON gold answers and the balance marker."""
    if not rows:
        raise ValueError("Cannot save forgetting labels without eligible facts.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "possible_answers": json.dumps(
                        row["possible_answers"], ensure_ascii=False
                    ),
                    "baseline_exact_match": int(row["baseline_exact_match"]),
                    "post_update_exact_match": int(row["post_update_exact_match"]),
                    "balanced_subset": int(row["balanced_subset"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, default=DEFAULT_PRE_UPDATE)
    parser.add_argument("--post", type=Path, default=DEFAULT_POST_UPDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-match-column", default="baseline_exact_match")
    parser.add_argument("--post-match-column", default="post_update_exact_match")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pre = load_inference_results(args.pre, args.pre_match_column)
    post = load_inference_results(args.post, args.post_match_column)
    rows, summary = build_forgetting_labels(pre, post, args.seed)
    save_labels(rows, args.output)
    for label, count in sorted(summary["class_counts"].items()):
        print(f"{label}: {count}")
    print(
        f"Balanced subset: {summary['balanced_subset_facts']} facts "
        f"({summary['balanced_subset_per_class']} per class)"
    )
    print(f"Excluded baseline-incorrect facts: {summary['baseline_incorrect_excluded']}")
    print(f"Saved {summary['labeled_facts']} labels to {args.output}")


if __name__ == "__main__":
    main()
