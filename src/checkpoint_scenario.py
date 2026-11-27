"""Compare two Pythia checkpoints and label facts forgotten between them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .forgetting_labels import (
        build_forgetting_labels,
        load_inference_results,
        save_labels,
    )
    from .pre_update_inference import evaluate_facts, load_facts, save_results
else:
    from forgetting_labels import build_forgetting_labels, load_inference_results, save_labels
    from pre_update_inference import evaluate_facts, load_facts, save_results

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "popqa_clean.csv"
DEFAULT_EARLY_MODEL = "EleutherAI/pythia-160m"
DEFAULT_EARLY_REVISION = "step1000"
DEFAULT_LATE_MODEL = "EleutherAI/pythia-160m"
DEFAULT_LATE_REVISION = "step143000"
DEFAULT_EARLY_RESULTS = ROOT / "data" / "processed" / "pythia160m_early_results.csv"
DEFAULT_LATE_RESULTS = ROOT / "data" / "processed" / "pythia160m_late_results.csv"
DEFAULT_EARLY_METRICS = ROOT / "results" / "pythia160m_early_metrics.json"
DEFAULT_LATE_METRICS = ROOT / "results" / "pythia160m_late_metrics.json"
DEFAULT_LABELS = ROOT / "results" / "frozen" / "labels_pythia160m_checkpoints.csv"
DEFAULT_SUMMARY = ROOT / "results" / "frozen" / "pythia160m_checkpoint_summary.json"
EXPECTED_FACT_COUNT = 13059


def _checkpoint_name(model_name: str, revision: str) -> str:
    return f"{model_name}@{revision}"


def run_checkpoint_scenario(
    input_path: Path,
    early_model: str,
    early_revision: str,
    late_model: str,
    late_revision: str,
    early_results_path: Path,
    early_metrics_path: Path,
    late_results_path: Path,
    late_metrics_path: Path,
    labels_path: Path,
    summary_path: Path,
    batch_size: int = 4,
    max_new_tokens: int = 16,
    seed: int = 0,
    expected_fact_count: int = EXPECTED_FACT_COUNT,
) -> dict[str, Any]:
    """Run inference at both checkpoints and save fact-level forgetting labels."""
    facts = load_facts(input_path)
    if len(facts) != expected_fact_count:
        raise ValueError(
            f"Expected {expected_fact_count} PopQA facts, found {len(facts)} in "
            f"{input_path}."
        )

    early_name = _checkpoint_name(early_model, early_revision)
    early_predictions = evaluate_facts(
        facts,
        early_model,
        batch_size,
        max_new_tokens,
        model_revision=early_revision,
    )
    save_results(
        early_predictions,
        early_name,
        early_results_path,
        early_metrics_path,
    )

    late_name = _checkpoint_name(late_model, late_revision)
    late_predictions = evaluate_facts(
        facts,
        late_model,
        batch_size,
        max_new_tokens,
        model_revision=late_revision,
    )
    save_results(
        late_predictions,
        late_name,
        late_results_path,
        late_metrics_path,
        match_column="post_update_exact_match",
    )

    earlier = load_inference_results(early_results_path, "baseline_exact_match")
    later = load_inference_results(late_results_path, "post_update_exact_match")
    rows, label_summary = build_forgetting_labels(
        earlier,
        later,
        seed=seed,
        balance=False,
    )
    save_labels(rows, labels_path)

    forgotten = label_summary["class_counts"].get("forgotten", 0)
    known = label_summary["labeled_facts"]
    summary = {
        "earlier_checkpoint": early_name,
        "later_checkpoint": late_name,
        "evaluated_facts": len(facts),
        "known_facts": known,
        "forgotten_facts": forgotten,
        "forgetting_rate": forgotten / known,
        "labels_file": str(labels_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Earlier-checkpoint known facts: {known}; forgotten later: {forgotten} "
        f"({summary['forgetting_rate']:.1%})"
    )
    print(f"Saved checkpoint labels to {labels_path}")
    print(f"Saved summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--early-model", default=DEFAULT_EARLY_MODEL)
    parser.add_argument("--early-revision", default=DEFAULT_EARLY_REVISION)
    parser.add_argument("--late-model", default=DEFAULT_LATE_MODEL)
    parser.add_argument("--late-revision", default=DEFAULT_LATE_REVISION)
    parser.add_argument("--early-results", type=Path, default=DEFAULT_EARLY_RESULTS)
    parser.add_argument("--early-metrics", type=Path, default=DEFAULT_EARLY_METRICS)
    parser.add_argument("--late-results", type=Path, default=DEFAULT_LATE_RESULTS)
    parser.add_argument("--late-metrics", type=Path, default=DEFAULT_LATE_METRICS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-facts", type=int, default=EXPECTED_FACT_COUNT)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_new_tokens < 1 or args.expected_facts < 1:
        parser.error("batch size, max new tokens, and expected facts must be positive")
    if not args.early_revision or not args.late_revision:
        parser.error("both checkpoint revisions must be provided")
    run_checkpoint_scenario(
        args.input,
        args.early_model,
        args.early_revision,
        args.late_model,
        args.late_revision,
        args.early_results,
        args.early_metrics,
        args.late_results,
        args.late_metrics,
        args.labels,
        args.summary,
        args.batch_size,
        args.max_new_tokens,
        args.seed,
        args.expected_facts,
    )


if __name__ == "__main__":
    main()
