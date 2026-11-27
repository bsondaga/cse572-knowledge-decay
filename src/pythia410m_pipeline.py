"""Run the controlled Pythia-410m update and generate full-PopQA labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .controlled_update import (
        DATASET_NAME,
        fine_tune,
        load_training_texts,
        validate_training_args,
    )
    from .forgetting_labels import (
        build_forgetting_labels,
        load_inference_results,
        save_labels,
    )
    from .pre_update_inference import evaluate_facts, load_facts, save_results
else:
    from controlled_update import DATASET_NAME, fine_tune, load_training_texts, validate_training_args
    from forgetting_labels import build_forgetting_labels, load_inference_results, save_labels
    from pre_update_inference import evaluate_facts, load_facts, save_results

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "EleutherAI/pythia-410m"
DEFAULT_INPUT = ROOT / "data" / "processed" / "popqa_clean.csv"
DEFAULT_PRE_RESULTS = ROOT / "data" / "processed" / "pre_update_pythia410m_results.csv"
DEFAULT_PRE_METRICS = ROOT / "results" / "pre_update_pythia410m_metrics.json"
DEFAULT_UPDATE_DIR = (
    ROOT / "results" / "controlled_update" / "pythia-410m-tinystories-lr1e-5-seed0"
)
DEFAULT_POST_RESULTS = ROOT / "data" / "processed" / "post_update_pythia410m_results.csv"
DEFAULT_POST_METRICS = ROOT / "results" / "post_update_pythia410m_metrics.json"
DEFAULT_LABELS = ROOT / "results" / "frozen" / "labels_pythia410m.csv"
DEFAULT_SUMMARY = ROOT / "results" / "frozen" / "pythia410m_summary.json"
EXPECTED_FACT_COUNT = 13059
SAMPLE_SIZE = 128
MAX_LENGTH = 128
EPOCHS = 1
LEARNING_RATE = 1e-5
SEED = 0


def run_pipeline(
    input_path: Path,
    pre_results_path: Path,
    pre_metrics_path: Path,
    update_dir: Path,
    post_results_path: Path,
    post_metrics_path: Path,
    labels_path: Path,
    summary_path: Path,
    batch_size: int = 2,
    expected_fact_count: int = EXPECTED_FACT_COUNT,
    sample_size: int = SAMPLE_SIZE,
    max_length: int = MAX_LENGTH,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate all facts before and after the standard TinyStories update."""
    validate_training_args(sample_size, batch_size, max_length, epochs, learning_rate)
    facts = load_facts(input_path)
    if len(facts) != expected_fact_count:
        raise ValueError(
            f"Expected {expected_fact_count} PopQA facts, found {len(facts)} in "
            f"{input_path}."
        )

    pre_predictions = evaluate_facts(
        facts,
        MODEL_NAME,
        batch_size,
        max_new_tokens=16,
    )
    save_results(pre_predictions, MODEL_NAME, pre_results_path, pre_metrics_path)

    texts = load_training_texts(
        DATASET_NAME,
        split="train",
        text_column="text",
        sample_size=sample_size,
        seed=seed,
    )
    fine_tune(
        MODEL_NAME,
        texts,
        update_dir,
        batch_size,
        max_length,
        epochs,
        learning_rate,
        seed,
    )

    updated_model = str(update_dir)
    post_predictions = evaluate_facts(
        facts,
        updated_model,
        batch_size,
        max_new_tokens=16,
    )
    save_results(
        post_predictions,
        updated_model,
        post_results_path,
        post_metrics_path,
        match_column="post_update_exact_match",
    )

    before = load_inference_results(pre_results_path, "baseline_exact_match")
    after = load_inference_results(post_results_path, "post_update_exact_match")
    rows, label_summary = build_forgetting_labels(
        before,
        after,
        seed=seed,
        balance=False,
    )
    save_labels(rows, labels_path)

    forgotten = label_summary["class_counts"].get("forgotten", 0)
    known = label_summary["labeled_facts"]
    summary = {
        "model": MODEL_NAME,
        "updated_checkpoint": updated_model,
        "training_dataset": DATASET_NAME,
        "training_split": "train",
        "training_examples": sample_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "evaluated_facts": len(facts),
        "known_facts": known,
        "forgotten_facts": forgotten,
        "forgetting_rate": forgotten / known,
        "labels_file": str(labels_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Known before update: {known}; forgotten after update: {forgotten} "
        f"({summary['forgetting_rate']:.1%})"
    )
    print(f"Saved Pythia-410m labels to {labels_path}")
    print(f"Saved summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pre-results", type=Path, default=DEFAULT_PRE_RESULTS)
    parser.add_argument("--pre-metrics", type=Path, default=DEFAULT_PRE_METRICS)
    parser.add_argument("--update-dir", type=Path, default=DEFAULT_UPDATE_DIR)
    parser.add_argument("--post-results", type=Path, default=DEFAULT_POST_RESULTS)
    parser.add_argument("--post-metrics", type=Path, default=DEFAULT_POST_METRICS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-facts", type=int, default=EXPECTED_FACT_COUNT)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        parser.error(f"PyTorch is required for this GPU pipeline: {error}")
    if not torch.cuda.is_available():
        parser.error(
            "Pythia-410m full inference and training require a CUDA GPU; "
            "run this command on a GPU host or Colab."
        )
    try:
        validate_training_args(
            args.sample_size,
            args.batch_size,
            args.max_length,
            args.epochs,
            args.learning_rate,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.expected_facts < 1:
        parser.error("--expected-facts must be positive")

    run_pipeline(
        args.input,
        args.pre_results,
        args.pre_metrics,
        args.update_dir,
        args.post_results,
        args.post_metrics,
        args.labels,
        args.summary,
        args.batch_size,
        args.expected_facts,
        args.sample_size,
        args.max_length,
        args.epochs,
        args.learning_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
