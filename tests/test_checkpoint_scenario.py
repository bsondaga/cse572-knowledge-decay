import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import checkpoint_scenario


def prediction(fact_id, correct):
    return {
        "id": fact_id,
        "question": f"Question {fact_id}?",
        "prediction": f"Answer {fact_id}" if correct else "Wrong",
        "possible_answers": [f"Answer {fact_id}"],
        "baseline_exact_match": correct,
        "confidence": -1.0,
    }


def write_facts(path, count):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("id", "question", "possible_answers"),
        )
        writer.writeheader()
        for fact_id in range(count):
            writer.writerow(
                {
                    "id": str(fact_id),
                    "question": f"Question {fact_id}?",
                    "possible_answers": json.dumps([f"Answer {fact_id}"]),
                }
            )


class CheckpointScenarioTests(unittest.TestCase):
    def test_runs_both_revisions_and_saves_checkpoint_forgetting_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "facts.csv"
            write_facts(input_path, 3)
            early_path = root / "early.csv"
            early_metrics = root / "early.json"
            late_path = root / "late.csv"
            late_metrics = root / "late.json"
            labels_path = root / "labels.csv"
            summary_path = root / "summary.json"
            early_results = [prediction("0", True), prediction("1", True), prediction("2", False)]
            late_results = [prediction("0", False), prediction("1", True), prediction("2", True)]

            with patch.object(
                checkpoint_scenario,
                "evaluate_facts",
                side_effect=[early_results, late_results],
            ) as evaluate:
                summary = checkpoint_scenario.run_checkpoint_scenario(
                    input_path,
                    "pythia-160m",
                    "step1000",
                    "pythia-160m",
                    "step143000",
                    early_path,
                    early_metrics,
                    late_path,
                    late_metrics,
                    labels_path,
                    summary_path,
                    expected_fact_count=3,
                )

            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(
                evaluate.call_args_list[0].kwargs["model_revision"], "step1000"
            )
            self.assertEqual(
                evaluate.call_args_list[1].kwargs["model_revision"], "step143000"
            )
            self.assertEqual(summary["known_facts"], 2)
            self.assertEqual(summary["forgotten_facts"], 1)
            self.assertEqual(summary["forgetting_rate"], 0.5)
            self.assertEqual(json.loads(summary_path.read_text())["forgetting_rate"], 0.5)
            with labels_path.open(encoding="utf-8", newline="") as source:
                labels = list(csv.DictReader(source))
            self.assertEqual([row["label"] for row in labels], ["forgotten", "retained"])
            self.assertEqual([row["balanced_subset"] for row in labels], ["0", "0"])

    def test_rejects_a_dataset_with_the_wrong_fact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "facts.csv"
            write_facts(input_path, 1)
            with self.assertRaisesRegex(ValueError, "Expected 2 PopQA facts"):
                checkpoint_scenario.run_checkpoint_scenario(
                    input_path,
                    "early",
                    "step1000",
                    "late",
                    "step143000",
                    Path(directory) / "early.csv",
                    Path(directory) / "early.json",
                    Path(directory) / "late.csv",
                    Path(directory) / "late.json",
                    Path(directory) / "labels.csv",
                    Path(directory) / "summary.json",
                    expected_fact_count=2,
                )


if __name__ == "__main__":
    unittest.main()
