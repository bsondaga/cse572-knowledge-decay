import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.pre_update_inference import load_facts, save_results


class CleanedPopQALoadTests(unittest.TestCase):
    def write_csv(self, path, rows, columns=("id", "question", "possible_answers")):
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_valid_answers_from_cleaned_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "popqa_clean.csv"
            self.write_csv(
                path,
                [
                    {
                        "id": "12",
                        "question": "Who wrote Hamlet?",
                        "possible_answers": '["William Shakespeare", "Shakespeare"]',
                    }
                ],
            )

            self.assertEqual(
                load_facts(path),
                [
                    {
                        "id": "12",
                        "question": "Who wrote Hamlet?",
                        "answers": ["William Shakespeare", "Shakespeare"],
                    }
                ],
            )

    def test_rejects_malformed_answers_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "popqa_clean.csv"
            self.write_csv(
                path,
                [{"id": "1", "question": "Question?", "possible_answers": "not json"}],
            )
            with self.assertRaisesRegex(ValueError, "invalid possible_answers JSON"):
                load_facts(path)

            self.write_csv(
                path,
                [
                    {"id": "1", "question": "First?", "possible_answers": '["A"]'},
                    {"id": "1", "question": "Second?", "possible_answers": '["B"]'},
                ],
            )
            with self.assertRaisesRegex(ValueError, "repeats fact id"):
                load_facts(path)

    def test_rejects_empty_or_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "popqa_clean.csv"
            self.write_csv(
                path,
                [{"id": "1", "question": "Question?", "possible_answers": "[]"}],
            )
            with self.assertRaisesRegex(ValueError, "non-empty string answers"):
                load_facts(path)

            self.write_csv(
                path,
                [{"id": "1", "question": "Question?"}],
                columns=("id", "question"),
            )
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_facts(path)


class BaselineOutputTests(unittest.TestCase):
    def test_saves_joinable_per_fact_results_and_accuracy_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "data" / "baseline.csv"
            metrics_path = root / "results" / "metrics.json"
            results = [
                {
                    "id": "1",
                    "question": "Who wrote Hamlet?",
                    "prediction": "William Shakespeare",
                    "possible_answers": ["William Shakespeare"],
                    "baseline_exact_match": True,
                },
                {
                    "id": "2",
                    "question": "What is 2 + 2?",
                    "prediction": "5",
                    "possible_answers": ["4"],
                    "baseline_exact_match": False,
                },
            ]

            metrics = save_results(
                results, "model", output_path, metrics_path
            )

            with output_path.open(encoding="utf-8", newline="") as source:
                saved_rows = list(csv.DictReader(source))
            self.assertEqual([row["id"] for row in saved_rows], ["1", "2"])
            self.assertEqual(
                [row["baseline_exact_match"] for row in saved_rows], ["1", "0"]
            )
            self.assertEqual(json.loads(saved_rows[0]["possible_answers"]), ["William Shakespeare"])
            self.assertEqual(metrics["evaluated_facts"], 2)
            self.assertEqual(metrics["correct_facts"], 1)
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertEqual(json.loads(metrics_path.read_text())["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
