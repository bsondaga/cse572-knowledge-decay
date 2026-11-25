import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import controlled_update
from src.controlled_update import (
    evaluate_known_facts,
    load_rehearsal_texts,
    select_training_texts,
    validate_training_args,
)


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


class TrainingDataSelectionTests(unittest.TestCase):
    def test_selects_non_empty_texts_and_stops_at_sample_size(self):
        records = [
            {"text": "  "},
            {"text": " First story. "},
            {"text": 12},
            {"text": "Second story."},
            {"text": "Extra story."},
        ]

        self.assertEqual(
            select_training_texts(records, "text", 2),
            ["First story.", "Second story."],
        )

    def test_reports_when_dataset_has_too_few_valid_examples(self):
        with self.assertRaisesRegex(ValueError, "Requested 2.*found 1"):
            select_training_texts([{"text": "Only story."}], "text", 2)

    def test_reports_missing_text_column_values(self):
        with self.assertRaisesRegex(ValueError, "column 'content'"):
            select_training_texts([{"text": "Story."}], "content", 1)


class TrainingArgumentTests(unittest.TestCase):
    def test_accepts_pilot_training_parameters(self):
        validate_training_args(128, 2, 128, 1, 5e-5)

    def test_rejects_non_positive_values(self):
        for values in (
            (0, 2, 128, 1, 5e-5),
            (1, 0, 128, 1, 5e-5),
            (1, 1, 1, 1, 5e-5),
            (1, 1, 128, 0, 5e-5),
            (1, 1, 128, 1, 0),
            (1, 1, 128, 1, float("nan")),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_training_args(*values)


class RehearsalDataTests(unittest.TestCase):
    def test_loads_50_selected_facts_as_inference_style_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            facts_path = Path(directory) / "facts.csv"
            selection_path = Path(directory) / "selection.csv"
            write_facts(facts_path, 50)
            with selection_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["fact_id"])
                writer.writerows([[str(fact_id)] for fact_id in range(50)])

            texts = load_rehearsal_texts(selection_path, facts_path)

            self.assertEqual(len(texts), 50)
            self.assertEqual(texts[0], "Q: Question 0?\nA: Answer 0")
            self.assertEqual(texts[-1], "Q: Question 49?\nA: Answer 49")

    def test_rejects_selection_ids_not_in_the_facts_file(self):
        with tempfile.TemporaryDirectory() as directory:
            facts_path = Path(directory) / "facts.csv"
            selection_path = Path(directory) / "selection.csv"
            write_facts(facts_path, 1)
            selection_path.write_text("fact_id\nmissing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "absent from"):
                load_rehearsal_texts(selection_path, facts_path)

    def test_main_mixes_dummy_selection_into_update_data(self):
        with tempfile.TemporaryDirectory() as directory:
            facts_path = Path(directory) / "facts.csv"
            selection_path = Path(directory) / "selection.csv"
            write_facts(facts_path, 50)
            with selection_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["fact_id"])
                writer.writerows([[str(fact_id)] for fact_id in range(50)])

            with (
                patch.object(
                    controlled_update,
                    "load_training_texts",
                    return_value=["story"],
                ),
                patch.object(controlled_update, "fine_tune") as fine_tune,
                patch.object(
                    controlled_update,
                    "evaluate_known_facts",
                ) as evaluate_known,
                patch.object(
                    sys,
                    "argv",
                    [
                        "controlled_update.py",
                        "--sample-size",
                        "1",
                        "--rehearsal-file",
                        str(selection_path),
                        "--facts-file",
                        str(facts_path),
                        "--only-known",
                        "--known-facts-count",
                        "50",
                    ],
                ),
            ):
                controlled_update.main()

            self.assertEqual(len(fine_tune.call_args.args[1]), 51)
            evaluate_known.assert_called_once()


class KnownFactEvaluationTests(unittest.TestCase):
    def test_evaluates_only_baseline_correct_facts_and_prints_forgetting_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            pre_path = Path(directory) / "pre.csv"
            with pre_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=(
                        "id",
                        "question",
                        "prediction",
                        "possible_answers",
                        "baseline_exact_match",
                    ),
                )
                writer.writeheader()
                for fact_id, match in (
                    ("known-1", "1"),
                    ("unknown", "0"),
                    ("known-2", "1"),
                ):
                    writer.writerow(
                        {
                            "id": fact_id,
                            "question": f"Question {fact_id}?",
                            "prediction": "answer",
                            "possible_answers": json.dumps(["answer"]),
                            "baseline_exact_match": match,
                        }
                    )

            evaluated = []

            def fake_evaluate(facts, model_name, batch_size, max_new_tokens):
                self.assertEqual(model_name, "updated-model")
                self.assertEqual(batch_size, 2)
                self.assertEqual(max_new_tokens, 16)
                evaluated.extend(facts)
                return [
                    {"baseline_exact_match": True},
                    {"baseline_exact_match": False},
                ]

            with (
                patch.object(controlled_update, "evaluate_facts", side_effect=fake_evaluate),
                patch.object(controlled_update, "save_results") as save_results,
                patch("builtins.print") as print_mock,
            ):
                rate = evaluate_known_facts(
                    "updated-model",
                    pre_path,
                    Path(directory) / "post.csv",
                    Path(directory) / "metrics.json",
                    batch_size=2,
                    expected_count=2,
                )

            self.assertEqual([fact["id"] for fact in evaluated], ["known-1", "known-2"])
            self.assertEqual(rate, 0.5)
            self.assertTrue(
                any(
                    "Forgetting rate: 1/2 = 50.0%" in str(call)
                    for call in print_mock.call_args_list
                )
            )
            save_results.assert_called_once()


if __name__ == "__main__":
    unittest.main()
