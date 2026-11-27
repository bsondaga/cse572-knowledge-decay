import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import pythia410m_pipeline


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


class Pythia410mPipelineTests(unittest.TestCase):
    def test_runs_full_flow_with_fixed_update_parameters_and_saves_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "facts.csv"
            write_facts(input_path, 2)
            pre_results = root / "pre.csv"
            pre_metrics = root / "pre.json"
            update_dir = root / "updated-model"
            post_results = root / "post.csv"
            post_metrics = root / "post.json"
            labels_path = root / "labels.csv"
            summary_path = root / "summary.json"
            pre_predictions = [prediction("0", True), prediction("1", True)]
            post_predictions = [prediction("0", False), prediction("1", True)]

            with (
                patch.object(
                    pythia410m_pipeline,
                    "evaluate_facts",
                    side_effect=[pre_predictions, post_predictions],
                ) as evaluate,
                patch.object(
                    pythia410m_pipeline,
                    "load_training_texts",
                    return_value=["story"],
                ) as load_texts,
                patch.object(pythia410m_pipeline, "fine_tune") as fine_tune,
            ):
                summary = pythia410m_pipeline.run_pipeline(
                    input_path,
                    pre_results,
                    pre_metrics,
                    update_dir,
                    post_results,
                    post_metrics,
                    labels_path,
                    summary_path,
                    batch_size=1,
                    expected_fact_count=2,
                    sample_size=3,
                    max_length=64,
                    epochs=1,
                    learning_rate=1e-5,
                    seed=7,
                )

            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args_list[0].args[1], "EleutherAI/pythia-410m")
            self.assertEqual(evaluate.call_args_list[1].args[1], str(update_dir))
            load_texts.assert_called_once_with(
                pythia410m_pipeline.DATASET_NAME,
                split="train",
                text_column="text",
                sample_size=3,
                seed=7,
            )
            fine_tune.assert_called_once_with(
                "EleutherAI/pythia-410m",
                ["story"],
                update_dir,
                1,
                64,
                1,
                1e-5,
                7,
            )
            self.assertEqual(summary["known_facts"], 2)
            self.assertEqual(summary["forgotten_facts"], 1)
            self.assertEqual(summary["forgetting_rate"], 0.5)
            self.assertEqual(json.loads(summary_path.read_text())["seed"], 7)
            with labels_path.open(encoding="utf-8", newline="") as source:
                labels = list(csv.DictReader(source))
            self.assertEqual([row["label"] for row in labels], ["forgotten", "retained"])

    def test_rejects_an_incomplete_popqa_input_before_running_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "facts.csv"
            write_facts(input_path, 1)
            with patch.object(pythia410m_pipeline, "evaluate_facts") as evaluate:
                with self.assertRaisesRegex(ValueError, "Expected 2 PopQA facts"):
                    pythia410m_pipeline.run_pipeline(
                        input_path,
                        root / "pre.csv",
                        root / "pre.json",
                        root / "updated",
                        root / "post.csv",
                        root / "post.json",
                        root / "labels.csv",
                        root / "summary.json",
                        expected_fact_count=2,
                    )
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
