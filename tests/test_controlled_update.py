import unittest

from src.controlled_update import select_training_texts, validate_training_args


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


if __name__ == "__main__":
    unittest.main()
