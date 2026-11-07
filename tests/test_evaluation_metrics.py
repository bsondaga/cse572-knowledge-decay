import unittest

import numpy as np

from src.evaluation_metrics import (
    compute_auprc,
    compute_f1,
    evaluate_predictions,
    recall_at_k,
)


class HandCheckedMetricTests(unittest.TestCase):
    def test_hand_checked_four_row_example(self):
        y_true = [1, 0, 1, 0]
        y_score = [0.9, 0.8, 0.3, 0.1]

        self.assertAlmostEqual(compute_auprc(y_true, y_score), 5 / 6)
        self.assertEqual(compute_f1(y_true, y_score, threshold=0.5), 0.5)
        self.assertEqual(recall_at_k(y_true, y_score, k=2), 0.5)

    def test_perfect_ranking(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]

        self.assertEqual(compute_auprc(y_true, y_score), 1.0)
        self.assertEqual(compute_f1(y_true, y_score), 1.0)
        self.assertEqual(recall_at_k(y_true, y_score, k=2), 1.0)

    def test_reversed_ranking(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.1, 0.2, 0.8, 0.9]

        self.assertAlmostEqual(compute_auprc(y_true, y_score), 5 / 12)
        self.assertEqual(recall_at_k(y_true, y_score, k=2), 0.0)

    def test_threshold_boundary_is_predicted_positive(self):
        self.assertEqual(compute_f1([1, 0], [0.5, 0.5], threshold=0.5), 2 / 3)

    def test_recall_at_one_and_full_dataset(self):
        y_true = [1, 0, 1, 0]
        y_score = [0.9, 0.8, 0.3, 0.1]

        self.assertEqual(recall_at_k(y_true, y_score, k=1), 0.5)
        self.assertEqual(recall_at_k(y_true, y_score, k=4), 1.0)

    def test_k_greater_than_dataset_size_clamps_to_full_dataset(self):
        self.assertEqual(recall_at_k([1, 0], [0.2, 0.8], k=10), 1.0)

    def test_tied_scores_preserve_input_order(self):
        y_true = [0, 1, 0, 1]
        y_score = [0.8, 0.8, 0.4, 0.4]

        self.assertEqual(recall_at_k(y_true, y_score, k=1), 0.0)
        self.assertEqual(recall_at_k(y_true, y_score, k=2), 0.5)

    def test_combined_evaluator_structure_and_values(self):
        results = evaluate_predictions(
            [1, 0, 1, 0],
            [0.9, 0.8, 0.3, 0.1],
            threshold=0.5,
            k_values=(1, 2, 10),
        )

        self.assertEqual(set(results), {"auprc", "f1", "threshold", "recall_at_k"})
        self.assertAlmostEqual(results["auprc"], 5 / 6)
        self.assertEqual(results["f1"], 0.5)
        self.assertEqual(results["threshold"], 0.5)
        self.assertEqual(results["recall_at_k"], {1: 0.5, 2: 0.5, 10: 1.0})


class MetricValidationTests(unittest.TestCase):
    def test_mismatched_lengths_fail(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            compute_auprc([1, 0], [0.5])

    def test_empty_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_f1([], [])

    def test_non_one_dimensional_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            compute_auprc([[1, 0]], [[0.8, 0.2]])

    def test_non_binary_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            compute_f1([0, 2], [0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            compute_f1(["1", "0"], [0.1, 0.9])

    def test_non_numeric_scores_fail(self):
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            compute_auprc([1, 0], ["0.9", "0.1"])

    def test_non_finite_scores_fail(self):
        for bad_score in (np.nan, np.inf, -np.inf):
            with self.subTest(score=bad_score):
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    compute_auprc([1, 0], [bad_score, 0.2])

    def test_scores_outside_probability_range_fail(self):
        for bad_score in (-0.01, 1.01):
            with self.subTest(score=bad_score):
                with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
                    compute_f1([1, 0], [bad_score, 0.2])

    def test_invalid_thresholds_fail(self):
        for threshold in (-0.1, 1.1, np.nan, np.inf, "bad", True):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    compute_f1([1, 0], [0.8, 0.2], threshold=threshold)

    def test_non_positive_k_fails(self):
        for k in (0, -1):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    recall_at_k([1, 0], [0.8, 0.2], k=k)

    def test_non_integer_k_fails(self):
        for k in (1.5, "2", True):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    recall_at_k([1, 0], [0.8, 0.2], k=k)

    def test_no_positive_behavior(self):
        labels = [0, 0]
        scores = [0.8, 0.2]

        self.assertEqual(compute_f1(labels, scores), 0.0)
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            compute_auprc(labels, scores)
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            recall_at_k(labels, scores, 1)
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            evaluate_predictions(labels, scores)


if __name__ == "__main__":
    unittest.main()