import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.logistic_regression_classifier import (
    generate_dummy_labels,
    select_features,
    train_and_evaluate,
)


class FeatureSelectionTests(unittest.TestCase):
    def test_selects_only_log_popularity_and_relation_features(self):
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "prop": ["author", "country"],
                "s_pop": [10, 20],
                "o_pop": [30, 40],
                "s_pop_log": [1.0, 1.3],
                "o_pop_log": [1.5, 1.6],
                "s_pop_z": [0.1, 0.2],
                "pop_bucket": [0, 1],
                "pop_bucket_0": [1, 0],
                "rel_author": [1, 0],
                "rel_country": [0, 1],
            }
        )

        features = select_features(df)

        self.assertEqual(
            list(features.columns),
            ["s_pop_log", "o_pop_log", "rel_author", "rel_country"],
        )


class DummyLabelTests(unittest.TestCase):
    def test_generation_is_deterministic_for_seed(self):
        first = generate_dummy_labels(1000, seed=572)
        second = generate_dummy_labels(1000, seed=572)

        np.testing.assert_array_equal(first, second)

    def test_labels_are_binary_and_about_one_quarter_forgotten(self):
        labels = generate_dummy_labels(4000, seed=572)

        self.assertEqual(set(np.unique(labels)), {0, 1})
        self.assertAlmostEqual(float(labels.mean()), 0.25, delta=0.025)


class LogisticRegressionTrainingTests(unittest.TestCase):
    def test_fits_and_evaluates_on_synthetic_feature_rows(self):
        rng = np.random.default_rng(42)
        row_count = 200
        features = pd.DataFrame(
            {
                "s_pop_log": rng.normal(size=row_count),
                "o_pop_log": rng.normal(size=row_count),
                "rel_author": rng.integers(0, 2, size=row_count),
                "rel_country": rng.integers(0, 2, size=row_count),
            }
        )
        labels = generate_dummy_labels(row_count, seed=572)

        results = train_and_evaluate(features, labels, seed=572, test_size=0.2)

        self.assertIsInstance(results["model"], Pipeline)
        self.assertEqual(results["train_size"], 160)
        self.assertEqual(results["test_size"], 40)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["f1"], 0.0)
        self.assertLessEqual(results["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()