import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.logistic_regression_classifier import (
    StratifiedSplit,
    create_stratified_split,
    generate_dummy_labels,
    select_features,
    train_and_evaluate,
    train_lightgbm_and_evaluate,
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


def make_training_data(row_count=800):
    rng = np.random.default_rng(42)
    relations = np.repeat(["author", "country", "director", "genre"], row_count // 4)
    labels = np.tile(np.repeat([0, 1], row_count // 8), 4)
    features = pd.DataFrame(
        {
            "s_pop_log": rng.normal(size=row_count),
            "o_pop_log": rng.normal(size=row_count),
            "rel_author": (relations == "author").astype(int),
            "rel_country": (relations == "country").astype(int),
            "rel_director": (relations == "director").astype(int),
            "rel_genre": (relations == "genre").astype(int),
        }
    )
    return features, labels, relations


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self):
        _, self.labels, self.relations = make_training_data()

    def test_split_is_deterministic_for_same_seed(self):
        first = create_stratified_split(self.labels, self.relations, seed=572)
        second = create_stratified_split(self.labels, self.relations, seed=572)

        for partition in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                getattr(first, partition), getattr(second, partition)
            )

    def test_split_sizes_are_approximately_70_15_15(self):
        split = create_stratified_split(self.labels, self.relations, seed=572)

        np.testing.assert_allclose(
            [len(split.train), len(split.validation), len(split.test)],
            [560, 120, 120],
            atol=1,
        )

    def test_indices_are_disjoint_and_cover_every_row(self):
        split = create_stratified_split(self.labels, self.relations, seed=572)
        partitions = [split.train, split.validation, split.test]
        combined = np.concatenate(partitions)

        self.assertEqual(len(np.unique(combined)), len(combined))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(self.labels)))

    def test_joint_label_and_relation_proportions_are_preserved(self):
        split = create_stratified_split(self.labels, self.relations, seed=572)
        keys = pd.Series(list(zip(self.labels, self.relations)))
        full_proportions = keys.value_counts(normalize=True)

        for indices in (split.train, split.validation, split.test):
            partition_proportions = keys.iloc[indices].value_counts(normalize=True)
            differences = (
                partition_proportions.reindex(full_proportions.index, fill_value=0)
                - full_proportions
            ).abs()
            self.assertLess(float(differences.max()), 0.01)

    def test_undersized_joint_stratum_has_useful_error(self):
        labels = np.array([0, 0, 0, 0, 1])
        relations = np.array(["author", "author", "country", "country", "country"])

        with self.assertRaisesRegex(
            ValueError, r"joint \(label, relation\) strata.*undersized"
        ):
            create_stratified_split(labels, relations, seed=572)

    def test_rejects_non_binary_label_values(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        relations = np.array(["author", "author", "author", "country", "country", "country"])

        with self.assertRaisesRegex(ValueError, "only binary values 0 and 1"):
            create_stratified_split(labels, relations, seed=572)

    def test_split_validator_rejects_overlap_and_incomplete_coverage(self):
        split = create_stratified_split(self.labels, self.relations, seed=572)
        overlapping = StratifiedSplit(
            train=np.concatenate([split.train[:-1], split.validation[:1]]),
            validation=split.validation,
            test=split.test,
        )
        incomplete = StratifiedSplit(
            train=split.train[:-1],
            validation=split.validation,
            test=split.test,
        )

        with self.assertRaisesRegex(ValueError, "must not overlap"):
            overlapping.validate(len(self.labels))
        with self.assertRaisesRegex(ValueError, "cover every input row exactly once"):
            incomplete.validate(len(self.labels))


class ClassifierTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.features, self.labels, self.relations = make_training_data()
        self.split = create_stratified_split(self.labels, self.relations, seed=572)

    def test_logistic_regression_fits_on_shared_split(self):
        results = train_and_evaluate(
            self.features, self.labels, self.split, seed=572
        )

        self.assertIsInstance(results["model"], Pipeline)
        self.assertEqual(results["train_size"], len(self.split.train))
        self.assertEqual(results["validation_size"], len(self.split.validation))
        self.assertEqual(results["test_size"], len(self.split.test))
        np.testing.assert_array_equal(results["test_indices"], self.split.test)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["f1"], 0.0)
        self.assertLessEqual(results["f1"], 1.0)

    def test_lightgbm_fits_on_shared_split(self):
        results = train_lightgbm_and_evaluate(
            self.features, self.labels, self.split, seed=572
        )

        self.assertEqual(results["train_size"], len(self.split.train))
        self.assertEqual(results["validation_size"], len(self.split.validation))
        self.assertEqual(results["test_size"], len(self.split.test))
        np.testing.assert_array_equal(results["test_indices"], self.split.test)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["f1"], 0.0)
        self.assertLessEqual(results["f1"], 1.0)

    def test_both_classifiers_evaluate_the_same_test_rows(self):
        logistic_results = train_and_evaluate(
            self.features, self.labels, self.split, seed=572
        )
        lightgbm_results = train_lightgbm_and_evaluate(
            self.features, self.labels, self.split, seed=572
        )

        np.testing.assert_array_equal(
            logistic_results["test_indices"], lightgbm_results["test_indices"]
        )
        np.testing.assert_array_equal(logistic_results["test_indices"], self.split.test)


if __name__ == "__main__":
    unittest.main()