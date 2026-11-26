import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.real_label_classifiers import (
    PREDICTOR_COLUMNS,
    FrozenSplitIndices,
    load_frozen_dataset,
    load_frozen_split,
    map_split_indices,
    select_population,
    select_predictor_features,
    validate_frozen_dataset,
)

RELATION_COLUMNS = tuple(column for column in PREDICTOR_COLUMNS if column.startswith("rel_"))


def make_dataset() -> pd.DataFrame:
    ids = [9001, 23, 501, 44, 66]
    guessable = [0, 1, 0, 0, 1]
    rows = []
    for index, fact_id in enumerate(ids):
        row = {
            "id": fact_id,
            "relation": "author",
            "subject_pop_log": 1.0 + index,
            "object_pop_log": 2.0 + index,
            "subject_pop": 100 + index,
            "object_pop": 50 + index,
            "degree": index,
            "guess_count": index,
            "guessable": guessable[index],
            "pop_bucket": index % 4,
            "s_pop_z": float(index),
            "forgotten_lr1e5": index % 2,
            "forgotten_default": (index + 1) % 2,
        }
        row.update(
            {
                column: int(column == RELATION_COLUMNS[index % len(RELATION_COLUMNS)])
                for column in RELATION_COLUMNS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_assignments(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [44, 501, 9001, 66, 23],
            "split": ["test", "val", "train", "val", "train"],
        }
    )


class FrozenDatasetTests(unittest.TestCase):
    def test_loads_valid_dataset_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "real_dataset.csv"
            make_dataset().to_csv(path, index=False)

            dataset = load_frozen_dataset(path)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(set(dataset["forgotten_lr1e5"].unique()), {0, 1})

    def test_selects_known_and_all_populations(self):
        dataset = validate_frozen_dataset(make_dataset())

        known = select_population(dataset, "known")
        all_rows = select_population(dataset, "all")

        self.assertEqual(known["id"].tolist(), [9001, 501, 44])
        self.assertTrue((known["guessable"] == 0).all())
        self.assertEqual(all_rows["id"].tolist(), dataset["id"].tolist())

    def test_selects_only_established_predictors(self):
        dataset = validate_frozen_dataset(make_dataset())

        features = select_predictor_features(dataset)

        self.assertEqual(tuple(features.columns), PREDICTOR_COLUMNS)
        self.assertEqual(features.shape[1], 18)
        self.assertNotIn("id", features.columns)
        self.assertNotIn("relation", features.columns)
        self.assertNotIn("guessable", features.columns)
        self.assertNotIn("forgotten_lr1e5", features.columns)
        self.assertNotIn("forgotten_default", features.columns)
        self.assertNotIn("guess_count", features.columns)
        self.assertNotIn("degree", features.columns)

    def test_duplicate_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[1, "id"] = dataset.loc[0, "id"]

        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            validate_frozen_dataset(dataset)

    def test_missing_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[0, "id"] = np.nan

        with self.assertRaisesRegex(ValueError, "missing fact IDs"):
            validate_frozen_dataset(dataset)

    def test_non_binary_target_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "forgotten_lr1e5"] = 2

        with self.assertRaisesRegex(ValueError, "forgotten_lr1e5.*binary"):
            validate_frozen_dataset(dataset)

    def test_invalid_guessable_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "guessable"] = -1

        with self.assertRaisesRegex(ValueError, "guessable.*binary"):
            validate_frozen_dataset(dataset)

    def test_non_finite_predictor_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "subject_pop_log"] = np.inf

        with self.assertRaisesRegex(ValueError, "subject_pop_log.*finite"):
            validate_frozen_dataset(dataset)

    def test_non_numeric_predictor_fails(self):
        dataset = make_dataset()
        dataset["object_pop_log"] = dataset["object_pop_log"].astype(object)
        dataset.loc[0, "object_pop_log"] = "unknown"

        with self.assertRaisesRegex(ValueError, "object_pop_log.*numeric"):
            validate_frozen_dataset(dataset)

    def test_missing_required_column_fails(self):
        dataset = make_dataset().drop(columns="rel_sport")

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_frozen_dataset(dataset)

    def test_empty_relation_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "relation"] = "  "

        with self.assertRaisesRegex(ValueError, "relation.*non-empty"):
            validate_frozen_dataset(dataset)


class FrozenSplitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_frozen_dataset(make_dataset())
        self.assignments = make_assignments(self.dataset)

    def test_loads_valid_split_and_maps_by_id_not_file_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir) / "all"
            split_dir.mkdir(parents=True)
            self.assignments.to_csv(split_dir / "seed0.csv", index=False)

            indices = load_frozen_split(
                self.dataset, "all", seed=0, splits_dir=Path(temp_dir)
            )

        self.assertEqual(indices.train.tolist(), [0, 1])
        self.assertEqual(indices.validation.tolist(), [2, 4])
        self.assertEqual(indices.test.tolist(), [3])

    def test_partitions_are_disjoint_and_cover_population(self):
        indices = map_split_indices(self.dataset, self.assignments)
        combined = np.concatenate([indices.train, indices.validation, indices.test])

        self.assertEqual(len(np.unique(combined)), len(self.dataset))
        np.testing.assert_array_equal(np.sort(combined), np.arange(len(self.dataset)))
        indices.validate(len(self.dataset))

    def test_duplicate_split_ids_fail(self):
        assignments = pd.concat(
            [self.assignments, self.assignments.iloc[[0]]], ignore_index=True
        )

        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            map_split_indices(self.dataset, assignments)

    def test_missing_split_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "missing .* selected population IDs"):
            map_split_indices(self.dataset, self.assignments.iloc[:-1])

    def test_extra_split_ids_fail(self):
        assignments = pd.concat(
            [
                self.assignments,
                pd.DataFrame({"id": [123456], "split": ["train"]}),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "outside the selected population"):
            map_split_indices(self.dataset, assignments)

    def test_invalid_membership_fails(self):
        assignments = self.assignments.copy()
        assignments.loc[0, "split"] = "validation"

        with self.assertRaisesRegex(ValueError, "invalid membership"):
            map_split_indices(self.dataset, assignments)

    def test_empty_partition_fails(self):
        assignments = self.assignments.copy()
        assignments["split"] = "train"

        with self.assertRaisesRegex(ValueError, "each be non-empty"):
            map_split_indices(self.dataset, assignments)

    def test_rejects_unsupported_population_or_seed(self):
        with self.assertRaisesRegex(ValueError, "population must be one of"):
            load_frozen_split(self.dataset, "unknown", seed=0)
        with self.assertRaisesRegex(ValueError, "seed must be one of"):
            load_frozen_split(self.dataset, "all", seed=3)


if __name__ == "__main__":
    unittest.main()