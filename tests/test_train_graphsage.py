import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.train_graphsage import (
    TARGET_COLUMN,
    build_graph_label_masks,
    load_frozen_split_assignments,
    load_graph_label_masks,
    load_known_labels,
    validate_split_assignments,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [9001, 23, 501, 44, 66, 77],
            "guessable": [0, 1, 0, 0, 1, 0],
            "forgotten_lr1e5": [1, 1, 0, 1, 0, 0],
            "forgotten_default": [0, 0, 1, 0, 1, 1],
        }
    )


def make_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [77, 9001, 44, 501],
            "split": ["test", "train", "val", "train"],
        }
    )


def make_graph_id_map() -> dict[int, int]:
    return {501: 0, 9001: 1, 66: 2, 77: 3, 23: 4, 44: 5}


class KnownLabelLoaderTests(unittest.TestCase):
    def test_filters_known_population_and_uses_lr1e5_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "real_dataset.csv"
            make_dataset().to_csv(path, index=False)

            known = load_known_labels(path)

        self.assertEqual(known["id"].tolist(), [9001, 501, 44, 77])
        self.assertEqual(known[TARGET_COLUMN].tolist(), [1, 0, 1, 0])
        self.assertNotIn("forgotten_default", known.columns)

    def test_duplicate_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[1, "id"] = dataset.loc[0, "id"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.csv"
            dataset.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
                load_known_labels(path)

    def test_missing_dataset_ids_fail(self):
        dataset = make_dataset()
        dataset.loc[0, "id"] = np.nan
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.csv"
            dataset.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing fact IDs"):
                load_known_labels(path)

    def test_non_binary_target_fails(self):
        dataset = make_dataset()
        dataset.loc[0, "forgotten_lr1e5"] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.csv"
            dataset.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "forgotten_lr1e5.*binary"):
                load_known_labels(path)


class FrozenMaskTests(unittest.TestCase):
    def setUp(self):
        self.dataset = make_dataset()
        self.known_labels = self.dataset.loc[
            self.dataset.guessable == 0, ["id", TARGET_COLUMN]
        ].copy()
        self.assignments = make_assignments()
        self.id_to_index = make_graph_id_map()

    def test_mask_mapping_uses_fact_ids_not_file_order(self):
        masks = build_graph_label_masks(
            self.id_to_index, self.known_labels, self.assignments, num_nodes=6
        )

        self.assertEqual(masks.train_mask.nonzero().flatten().tolist(), [0, 1])
        self.assertEqual(masks.validation_mask.nonzero().flatten().tolist(), [5])
        self.assertEqual(masks.test_mask.nonzero().flatten().tolist(), [3])
        self.assertEqual(masks.node_ids, (501, 9001, 66, 77, 23, 44))
        self.assertEqual(masks.labels[1].item(), 1.0)
        self.assertEqual(masks.labels[0].item(), 0.0)

    def test_unlabeled_nodes_remain_outside_all_supervised_masks(self):
        masks = build_graph_label_masks(
            self.id_to_index, self.known_labels, self.assignments, num_nodes=6
        )
        unlabeled = ~masks.labeled_mask

        self.assertEqual(int(masks.labeled_mask.sum()), 4)
        self.assertEqual(torch.where(unlabeled)[0].tolist(), [2, 4])
        self.assertFalse(torch.any(masks.train_mask & unlabeled))
        self.assertFalse(torch.any(masks.validation_mask & unlabeled))
        self.assertFalse(torch.any(masks.test_mask & unlabeled))
        self.assertEqual(masks.labels[2].item(), 0.0)
        self.assertFalse(masks.labeled_mask[2])

    def test_masks_have_expected_types_pairwise_disjointness_and_coverage(self):
        masks = build_graph_label_masks(
            self.id_to_index, self.known_labels, self.assignments, num_nodes=6
        )

        self.assertEqual(masks.labels.dtype, torch.float32)
        for mask in (
            masks.labeled_mask,
            masks.train_mask,
            masks.validation_mask,
            masks.test_mask,
        ):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tuple(mask.shape), (6,))
        self.assertFalse(torch.any(masks.train_mask & masks.validation_mask))
        self.assertFalse(torch.any(masks.train_mask & masks.test_mask))
        self.assertFalse(torch.any(masks.validation_mask & masks.test_mask))
        self.assertTrue(
            torch.equal(
                masks.train_mask | masks.validation_mask | masks.test_mask,
                masks.labeled_mask,
            )
        )

    def test_unknown_labeled_graph_id_fails(self):
        labels = pd.concat(
            [self.known_labels, pd.DataFrame({"id": [99999], TARGET_COLUMN: [0]})],
            ignore_index=True,
        )
        assignments = pd.concat(
            [
                self.assignments,
                pd.DataFrame({"id": [99999], "split": ["test"]}),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "not present in the graph"):
            build_graph_label_masks(self.id_to_index, labels, assignments, num_nodes=6)

    def test_duplicate_known_ids_fail(self):
        labels = pd.concat([self.known_labels, self.known_labels.iloc[[0]]], ignore_index=True)
        assignments = pd.concat([self.assignments, self.assignments.iloc[[1]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            build_graph_label_masks(self.id_to_index, labels, assignments, num_nodes=6)

    def test_duplicate_split_ids_fail(self):
        assignments = pd.concat([self.assignments, self.assignments.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            validate_split_assignments(assignments, self.known_labels.id)

    def test_missing_split_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "missing .* known fact IDs"):
            validate_split_assignments(self.assignments.iloc[:-1], self.known_labels.id)

    def test_extra_split_ids_fail(self):
        assignments = pd.concat(
            [self.assignments, pd.DataFrame({"id": [999], "split": ["train"]})],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "outside the known population"):
            validate_split_assignments(assignments, self.known_labels.id)

    def test_invalid_split_membership_fails(self):
        assignments = self.assignments.copy()
        assignments.loc[0, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "invalid membership"):
            validate_split_assignments(assignments, self.known_labels.id)

    def test_unsupported_seed_fails(self):
        with self.assertRaisesRegex(ValueError, "seed must be one of"):
            load_frozen_split_assignments(3, self.known_labels.id)

    def test_loader_reads_authoritative_seed_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            known_dir = Path(temp_dir) / "known"
            known_dir.mkdir()
            self.assignments.to_csv(known_dir / "seed0.csv", index=False)

            loaded = load_frozen_split_assignments(
                0, self.known_labels.id, splits_dir=Path(temp_dir)
            )

        pd.testing.assert_frame_equal(
            loaded, self.assignments[["id", "split"]].reset_index(drop=True)
        )

    def test_real_style_285_node_population_counts(self):
        graph_ids = list(range(5000, 18059))
        fact_ids = graph_ids[:285]
        id_to_index = {fact_id: index for index, fact_id in enumerate(graph_ids)}
        labels = pd.DataFrame(
            {
                "id": fact_ids,
                TARGET_COLUMN: [1] * 153 + [0] * 132,
            }
        )
        split_names = ["train"] * 199 + ["val"] * 43 + ["test"] * 43
        assignments = pd.DataFrame({"id": fact_ids, "split": split_names})

        masks = build_graph_label_masks(
            id_to_index, labels, assignments, num_nodes=13059
        )

        self.assertEqual(int(masks.labeled_mask.sum()), 285)
        self.assertEqual(int(masks.train_mask.sum()), 199)
        self.assertEqual(int(masks.validation_mask.sum()), 43)
        self.assertEqual(int(masks.test_mask.sum()), 43)
        self.assertEqual(int((~masks.labeled_mask).sum()), 13059 - 285)

    def test_real_seed_loader_maps_full_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "real_dataset.csv"
            self.dataset.to_csv(path, index=False)
            split_dir = Path(temp_dir) / "splits" / "known"
            split_dir.mkdir(parents=True)
            self.assignments.to_csv(split_dir / "seed0.csv", index=False)

            masks = load_graph_label_masks(
                self.id_to_index,
                num_nodes=6,
                seed=0,
                labels_path=path,
                splits_dir=Path(temp_dir) / "splits",
            )

        self.assertEqual(int(masks.labeled_mask.sum()), 4)
        self.assertEqual(int(masks.train_mask.sum()), 2)
        self.assertEqual(int(masks.validation_mask.sum()), 1)
        self.assertEqual(int(masks.test_mask.sum()), 1)


if __name__ == "__main__":
    unittest.main()