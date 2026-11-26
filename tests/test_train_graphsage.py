import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data

from src.train_graphsage import (
    GraphLabelMasks,
    GraphSAGEClassifier,
    GraphSAGETrainingConfig,
    TARGET_COLUMN,
    build_graph_label_masks,
    compute_train_pos_weight,
    load_frozen_split_assignments,
    load_graph_label_masks,
    load_known_labels,
    masked_bce_loss,
    predict_graph_probabilities,
    seed_everything,
    train_graphsage,
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


def make_training_graph_and_masks(labels=None) -> tuple[Data, GraphLabelMasks]:
    if labels is None:
        labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    x = torch.zeros((10, 18), dtype=torch.float32)
    x[:, 0] = 1.0
    x[:, 1] = torch.arange(10, dtype=torch.float32) / 10
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 0, 2, 1, 3, 2, 5, 4, 7, 6, 9, 8]],
        dtype=torch.long,
    )
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    labeled = torch.ones(10, dtype=torch.bool)
    train = torch.tensor([True] * 6 + [False] * 4)
    validation = torch.tensor([False] * 6 + [True, True, False, False])
    test = torch.tensor([False] * 8 + [True, True])
    masks = GraphLabelMasks(
        labels=labels_tensor,
        labeled_mask=labeled,
        train_mask=train,
        validation_mask=validation,
        test_mask=test,
        node_ids=tuple(range(10)),
    )
    return Data(x=x, edge_index=edge_index), masks


class ConstantLogitModel(nn.Module):
    def __init__(self, initial_logit=-0.2):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(initial_logit)))
        self.seen_nodes = []
        self.seen_edges = []

    def forward(self, x, edge_index):
        self.seen_nodes.append(x.shape[0])
        self.seen_edges.append(edge_index.shape[1])
        return self.logit.expand(x.shape[0])


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x, edge_index):
        self.seen = (x.shape, edge_index.shape)
        return x[:, :16]


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


class GraphSAGETrainingTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.masks = make_training_graph_and_masks()

    def test_classifier_returns_one_logit_per_full_graph_node(self):
        encoder = RecordingEncoder()
        model = GraphSAGEClassifier(encoder=encoder)

        logits = model(self.graph.x, self.graph.edge_index)

        self.assertEqual(tuple(logits.shape), (10,))
        self.assertEqual(encoder.seen, (self.graph.x.shape, self.graph.edge_index.shape))

    def test_masked_loss_ignores_labels_outside_the_selected_mask(self):
        logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        train_mask = torch.tensor([True, True, False, False])
        criterion = nn.BCEWithLogitsLoss()
        original = masked_bce_loss(logits, labels, train_mask, criterion)
        changed = labels.clone()
        changed[~train_mask] = 1 - changed[~train_mask]

        changed_loss = masked_bce_loss(logits, changed, train_mask, criterion)

        torch.testing.assert_close(original, changed_loss)

    def test_pos_weight_uses_train_labels_only_and_matches_real_ratio(self):
        pos_weight = compute_train_pos_weight(self.masks)
        self.assertAlmostEqual(pos_weight.item(), 2 / 4)
        changed_non_train_labels = self.masks.labels.clone()
        changed_non_train_labels[~self.masks.train_mask] = (
            1 - changed_non_train_labels[~self.masks.train_mask]
        )
        changed_masks = GraphLabelMasks(
            labels=changed_non_train_labels,
            labeled_mask=self.masks.labeled_mask.clone(),
            train_mask=self.masks.train_mask.clone(),
            validation_mask=self.masks.validation_mask.clone(),
            test_mask=self.masks.test_mask.clone(),
            node_ids=self.masks.node_ids,
        )
        torch.testing.assert_close(compute_train_pos_weight(changed_masks), pos_weight)

        real_style = torch.tensor([1.0] * 107 + [0.0] * 94)
        train_mask = torch.tensor([True] * 199 + [False, False])
        validation_mask = torch.zeros(201, dtype=torch.bool)
        validation_mask[199] = True
        test_mask = torch.zeros(201, dtype=torch.bool)
        test_mask[200] = True
        real_masks = GraphLabelMasks(
            labels=real_style,
            labeled_mask=torch.ones(201, dtype=torch.bool),
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            node_ids=tuple(range(201)),
        )
        real_weight = compute_train_pos_weight(real_masks)
        self.assertAlmostEqual(real_weight.item(), 92 / 107, places=6)

    def test_train_requires_both_target_classes(self):
        graph, masks = make_training_graph_and_masks([1] * 10)
        with self.assertRaisesRegex(ValueError, "both forgotten and retained"):
            compute_train_pos_weight(masks)
        with self.assertRaisesRegex(ValueError, "both forgotten and retained"):
            train_graphsage(graph, masks, seed=0, config=GraphSAGETrainingConfig(max_epochs=1))

    def test_validation_loss_uses_validation_mask_only(self):
        logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
        validation_mask = torch.tensor([False, False, True, True])
        criterion = nn.BCEWithLogitsLoss()
        original = masked_bce_loss(logits, labels, validation_mask, criterion)
        changed = labels.clone()
        changed[:2] = 1 - changed[:2]

        changed_loss = masked_bce_loss(logits, changed, validation_mask, criterion)

        torch.testing.assert_close(original, changed_loss)

    def test_training_uses_full_graph_and_does_not_consult_test_labels(self):
        config = GraphSAGETrainingConfig(max_epochs=4, patience=4)
        model = ConstantLogitModel()
        result = train_graphsage(self.graph, self.masks, seed=11, config=config, model=model)
        self.assertTrue(all(node_count == self.graph.num_nodes for node_count in model.seen_nodes))
        self.assertTrue(all(edge_count == self.graph.edge_index.shape[1] for edge_count in model.seen_edges))

        changed_labels = self.masks.labels.clone()
        changed_labels[self.masks.test_mask] = 1 - changed_labels[self.masks.test_mask]
        changed_masks = GraphLabelMasks(
            labels=changed_labels,
            labeled_mask=self.masks.labeled_mask.clone(),
            train_mask=self.masks.train_mask.clone(),
            validation_mask=self.masks.validation_mask.clone(),
            test_mask=self.masks.test_mask.clone(),
            node_ids=self.masks.node_ids,
        )
        repeated = train_graphsage(
            self.graph,
            changed_masks,
            seed=11,
            config=config,
            model=ConstantLogitModel(),
        )
        self.assertEqual(result.best_epoch, repeated.best_epoch)
        self.assertEqual(result.best_validation_loss, repeated.best_validation_loss)
        self.assertEqual(result.loss_history, repeated.loss_history)

    def test_early_stopping_records_history_and_restores_best_weights(self):
        early_config = GraphSAGETrainingConfig(
            max_epochs=10, patience=2, min_delta=1e-6, learning_rate=0.1, weight_decay=0
        )
        early = train_graphsage(
            self.graph,
            self.masks,
            seed=5,
            config=early_config,
            model=ConstantLogitModel(),
        )
        first_epoch = train_graphsage(
            self.graph,
            self.masks,
            seed=5,
            config=GraphSAGETrainingConfig(
                max_epochs=1, patience=2, min_delta=1e-6, learning_rate=0.1, weight_decay=0
            ),
            model=ConstantLogitModel(),
        )

        self.assertTrue(early.stopped_early)
        self.assertLess(len(early.loss_history), early_config.max_epochs)
        self.assertEqual(early.best_epoch, 1)
        self.assertAlmostEqual(early.model.logit.item(), first_epoch.model.logit.item(), places=7)
        self.assertTrue(any(row["is_best"] for row in early.loss_history))
        self.assertTrue(
            all(set(row) == {"epoch", "train_loss", "validation_loss", "is_best"}
                for row in early.loss_history)
        )

    def test_seeded_training_is_deterministic(self):
        config = GraphSAGETrainingConfig(max_epochs=5, patience=5)
        first = train_graphsage(self.graph, self.masks, seed=37, config=config)
        second = train_graphsage(self.graph, self.masks, seed=37, config=config)

        self.assertEqual(first.best_epoch, second.best_epoch)
        self.assertEqual(first.loss_history, second.loss_history)
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[name])

    def test_prediction_returns_finite_probabilities_and_applies_mask(self):
        model = ConstantLogitModel()
        all_probabilities = predict_graph_probabilities(model, self.graph)
        test_probabilities = predict_graph_probabilities(
            model, self.graph, self.masks.test_mask
        )

        self.assertEqual(tuple(all_probabilities.shape), (self.graph.num_nodes,))
        self.assertEqual(len(test_probabilities), int(self.masks.test_mask.sum()))
        self.assertTrue(torch.isfinite(all_probabilities).all())
        self.assertTrue(torch.all((all_probabilities >= 0) & (all_probabilities <= 1)))


if __name__ == "__main__":
    unittest.main()