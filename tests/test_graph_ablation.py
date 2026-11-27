import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data

from src.graph_ablation import (
    ABLATION_METRIC_COLUMNS,
    EdgeShuffleMetadata,
    NeighborLightGBMResult,
    NEIGHBOR_FEATURE_COLUMNS,
    PAIRED_DIFFERENCE_COLUMNS,
    PAIRED_SUMMARY_COLUMNS,
    ShuffledGraphResult,
    TASK13_METRICS_COLUMNS,
    TASK13_PAIRED_COLUMNS,
    TASK13_SCORE_COLUMNS,
    TASK13_CONDITIONS,
    TASK8_SCORE_INPUT_COLUMNS,
    build_neighbor_features,
    build_neighbor_predictor_table,
    evaluate_neighbor_lightgbm_seed,
    evaluate_risk_scores,
    load_task5_lightgbm_baseline,
    load_task8_baseline,
    load_task8_test_scores_from_frame,
    make_no_edge_graph,
    make_shuffled_edge_graph,
    paired_metric_differences,
    run_task13_experiment,
    summarize_condition_metrics,
    write_task13_outputs,
)
from src.real_label_classifiers import METRIC_NAMES, PREDICTOR_COLUMNS
from src.train_graphsage import (
    GraphLabelMasks,
    GraphSAGEClassifier,
    GraphSAGETrainingResult,
    build_graph_label_masks,
)


def make_graph_and_edges():
    node_ids = [901, 104, 3307, 52, 8800, 711, 640, 4500]
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    undirected_edges = [
        (104, 901), (3307, 104), (52, 3307), (901, 52),
        (711, 8800), (640, 711), (4500, 640), (8800, 4500),
        (901, 8800), (104, 711), (3307, 640), (52, 4500),
    ]
    rows = [(min(a, b), max(a, b), "head") for a, b in undirected_edges]
    edges = pd.DataFrame(rows, columns=["src_id", "dst_id", "edge_type"])
    directed = []
    for source_id, destination_id, _ in rows:
        source, destination = id_to_index[source_id], id_to_index[destination_id]
        directed.extend([(source, destination), (destination, source)])
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    graph = Data(
        x=torch.arange(8 * 18, dtype=torch.float32).reshape(8, 18),
        edge_index=edge_index,
    )
    return graph, edges, id_to_index


def degrees(edges, node_ids):
    degree = {node_id: 0 for node_id in node_ids}
    for row in edges.itertuples(index=False):
        degree[row.src_id] += 1
        degree[row.dst_id] += 1
    return degree


def make_real_label_dataset(row_count=60):
    rows = []
    relation_columns = [
        column for column in PREDICTOR_COLUMNS if column.startswith("rel_")
    ]
    for index in range(row_count):
        row = {
            "id": 50000 + 19 * index,
            "relation": "author" if index % 2 else "country",
            "subject_pop_log": float(index % 13),
            "object_pop_log": float(index % 7),
            "guessable": 0,
            "forgotten_lr1e5": index % 2,
            "forgotten_default": (index + 1) % 2,
        }
        row.update(
            {
                column: int(column == relation_columns[index % len(relation_columns)])
                for column in relation_columns
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_neighbor_rows(ids):
    return pd.DataFrame(
        {
            "id": ids,
            "neighbor_count": np.arange(len(ids), dtype=int),
            "neighbor_mean_subject_pop_log": np.arange(len(ids), dtype=float) + 10,
            "neighbor_mean_object_pop_log": np.arange(len(ids), dtype=float) + 20,
        }
    )


def make_metric_rows():
    rows = []
    for condition, values in (
        ("real_graph", (0.2, 0.3, 0.4)),
        ("no_edge", (0.3, 0.5, 0.7)),
        ("shuffled_edges", (0.1, 0.2, 0.3)),
        ("task5_lightgbm_full", (0.4, 0.5, 0.6)),
        ("neighbor_lightgbm", (0.5, 0.7, 0.9)),
    ):
        for seed, value in reversed(list(enumerate(values))):
            rows.append(
                {
                    "population": "known",
                    "condition": condition,
                    "seed": seed,
                    **{metric: value for metric in METRIC_NAMES},
                }
            )
    return pd.DataFrame(rows, columns=ABLATION_METRIC_COLUMNS)


def make_task13_fixtures():
    graph, stored_edges, id_to_index = make_graph_and_edges()
    ids = list(id_to_index)
    dataset_rows = []
    relation_columns = [
        column for column in PREDICTOR_COLUMNS if column.startswith("rel_")
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    for index, fact_id in enumerate(ids):
        row = {
            "id": fact_id,
            "relation": "author" if index % 2 else "country",
            "subject_pop_log": float(index),
            "object_pop_log": float(index % 3),
            "guessable": 0,
            "forgotten_lr1e5": labels[index],
            "forgotten_default": 1 - labels[index],
        }
        row.update(
            {
                column: int(column == relation_columns[index % len(relation_columns)])
                for column in relation_columns
            }
        )
        dataset_rows.append(row)
    real_dataset = pd.DataFrame(dataset_rows)

    split_indices = {
        0: ([0, 1, 2, 3], [4, 5], [6, 7]),
        1: ([2, 3, 4, 5], [6, 7], [0, 1]),
        2: ([4, 5, 6, 7], [0, 1], [2, 3]),
    }
    masks_by_seed = {}
    for seed, (train, validation, test) in split_indices.items():
        names = [None] * len(ids)
        for index in train:
            names[index] = "train"
        for index in validation:
            names[index] = "val"
        for index in test:
            names[index] = "test"
        assignments = pd.DataFrame({"id": ids, "split": names})
        masks_by_seed[seed] = build_graph_label_masks(
            id_to_index,
            real_dataset[["id", "forgotten_lr1e5"]],
            assignments,
            num_nodes=len(ids),
        )

    task8_metrics = pd.DataFrame(
        [
            {
                "population": "known",
                "condition": "real_graph",
                "seed": seed,
                **{metric: 0.5 for metric in METRIC_NAMES},
            }
            for seed in range(3)
        ]
    )
    task5_metrics = pd.DataFrame(
        [
            {
                "population": "known",
                "condition": "task5_lightgbm_full",
                "seed": seed,
                **{metric: 0.4 for metric in METRIC_NAMES},
            }
            for seed in range(3)
        ]
    )
    task8_scores = pd.DataFrame(
        [
            {
                "population": "known",
                "seed": seed,
                "id": masks.node_ids[index],
                "split": "test",
                "true_label": int(masks.labels[index]),
                "forgetting_risk": 0.5,
            }
            for seed, masks in masks_by_seed.items()
            for index in torch.where(masks.test_mask)[0].tolist()
        ],
        columns=TASK8_SCORE_INPUT_COLUMNS,
    )
    graph_features = pd.DataFrame(
        {
            "id": ids,
            "s_pop_log": np.arange(len(ids), dtype=float),
            "o_pop_log": np.arange(len(ids), dtype=float) / 2,
        }
    )
    neighbor_features = build_neighbor_features(graph_features, stored_edges)
    return (
        graph,
        stored_edges,
        id_to_index,
        real_dataset,
        masks_by_seed,
        task8_metrics,
        task8_scores,
        task5_metrics,
        neighbor_features,
    )


class StubLogitModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x, edge_index):
        return self.value.expand(x.shape[0])


class NoEdgeGraphTests(unittest.TestCase):
    def test_empty_edges_preserve_features_nodes_and_original_graph(self):
        graph, _, _ = make_graph_and_edges()
        original_x = graph.x.clone()
        original_edge_index = graph.edge_index.clone()

        no_edge = make_no_edge_graph(graph)

        self.assertEqual(tuple(no_edge.edge_index.shape), (2, 0))
        self.assertEqual(no_edge.edge_index.dtype, torch.long)
        self.assertEqual(no_edge.num_nodes, graph.num_nodes)
        torch.testing.assert_close(no_edge.x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(graph.x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(graph.edge_index, original_edge_index, rtol=0, atol=0)
        self.assertIsNot(no_edge, graph)
        self.assertIsNot(no_edge.x, graph.x)

    def test_existing_graphsage_classifier_forwards_on_no_edge_graph(self):
        graph, _, _ = make_graph_and_edges()
        no_edge = make_no_edge_graph(graph)
        model = GraphSAGEClassifier()

        logits = model(no_edge.x, no_edge.edge_index)

        self.assertEqual(tuple(logits.shape), (graph.num_nodes,))
        self.assertTrue(torch.isfinite(logits).all())


class ShuffledEdgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.edges, self.id_to_index = make_graph_and_edges()
        self.node_ids = list(self.id_to_index)

    def shuffle(self, seed):
        return make_shuffled_edge_graph(
            self.graph,
            self.edges,
            self.id_to_index,
            seed=seed,
            target_swaps=8,
            max_attempts=5000,
        )

    def test_same_seed_is_deterministic_and_different_seeds_change_edges(self):
        first = self.shuffle(seed=11)
        repeated = self.shuffle(seed=11)
        another = self.shuffle(seed=12)

        pd.testing.assert_frame_equal(first.stored_edges, repeated.stored_edges)
        self.assertNotEqual(
            set(map(tuple, first.stored_edges.to_numpy())),
            set(map(tuple, another.stored_edges.to_numpy())),
        )

    def test_preserves_edge_count_degrees_and_directed_count(self):
        result = self.shuffle(seed=11)

        self.assertEqual(result.metadata.requested_swaps, 8)
        self.assertEqual(result.metadata.successful_swaps, 8)
        self.assertEqual(result.metadata.original_edge_count, len(self.edges))
        self.assertEqual(result.metadata.shuffled_edge_count, len(self.edges))
        self.assertEqual(len(result.stored_edges), len(self.edges))
        self.assertEqual(result.graph.edge_index.shape[1], 2 * len(self.edges))
        self.assertEqual(degrees(result.stored_edges, self.node_ids), degrees(self.edges, self.node_ids))

    def test_no_self_loops_duplicates_and_pairs_are_canonical(self):
        result = self.shuffle(seed=11)
        shuffled = result.stored_edges

        self.assertFalse((shuffled.src_id == shuffled.dst_id).any())
        self.assertTrue((shuffled.src_id < shuffled.dst_id).all())
        self.assertFalse(shuffled.duplicated(["src_id", "dst_id"]).any())
        self.assertEqual(
            len(set(map(tuple, result.graph.edge_index.t().tolist()))),
            result.graph.edge_index.shape[1],
        )

    def test_sparse_ids_map_to_contiguous_directed_indices(self):
        result = self.shuffle(seed=11)
        directed = set(map(tuple, result.graph.edge_index.t().tolist()))
        for source_id, destination_id in result.stored_edges.itertuples(index=False, name=None):
            source_index = self.id_to_index[source_id]
            destination_index = self.id_to_index[destination_id]
            self.assertIn((source_index, destination_index), directed)
            self.assertIn((destination_index, source_index), directed)
        self.assertTrue(
            all(0 <= node_index < self.graph.num_nodes for edge in directed for node_index in edge)
        )

    def test_original_graph_and_input_dataframe_are_unchanged(self):
        original_edges = self.edges.copy(deep=True)
        original_x = self.graph.x.clone()
        original_edge_index = self.graph.edge_index.clone()

        result = self.shuffle(seed=11)

        pd.testing.assert_frame_equal(self.edges, original_edges)
        torch.testing.assert_close(self.graph.x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(self.graph.edge_index, original_edge_index, rtol=0, atol=0)
        self.assertIsNot(result.graph, self.graph)

    def test_failed_attempt_budget_reports_achieved_and_requested_swaps(self):
        with self.assertRaisesRegex(RuntimeError, r"achieved 0/1 successful swaps"):
            make_shuffled_edge_graph(
                self.graph,
                self.edges,
                self.id_to_index,
                seed=1,
                target_swaps=1,
                max_attempts=0,
            )

    def test_shuffle_api_accepts_no_label_arguments(self):
        result = self.shuffle(seed=5)
        self.assertEqual(result.metadata.successful_swaps, 8)


class NeighborFeatureTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            {
                "id": [30, 10, 40, 20],
                "s_pop_log": [6.0, 2.0, 100.0, 4.0],
                "o_pop_log": [5.0, 1.0, 9.0, 3.0],
            }
        )
        self.edges = pd.DataFrame(
            {"src_id": [10, 10], "dst_id": [20, 30], "edge_type": ["head", "tail"]}
        )

    def test_hand_calculates_neighbor_counts_means_and_isolate_zeros(self):
        result = build_neighbor_features(self.features, self.edges).set_index("id")

        self.assertEqual(result.loc[10, "neighbor_count"], 2)
        self.assertEqual(result.loc[10, "neighbor_mean_subject_pop_log"], 5.0)
        self.assertEqual(result.loc[10, "neighbor_mean_object_pop_log"], 4.0)
        self.assertEqual(result.loc[20, "neighbor_count"], 1)
        self.assertEqual(result.loc[20, "neighbor_mean_subject_pop_log"], 2.0)
        self.assertEqual(result.loc[20, "neighbor_mean_object_pop_log"], 1.0)
        self.assertEqual(result.loc[30, "neighbor_count"], 1)
        self.assertEqual(result.loc[30, "neighbor_mean_subject_pop_log"], 2.0)
        self.assertEqual(result.loc[30, "neighbor_mean_object_pop_log"], 1.0)
        self.assertEqual(result.loc[40, "neighbor_count"], 0)
        self.assertEqual(result.loc[40, "neighbor_mean_subject_pop_log"], 0.0)
        self.assertEqual(result.loc[40, "neighbor_mean_object_pop_log"], 0.0)

    def test_preserves_input_order_and_does_not_mutate_inputs(self):
        original_features = self.features.copy(deep=True)
        original_edges = self.edges.copy(deep=True)

        result = build_neighbor_features(self.features, self.edges)

        self.assertEqual(result.id.tolist(), self.features.id.tolist())
        pd.testing.assert_frame_equal(self.features, original_features)
        pd.testing.assert_frame_equal(self.edges, original_edges)
        self.assertEqual(tuple(result.columns[-3:]), NEIGHBOR_FEATURE_COLUMNS)
        self.assertTrue(np.isfinite(result[list(NEIGHBOR_FEATURE_COLUMNS)]).to_numpy().all())

    def test_rejects_unknown_endpoints_and_non_finite_source_features(self):
        unknown = pd.DataFrame({"src_id": [10], "dst_id": [9999]})
        with self.assertRaisesRegex(ValueError, "not present in id_to_index"):
            build_neighbor_features(self.features, unknown)

        bad_features = self.features.copy()
        bad_features.loc[0, "s_pop_log"] = np.inf
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            build_neighbor_features(bad_features, self.edges)

    def test_rejects_duplicate_ids_and_self_loops(self):
        duplicate_ids = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            build_neighbor_features(duplicate_ids, self.edges)
        loop = pd.DataFrame({"src_id": [10], "dst_id": [10]})
        with self.assertRaisesRegex(ValueError, "self-loop"):
            build_neighbor_features(self.features, loop)


class NeighborPredictorTests(unittest.TestCase):
    def test_join_by_id_has_exactly_21_predictors_and_no_target_columns(self):
        dataset = make_real_label_dataset(12)
        neighbor = make_neighbor_rows(dataset.id.tolist()[::-1])

        known, predictors = build_neighbor_predictor_table(dataset, neighbor)

        self.assertEqual(known.id.tolist(), dataset.id.tolist())
        self.assertEqual(predictors.shape, (12, 21))
        self.assertEqual(
            tuple(predictors.columns), (*PREDICTOR_COLUMNS, *NEIGHBOR_FEATURE_COLUMNS)
        )
        self.assertFalse(
            set(predictors.columns)
            & {"id", "relation", "guessable", "forgotten_lr1e5", "forgotten_default"}
        )
        self.assertEqual(predictors.iloc[0]["neighbor_mean_subject_pop_log"], 21)

    def test_missing_or_duplicate_neighbor_ids_fail(self):
        dataset = make_real_label_dataset(12)
        with self.assertRaisesRegex(ValueError, "missing 1 known IDs"):
            build_neighbor_predictor_table(dataset, make_neighbor_rows(dataset.id.iloc[1:].tolist()))
        duplicate = pd.concat(
            [make_neighbor_rows(dataset.id.tolist()), make_neighbor_rows(dataset.id.iloc[[0]].tolist())],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            build_neighbor_predictor_table(dataset, duplicate)

    def test_neighbor_lightgbm_uses_frozen_train_test_split_and_returns_scores(self):
        dataset = make_real_label_dataset(60)
        ids = dataset.id.tolist()
        neighbor = make_neighbor_rows(ids[::-1])
        split_root = Path(tempfile.mkdtemp())
        (split_root / "known").mkdir()
        memberships = ["train"] * 42 + ["val"] * 9 + ["test"] * 9
        assignments = pd.DataFrame({"id": ids, "split": memberships})
        assignments.iloc[::-1].to_csv(split_root / "known" / "seed0.csv", index=False)
        try:
            result = evaluate_neighbor_lightgbm_seed(
                dataset, neighbor, seed=0, splits_dir=split_root
            )
        finally:
            for path in (split_root / "known").iterdir():
                path.unlink()
            (split_root / "known").rmdir()
            split_root.rmdir()

        self.assertEqual(result.predictors.shape, (60, 21))
        self.assertEqual(len(result.scores), 9)
        self.assertEqual(set(result.scores.split), {"test"})
        self.assertEqual(set(result.scores.id), set(ids[51:]))
        self.assertTrue(result.scores.forgetting_risk.between(0, 1).all())
        self.assertTrue(np.isfinite(result.scores.forgetting_risk).all())
        self.assertEqual(set(result.metrics), set(METRIC_NAMES))


class MetricAndPairingTests(unittest.TestCase):
    def test_metric_helper_returns_existing_headline_metrics(self):
        results = evaluate_risk_scores([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1])

        self.assertAlmostEqual(results["auprc"], 5 / 6)
        self.assertEqual(results["f1"], 0.5)
        self.assertEqual(set(results), set(METRIC_NAMES))

    def test_condition_summary_hand_checks_mean_and_sample_sd(self):
        rows = []
        for seed, value in enumerate((0.2, 0.4, 0.6)):
            rows.append(
                {
                    "population": "known",
                    "condition": "real_graph",
                    "seed": seed,
                    **{metric: value for metric in METRIC_NAMES},
                }
            )
        summary = summarize_condition_metrics(pd.DataFrame(rows))

        self.assertEqual(summary.loc[0, "n_seeds"], 3)
        self.assertAlmostEqual(summary.loc[0, "auprc_mean"], 0.4)
        self.assertAlmostEqual(summary.loc[0, "auprc_std"], 0.2)

    def test_paired_differences_use_seed_pairing_and_candidate_minus_reference(self):
        metrics = make_metric_rows()
        paired, summary = paired_metric_differences(
            metrics,
            (("no_edge", "real_graph"),),
        )
        auprc = paired.loc[paired.metric == "auprc"].sort_values("seed")

        np.testing.assert_allclose(auprc.paired_difference.to_numpy(), [0.1, 0.2, 0.3])
        self.assertAlmostEqual(summary.loc[summary.metric == "auprc", "mean_difference"].iloc[0], 0.2)
        self.assertAlmostEqual(summary.loc[summary.metric == "auprc", "std_difference"].iloc[0], 0.1)
        self.assertEqual(list(paired.columns), list(PAIRED_DIFFERENCE_COLUMNS))
        self.assertEqual(list(summary.columns), list(PAIRED_SUMMARY_COLUMNS))

    def test_pairing_rejects_mismatched_seed_sets_duplicates_and_non_finite_metrics(self):
        metrics = make_metric_rows()
        missing_seed = metrics.loc[
            ~((metrics.condition == "no_edge") & (metrics.seed == 2))
        ]
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            paired_metric_differences(missing_seed, (("no_edge", "real_graph"),))

        duplicate = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate population/condition/seed"):
            paired_metric_differences(duplicate, (("no_edge", "real_graph"),))

        invalid = metrics.copy()
        invalid.loc[0, "auprc"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_metric_differences(invalid, (("no_edge", "real_graph"),))

    def test_baseline_loaders_map_task_conditions_and_select_known_lightgbm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task8_path = Path(temp_dir) / "task8.csv"
            task5_path = Path(temp_dir) / "task5.csv"
            task8_rows = [
                {"population": "known", "seed": seed, "model": "graphsage", **{m: 0.5 for m in METRIC_NAMES}}
                for seed in range(3)
            ]
            task5_rows = [
                {"population": population, "seed": seed, "model": model, **{m: 0.4 for m in METRIC_NAMES}}
                for population, model in (("known", "lightgbm_full"), ("known", "logistic_regression_full"), ("all", "lightgbm_full"))
                for seed in range(3)
            ]
            pd.DataFrame(task8_rows).to_csv(task8_path, index=False)
            pd.DataFrame(task5_rows).to_csv(task5_path, index=False)

            task8 = load_task8_baseline(task8_path)
            task5 = load_task5_lightgbm_baseline(task5_path)

        self.assertEqual(task8.condition.unique().tolist(), ["real_graph"])
        self.assertEqual(task5.condition.unique().tolist(), ["task5_lightgbm_full"])
        self.assertEqual(task8.seed.tolist(), [0, 1, 2])
        self.assertEqual(task5.seed.tolist(), [0, 1, 2])
        self.assertEqual(len(task5), 3)


class Task8ScoreLoaderTests(unittest.TestCase):
    def test_loads_real_six_column_task8_schema_and_adds_condition(self):
        fixtures = make_task13_fixtures()
        masks_by_seed, task8_scores = fixtures[4], fixtures[6]
        source = task8_scores.copy()
        source["forgetting_risk"] = np.linspace(0.1, 0.9, len(source))
        self.assertEqual(
            list(source.columns),
            ["population", "seed", "id", "split", "true_label", "forgetting_risk"],
        )
        self.assertNotIn("condition", source.columns)

        loaded = load_task8_test_scores_from_frame(source, masks_by_seed)

        self.assertEqual(list(loaded.columns), list(TASK13_SCORE_COLUMNS))
        self.assertEqual(loaded.condition.unique().tolist(), ["real_graph"])
        self.assertEqual(len(loaded), len(source))
        merged = loaded.merge(
            source, on=["seed", "id"], suffixes=("_loaded", "_source"), validate="one_to_one"
        )
        self.assertEqual(len(merged), len(source))
        np.testing.assert_allclose(
            merged.forgetting_risk_loaded, merged.forgetting_risk_source
        )
        self.assertTrue((merged.true_label_loaded == merged.true_label_source).all())
        self.assertEqual(sorted(loaded.seed.unique().tolist()), [0, 1, 2])
        for seed in (0, 1, 2):
            masks = masks_by_seed[seed]
            expected_ids = {
                masks.node_ids[index] for index in torch.where(masks.test_mask)[0].tolist()
            }
            self.assertEqual(set(loaded.loc[loaded.seed == seed, "id"]), expected_ids)


class Task13ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = make_task13_fixtures()

    def _run_experiment(self):
        (
            graph,
            stored_edges,
            id_to_index,
            real_dataset,
            masks_by_seed,
            task8_metrics,
            task8_scores,
            task5_metrics,
            neighbor_features,
        ) = self.fixtures
        train_calls = []
        shuffle_calls = []
        neighbor_calls = []

        def fake_training(graph_arg, masks, seed, config):
            train_calls.append((seed, graph_arg.edge_index.clone(), masks))
            logit = -0.2 if graph_arg.edge_index.shape[1] else 0.2
            history = [
                {"epoch": 1, "train_loss": 0.7, "validation_loss": 0.6, "is_best": True}
            ]
            return GraphSAGETrainingResult(
                model=StubLogitModel(logit),
                best_epoch=1,
                best_validation_loss=0.6,
                stopped_early=False,
                loss_history=history,
                pos_weight=1.0,
            )

        def fake_shuffle(graph_arg, edge_frame, mapping, seed, target_swaps):
            shuffle_calls.append((seed, target_swaps))
            endpoints = edge_frame[["src_id", "dst_id"]].copy()
            return ShuffledGraphResult(
                graph=graph_arg.clone(),
                stored_edges=endpoints,
                metadata=EdgeShuffleMetadata(
                    requested_swaps=target_swaps,
                    successful_swaps=target_swaps,
                    attempts=target_swaps,
                    original_edge_count=len(endpoints),
                    shuffled_edge_count=len(endpoints),
                ),
            )

        def fake_neighbor_eval(dataset, graph_neighbor, seed, splits_dir):
            neighbor_calls.append(seed)
            masks = masks_by_seed[seed]
            score_rows = [
                {
                    "population": "known",
                    "condition": "neighbor_lightgbm",
                    "seed": seed,
                    "id": masks.node_ids[index],
                    "split": "test",
                    "true_label": int(masks.labels[index]),
                    "forgetting_risk": 0.65,
                }
                for index in torch.where(masks.test_mask)[0].tolist()
            ]
            return NeighborLightGBMResult(
                metrics={metric: 0.6 for metric in METRIC_NAMES},
                scores=pd.DataFrame(score_rows, columns=TASK13_SCORE_COLUMNS),
                predictors=pd.DataFrame(),
            )

        outputs = run_task13_experiment(
            graph=graph,
            stored_edges=stored_edges,
            id_to_index=id_to_index,
            real_dataset=real_dataset,
            masks_by_seed=masks_by_seed,
            task8_metrics=task8_metrics,
            task8_scores=task8_scores,
            task5_metrics=task5_metrics,
            graph_neighbor_features=neighbor_features,
            training_function=fake_training,
            shuffle_function=fake_shuffle,
            neighbor_evaluation_function=fake_neighbor_eval,
        )
        return outputs, train_calls, shuffle_calls, neighbor_calls

    def test_orchestration_reuses_baselines_and_runs_all_conditions_per_seed(self):
        (metrics, summary, paired, scores), train_calls, shuffle_calls, neighbor_calls = self._run_experiment()

        self.assertEqual(list(metrics.columns), list(TASK13_METRICS_COLUMNS))
        self.assertEqual(len(metrics), 15)
        self.assertEqual(metrics.groupby("condition").size().to_dict(), {
            "real_graph": 3,
            "no_edge": 3,
            "shuffled_edges": 3,
            "task5_lightgbm_full": 3,
            "neighbor_lightgbm": 3,
        })
        self.assertEqual(metrics.groupby("condition", sort=False).seed.apply(list).to_dict(), {
            condition: [0, 1, 2] for condition in TASK13_CONDITIONS
        })
        self.assertEqual(list(summary.condition), list(TASK13_CONDITIONS))
        self.assertEqual(len(summary), 5)
        self.assertEqual(len(paired), 15)
        self.assertEqual(len(scores), 4 * 3 * 2)
        self.assertEqual(list(scores.columns), list(TASK13_SCORE_COLUMNS))
        self.assertEqual(len(train_calls), 6)
        self.assertEqual(neighbor_calls, [0, 1, 2])
        self.assertEqual(shuffle_calls, [(10000, 120), (10001, 120), (10002, 120)])
        no_edge_calls = [call for call in train_calls if call[1].shape[1] == 0]
        self.assertEqual(len(no_edge_calls), 3)

    def test_reuses_task5_and_task8_metrics_and_excludes_task5_scores(self):
        (metrics, _, _, scores), _, _, _ = self._run_experiment()
        task5_rows = metrics.loc[metrics.condition == "task5_lightgbm_full"]
        real_rows = metrics.loc[metrics.condition == "real_graph"]

        self.assertEqual(task5_rows.auprc.tolist(), [0.4] * 3)
        self.assertEqual(real_rows.auprc.tolist(), [0.5] * 3)
        self.assertNotIn("task5_lightgbm_full", set(scores.condition))
        self.assertEqual(set(scores.condition), {
            "real_graph", "no_edge", "shuffled_edges", "neighbor_lightgbm"
        })

    def test_score_ids_match_each_frozen_test_mask_only(self):
        (_, _, _, scores), _, _, _ = self._run_experiment()
        masks_by_seed = self.fixtures[4]

        self.assertTrue(scores.split.eq("test").all())
        self.assertFalse(scores.duplicated(["condition", "seed", "id"]).any())
        self.assertTrue(scores.forgetting_risk.between(0, 1).all())
        self.assertTrue(np.isfinite(scores.forgetting_risk).all())
        for seed in (0, 1, 2):
            expected_ids = {
                masks_by_seed[seed].node_ids[index]
                for index in torch.where(masks_by_seed[seed].test_mask)[0].tolist()
            }
            for condition in ("real_graph", "no_edge", "shuffled_edges", "neighbor_lightgbm"):
                rows = scores.loc[(scores.seed == seed) & (scores.condition == condition)]
                self.assertEqual(set(rows.id), expected_ids)
                for row in rows.itertuples(index=False):
                    graph_index = masks_by_seed[seed].node_ids.index(row.id)
                    self.assertEqual(row.true_label, int(masks_by_seed[seed].labels[graph_index]))

    def test_paired_table_has_required_names_signs_and_means(self):
        (_, _, paired, _), _, _, _ = self._run_experiment()

        self.assertEqual(list(paired.columns), list(TASK13_PAIRED_COLUMNS))
        self.assertEqual(len(paired), 15)
        comparisons = set(paired.comparison)
        self.assertEqual(
            comparisons,
            {
                "no_edge_minus_real_graph",
                "shuffled_edges_minus_real_graph",
                "neighbor_lightgbm_minus_task5_lightgbm_full",
            },
        )
        noedge_f1 = paired.loc[
            (paired.comparison == "no_edge_minus_real_graph") & (paired.metric == "f1")
        ].iloc[0]
        self.assertGreater(noedge_f1.seed0_difference, 0)
        self.assertAlmostEqual(
            noedge_f1.mean_difference,
            np.mean([noedge_f1.seed0_difference, noedge_f1.seed1_difference, noedge_f1.seed2_difference]),
        )

    def test_writer_creates_only_four_task13_artifacts(self):
        outputs, _, _, _ = self._run_experiment()
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_task13_outputs(*outputs, output_dir=Path(temp_dir))
            names = sorted(path.name for path in Path(temp_dir).iterdir())

        self.assertEqual(
            names,
            sorted(
                [
                    "task13_graph_ablation_metrics_by_seed.csv",
                    "task13_graph_ablation_summary.csv",
                    "task13_graph_ablation_paired_differences.csv",
                    "task13_graph_ablation_scores.csv",
                ]
            ),
        )
        self.assertEqual(set(paths), {"metrics_by_seed", "summary", "paired_differences", "scores"})


if __name__ == "__main__":
    unittest.main()