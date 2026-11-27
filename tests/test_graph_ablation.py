import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph_ablation import (
    ABLATION_METRIC_COLUMNS,
    NEIGHBOR_FEATURE_COLUMNS,
    PAIRED_DIFFERENCE_COLUMNS,
    PAIRED_SUMMARY_COLUMNS,
    build_neighbor_features,
    build_neighbor_predictor_table,
    evaluate_neighbor_lightgbm_seed,
    evaluate_risk_scores,
    load_task5_lightgbm_baseline,
    load_task8_baseline,
    make_no_edge_graph,
    make_shuffled_edge_graph,
    paired_metric_differences,
    summarize_condition_metrics,
)
from src.real_label_classifiers import METRIC_NAMES, PREDICTOR_COLUMNS
from src.train_graphsage import GraphSAGEClassifier


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


if __name__ == "__main__":
    unittest.main()