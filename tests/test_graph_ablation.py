import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph_ablation import make_no_edge_graph, make_shuffled_edge_graph
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


if __name__ == "__main__":
    unittest.main()