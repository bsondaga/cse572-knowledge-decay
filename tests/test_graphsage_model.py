import unittest

import numpy as np
import pandas as pd
import torch

from src.graphsage_model import GraphSAGEModel, build_graph_data

RELATION_COLUMNS = (
    "rel_author",
    "rel_capital",
    "rel_capital_of",
    "rel_color",
    "rel_composer",
    "rel_country",
    "rel_director",
    "rel_father",
    "rel_genre",
    "rel_mother",
    "rel_occupation",
    "rel_place_of_birth",
    "rel_producer",
    "rel_religion",
    "rel_screenwriter",
    "rel_sport",
)


def make_features(ids=(9001, 23, 501)):
    rows = []
    for index, fact_id in enumerate(ids):
        row = {
            "id": fact_id,
            "prop": "author",
            "s_pop": 10 + index,
            "o_pop": 20 + index,
            "s_pop_log": 1.0 + index,
            "o_pop_log": 2.0 + index,
            "s_pop_z": 100.0 + index,
            "pop_bucket": index,
            "pop_bucket_0": int(index == 0),
        }
        row.update({column: int(column == RELATION_COLUMNS[index % 16]) for column in RELATION_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def make_edges(rows=None):
    if rows is None:
        rows = [(9001, 501, "head")]
    return pd.DataFrame(rows, columns=["src_id", "dst_id", "edge_type"])


class GraphInputTests(unittest.TestCase):
    def test_sparse_fact_ids_map_in_feature_row_order(self):
        data, id_to_index = build_graph_data(make_features(), make_edges())

        self.assertEqual(id_to_index, {9001: 0, 23: 1, 501: 2})
        self.assertEqual(data.x.shape, (3, 18))
        self.assertEqual(data.x[0, 0].item(), 1.0)
        self.assertEqual(data.x[1, 0].item(), 2.0)
        self.assertEqual(data.x[2, 0].item(), 3.0)

    def test_id_prop_and_non_predictor_columns_are_excluded(self):
        data, _ = build_graph_data(make_features(), make_edges())

        self.assertEqual(data.x.shape[1], 18)
        self.assertEqual(data.x[0, 0:2].tolist(), [1.0, 2.0])
        self.assertEqual(data.x[0, 2:].tolist(), [1.0] + [0.0] * 15)

    def test_both_edge_directions_are_added(self):
        data, id_to_index = build_graph_data(make_features(), make_edges())

        directed_edges = set(map(tuple, data.edge_index.t().tolist()))
        self.assertEqual(
            directed_edges,
            {(id_to_index[9001], id_to_index[501]), (id_to_index[501], id_to_index[9001])},
        )

    def test_isolated_fact_remains_in_node_features(self):
        data, id_to_index = build_graph_data(make_features(), make_edges())

        self.assertIn(23, id_to_index)
        self.assertEqual(data.x.shape[0], 3)
        self.assertNotIn(id_to_index[23], data.edge_index.tolist())

    def test_edge_tensor_has_expected_dtype_and_shape(self):
        data, _ = build_graph_data(
            make_features(), make_edges([(9001, 23, "head"), (9001, 501, "tail")])
        )

        self.assertEqual(data.x.dtype, torch.float32)
        self.assertEqual(tuple(data.x.shape), (3, 18))
        self.assertEqual(data.edge_index.dtype, torch.long)
        self.assertEqual(tuple(data.edge_index.shape), (2, 4))

    def test_rejects_missing_feature_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_graph_data(make_features().drop(columns="s_pop_log"), make_edges())

    def test_rejects_non_numeric_and_non_finite_features(self):
        non_numeric = make_features()
        non_numeric["s_pop_log"] = non_numeric["s_pop_log"].astype(object)
        non_numeric.loc[0, "s_pop_log"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            build_graph_data(non_numeric, make_edges())

        non_finite = make_features()
        non_finite.loc[0, "o_pop_log"] = np.inf
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            build_graph_data(non_finite, make_edges())

    def test_rejects_duplicate_or_missing_fact_ids(self):
        duplicate_ids = make_features()
        duplicate_ids.loc[1, "id"] = duplicate_ids.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "duplicate fact IDs"):
            build_graph_data(duplicate_ids, make_edges())

        missing_ids = make_features()
        missing_ids.loc[1, "id"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing fact IDs"):
            build_graph_data(missing_ids, make_edges())

    def test_rejects_missing_edge_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_graph_data(make_features(), make_edges().drop(columns="edge_type"))

    def test_rejects_unknown_edge_endpoint(self):
        with self.assertRaisesRegex(ValueError, "not present in the feature table"):
            build_graph_data(make_features(), make_edges([(9001, 70000, "head")]))

    def test_rejects_self_loop(self):
        with self.assertRaisesRegex(ValueError, "self-loop"):
            build_graph_data(make_features(), make_edges([(9001, 9001, "head")]))

    def test_rejects_duplicate_undirected_stored_pairs(self):
        duplicate_edges = make_edges(
            [(9001, 501, "head"), (501, 9001, "tail")]
        )
        with self.assertRaisesRegex(ValueError, "duplicate undirected stored edge pair"):
            build_graph_data(make_features(), duplicate_edges)


class GraphSAGEForwardTests(unittest.TestCase):
    def test_forward_pass_returns_finite_node_embeddings(self):
        data, _ = build_graph_data(make_features(), make_edges())
        model = GraphSAGEModel()

        output = model(data.x, data.edge_index)

        self.assertEqual(tuple(output.shape), (3, 16))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()