from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "processed" / "popqa_features.csv"
DEFAULT_EDGES = ROOT / "data" / "processed" / "popqa_graph_edges.csv"
BASE_FEATURES = ("s_pop_log", "o_pop_log")
RELATION_FEATURE_COUNT = 16
INPUT_DIM = len(BASE_FEATURES) + RELATION_FEATURE_COUNT


def build_graph_data(
    features: pd.DataFrame, edges: pd.DataFrame
) -> tuple[Data, dict[object, int]]:
    """Build PyG tensors and an original-fact-ID to contiguous-index mapping."""
    missing_features = {"id", *BASE_FEATURES} - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Feature table is missing required columns: {sorted(missing_features)}"
        )

    relation_features = [
        column for column in features.columns if column.startswith("rel_")
    ]
    if len(relation_features) != RELATION_FEATURE_COUNT:
        raise ValueError(
            "Feature table must contain exactly 16 rel_ columns; "
            f"found {len(relation_features)}"
        )

    fact_ids = features["id"]
    if fact_ids.isna().any():
        raise ValueError("Feature table contains missing fact IDs")
    duplicate_ids = fact_ids[fact_ids.duplicated()].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Feature table contains duplicate fact IDs: {duplicate_ids[:5]}")

    id_to_index = {
        fact_id: index for index, fact_id in enumerate(fact_ids.tolist())
    }
    feature_columns = [*BASE_FEATURES, *relation_features]
    numeric_features = features[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    try:
        feature_values = numeric_features.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Node feature values must be numeric and finite") from error
    if not np.isfinite(feature_values).all():
        raise ValueError("Node feature values must be numeric and finite")
    x = torch.tensor(feature_values, dtype=torch.float32)

    required_edge_columns = {"src_id", "dst_id", "edge_type"}
    missing_edge_columns = required_edge_columns - set(edges.columns)
    if missing_edge_columns:
        raise ValueError(
            f"Edge table is missing required columns: {sorted(missing_edge_columns)}"
        )

    directed_pairs: list[tuple[int, int]] = []
    stored_pairs: set[tuple[int, int]] = set()
    for source_id, destination_id in edges[["src_id", "dst_id"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(source_id) or pd.isna(destination_id):
            raise ValueError("Edge table contains a missing endpoint fact ID")
        if source_id not in id_to_index or destination_id not in id_to_index:
            missing_ids = [
                fact_id
                for fact_id in (source_id, destination_id)
                if fact_id not in id_to_index
            ]
            raise ValueError(
                f"Edge endpoint fact IDs are not present in the feature table: "
                f"{missing_ids[:5]}"
            )

        source_index = id_to_index[source_id]
        destination_index = id_to_index[destination_id]
        if source_index == destination_index:
            raise ValueError(f"Stored graph contains a self-loop for fact ID {source_id}")

        pair = (min(source_index, destination_index), max(source_index, destination_index))
        if pair in stored_pairs:
            raise ValueError(
                "Edge table contains a duplicate undirected stored edge pair: "
                f"{source_id}, {destination_id}"
            )
        stored_pairs.add(pair)
        directed_pairs.extend(
            [(source_index, destination_index), (destination_index, source_index)]
        )

    if directed_pairs:
        edge_index = torch.tensor(directed_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    validate_graph_data(data)
    return data, id_to_index


def validate_graph_data(data: Data) -> None:
    """Validate tensor structure, bounds, and directed-edge uniqueness."""
    if data.x.ndim != 2 or data.x.dtype != torch.float32:
        raise ValueError("x must be a rank-2 torch.float32 tensor")
    if data.x.shape[1] != INPUT_DIM:
        raise ValueError(f"x must contain exactly {INPUT_DIM} node features")
    if not torch.isfinite(data.x).all():
        raise ValueError("x contains non-finite node feature values")
    if (
        data.edge_index.dtype != torch.long
        or data.edge_index.ndim != 2
        or data.edge_index.shape[0] != 2
    ):
        raise ValueError("edge_index must be a torch.long tensor with shape [2, E]")

    num_nodes = data.x.shape[0]
    if data.edge_index.numel() and (
        int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains node indices outside [0, num_nodes)")
    if data.edge_index.shape[1]:
        directed_edges = data.edge_index.t()
        if torch.unique(directed_edges, dim=0).shape[0] != directed_edges.shape[0]:
            raise ValueError("edge_index contains duplicate directed edges")


class GraphSAGEModel(nn.Module):
    """Two-layer GraphSAGE encoder that returns node embeddings."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        self.input_dim = input_dim
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape [num_nodes, {self.input_dim}]")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a torch.long tensor with shape [2, E]")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]
        ):
            raise ValueError("edge_index contains node indices outside [0, num_nodes)")

        embeddings = self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index)
        if not torch.isfinite(embeddings).all():
            raise ValueError("GraphSAGE produced non-finite node embeddings")
        return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one untrained GraphSAGE forward pass.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    edges = pd.read_csv(args.edges)
    data, _ = build_graph_data(features, edges)
    model = GraphSAGEModel()
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)

    output_is_finite = bool(torch.isfinite(output).all())
    print("GraphSAGE skeleton forward pass only — no training performed.")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Stored undirected edges: {len(edges)}")
    print(f"Directed edge count: {data.edge_index.shape[1]}")
    print(f"Number of input features: {data.x.shape[1]}")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"Output embedding shape: {tuple(output.shape)}")
    print(f"Output is finite: {output_is_finite}")


if __name__ == "__main__":
    main()