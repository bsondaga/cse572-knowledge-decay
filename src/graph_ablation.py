from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class EdgeShuffleMetadata:
    requested_swaps: int
    successful_swaps: int
    attempts: int
    original_edge_count: int
    shuffled_edge_count: int


@dataclass(frozen=True)
class ShuffledGraphResult:
    graph: Data
    stored_edges: pd.DataFrame
    metadata: EdgeShuffleMetadata


def make_no_edge_graph(graph: Data) -> Data:
    """Return a cloned graph with no message-passing edges."""
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if not hasattr(graph, "x") or graph.x is None or graph.x.ndim != 2:
        raise ValueError("graph must contain a rank-2 x tensor")
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("graph must contain edge_index")
    original_nodes = graph.num_nodes
    no_edge_graph = graph.clone()
    no_edge_graph.edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=graph.edge_index.device
    )
    if no_edge_graph.num_nodes != original_nodes:
        raise ValueError("no-edge graph changed the number of nodes")
    return no_edge_graph


def _validate_mapping(id_to_index: dict[object, int], num_nodes: int) -> None:
    if len(id_to_index) != num_nodes:
        raise ValueError("id_to_index must contain exactly one entry per graph node")
    values = list(id_to_index.values())
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in values):
        raise ValueError("id_to_index values must be integer node indices")
    if set(int(index) for index in values) != set(range(num_nodes)):
        raise ValueError("id_to_index values must be contiguous indices from 0 to num_nodes - 1")


def _canonical_stored_pairs(
    stored_edges: pd.DataFrame,
    id_to_index: dict[object, int],
) -> list[tuple[object, object]]:
    if not isinstance(stored_edges, pd.DataFrame):
        raise TypeError("stored_edges must be a pandas DataFrame")
    missing_columns = {"src_id", "dst_id"} - set(stored_edges.columns)
    if missing_columns:
        raise ValueError(f"stored_edges is missing endpoint columns: {sorted(missing_columns)}")

    pairs: list[tuple[object, object]] = []
    seen: set[tuple[object, object]] = set()
    for source_id, destination_id in stored_edges[["src_id", "dst_id"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(source_id) or pd.isna(destination_id):
            raise ValueError("stored_edges contains a missing endpoint ID")
        unknown = [
            fact_id
            for fact_id in (source_id, destination_id)
            if fact_id not in id_to_index
        ]
        if unknown:
            raise ValueError(f"stored_edges contains IDs not present in id_to_index: {unknown[:5]}")
        if source_id == destination_id:
            raise ValueError(f"stored_edges contains a self-loop for fact ID {source_id}")
        try:
            pair = (source_id, destination_id) if source_id < destination_id else (destination_id, source_id)
        except TypeError as error:
            raise ValueError("fact IDs must be mutually orderable for canonical undirected pairs") from error
        if pair in seen:
            raise ValueError(f"stored_edges contains duplicate undirected pair: {pair}")
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _degree_by_id(
    pairs: list[tuple[object, object]], id_to_index: dict[object, int]
) -> dict[object, int]:
    degrees = {fact_id: 0 for fact_id in id_to_index}
    for source_id, destination_id in pairs:
        degrees[source_id] += 1
        degrees[destination_id] += 1
    return degrees


def _directed_edge_index(
    pairs: list[tuple[object, object]],
    id_to_index: dict[object, int],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    directed = [
        (int(id_to_index[source_id]), int(id_to_index[destination_id]))
        for source_id, destination_id in pairs
    ]
    directed.extend((destination, source) for source, destination in directed.copy())
    if directed:
        edge_index = torch.tensor(directed, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("directed edge indices are outside [0, num_nodes)")
    if edge_index.shape != (2, 2 * len(pairs)):
        raise ValueError("directed edge_index must contain both directions for every stored pair")
    if torch.unique(edge_index.t(), dim=0).shape[0] != edge_index.shape[1]:
        raise ValueError("directed edge_index contains duplicate directed edges")
    return edge_index


def make_shuffled_edge_graph(
    graph: Data,
    stored_edges: pd.DataFrame,
    id_to_index: dict[object, int],
    seed: int,
    target_swaps: int | None = None,
    max_attempts: int | None = None,
) -> ShuffledGraphResult:
    """Rewire stored undirected edges with deterministic degree-preserving swaps."""
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if not hasattr(graph, "x") or graph.x is None or graph.x.ndim != 2:
        raise ValueError("graph must contain a rank-2 x tensor")
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("graph must contain edge_index")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    num_nodes = graph.x.shape[0]
    _validate_mapping(id_to_index, num_nodes)
    pairs = _canonical_stored_pairs(stored_edges, id_to_index)
    original_pairs = list(pairs)
    edge_count = len(pairs)
    requested = 10 * edge_count if target_swaps is None else target_swaps
    if isinstance(requested, bool) or not isinstance(requested, (int, np.integer)) or requested < 0:
        raise ValueError("target_swaps must be a non-negative integer")
    requested = int(requested)
    if max_attempts is None:
        max_attempts = max(1_000, requested * 20)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)) or max_attempts < 0:
        raise ValueError("max_attempts must be a non-negative integer")
    max_attempts = int(max_attempts)

    original_degrees = _degree_by_id(pairs, id_to_index)
    edge_set = set(pairs)
    rng = np.random.default_rng(int(seed))
    successful_swaps = 0
    attempts = 0

    while successful_swaps < requested and attempts < max_attempts:
        attempts += 1
        if edge_count < 2:
            break

        first_index = int(rng.integers(edge_count))
        second_index = int(rng.integers(edge_count - 1))
        if second_index >= first_index:
            second_index += 1
        first_source, first_destination = pairs[first_index]
        second_source, second_destination = pairs[second_index]
        if len({first_source, first_destination, second_source, second_destination}) != 4:
            continue

        if rng.integers(2) == 0:
            proposed = (
                (first_source, second_destination),
                (second_source, first_destination),
            )
        else:
            proposed = (
                (first_source, second_source),
                (first_destination, second_destination),
            )
        new_pairs = tuple(
            (left, right) if left < right else (right, left)
            for left, right in proposed
        )
        if new_pairs[0] == new_pairs[1]:
            continue
        if any(pair in edge_set for pair in new_pairs):
            continue
        if set(new_pairs) == {pairs[first_index], pairs[second_index]}:
            continue

        edge_set.remove(pairs[first_index])
        edge_set.remove(pairs[second_index])
        edge_set.update(new_pairs)
        pairs[first_index], pairs[second_index] = new_pairs
        successful_swaps += 1

    if successful_swaps != requested:
        raise RuntimeError(
            "degree-preserving edge shuffle exhausted attempts: "
            f"achieved {successful_swaps}/{requested} successful swaps "
            f"in {attempts}/{max_attempts} attempts"
        )

    pairs.sort()
    if len(pairs) != edge_count:
        raise RuntimeError("edge shuffle changed stored edge count")
    shuffled_degrees = _degree_by_id(pairs, id_to_index)
    if shuffled_degrees != original_degrees:
        raise RuntimeError("edge shuffle failed to preserve the exact node degree sequence")
    if any(source_id >= destination_id for source_id, destination_id in pairs):
        raise RuntimeError("shuffled stored edges are not in canonical src_id < dst_id order")

    edge_index = _directed_edge_index(
        pairs,
        id_to_index,
        num_nodes,
        graph.edge_index.device,
    )
    shuffled_graph = graph.clone()
    shuffled_graph.edge_index = edge_index
    if shuffled_graph.num_nodes != graph.num_nodes or not torch.equal(shuffled_graph.x, graph.x):
        raise RuntimeError("shuffled graph changed original node features or node count")

    shuffled_frame = pd.DataFrame(pairs, columns=["src_id", "dst_id"])
    return ShuffledGraphResult(
        graph=shuffled_graph,
        stored_edges=shuffled_frame,
        metadata=EdgeShuffleMetadata(
            requested_swaps=requested,
            successful_swaps=successful_swaps,
            attempts=attempts,
            original_edge_count=edge_count,
            shuffled_edge_count=len(shuffled_frame),
        ),
    )