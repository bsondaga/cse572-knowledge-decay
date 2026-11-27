from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.evaluation_metrics import evaluate_predictions
from src.real_label_classifiers import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SPLITS_DIR,
    METRIC_NAMES,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    fit_predict_indices,
    load_frozen_dataset,
    load_frozen_split,
    make_lightgbm_classifier,
    select_population,
    select_predictor_features,
    validate_frozen_dataset,
)
from src.graphsage_model import DEFAULT_EDGES, DEFAULT_FEATURES, build_graph_data
from src.train_graphsage import (
    DEFAULT_LABELS_PATH,
    GraphLabelMasks,
    GraphSAGETrainingConfig,
    GraphSAGETrainingResult,
    load_graph_label_masks,
    predict_graph_probabilities,
    train_graphsage,
)

ROOT = Path(__file__).resolve().parents[1]
TASK5_METRICS_PATH = ROOT / "results" / "task5_metrics_by_seed.csv"
TASK8_METRICS_PATH = ROOT / "results" / "task8_graphsage_metrics_by_seed.csv"
TASK8_SCORES_PATH = ROOT / "results" / "task8_graphsage_scores.csv"
NEIGHBOR_FEATURE_COLUMNS = (
    "neighbor_count",
    "neighbor_mean_subject_pop_log",
    "neighbor_mean_object_pop_log",
)
ABLATION_METRIC_COLUMNS = (
    "population",
    "condition",
    "seed",
    *METRIC_NAMES,
)
PAIRED_DIFFERENCE_COLUMNS = (
    "population",
    "candidate_condition",
    "reference_condition",
    "seed",
    "metric",
    "candidate_value",
    "reference_value",
    "paired_difference",
)
PAIRED_SUMMARY_COLUMNS = (
    "population",
    "candidate_condition",
    "reference_condition",
    "metric",
    "n_seeds",
    "mean_difference",
    "std_difference",
)
TASK13_CONDITIONS = (
    "real_graph",
    "no_edge",
    "shuffled_edges",
    "task5_lightgbm_full",
    "neighbor_lightgbm",
)
TASK13_COMPARISONS = (
    ("no_edge", "real_graph"),
    ("shuffled_edges", "real_graph"),
    ("neighbor_lightgbm", "task5_lightgbm_full"),
)
TASK13_PAIRED_COLUMNS = (
    "population",
    "comparison",
    "candidate_condition",
    "reference_condition",
    "metric",
    "seed0_difference",
    "seed1_difference",
    "seed2_difference",
    "mean_difference",
    "std_difference",
)
TASK13_METRICS_COLUMNS = (
    "population",
    "condition",
    "seed",
    "n_train",
    "n_validation",
    "n_test",
    "best_epoch",
    "best_validation_loss",
    "stopped_early",
    "shuffle_seed",
    "shuffle_requested_swaps",
    "shuffle_successful_swaps",
    "shuffle_attempts",
    "shuffle_changed_edges",
    "shuffle_changed_percent",
    *METRIC_NAMES,
)
TASK8_SCORE_INPUT_COLUMNS = (
    "population",
    "seed",
    "id",
    "split",
    "true_label",
    "forgetting_risk",
)
TASK13_SCORE_COLUMNS = (
    "population",
    "condition",
    "seed",
    "id",
    "split",
    "true_label",
    "forgetting_risk",
)
TASK13_RESULTS_DIR = ROOT / "results"
TASK13_METRICS_OUTPUT = TASK13_RESULTS_DIR / "task13_graph_ablation_metrics_by_seed.csv"
TASK13_SUMMARY_OUTPUT = TASK13_RESULTS_DIR / "task13_graph_ablation_summary.csv"
TASK13_PAIRED_OUTPUT = TASK13_RESULTS_DIR / "task13_graph_ablation_paired_differences.csv"
TASK13_SCORES_OUTPUT = TASK13_RESULTS_DIR / "task13_graph_ablation_scores.csv"


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


@dataclass(frozen=True)
class NeighborLightGBMResult:
    metrics: dict[str, float | int | str]
    scores: pd.DataFrame
    predictors: pd.DataFrame


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


def build_neighbor_features(
    features: pd.DataFrame,
    stored_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Append count and mean popularity features for each node's graph neighbors."""
    required_features = {"id", "s_pop_log", "o_pop_log"}
    missing_features = required_features - set(features.columns)
    if missing_features:
        raise ValueError(f"features is missing required columns: {sorted(missing_features)}")
    if features.empty:
        raise ValueError("features must not be empty")
    if features["id"].isna().any():
        raise ValueError("features contains missing fact IDs")
    if features["id"].duplicated().any():
        raise ValueError("features contains duplicate fact IDs")

    try:
        source_values = features[["s_pop_log", "o_pop_log"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("s_pop_log and o_pop_log must be numeric and finite") from error
    if not np.isfinite(source_values).all():
        raise ValueError("s_pop_log and o_pop_log must be numeric and finite")

    id_to_index = {
        fact_id: index for index, fact_id in enumerate(features["id"].tolist())
    }
    pairs = _canonical_stored_pairs(stored_edges, id_to_index)
    source_indices = np.fromiter(
        (id_to_index[source_id] for source_id, _ in pairs), dtype=np.int64, count=len(pairs)
    )
    destination_indices = np.fromiter(
        (id_to_index[destination_id] for _, destination_id in pairs),
        dtype=np.int64,
        count=len(pairs),
    )

    neighbor_count = np.zeros(len(features), dtype=np.int64)
    neighbor_sums = np.zeros((len(features), 2), dtype=np.float64)
    np.add.at(neighbor_count, source_indices, 1)
    np.add.at(neighbor_count, destination_indices, 1)
    np.add.at(neighbor_sums, source_indices, source_values[destination_indices])
    np.add.at(neighbor_sums, destination_indices, source_values[source_indices])
    neighbor_means = np.zeros_like(neighbor_sums)
    np.divide(
        neighbor_sums,
        neighbor_count[:, None],
        out=neighbor_means,
        where=neighbor_count[:, None] > 0,
    )
    if not np.isfinite(neighbor_means).all():
        raise ValueError("computed neighbor features contain non-finite values")

    result = features.copy()
    result["neighbor_count"] = neighbor_count
    result["neighbor_mean_subject_pop_log"] = neighbor_means[:, 0]
    result["neighbor_mean_object_pop_log"] = neighbor_means[:, 1]
    if not np.isfinite(
        result[list(NEIGHBOR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("computed neighbor features contain non-finite values")
    return result


def build_neighbor_predictor_table(
    real_dataset: pd.DataFrame,
    graph_neighbor_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join graph aggregates by ID and return known facts plus their 21 predictors."""
    dataset = validate_frozen_dataset(real_dataset)
    known = select_population(dataset, "known")
    required_neighbor = {"id", *NEIGHBOR_FEATURE_COLUMNS}
    missing_columns = required_neighbor - set(graph_neighbor_features.columns)
    if missing_columns:
        raise ValueError(
            f"graph neighbor table is missing columns: {sorted(missing_columns)}"
        )
    if graph_neighbor_features["id"].isna().any():
        raise ValueError("graph neighbor table contains missing IDs")
    if graph_neighbor_features["id"].duplicated().any():
        raise ValueError("graph neighbor table contains duplicate IDs")

    known_ids = set(known["id"])
    graph_ids = set(graph_neighbor_features["id"])
    missing_ids = known_ids - graph_ids
    if missing_ids:
        raise ValueError(
            f"graph neighbor table is missing {len(missing_ids)} known IDs: "
            f"{sorted(missing_ids, key=str)[:5]}"
        )

    neighbor_subset = graph_neighbor_features.loc[:, ["id", *NEIGHBOR_FEATURE_COLUMNS]]
    joined = known.merge(
        neighbor_subset,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined[list(NEIGHBOR_FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("known facts have missing graph neighbor features")
    if not np.isfinite(
        joined[list(NEIGHBOR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("graph neighbor features must be finite")

    base_predictors = select_predictor_features(joined)
    predictors = pd.concat(
        [base_predictors.reset_index(drop=True), joined[list(NEIGHBOR_FEATURE_COLUMNS)].reset_index(drop=True)],
        axis=1,
    )
    if predictors.shape[1] != len(PREDICTOR_COLUMNS) + len(NEIGHBOR_FEATURE_COLUMNS):
        raise ValueError("neighbor predictor table must contain exactly 21 features")
    if any(column in predictors.columns for column in ("id", TARGET_COLUMN, "forgotten_default", "guessable")):
        raise ValueError("predictor table contains an identifier or outcome column")
    return joined, predictors


def evaluate_risk_scores(y_true, y_score) -> dict[str, float]:
    """Evaluate supplied labels and risk scores without training-related behavior."""
    results = evaluate_predictions(
        y_true,
        y_score,
        threshold=0.5,
        k_values=(1, 5, 10),
    )
    return {
        "auprc": results["auprc"],
        "f1": results["f1"],
        "recall_at_1": results["recall_at_k"][1],
        "recall_at_5": results["recall_at_k"][5],
        "recall_at_10": results["recall_at_k"][10],
    }


def evaluate_neighbor_lightgbm_seed(
    real_dataset: pd.DataFrame,
    graph_neighbor_features: pd.DataFrame,
    seed: int,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> NeighborLightGBMResult:
    """Fit Task 5 LightGBM settings on frozen training rows with 21 predictors."""
    known, predictors = build_neighbor_predictor_table(
        real_dataset, graph_neighbor_features
    )
    split = load_frozen_split(known, "known", seed, splits_dir=splits_dir)
    labels = known[TARGET_COLUMN].to_numpy(dtype=int)
    risk = fit_predict_indices(
        make_lightgbm_classifier(seed),
        predictors,
        labels,
        split.train,
        split.test,
    )
    metrics = evaluate_risk_scores(labels[split.test], risk)
    scores = pd.DataFrame(
        {
            "population": "known",
            "condition": "neighbor_lightgbm",
            "seed": seed,
            "id": known.iloc[split.test]["id"].to_numpy(),
            "split": "test",
            "true_label": labels[split.test],
            "forgetting_risk": risk,
        }
    )
    return NeighborLightGBMResult(metrics=metrics, scores=scores, predictors=predictors)


def summarize_condition_metrics(per_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed ablation metrics with sample SD for each condition."""
    required = {"population", "condition", "seed", *METRIC_NAMES}
    missing = required - set(per_seed_metrics.columns)
    if missing:
        raise ValueError(f"per-seed metrics are missing columns: {sorted(missing)}")
    if per_seed_metrics.empty:
        raise ValueError("per-seed metrics must not be empty")
    if per_seed_metrics.duplicated(["population", "condition", "seed"]).any():
        raise ValueError("per-seed metrics contain duplicate population/condition/seed rows")
    values = per_seed_metrics[list(METRIC_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("per-seed metric values must be finite")

    rows = []
    for (population, condition), group in per_seed_metrics.groupby(
        ["population", "condition"], sort=False
    ):
        row = {
            "population": population,
            "condition": condition,
            "n_seeds": group["seed"].nunique(),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def paired_metric_differences(
    per_seed_metrics: pd.DataFrame,
    comparisons: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-seed candidate-minus-reference deltas and their mean/sample SD."""
    required = {"population", "condition", "seed", *METRIC_NAMES}
    missing = required - set(per_seed_metrics.columns)
    if missing:
        raise ValueError(f"per-seed metrics are missing columns: {sorted(missing)}")
    if per_seed_metrics.duplicated(["population", "condition", "seed"]).any():
        raise ValueError("per-seed metrics contain duplicate population/condition/seed rows")
    if not np.isfinite(per_seed_metrics[list(METRIC_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("per-seed metric values must be finite")

    paired_rows = []
    for candidate, reference in comparisons:
        for population in per_seed_metrics["population"].drop_duplicates():
            candidate_rows = per_seed_metrics.loc[
                (per_seed_metrics["population"] == population)
                & (per_seed_metrics["condition"] == candidate)
            ].set_index("seed")
            reference_rows = per_seed_metrics.loc[
                (per_seed_metrics["population"] == population)
                & (per_seed_metrics["condition"] == reference)
            ].set_index("seed")
            if candidate_rows.empty or reference_rows.empty:
                raise ValueError(
                    f"missing paired condition rows for {candidate} vs {reference} in {population}"
                )
            if set(candidate_rows.index) != set(reference_rows.index):
                raise ValueError(
                    f"paired conditions must have identical seed sets: {candidate} vs {reference}"
                )
            for seed in sorted(candidate_rows.index):
                for metric in METRIC_NAMES:
                    candidate_value = float(candidate_rows.loc[seed, metric])
                    reference_value = float(reference_rows.loc[seed, metric])
                    paired_rows.append(
                        {
                            "population": population,
                            "candidate_condition": candidate,
                            "reference_condition": reference,
                            "seed": seed,
                            "metric": metric,
                            "candidate_value": candidate_value,
                            "reference_value": reference_value,
                            "paired_difference": candidate_value - reference_value,
                        }
                    )

    paired = pd.DataFrame(paired_rows, columns=PAIRED_DIFFERENCE_COLUMNS)
    summaries = []
    for keys, group in paired.groupby(
        ["population", "candidate_condition", "reference_condition", "metric"],
        sort=False,
    ):
        population, candidate, reference, metric = keys
        summaries.append(
            {
                "population": population,
                "candidate_condition": candidate,
                "reference_condition": reference,
                "metric": metric,
                "n_seeds": group["seed"].nunique(),
                "mean_difference": group["paired_difference"].mean(),
                "std_difference": group["paired_difference"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summaries, columns=PAIRED_SUMMARY_COLUMNS)
    return paired, summary


def load_task8_baseline(path: Path = TASK8_METRICS_PATH) -> pd.DataFrame:
    """Load Task 8 real-graph metrics and normalize the condition name."""
    frame = pd.read_csv(path)
    required = {"population", "seed", "model", *METRIC_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Task 8 metrics are missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["population"] == "known") & (frame["model"] == "graphsage"),
        ["population", "seed", *METRIC_NAMES],
    ].copy()
    _validate_baseline_seeds(selected, "Task 8 real-graph")
    selected.insert(1, "condition", "real_graph")
    return selected.reset_index(drop=True)


def load_task5_lightgbm_baseline(path: Path = TASK5_METRICS_PATH) -> pd.DataFrame:
    """Load Task 5 known full-LightGBM metrics and normalize the condition name."""
    frame = pd.read_csv(path)
    required = {"population", "seed", "model", *METRIC_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Task 5 metrics are missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["population"] == "known") & (frame["model"] == "lightgbm_full"),
        ["population", "seed", *METRIC_NAMES],
    ].copy()
    _validate_baseline_seeds(selected, "Task 5 known LightGBM")
    selected.insert(1, "condition", "task5_lightgbm_full")
    return selected.reset_index(drop=True)


def _validate_baseline_seeds(frame: pd.DataFrame, description: str) -> None:
    if frame.duplicated(["population", "seed"]).any():
        raise ValueError(f"{description} has duplicate seed rows")
    if set(frame["seed"]) != {0, 1, 2} or len(frame) != 3:
        raise ValueError(f"{description} must contain exactly seeds 0, 1, and 2")
    if not np.isfinite(frame[list(METRIC_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError(f"{description} metrics must be finite")


def load_task8_test_scores_from_frame(
    scores: pd.DataFrame,
    masks_by_seed: dict[int, object],
) -> pd.DataFrame:
    required = set(TASK8_SCORE_INPUT_COLUMNS)
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Task 8 scores are missing columns: {sorted(missing)}")
    if set(scores["population"]) != {"known"} or set(scores["seed"]) != {0, 1, 2}:
        raise ValueError("Task 8 scores must contain only known seeds 0, 1, and 2")
    rows = []
    for seed in (0, 1, 2):
        seed_rows = scores.loc[scores["seed"] == seed]
        if seed_rows["id"].duplicated().any() or not seed_rows["split"].eq("test").all():
            raise ValueError(f"Task 8 seed {seed} scores must be unique test-only rows")
        masks = masks_by_seed[seed]
        test_indices = torch.where(masks.test_mask.detach().cpu())[0].tolist()
        expected_ids = {masks.node_ids[index] for index in test_indices}
        if len(seed_rows) != len(expected_ids) or set(seed_rows["id"]) != expected_ids:
            raise ValueError(f"Task 8 seed {seed} scores do not match frozen test IDs")
        by_id = seed_rows.set_index("id")
        for node_index in test_indices:
            fact_id = masks.node_ids[node_index]
            source = by_id.loc[fact_id]
            label = int(masks.labels[node_index])
            risk = float(source["forgetting_risk"])
            if int(source["true_label"]) != label:
                raise ValueError(f"Task 8 label mismatch for seed {seed}, ID {fact_id}")
            if not np.isfinite(risk) or not 0 <= risk <= 1:
                raise ValueError("Task 8 risks must be finite and in [0, 1]")
            rows.append(
                {
                    "population": "known",
                    "condition": "real_graph",
                    "seed": seed,
                    "id": fact_id,
                    "split": "test",
                    "true_label": label,
                    "forgetting_risk": risk,
                }
            )
    return pd.DataFrame(rows, columns=TASK13_SCORE_COLUMNS)


def _graphsage_metric_record(
    condition: str,
    seed: int,
    masks,
    training_result,
    probabilities: torch.Tensor,
    *,
    shuffle_seed: int | None = None,
    shuffle_metadata: EdgeShuffleMetadata | None = None,
    shuffle_changed_edges: int | None = None,
    shuffle_changed_percent: float | None = None,
) -> dict[str, object]:
    test_labels = masks.labels[masks.test_mask].detach().cpu().numpy()
    risk = probabilities.detach().cpu().numpy()
    metrics = evaluate_risk_scores(test_labels, risk)
    return {
        "population": "known",
        "condition": condition,
        "seed": seed,
        "n_train": int(masks.train_mask.sum()),
        "n_validation": int(masks.validation_mask.sum()),
        "n_test": int(masks.test_mask.sum()),
        "best_epoch": training_result.best_epoch,
        "best_validation_loss": training_result.best_validation_loss,
        "stopped_early": training_result.stopped_early,
        "shuffle_seed": shuffle_seed,
        "shuffle_requested_swaps": shuffle_metadata.requested_swaps if shuffle_metadata else np.nan,
        "shuffle_successful_swaps": shuffle_metadata.successful_swaps if shuffle_metadata else np.nan,
        "shuffle_attempts": shuffle_metadata.attempts if shuffle_metadata else np.nan,
        "shuffle_changed_edges": shuffle_changed_edges if shuffle_changed_edges is not None else np.nan,
        "shuffle_changed_percent": shuffle_changed_percent if shuffle_changed_percent is not None else np.nan,
        **metrics,
    }


def _record_scores(condition: str, seed: int, masks, probabilities: torch.Tensor) -> pd.DataFrame:
    test_indices = torch.where(masks.test_mask.detach().cpu())[0].tolist()
    values = probabilities.detach().cpu().numpy()
    if values.ndim != 1 or len(values) != len(test_indices):
        raise ValueError(f"{condition} seed {seed} must return one score per test fact")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{condition} seed {seed} scores must be finite and in [0,1]")
    return pd.DataFrame(
        [
            {
                "population": "known",
                "condition": condition,
                "seed": seed,
                "id": masks.node_ids[node_index],
                "split": "test",
                "true_label": int(masks.labels[node_index]),
                "forgetting_risk": float(risk),
            }
            for node_index, risk in zip(test_indices, values)
        ],
        columns=TASK13_SCORE_COLUMNS,
    )


def build_task13_paired_differences(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Format the required three paired comparisons into one row per metric."""
    paired, summary = paired_metric_differences(metrics_by_seed, TASK13_COMPARISONS)
    wide = paired.pivot(
        index=["population", "candidate_condition", "reference_condition", "metric"],
        columns="seed",
        values="paired_difference",
    ).reset_index()
    wide = wide.rename(columns={seed: f"seed{seed}_difference" for seed in (0, 1, 2)})
    if any(f"seed{seed}_difference" not in wide for seed in (0, 1, 2)):
        raise ValueError("paired differences must include seeds 0, 1, and 2")
    wide["comparison"] = wide["candidate_condition"] + "_minus_" + wide["reference_condition"]
    wide = wide.merge(
        summary[[
            "population", "candidate_condition", "reference_condition", "metric",
            "mean_difference", "std_difference",
        ]],
        on=["population", "candidate_condition", "reference_condition", "metric"],
        validate="one_to_one",
    )
    comparison_order = {
        f"{candidate}_minus_{reference}": order
        for order, (candidate, reference) in enumerate(TASK13_COMPARISONS)
    }
    metric_order = {metric: order for order, metric in enumerate(METRIC_NAMES)}
    wide["_comparison_order"] = wide["comparison"].map(comparison_order)
    wide["_metric_order"] = wide["metric"].map(metric_order)
    wide = wide.sort_values(["_comparison_order", "_metric_order"], kind="stable")
    return wide.loc[:, TASK13_PAIRED_COLUMNS].reset_index(drop=True)


def run_task13_experiment(
    *,
    graph: Data | None = None,
    stored_edges: pd.DataFrame | None = None,
    id_to_index: dict[object, int] | None = None,
    real_dataset: pd.DataFrame | None = None,
    masks_by_seed: dict[int, object] | None = None,
    task8_metrics: pd.DataFrame | None = None,
    task8_scores: pd.DataFrame | None = None,
    task5_metrics: pd.DataFrame | None = None,
    graph_neighbor_features: pd.DataFrame | None = None,
    features_path: Path = DEFAULT_FEATURES,
    edges_path: Path = DEFAULT_EDGES,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    task8_metrics_path: Path = TASK8_METRICS_PATH,
    task8_scores_path: Path = TASK8_SCORES_PATH,
    task5_metrics_path: Path = TASK5_METRICS_PATH,
    seeds: tuple[int, ...] = (0, 1, 2),
    config: GraphSAGETrainingConfig | None = None,
    training_function=None,
    shuffle_function=None,
    neighbor_evaluation_function=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all Task 13 conditions and return per-seed, summary, paired, score tables."""
    if tuple(seeds) != (0, 1, 2):
        raise ValueError("Task 13 requires frozen seeds (0, 1, 2)")
    if graph is None:
        if stored_edges is not None or id_to_index is not None:
            raise ValueError("graph, stored_edges, and id_to_index must be supplied together")
        graph_features = pd.read_csv(features_path)
        stored_edges = pd.read_csv(edges_path)
        graph, id_to_index = build_graph_data(graph_features, stored_edges)
    elif stored_edges is None or id_to_index is None:
        raise ValueError("stored_edges and id_to_index are required with a supplied graph")
    else:
        graph_features = None

    if real_dataset is None:
        real_dataset = load_frozen_dataset(dataset_path)
    else:
        real_dataset = validate_frozen_dataset(real_dataset)
    if graph_neighbor_features is None:
        if graph_features is None:
            graph_features = pd.read_csv(features_path)
        graph_neighbor_features = build_neighbor_features(graph_features, stored_edges)
    if masks_by_seed is None:
        masks_by_seed = {
            seed: load_graph_label_masks(
                id_to_index,
                graph.num_nodes,
                seed,
                labels_path=dataset_path,
                splits_dir=splits_dir,
            )
            for seed in seeds
        }
    if set(masks_by_seed) != set(seeds):
        raise ValueError(f"masks_by_seed must contain exactly seeds {seeds}")

    if task8_metrics is None:
        task8_metrics = load_task8_baseline(task8_metrics_path)
    if task5_metrics is None:
        task5_metrics = load_task5_lightgbm_baseline(task5_metrics_path)
    if task8_scores is None:
        task8_scores = pd.read_csv(task8_scores_path)
    real_scores = load_task8_test_scores_from_frame(task8_scores, masks_by_seed)

    task8_by_seed = task8_metrics.set_index("seed")
    task5_by_seed = task5_metrics.set_index("seed")
    graph_train = training_function or train_graphsage
    edge_shuffle = shuffle_function or make_shuffled_edge_graph
    neighbor_evaluate = neighbor_evaluation_function or evaluate_neighbor_lightgbm_seed
    task_config = config or GraphSAGETrainingConfig()
    task_config.validate()

    original_pairs = set(_canonical_stored_pairs(stored_edges, id_to_index))
    full_shuffle_swaps = 10 * len(stored_edges)
    no_edge = make_no_edge_graph(graph)
    metric_rows: list[dict[str, object]] = []
    score_frames = [real_scores]

    for seed in seeds:
        masks = masks_by_seed[seed]
        masks.validate(graph.num_nodes)
        counts = {
            "n_train": int(masks.train_mask.sum()),
            "n_validation": int(masks.validation_mask.sum()),
            "n_test": int(masks.test_mask.sum()),
        }

        real_row = task8_by_seed.loc[seed]
        metric_rows.append(
            {
                "population": "known",
                "condition": "real_graph",
                "seed": seed,
                **counts,
                "best_epoch": real_row.get("best_epoch", np.nan),
                "best_validation_loss": real_row.get("best_validation_loss", np.nan),
                "stopped_early": real_row.get("stopped_early", np.nan),
                "shuffle_seed": np.nan,
                "shuffle_requested_swaps": np.nan,
                "shuffle_successful_swaps": np.nan,
                "shuffle_attempts": np.nan,
                "shuffle_changed_edges": np.nan,
                "shuffle_changed_percent": np.nan,
                **{metric: float(real_row[metric]) for metric in METRIC_NAMES},
            }
        )

        no_edge_result = graph_train(no_edge, masks, seed=seed, config=task_config)
        no_edge_risk = predict_graph_probabilities(no_edge_result.model, no_edge, masks.test_mask)
        metric_rows.append(
            _graphsage_metric_record("no_edge", seed, masks, no_edge_result, no_edge_risk)
        )
        score_frames.append(_record_scores("no_edge", seed, masks, no_edge_risk))

        shuffle_seed = 10_000 + seed
        shuffled = edge_shuffle(
            graph,
            stored_edges,
            id_to_index,
            seed=shuffle_seed,
            target_swaps=full_shuffle_swaps,
        )
        shuffled_pairs = set(
            tuple(pair)
            for pair in shuffled.stored_edges[["src_id", "dst_id"]].itertuples(
                index=False, name=None
            )
        )
        changed_edges = len(original_pairs - shuffled_pairs)
        changed_percent = 100.0 * changed_edges / len(original_pairs)
        shuffled_result = graph_train(
            shuffled.graph, masks, seed=seed, config=task_config
        )
        shuffled_risk = predict_graph_probabilities(
            shuffled_result.model, shuffled.graph, masks.test_mask
        )
        metric_rows.append(
            _graphsage_metric_record(
                "shuffled_edges",
                seed,
                masks,
                shuffled_result,
                shuffled_risk,
                shuffle_seed=shuffle_seed,
                shuffle_metadata=shuffled.metadata,
                shuffle_changed_edges=changed_edges,
                shuffle_changed_percent=changed_percent,
            )
        )
        score_frames.append(_record_scores("shuffled_edges", seed, masks, shuffled_risk))

        t5_row = task5_by_seed.loc[seed]
        metric_rows.append(
            {
                "population": "known",
                "condition": "task5_lightgbm_full",
                "seed": seed,
                **counts,
                "best_epoch": np.nan,
                "best_validation_loss": np.nan,
                "stopped_early": np.nan,
                "shuffle_seed": np.nan,
                "shuffle_requested_swaps": np.nan,
                "shuffle_successful_swaps": np.nan,
                "shuffle_attempts": np.nan,
                "shuffle_changed_edges": np.nan,
                "shuffle_changed_percent": np.nan,
                **{metric: float(t5_row[metric]) for metric in METRIC_NAMES},
            }
        )

        neighbor_result = neighbor_evaluate(
            real_dataset,
            graph_neighbor_features,
            seed=seed,
            splits_dir=splits_dir,
        )
        metric_rows.append(
            {
                "population": "known",
                "condition": "neighbor_lightgbm",
                "seed": seed,
                **counts,
                "best_epoch": np.nan,
                "best_validation_loss": np.nan,
                "stopped_early": np.nan,
                "shuffle_seed": np.nan,
                "shuffle_requested_swaps": np.nan,
                "shuffle_successful_swaps": np.nan,
                "shuffle_attempts": np.nan,
                "shuffle_changed_edges": np.nan,
                "shuffle_changed_percent": np.nan,
                **neighbor_result.metrics,
            }
        )
        score_frames.append(neighbor_result.scores.loc[:, TASK13_SCORE_COLUMNS])

    condition_order = {name: index for index, name in enumerate(TASK13_CONDITIONS)}
    metrics_by_seed = pd.DataFrame(metric_rows, columns=TASK13_METRICS_COLUMNS)
    metrics_by_seed["_order"] = metrics_by_seed["condition"].map(condition_order)
    metrics_by_seed = metrics_by_seed.sort_values(["_order", "seed"], kind="stable").drop(
        columns="_order"
    ).reset_index(drop=True)
    summary = summarize_condition_metrics(metrics_by_seed)
    summary["_order"] = summary["condition"].map(condition_order)
    summary = summary.sort_values("_order", kind="stable").drop(columns="_order").reset_index(
        drop=True
    )
    paired = build_task13_paired_differences(metrics_by_seed)
    scores = pd.concat(score_frames, ignore_index=True)
    scores = scores.sort_values(["condition", "seed", "id"], kind="stable").reset_index(
        drop=True
    )
    return metrics_by_seed, summary, paired, scores


def write_task13_outputs(
    metrics_by_seed: pd.DataFrame,
    summary: pd.DataFrame,
    paired_differences: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path = TASK13_RESULTS_DIR,
) -> dict[str, Path]:
    """Write exactly the four approved Task 13 result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_by_seed": output_dir / TASK13_METRICS_OUTPUT.name,
        "summary": output_dir / TASK13_SUMMARY_OUTPUT.name,
        "paired_differences": output_dir / TASK13_PAIRED_OUTPUT.name,
        "scores": output_dir / TASK13_SCORES_OUTPUT.name,
    }
    metrics_by_seed.to_csv(paths["metrics_by_seed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired_differences.to_csv(paths["paired_differences"], index=False)
    scores.to_csv(paths["scores"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 13 graph ablation experiments.")
    parser.add_argument("--run-experiment", action="store_true")
    args = parser.parse_args()
    if not args.run_experiment:
        parser.print_help()
        return

    metrics, summary, paired, _scores = run_task13_experiment()
    paths = write_task13_outputs(metrics, summary, paired, _scores)
    print(metrics[["condition", "seed", *METRIC_NAMES]].to_string(index=False))
    print("\nPaired AUPRC differences (candidate - reference):")
    print(paired.loc[paired["metric"] == "auprc"].to_string(index=False))
    for output_path in paths.values():
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()