from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = ROOT / "results" / "frozen" / "real_dataset.csv"
DEFAULT_SPLITS_DIR = ROOT / "results" / "splits"
KNOWN_POPULATION = "known"
TARGET_COLUMN = "forgotten_lr1e5"
SUPPORTED_SEEDS = (0, 1, 2)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class GraphLabelMasks:
    """Full-graph labels and frozen split masks, indexed in graph node order."""

    labels: Tensor
    labeled_mask: Tensor
    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor
    node_ids: tuple[object, ...]

    def validate(self, num_nodes: int) -> None:
        tensors = (
            self.labels,
            self.labeled_mask,
            self.train_mask,
            self.validation_mask,
            self.test_mask,
        )
        if any(tensor.ndim != 1 or len(tensor) != num_nodes for tensor in tensors):
            raise ValueError("labels and masks must be one-dimensional with length num_nodes")
        if self.labels.dtype != torch.float32:
            raise ValueError("labels must use torch.float32")
        if any(
            mask.dtype != torch.bool
            for mask in (
                self.labeled_mask,
                self.train_mask,
                self.validation_mask,
                self.test_mask,
            )
        ):
            raise ValueError("labeled and split masks must use torch.bool")
        if len(self.node_ids) != num_nodes or len(set(self.node_ids)) != num_nodes:
            raise ValueError("node_ids must contain one unique fact ID per graph node")

        supervised = (self.train_mask, self.validation_mask, self.test_mask)
        if any(torch.any(mask & ~self.labeled_mask) for mask in supervised):
            raise ValueError("split masks must select only labeled graph nodes")
        if any(
            torch.any(left & right)
            for index, left in enumerate(supervised)
            for right in supervised[index + 1 :]
        ):
            raise ValueError("train, validation, and test masks must be disjoint")
        if not torch.equal(
            self.train_mask | self.validation_mask | self.test_mask,
            self.labeled_mask,
        ):
            raise ValueError("train, validation, and test masks must cover labeled nodes")
        if any(not torch.any(mask) for mask in supervised):
            raise ValueError("train, validation, and test masks must all be non-empty")


def load_known_labels(path: Path = DEFAULT_LABELS_PATH) -> pd.DataFrame:
    """Load, validate, and return only known fact IDs with the Task 8 target."""
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen labels from {path}: {error}") from error
    if dataset.empty:
        raise ValueError("frozen label dataset must not be empty")

    required = {"id", "guessable", TARGET_COLUMN}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Frozen label dataset is missing required columns: {sorted(missing)}")
    if dataset["id"].isna().any():
        raise ValueError("Frozen label dataset contains missing fact IDs")
    duplicate_ids = dataset.loc[dataset["id"].duplicated(), "id"].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen label dataset contains duplicate fact IDs: {duplicate_ids[:5]}")

    known = dataset.copy()
    for column in ("guessable", TARGET_COLUMN):
        try:
            values = pd.to_numeric(known[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain only binary 0/1 values") from error
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0/1 values")
        known[column] = values.astype(int)

    known = known.loc[known["guessable"] == 0, ["id", TARGET_COLUMN]].copy()
    if known.empty:
        raise ValueError("frozen dataset contains no known facts (guessable == 0)")
    return known.reset_index(drop=True)


def validate_split_assignments(
    assignments: pd.DataFrame, known_ids
) -> pd.DataFrame:
    """Validate frozen ID-to-split rows against the exact known ID set."""
    if not isinstance(assignments, pd.DataFrame):
        raise TypeError("split assignments must be a pandas DataFrame")
    if assignments.empty:
        raise ValueError("frozen split assignments must not be empty")
    missing_columns = {"id", "split"} - set(assignments.columns)
    if missing_columns:
        raise ValueError(f"Frozen split is missing required columns: {sorted(missing_columns)}")
    if assignments["id"].isna().any():
        raise ValueError("Frozen split contains missing fact IDs")
    duplicate_ids = assignments.loc[
        assignments["id"].duplicated(), "id"
    ].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen split contains duplicate IDs: {duplicate_ids[:5]}")
    if assignments["split"].isna().any():
        raise ValueError("Frozen split contains missing membership values")

    invalid_memberships = sorted(set(assignments["split"].astype(str)) - set(SPLIT_NAMES))
    if invalid_memberships:
        raise ValueError(f"Frozen split contains invalid membership values: {invalid_memberships}")

    known_id_set = set(known_ids)
    split_id_set = set(assignments["id"].tolist())
    missing_ids = known_id_set - split_id_set
    extra_ids = split_id_set - known_id_set
    if missing_ids:
        raise ValueError(
            f"Frozen split is missing {len(missing_ids)} known fact IDs: "
            f"{sorted(missing_ids, key=str)[:5]}"
        )
    if extra_ids:
        raise ValueError(
            f"Frozen split contains {len(extra_ids)} IDs outside the known population: "
            f"{sorted(extra_ids, key=str)[:5]}"
        )
    empty_partitions = [
        name for name in SPLIT_NAMES if not (assignments["split"] == name).any()
    ]
    if empty_partitions:
        raise ValueError(f"Frozen split partitions must be non-empty: {empty_partitions}")
    return assignments.loc[:, ["id", "split"]].copy().reset_index(drop=True)


def load_frozen_split_assignments(
    seed: int,
    known_ids,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> pd.DataFrame:
    """Load one authoritative known-population split; never generate assignments."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed not in SUPPORTED_SEEDS:
        raise ValueError(f"seed must be one of {SUPPORTED_SEEDS}")
    path = Path(splits_dir) / KNOWN_POPULATION / f"seed{seed}.csv"
    try:
        assignments = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen split from {path}: {error}") from error
    return validate_split_assignments(assignments, known_ids)


def build_graph_label_masks(
    id_to_index: dict[object, int],
    known_labels: pd.DataFrame,
    assignments: pd.DataFrame,
    num_nodes: int,
) -> GraphLabelMasks:
    """Map known targets and frozen split memberships onto full-graph positions."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    if len(id_to_index) != num_nodes:
        raise ValueError("id_to_index must contain exactly one entry per graph node")
    if set(id_to_index.values()) != set(range(num_nodes)):
        raise ValueError("id_to_index values must be contiguous indices from 0 to num_nodes - 1")
    if known_labels.empty:
        raise ValueError("known labels must not be empty")
    missing_columns = {"id", TARGET_COLUMN} - set(known_labels.columns)
    if missing_columns:
        raise ValueError(f"Known labels are missing required columns: {sorted(missing_columns)}")
    if known_labels["id"].isna().any():
        raise ValueError("Known labels contain missing fact IDs")
    if known_labels["id"].duplicated().any():
        raise ValueError("Known labels contain duplicate IDs")

    try:
        target_values = pd.to_numeric(known_labels[TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values") from error
    target_array = target_values.to_numpy(dtype=float)
    if not np.isfinite(target_array).all() or not np.isin(target_array, [0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values")

    known_ids = known_labels["id"].tolist()
    unknown_ids = [fact_id for fact_id in known_ids if fact_id not in id_to_index]
    if unknown_ids:
        raise ValueError(f"Known fact IDs are not present in the graph: {unknown_ids[:5]}")
    assignments = validate_split_assignments(assignments, known_ids)
    split_by_id = dict(zip(assignments["id"], assignments["split"]))

    labels = torch.zeros(num_nodes, dtype=torch.float32)
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_by_split = {
        "train": train_mask,
        "val": validation_mask,
        "test": test_mask,
    }
    for fact_id, target in zip(known_ids, target_array):
        node_index = id_to_index[fact_id]
        labels[node_index] = float(target)
        labeled_mask[node_index] = True
        mask_by_split[split_by_id[fact_id]][node_index] = True

    inverse_mapping: list[object | None] = [None] * num_nodes
    for fact_id, node_index in id_to_index.items():
        inverse_mapping[node_index] = fact_id
    if any(fact_id is None for fact_id in inverse_mapping):
        raise ValueError("id_to_index does not map every graph node")

    masks = GraphLabelMasks(
        labels=labels,
        labeled_mask=labeled_mask,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        node_ids=tuple(inverse_mapping),
    )
    masks.validate(num_nodes)
    return masks


def load_graph_label_masks(
    id_to_index: dict[object, int],
    num_nodes: int,
    seed: int,
    labels_path: Path = DEFAULT_LABELS_PATH,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> GraphLabelMasks:
    """Load the known labels and frozen seed split, then map them to graph masks."""
    known_labels = load_known_labels(labels_path)
    assignments = load_frozen_split_assignments(
        seed, known_labels["id"].tolist(), splits_dir=splits_dir
    )
    return build_graph_label_masks(
        id_to_index, known_labels, assignments, num_nodes=num_nodes
    )