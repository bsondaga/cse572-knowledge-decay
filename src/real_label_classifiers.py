from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "results" / "frozen" / "real_dataset.csv"
DEFAULT_SPLITS_DIR = ROOT / "results" / "splits"
POPULATIONS = ("known", "all")
SEEDS = (0, 1, 2)
TARGET_COLUMN = "forgotten_lr1e5"
PREDICTOR_BASE_COLUMNS = ("subject_pop_log", "object_pop_log")
RELATION_FEATURES = (
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
PREDICTOR_COLUMNS = (*PREDICTOR_BASE_COLUMNS, *RELATION_FEATURES)
REQUIRED_DATASET_COLUMNS = {
    "id",
    "relation",
    TARGET_COLUMN,
    "guessable",
    *PREDICTOR_COLUMNS,
}
SPLIT_COLUMNS = {"id", "split"}
SPLIT_MEMBERSHIPS = ("train", "val", "test")


@dataclass(frozen=True)
class FrozenSplitIndices:
    """Positional row indices derived from a frozen ID-to-split assignment."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, n_rows: int) -> None:
        partitions = (self.train, self.validation, self.test)
        if any(indices.ndim != 1 for indices in partitions):
            raise ValueError("split indices must be one-dimensional")
        if any(not np.issubdtype(indices.dtype, np.integer) for indices in partitions):
            raise ValueError("split indices must contain integers")
        if any(len(indices) == 0 for indices in partitions):
            raise ValueError("train, val, and test partitions must each be non-empty")

        combined = np.concatenate(partitions)
        if np.any(combined < 0) or np.any(combined >= n_rows):
            raise ValueError("split indices must be within the selected population")
        if len(np.unique(combined)) != len(combined):
            raise ValueError("train, val, and test partitions must not overlap")
        if len(combined) != n_rows or not np.array_equal(
            np.sort(combined), np.arange(n_rows)
        ):
            raise ValueError("split indices must cover the selected population exactly once")


def load_frozen_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load and validate the frozen real-label dataset."""
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen dataset from {path}: {error}") from error
    return validate_frozen_dataset(dataset)


def validate_frozen_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate required frozen fields and return a normalized copy."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    if dataset.empty:
        raise ValueError("frozen dataset must not be empty")

    missing_columns = REQUIRED_DATASET_COLUMNS - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Frozen dataset is missing required columns: {sorted(missing_columns)}"
        )

    result = dataset.copy()
    if result["id"].isna().any():
        raise ValueError("Frozen dataset contains missing fact IDs")
    duplicate_ids = result.loc[result["id"].duplicated(), "id"].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen dataset contains duplicate fact IDs: {duplicate_ids[:5]}")

    for column in (TARGET_COLUMN, "guessable"):
        try:
            numeric_values = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain binary 0/1 values") from error
        values = numeric_values.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{column} must contain binary 0/1 values")
        result[column] = numeric_values.astype(int)

    relation_values = result["relation"]
    if relation_values.isna().any() or relation_values.astype(str).str.strip().eq("").any():
        raise ValueError("relation must contain non-empty values")

    for column in PREDICTOR_COLUMNS:
        try:
            numeric_values = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Predictor feature {column} must be numeric and finite") from error
        values = numeric_values.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Predictor feature {column} must be numeric and finite")
        result[column] = numeric_values

    return result


def select_population(dataset: pd.DataFrame, population: str) -> pd.DataFrame:
    """Select all frozen rows or only non-guessable (known) rows."""
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}")
    if "guessable" not in dataset.columns:
        raise ValueError("dataset is missing required column: guessable")
    if population == "all":
        return dataset.copy().reset_index(drop=True)
    return dataset.loc[dataset["guessable"] == 0].copy().reset_index(drop=True)


def select_predictor_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select the established 18 Task 10/13 predictors, excluding IDs and outcomes."""
    missing_columns = set(PREDICTOR_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required predictor columns: {sorted(missing_columns)}"
        )
    return dataset.loc[:, PREDICTOR_COLUMNS].copy()


def map_split_indices(
    dataset: pd.DataFrame, assignments: pd.DataFrame
) -> FrozenSplitIndices:
    """Map frozen ID assignments to positions in dataset row order."""
    if "id" not in dataset.columns:
        raise ValueError("Selected population is missing required column: id")
    if dataset["id"].isna().any() or dataset["id"].duplicated().any():
        raise ValueError("Selected population IDs must be unique and non-null")
    if assignments.empty:
        raise ValueError("Frozen split must not be empty")

    missing_columns = SPLIT_COLUMNS - set(assignments.columns)
    if missing_columns:
        raise ValueError(f"Frozen split is missing required columns: {sorted(missing_columns)}")
    if assignments["id"].isna().any():
        raise ValueError("Frozen split contains missing fact IDs")
    duplicate_ids = assignments.loc[
        assignments["id"].duplicated(), "id"
    ].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Frozen split contains duplicate fact IDs: {duplicate_ids[:5]}")
    if assignments["split"].isna().any():
        raise ValueError("Frozen split contains missing membership values")
    invalid_memberships = sorted(
        set(assignments["split"].astype(str)) - set(SPLIT_MEMBERSHIPS)
    )
    if invalid_memberships:
        raise ValueError(
            f"Frozen split has invalid membership values: {invalid_memberships}"
        )

    dataset_ids = set(dataset["id"].tolist())
    split_ids = set(assignments["id"].tolist())
    missing_ids = dataset_ids - split_ids
    extra_ids = split_ids - dataset_ids
    if missing_ids:
        raise ValueError(
            f"Frozen split is missing {len(missing_ids)} selected population IDs: "
            f"{sorted(missing_ids, key=str)[:5]}"
        )
    if extra_ids:
        raise ValueError(
            f"Frozen split contains {len(extra_ids)} IDs outside the selected population: "
            f"{sorted(extra_ids, key=str)[:5]}"
        )

    row_by_id = {fact_id: index for index, fact_id in enumerate(dataset["id"].tolist())}

    def indices_for(membership: str) -> np.ndarray:
        row_ids = assignments.loc[
            assignments["split"] == membership, "id"
        ].tolist()
        return np.asarray(sorted(row_by_id[fact_id] for fact_id in row_ids), dtype=int)

    split_indices = FrozenSplitIndices(
        train=indices_for("train"),
        validation=indices_for("val"),
        test=indices_for("test"),
    )
    split_indices.validate(len(dataset))
    return split_indices


def load_frozen_split(
    dataset: pd.DataFrame,
    population: str,
    seed: int,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> FrozenSplitIndices:
    """Load one authoritative frozen split and map it to selected-row indices."""
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}")
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")

    selected = select_population(dataset, population)
    split_path = Path(splits_dir) / population / f"seed{seed}.csv"
    try:
        assignments = pd.read_csv(split_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load frozen split from {split_path}: {error}") from error
    return map_split_indices(selected, assignments)