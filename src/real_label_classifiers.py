from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.evaluation_metrics import evaluate_predictions
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation_metrics import evaluate_predictions

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
MODEL_NAMES = (
    "logistic_regression_full",
    "lightgbm_full",
    "logistic_regression_popularity_only",
    "prevalence",
)
METRIC_NAMES = ("auprc", "f1", "recall_at_1", "recall_at_5", "recall_at_10")
PER_SEED_COLUMNS = (
    "population",
    "seed",
    "model",
    "n_train",
    "n_val",
    "n_test",
    "train_prevalence",
    *METRIC_NAMES,
)
SUMMARY_COLUMNS = (
    "population",
    "model",
    "n_seeds",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")),
)
RESULTS_DIR = ROOT / "results"
PER_SEED_OUTPUT = RESULTS_DIR / "task5_metrics_by_seed.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "task5_summary.csv"


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


def select_popularity_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select only the two log-popularity predictors for the ablation baseline."""
    missing_columns = set(PREDICTOR_BASE_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing popularity predictor columns: {sorted(missing_columns)}"
        )
    return dataset.loc[:, PREDICTOR_BASE_COLUMNS].copy()


def make_logistic_pipeline(seed: int) -> Pipeline:
    """Create the Task 5 Logistic Regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=seed
                ),
            ),
        ]
    )


def make_lightgbm_classifier(seed: int) -> LGBMClassifier:
    """Create the fixed LightGBM baseline used by Task 5."""
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def fit_predict_test_scores(
    model,
    features: pd.DataFrame,
    labels: np.ndarray,
    split: FrozenSplitIndices,
) -> np.ndarray:
    """Fit on frozen train rows only and return forgotten-class test probabilities."""
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    split.validate(len(features))

    train_features = features.iloc[split.train]
    test_features = features.iloc[split.test]
    train_labels = labels[split.train]
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("frozen training partition must contain both binary classes")

    model.fit(train_features, train_labels)
    probabilities = np.asarray(model.predict_proba(test_features), dtype=float)
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(split.test):
        raise ValueError("predict_proba must return one probability row per test fact")
    if len(positive_columns) != 1:
        raise ValueError("fitted model does not expose a unique forgotten class 1")

    scores = probabilities[:, positive_columns[0]]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("predicted forgotten-class probabilities must be finite in [0, 1]")
    return scores


def prevalence_test_scores(train_labels: np.ndarray, n_test: int) -> np.ndarray:
    """Return a constant risk equal to train-only forgotten-label prevalence."""
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("train_labels must be a non-empty one-dimensional array")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("train_labels must contain only binary 0/1 values")
    if isinstance(n_test, bool) or not isinstance(n_test, (int, np.integer)) or n_test < 1:
        raise ValueError("n_test must be a positive integer")
    return np.full(n_test, float(labels.mean()), dtype=float)


def evaluate_test_scores(
    labels: np.ndarray,
    split: FrozenSplitIndices,
    test_scores: np.ndarray,
) -> dict[str, float]:
    """Evaluate risk scores against frozen test labels only."""
    labels = np.asarray(labels)
    split.validate(len(labels))
    test_labels = labels[split.test]
    scores = np.asarray(test_scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(split.test):
        raise ValueError("test_scores must contain exactly one score per frozen test row")
    metrics = evaluate_predictions(
        test_labels,
        scores,
        threshold=0.5,
        k_values=(1, 5, 10),
    )
    return {
        "auprc": metrics["auprc"],
        "f1": metrics["f1"],
        "recall_at_1": metrics["recall_at_k"][1],
        "recall_at_5": metrics["recall_at_k"][5],
        "recall_at_10": metrics["recall_at_k"][10],
    }


def evaluate_frozen_seed(
    dataset: pd.DataFrame,
    population: str,
    seed: int,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> list[dict[str, int | float | str]]:
    """Fit requested models and evaluate their risks on one frozen test split."""
    selected = select_population(dataset, population)
    split = load_frozen_split(dataset, population, seed, splits_dir=splits_dir)
    labels = selected[TARGET_COLUMN].to_numpy(dtype=int)
    features = select_predictor_features(selected)
    popularity_features = select_popularity_features(selected)
    train_labels = labels[split.train]
    train_prevalence = float(train_labels.mean())

    model_features = (
        ("logistic_regression_full", make_logistic_pipeline(seed), features),
        ("lightgbm_full", make_lightgbm_classifier(seed), features),
        (
            "logistic_regression_popularity_only",
            make_logistic_pipeline(seed),
            popularity_features,
        ),
    )
    records: list[dict[str, int | float | str]] = []
    for model_name, model, predictors in model_features:
        test_scores = fit_predict_test_scores(model, predictors, labels, split)
        metrics = evaluate_test_scores(labels, split, test_scores)
        records.append(
            _result_record(
                population, seed, model_name, split, train_prevalence, metrics
            )
        )

    baseline_scores = prevalence_test_scores(train_labels, len(split.test))
    baseline_metrics = evaluate_test_scores(labels, split, baseline_scores)
    records.append(
        _result_record(
            population, seed, "prevalence", split, train_prevalence, baseline_metrics
        )
    )
    return records


def _result_record(
    population: str,
    seed: int,
    model_name: str,
    split: FrozenSplitIndices,
    train_prevalence: float,
    metrics: dict[str, float],
) -> dict[str, int | float | str]:
    return {
        "population": population,
        "seed": seed,
        "model": model_name,
        "n_train": len(split.train),
        "n_val": len(split.validation),
        "n_test": len(split.test),
        "train_prevalence": train_prevalence,
        **metrics,
    }


def aggregate_seed_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by population/model using mean and sample SD (ddof=1)."""
    missing_columns = set(PER_SEED_COLUMNS) - set(per_seed.columns)
    if missing_columns:
        raise ValueError(f"Per-seed results are missing columns: {sorted(missing_columns)}")
    if per_seed.empty:
        raise ValueError("Per-seed results must not be empty")

    rows = []
    for (population, model_name), group in per_seed.groupby(
        ["population", "model"], sort=False
    ):
        row: dict[str, int | float | str] = {
            "population": population,
            "model": model_name,
            "n_seeds": group["seed"].nunique(),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    population_order = {population: index for index, population in enumerate(POPULATIONS)}
    model_order = {model: index for index, model in enumerate(MODEL_NAMES)}
    summary["_population_order"] = summary["population"].map(population_order)
    summary["_model_order"] = summary["model"].map(model_order)
    return (
        summary.sort_values(["_population_order", "_model_order"], kind="stable")
        .drop(columns=["_population_order", "_model_order"])
        .reset_index(drop=True)
    )


def run_experiments(
    dataset: pd.DataFrame | None = None,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    populations: tuple[str, ...] = POPULATIONS,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frozen-split Task 5 models for requested populations and seeds."""
    if dataset is None:
        dataset = load_frozen_dataset(dataset_path)
    else:
        dataset = validate_frozen_dataset(dataset)
    invalid_populations = set(populations) - set(POPULATIONS)
    if invalid_populations:
        raise ValueError(f"unsupported populations: {sorted(invalid_populations)}")
    invalid_seeds = set(seeds) - set(SEEDS)
    if invalid_seeds:
        raise ValueError(f"unsupported seeds: {sorted(invalid_seeds)}")

    requested_populations = [
        population for population in POPULATIONS if population in populations
    ]
    records = [
        record
        for population in requested_populations
        for seed in seeds
        for record in evaluate_frozen_seed(
            dataset, population, seed, splits_dir=splits_dir
        )
    ]
    per_seed = pd.DataFrame(records, columns=PER_SEED_COLUMNS)
    return per_seed, aggregate_seed_results(per_seed)


def main() -> None:
    per_seed, summary = run_experiments()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(PER_SEED_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"Saved per-seed metrics to {PER_SEED_OUTPUT}")
    print(f"Saved summary metrics to {SUMMARY_OUTPUT}")
    print(per_seed.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()