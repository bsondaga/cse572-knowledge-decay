from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.evaluation_metrics import evaluate_predictions
    from src.make_real_dataset import MIN_STRATUM, merge_rare_relations
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation_metrics import evaluate_predictions
    from src.make_real_dataset import MIN_STRATUM, merge_rare_relations

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
CV_SEEDS = (0, 1, 2)
CV_FOLDS = 5
CV_SCORE_COLUMNS = (
    "id",
    "population",
    "model",
    "seed",
    "fold",
    "true_label",
    "forgetting_risk",
)
CV_METRIC_COLUMNS = (
    "population",
    "model",
    "seed",
    "fold",
    "n_train",
    "n_validation",
    *METRIC_NAMES,
)
CV_SUMMARY_COLUMNS = (
    "population",
    "model",
    "n_evaluations",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")),
)
OOF_SCORE_OUTPUT = RESULTS_DIR / "risk_scores_by_model.csv"
FINAL_RISK_OUTPUT = RESULTS_DIR / "risk_scores.csv"
CV_METRICS_OUTPUT = RESULTS_DIR / "task5_cv_metrics_by_fold.csv"
CV_SUMMARY_OUTPUT = RESULTS_DIR / "task5_cv_summary.csv"


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


@dataclass(frozen=True)
class CrossValidationFold:
    train: np.ndarray
    validation: np.ndarray

    def validate(self, n_rows: int) -> None:
        for indices in (self.train, self.validation):
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("cross-validation indices must be one-dimensional integers")
            if len(indices) == 0:
                raise ValueError("cross-validation train and validation folds must be non-empty")
            if np.any(indices < 0) or np.any(indices >= n_rows):
                raise ValueError("cross-validation indices are outside the dataset")
        if np.intersect1d(self.train, self.validation).size:
            raise ValueError("cross-validation train and validation rows must not overlap")


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


def group_rare_relations(
    labels: np.ndarray, relations: pd.Series | np.ndarray
) -> np.ndarray:
    """Apply Task 1's minimum-per-class relation grouping convention."""
    return merge_rare_relations(labels, relations, min_stratum=MIN_STRATUM)


def make_cv_folds(
    labels: np.ndarray,
    relations: pd.Series | np.ndarray,
    seed: int,
    n_splits: int = CV_FOLDS,
) -> list[CrossValidationFold]:
    """Make frozen-seed-compatible CV folds, stratified by label and grouped relation."""
    labels = np.asarray(labels)
    relations = np.asarray(relations, dtype=object)
    if labels.ndim != 1 or relations.ndim != 1:
        raise ValueError("labels and relations must be one-dimensional")
    if labels.size == 0 or labels.size != relations.size:
        raise ValueError("labels and relations must be non-empty and have equal length")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("labels must contain both binary classes 0 and 1")
    if pd.isna(relations).any():
        raise ValueError("relations must not contain missing values")
    if isinstance(n_splits, bool) or n_splits != CV_FOLDS:
        raise ValueError(f"Task 5 OOF generation requires n_splits={CV_FOLDS}")

    relation_groups = group_rare_relations(labels, relations)
    joint_strata = list(zip(labels.tolist(), relation_groups.tolist()))
    strata_codes, unique_strata = pd.factorize(
        pd.Series(joint_strata, dtype=object), sort=False
    )
    counts = np.bincount(strata_codes, minlength=len(unique_strata))
    if np.any(counts < n_splits):
        sparse = [
            (joint_strata[index], int(count))
            for index, count in enumerate(counts)
            if count < n_splits
        ]
        raise ValueError(
            "joint (forgotten_lr1e5, relation_group) strata must each contain "
            f"at least {n_splits} rows after Task 1 rare-relation grouping; "
            f"undersized strata (stratum, count): {sparse}"
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = [
        CrossValidationFold(
            train=np.asarray(train_indices, dtype=int),
            validation=np.asarray(validation_indices, dtype=int),
        )
        for train_indices, validation_indices in splitter.split(
            np.zeros(labels.size), strata_codes
        )
    ]
    validation_counts = np.zeros(labels.size, dtype=int)
    for fold in folds:
        fold.validate(labels.size)
        validation_counts[fold.validation] += 1
    if not np.all(validation_counts == 1):
        raise ValueError("each row must appear in exactly one OOF validation fold")
    return folds


def fit_predict_indices(
    model,
    features: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
) -> np.ndarray:
    """Fit on the supplied training indices and return class-1 probabilities."""
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    train_indices = np.asarray(train_indices)
    prediction_indices = np.asarray(prediction_indices)
    if train_indices.ndim != 1 or prediction_indices.ndim != 1:
        raise ValueError("train and prediction indices must be one-dimensional")
    if (
        not np.issubdtype(train_indices.dtype, np.integer)
        or not np.issubdtype(prediction_indices.dtype, np.integer)
    ):
        raise ValueError("train and prediction indices must contain integers")
    if train_indices.size == 0 or prediction_indices.size == 0:
        raise ValueError("train and prediction indices must be non-empty")
    if np.any(train_indices < 0) or np.any(train_indices >= len(features)):
        raise ValueError("train indices are outside the dataset")
    if np.any(prediction_indices < 0) or np.any(prediction_indices >= len(features)):
        raise ValueError("prediction indices are outside the dataset")
    if np.intersect1d(train_indices, prediction_indices).size:
        raise ValueError("training and prediction rows must not overlap")

    train_labels = labels[train_indices]
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("training rows must contain both binary classes")
    model.fit(features.iloc[train_indices], train_labels)
    probabilities = np.asarray(
        model.predict_proba(features.iloc[prediction_indices]), dtype=float
    )
    if probabilities.ndim != 2 or probabilities.shape[0] != len(prediction_indices):
        raise ValueError("predict_proba must return one row per held-out fact")
    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) != 1:
        raise ValueError("fitted model does not expose a unique forgotten class 1")
    scores = probabilities[:, positive_columns[0]]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("predicted forgotten-class probabilities must be finite in [0, 1]")
    return scores


def aggregate_cv_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CV fold metrics by model using sample standard deviation."""
    required = set(CV_METRIC_COLUMNS)
    missing_columns = required - set(fold_metrics.columns)
    if missing_columns:
        raise ValueError(f"CV metrics are missing columns: {sorted(missing_columns)}")
    if fold_metrics.empty:
        raise ValueError("CV metrics must not be empty")

    rows = []
    for (population, model_name), group in fold_metrics.groupby(
        ["population", "model"], sort=False
    ):
        row: dict[str, int | float | str] = {
            "population": population,
            "model": model_name,
            "n_evaluations": len(group),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows, columns=CV_SUMMARY_COLUMNS)
    model_order = {"logistic_regression_full": 0, "lightgbm_full": 1}
    summary["_model_order"] = summary["model"].map(model_order)
    return (
        summary.sort_values("_model_order", kind="stable")
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def select_final_risk_model(commit2_summary: pd.DataFrame) -> str:
    """Select final rehearsal model only from frozen known test mean AP results."""
    required = {"population", "model", "auprc_mean"}
    missing_columns = required - set(commit2_summary.columns)
    if missing_columns:
        raise ValueError(
            f"Commit 2 summary is missing columns: {sorted(missing_columns)}"
        )
    candidates = commit2_summary.loc[
        (commit2_summary["population"] == "known")
        & commit2_summary["model"].isin(
            ("logistic_regression_full", "lightgbm_full")
        ),
        ["model", "auprc_mean"],
    ].copy()
    candidates["auprc_mean"] = pd.to_numeric(candidates["auprc_mean"], errors="coerce")
    if (
        len(candidates) != 2
        or candidates["auprc_mean"].isna().any()
        or not np.isfinite(candidates["auprc_mean"].to_numpy(dtype=float)).all()
    ):
        raise ValueError(
            "Commit 2 summary must contain finite known-population mean AUPRC "
            "for logistic_regression_full and lightgbm_full"
        )
    return str(candidates.sort_values("auprc_mean", ascending=False).iloc[0]["model"])


def validate_oof_score_artifact(
    scores: pd.DataFrame,
    known_ids,
    seeds: tuple[int, ...] = CV_SEEDS,
) -> None:
    """Validate complete unique OOF coverage for both models and every seed."""
    missing_columns = set(CV_SCORE_COLUMNS) - set(scores.columns)
    if missing_columns:
        raise ValueError(f"OOF score table is missing columns: {sorted(missing_columns)}")
    known_id_set = set(known_ids)
    expected_models = ("logistic_regression_full", "lightgbm_full")
    if set(scores["population"].unique()) != {"known"}:
        raise ValueError("OOF score table must contain only the known population")
    if set(scores["model"].unique()) != set(expected_models):
        raise ValueError(f"OOF score table must contain exactly these models: {expected_models}")
    if set(scores["seed"].unique()) != set(seeds):
        raise ValueError(f"OOF score table must contain exactly seeds {seeds}")
    if scores.duplicated(["model", "seed", "id"]).any():
        raise ValueError("OOF score table contains duplicate model/seed/id rows")
    expected_rows = len(known_id_set) * len(expected_models) * len(seeds)
    if len(scores) != expected_rows:
        raise ValueError(f"OOF score table must contain exactly {expected_rows} rows")
    if scores["id"].isna().any() or scores["true_label"].isna().any():
        raise ValueError("OOF score table contains missing IDs or true labels")
    if not np.isin(scores["true_label"].to_numpy(), [0, 1]).all():
        raise ValueError("OOF score table true_label must be binary 0/1")
    risk = pd.to_numeric(scores["forgetting_risk"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("OOF risk values must be finite and in [0, 1]")

    for model_name in expected_models:
        for seed in seeds:
            rows = scores.loc[
                (scores["model"] == model_name) & (scores["seed"] == seed)
            ]
            if set(rows["id"]) != known_id_set or len(rows) != len(known_id_set):
                raise ValueError(
                    f"OOF scores for {model_name} seed {seed} do not cover known IDs exactly once"
                )
    fold_assignments = scores.groupby(["seed", "id"])["fold"].nunique()
    if not (fold_assignments == 1).all():
        raise ValueError("each fact must have the same OOF fold across both models per seed")


def build_final_risk_scores(
    oof_scores: pd.DataFrame,
    known_ids,
    final_model: str,
    seeds: tuple[int, ...] = CV_SEEDS,
) -> pd.DataFrame:
    """Average the selected model's seed 0/1/2 OOF risks for each known fact."""
    if final_model not in ("logistic_regression_full", "lightgbm_full"):
        raise ValueError("final_model must be one of the two full-feature classifiers")
    validate_oof_score_artifact(oof_scores, known_ids, seeds)
    selected = oof_scores.loc[oof_scores["model"] == final_model].copy()
    if set(selected["seed"].unique()) != set(seeds):
        raise ValueError(f"final risk requires OOF scores from seeds {seeds}")

    per_id = selected.groupby("id").agg(
        forgetting_risk=("forgetting_risk", "mean"),
        seed_count=("seed", "nunique"),
    )
    if not (per_id["seed_count"] == len(seeds)).all():
        raise ValueError("final risk requires exactly one score per known ID from every seed")
    if set(per_id.index) != set(known_ids) or len(per_id) != len(set(known_ids)):
        raise ValueError("final risk scores do not cover known IDs exactly once")
    result = per_id[["forgetting_risk"]].reset_index().sort_values("id").reset_index(drop=True)
    result["forgetting_risk"] = pd.to_numeric(
        result["forgetting_risk"], errors="raise"
    ).astype(float)
    if not np.isfinite(result["forgetting_risk"]).all() or not result[
        "forgetting_risk"
    ].between(0, 1).all():
        raise ValueError("final risk values must be finite and in [0, 1]")
    return result[["id", "forgetting_risk"]]


def run_oof_experiment(
    dataset: pd.DataFrame | None = None,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    commit2_summary: pd.DataFrame | None = None,
    commit2_summary_path: Path = SUMMARY_OUTPUT,
    seeds: tuple[int, ...] = CV_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run known-only 5-fold OOF model evaluation and build rehearsal risks."""
    if tuple(seeds) != CV_SEEDS:
        raise ValueError(f"Task 5 OOF generation requires exactly seeds {CV_SEEDS}")
    if dataset is None:
        dataset = load_frozen_dataset(dataset_path)
    else:
        dataset = validate_frozen_dataset(dataset)
    known = select_population(dataset, "known")
    if known.empty:
        raise ValueError("known population is empty")
    if commit2_summary is None:
        commit2_summary = pd.read_csv(commit2_summary_path)
    final_model = select_final_risk_model(commit2_summary)

    labels = known[TARGET_COLUMN].to_numpy(dtype=int)
    relations = known["relation"].to_numpy(dtype=object)
    features = select_predictor_features(known)
    known_ids = known["id"].tolist()
    score_records: list[dict[str, int | float | str]] = []
    metric_records: list[dict[str, int | float | str]] = []

    for seed in seeds:
        folds = make_cv_folds(labels, relations, seed=seed, n_splits=CV_FOLDS)
        oof_counts = np.zeros(len(known), dtype=int)
        for fold_index, fold in enumerate(folds):
            fold.validate(len(known))
            oof_counts[fold.validation] += 1
            model_definitions = (
                ("logistic_regression_full", make_logistic_pipeline(seed)),
                ("lightgbm_full", make_lightgbm_classifier(seed)),
            )
            for model_name, model in model_definitions:
                validation_scores = fit_predict_indices(
                    model,
                    features,
                    labels,
                    fold.train,
                    fold.validation,
                )
                validation_labels = labels[fold.validation]
                metrics = evaluate_predictions(
                    validation_labels,
                    validation_scores,
                    threshold=0.5,
                    k_values=(1, 5, 10),
                )
                score_records.extend(
                    {
                        "id": known_ids[row_index],
                        "population": "known",
                        "model": model_name,
                        "seed": seed,
                        "fold": fold_index,
                        "true_label": int(labels[row_index]),
                        "forgetting_risk": float(score),
                    }
                    for row_index, score in zip(fold.validation, validation_scores)
                )
                metric_records.append(
                    {
                        "population": "known",
                        "model": model_name,
                        "seed": seed,
                        "fold": fold_index,
                        "n_train": len(fold.train),
                        "n_validation": len(fold.validation),
                        "auprc": metrics["auprc"],
                        "f1": metrics["f1"],
                        "recall_at_1": metrics["recall_at_k"][1],
                        "recall_at_5": metrics["recall_at_k"][5],
                        "recall_at_10": metrics["recall_at_k"][10],
                    }
                )
        if not np.all(oof_counts == 1):
            raise ValueError(f"seed {seed}: each known fact must be OOF exactly once")

    oof_scores = pd.DataFrame(score_records, columns=CV_SCORE_COLUMNS)
    oof_scores["_model_order"] = oof_scores["model"].map(
        {"logistic_regression_full": 0, "lightgbm_full": 1}
    )
    oof_scores = (
        oof_scores.sort_values(["_model_order", "seed", "fold", "id"], kind="stable")
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )
    validate_oof_score_artifact(oof_scores, known_ids, seeds)

    fold_metrics = pd.DataFrame(metric_records, columns=CV_METRIC_COLUMNS)
    fold_metrics["_model_order"] = fold_metrics["model"].map(
        {"logistic_regression_full": 0, "lightgbm_full": 1}
    )
    fold_metrics = (
        fold_metrics.sort_values(["_model_order", "seed", "fold"], kind="stable")
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )
    cv_summary = aggregate_cv_metrics(fold_metrics)
    final_risk = build_final_risk_scores(oof_scores, known_ids, final_model, seeds)
    return oof_scores, fold_metrics, cv_summary, final_risk, final_model


def save_oof_outputs(
    oof_scores: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    cv_summary: pd.DataFrame,
    final_risk: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Write only the four approved Commit 3 result artifacts."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    oof_scores.to_csv(results_dir / OOF_SCORE_OUTPUT.name, index=False)
    final_risk.to_csv(results_dir / FINAL_RISK_OUTPUT.name, index=False)
    fold_metrics.to_csv(results_dir / CV_METRICS_OUTPUT.name, index=False)
    cv_summary.to_csv(results_dir / CV_SUMMARY_OUTPUT.name, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 5 frozen evaluation or OOF integration.")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="run known-population 5-fold OOF integration without changing Commit 2 tables",
    )
    args = parser.parse_args()
    if args.cv:
        oof_scores, fold_metrics, cv_summary, final_risk, final_model = run_oof_experiment()
        save_oof_outputs(oof_scores, fold_metrics, cv_summary, final_risk)
        print(f"Selected final rehearsal risk model from Commit 2 known mean AUPRC: {final_model}")
        print(f"Saved OOF scores by model: {OOF_SCORE_OUTPUT} ({len(oof_scores)} rows)")
        print(f"Saved final rehearsal risk scores: {FINAL_RISK_OUTPUT} ({len(final_risk)} rows)")
        print(f"Saved CV metrics by fold: {CV_METRICS_OUTPUT} ({len(fold_metrics)} rows)")
        print(f"Saved CV summary: {CV_SUMMARY_OUTPUT} ({len(cv_summary)} rows)")
        print(cv_summary.to_string(index=False))
        return

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