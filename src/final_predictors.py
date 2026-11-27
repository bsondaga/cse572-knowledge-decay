"""Task 17: final enriched predictors on the frozen real-label splits.

Commit 1 scope: enriched-data loading and validation, the leakage guard, the
frozen-split one-seed runner (reusing the Task 5 model factories, split mapping
and evaluation), and infrastructure for reproducing the committed Task 5
known-population baseline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from src.enrich_features import ENRICHED_PATH, ENTITY_TYPES, TYPE_NAMES
    from src.real_label_classifiers import (
        DEFAULT_SPLITS_DIR,
        METRIC_NAMES,
        PREDICTOR_COLUMNS,
        RELATION_FEATURES,
        TARGET_COLUMN,
        FrozenSplitIndices,
        evaluate_test_scores,
        fit_predict_test_scores,
        load_frozen_split,
        make_lightgbm_classifier,
        make_logistic_pipeline,
        select_population,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.enrich_features import ENRICHED_PATH, ENTITY_TYPES, TYPE_NAMES
    from src.real_label_classifiers import (
        DEFAULT_SPLITS_DIR,
        METRIC_NAMES,
        PREDICTOR_COLUMNS,
        RELATION_FEATURES,
        TARGET_COLUMN,
        FrozenSplitIndices,
        evaluate_test_scores,
        fit_predict_test_scores,
        load_frozen_split,
        make_lightgbm_classifier,
        make_logistic_pipeline,
        select_population,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHED_PATH = ENRICHED_PATH
TASK5_METRICS_PATH = ROOT / "results" / "task5_metrics_by_seed.csv"

POPULATION = "known"
SEEDS = (0, 1, 2)
EXPECTED_TOTAL_ROWS = 929
EXPECTED_KNOWN_ROWS = 285

# Task 5 predictors, in the exact Task 5 order.
ORIGINAL_PREDICTOR_COLUMNS = tuple(PREDICTOR_COLUMNS)
ENTITY_TYPE_FEATURES = tuple(f"et_{name}" for name in TYPE_NAMES)
TASK7_ADDED_COLUMNS = (
    "question_length",
    "question_chars",
    "answer_length",
    "answer_chars",
    *ENTITY_TYPE_FEATURES,
)
ENRICHED_PREDICTOR_COLUMNS = (*ORIGINAL_PREDICTOR_COLUMNS, *TASK7_ADDED_COLUMNS)
FEATURE_SETS = {
    "original": ORIGINAL_PREDICTOR_COLUMNS,
    "enriched": ENRICHED_PREDICTOR_COLUMNS,
}
BINARY_PREDICTOR_COLUMNS = (*RELATION_FEATURES, *ENTITY_TYPE_FEATURES)

# Columns that must never be used as predictors.
PROHIBITED_PREDICTOR_COLUMNS = frozenset(
    {
        "id",
        "relation",
        "entity_type",
        TARGET_COLUMN,
        "forgotten_default",
        "guessable",
        "guess_count",
        "degree",
        "subject_pop",
        "object_pop",
        "pop_bucket",
        "forgetting_risk",
        "true_label",
        "split",
    }
)
PROHIBITED_PREFIXES = ("forgotten", "pop_bucket")
PROHIBITED_SUFFIXES = ("_z",)
PROHIBITED_SUBSTRINGS = ("risk",)

REQUIRED_ENRICHED_COLUMNS = frozenset(
    {"id", "relation", "guessable", TARGET_COLUMN, *ENRICHED_PREDICTOR_COLUMNS}
)

MODEL_FACTORIES: dict[str, Callable[[int], object]] = {
    "logistic_regression": make_logistic_pipeline,
    "lightgbm": make_lightgbm_classifier,
}
MODEL_NAMES = tuple(MODEL_FACTORIES)
TASK5_MODEL_NAMES = {
    "logistic_regression_full": "logistic_regression",
    "lightgbm_full": "lightgbm",
}

METRIC_RECORD_COLUMNS = (
    "population",
    "model",
    "feature_set",
    "seed",
    "n_features",
    "n_train",
    "n_validation",
    "n_test",
    *METRIC_NAMES,
)
BASELINE_COLUMNS = (
    "population",
    "model",
    "feature_set",
    "seed",
    "n_train",
    "n_validation",
    "n_test",
    *METRIC_NAMES,
)
BASELINE_TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Leakage guard and predictor selection
# --------------------------------------------------------------------------


def _prohibited_reason(column: str) -> str | None:
    name = str(column)
    lowered = name.lower()
    if name in PROHIBITED_PREDICTOR_COLUMNS:
        return "ID, target, cohort, graph or raw-popularity column"
    if lowered.startswith(PROHIBITED_PREFIXES):
        return "target-derived or popularity-bucket column"
    if lowered.endswith(PROHIBITED_SUFFIXES):
        return "z-scored duplicate of a popularity column"
    if any(part in lowered for part in PROHIBITED_SUBSTRINGS):
        return "model risk-score column"
    return None


def validate_predictor_columns(columns) -> tuple[str, ...]:
    """Reject prohibited or unapproved predictors; return the columns as a tuple."""
    if isinstance(columns, str):
        raise TypeError("columns must be a sequence of column names, not a string")
    columns = tuple(columns)
    if not columns:
        raise ValueError("predictor list must not be empty")
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"predictor list contains duplicates: {duplicates}")
    prohibited = {c: reason for c in columns if (reason := _prohibited_reason(c))}
    if prohibited:
        details = ", ".join(f"{c!r} ({reason})" for c, reason in prohibited.items())
        raise ValueError(f"prohibited predictor columns: {details}")
    unapproved = [c for c in columns if c not in ENRICHED_PREDICTOR_COLUMNS]
    if unapproved:
        raise ValueError(f"predictor columns are not in the approved Task 17 set: {unapproved}")
    return columns


def predictor_columns(feature_set: str) -> tuple[str, ...]:
    """Return the fixed, ordered predictor list for 'original' (18) or 'enriched' (26)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {tuple(FEATURE_SETS)}")
    return validate_predictor_columns(FEATURE_SETS[feature_set])


def select_feature_set(dataset: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Return exactly the approved predictors for a feature set, in fixed order."""
    columns = predictor_columns(feature_set)
    missing = [c for c in columns if c not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing predictor columns: {missing}")
    return dataset.loc[:, list(columns)].copy()


def select_enriched_predictors(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the 26 approved enriched predictors."""
    return select_feature_set(dataset, "enriched")


# --------------------------------------------------------------------------
# Enriched dataset loading and validation
# --------------------------------------------------------------------------


def _binary_column(values: pd.Series, name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain binary 0/1 values") from error
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain binary 0/1 values")
    return numeric.astype(int)


def validate_enriched_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate the enriched frozen table and return a normalized copy (no rows dropped)."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    if dataset.empty:
        raise ValueError("enriched dataset must not be empty")
    missing = sorted(REQUIRED_ENRICHED_COLUMNS - set(dataset.columns))
    if missing:
        raise ValueError(f"enriched dataset is missing required columns: {missing}")

    result = dataset.copy()
    if result["id"].isna().any():
        raise ValueError("enriched dataset contains missing fact IDs")
    duplicate_ids = result.loc[result["id"].duplicated(), "id"].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"enriched dataset contains duplicate fact IDs: {duplicate_ids[:5]}")

    for column in ("guessable", TARGET_COLUMN):
        result[column] = _binary_column(result[column], column)

    relation = result["relation"]
    if relation.isna().any() or relation.astype(str).str.strip().eq("").any():
        raise ValueError("relation must contain non-empty values")

    for column in ENRICHED_PREDICTOR_COLUMNS:
        if result[column].isna().any():
            raise ValueError(f"predictor {column} has missing values")
        try:
            numeric = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"predictor {column} must be numeric") from error
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"predictor {column} must be finite")
        result[column] = numeric

    for column in BINARY_PREDICTOR_COLUMNS:
        result[column] = _binary_column(result[column], column)

    relation_names = relation.astype(str)
    expected_relation = "rel_" + relation_names.str.replace(" ", "_", regex=False)
    unknown = sorted(set(expected_relation) - set(RELATION_FEATURES))
    if unknown:
        raise ValueError(f"relations without a rel_* indicator column: {unknown}")
    relation_onehot = result[list(RELATION_FEATURES)].to_numpy()
    if not (relation_onehot.sum(axis=1) == 1).all():
        raise ValueError("rel_* indicators must have exactly one 1 per row")
    relation_hits = relation_onehot[
        np.arange(len(result)),
        [RELATION_FEATURES.index(name) for name in expected_relation],
    ]
    if not (relation_hits == 1).all():
        raise ValueError("rel_* indicators do not match the relation column")

    entity_onehot = result[list(ENTITY_TYPE_FEATURES)].to_numpy()
    if not (entity_onehot.sum(axis=1) == 1).all():
        raise ValueError("et_* indicators must have exactly one 1 per row")
    expected_entity = relation_names.map(ENTITY_TYPES).fillna("other")
    entity_hits = entity_onehot[
        np.arange(len(result)),
        [ENTITY_TYPE_FEATURES.index(f"et_{name}") for name in expected_entity],
    ]
    if not (entity_hits == 1).all():
        raise ValueError("et_* indicators do not match the Task 7 entity type of relation")
    return result


def check_expected_counts(dataset: pd.DataFrame) -> None:
    """Assert the committed enriched table has 929 rows, 285 of them known."""
    known = int((dataset["guessable"] == 0).sum())
    if len(dataset) != EXPECTED_TOTAL_ROWS or known != EXPECTED_KNOWN_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_ROWS} rows with {EXPECTED_KNOWN_ROWS} known; "
            f"found {len(dataset)} rows with {known} known"
        )


def load_enriched_dataset(
    path: Path = DEFAULT_ENRICHED_PATH, *, expect_counts: bool = True
) -> pd.DataFrame:
    """Load and validate the frozen Task 7 enriched table."""
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load enriched dataset from {path}: {error}") from error
    dataset = validate_enriched_dataset(dataset)
    if expect_counts:
        check_expected_counts(dataset)
    return dataset


def select_known_population(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the primary population (guessable == 0) using the Task 5 selector."""
    return select_population(dataset, POPULATION)


# --------------------------------------------------------------------------
# One-seed frozen-split runner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRunResult:
    model: object
    model_name: str
    feature_set: str
    seed: int
    predictor_names: tuple[str, ...]
    split: FrozenSplitIndices
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    test_labels: np.ndarray
    test_scores: np.ndarray
    metrics: dict[str, float]


def run_frozen_seed(
    dataset: pd.DataFrame,
    seed: int,
    feature_set: str,
    model_name: str,
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    model_factories: dict[str, Callable[[int], object]] | None = None,
) -> SeedRunResult:
    """Fit one model on frozen train rows and evaluate on frozen test rows only."""
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    factories = MODEL_FACTORIES if model_factories is None else model_factories
    if model_name not in MODEL_NAMES or model_name not in factories:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}")
    predictors = predictor_columns(feature_set)

    known = select_known_population(dataset)
    split = load_frozen_split(dataset, POPULATION, seed, splits_dir=splits_dir)
    labels = known[TARGET_COLUMN].to_numpy(dtype=int)
    features = select_feature_set(known, feature_set)

    model = factories[model_name](seed)
    scores = fit_predict_test_scores(model, features, labels, split)
    metrics = evaluate_test_scores(labels, split, scores)
    ids = known["id"].to_numpy()
    return SeedRunResult(
        model=model,
        model_name=model_name,
        feature_set=feature_set,
        seed=seed,
        predictor_names=predictors,
        split=split,
        train_ids=ids[split.train],
        validation_ids=ids[split.validation],
        test_ids=ids[split.test],
        test_labels=labels[split.test],
        test_scores=scores,
        metrics=metrics,
    )


def metric_record(result: SeedRunResult) -> dict[str, object]:
    """Flatten a one-seed result into the Task 17 per-seed metric schema."""
    return {
        "population": POPULATION,
        "model": result.model_name,
        "feature_set": result.feature_set,
        "seed": result.seed,
        "n_features": len(result.predictor_names),
        "n_train": len(result.split.train),
        "n_validation": len(result.split.validation),
        "n_test": len(result.split.test),
        **{metric: float(result.metrics[metric]) for metric in METRIC_NAMES},
    }


def metric_records_frame(results) -> pd.DataFrame:
    return pd.DataFrame([metric_record(r) for r in results], columns=METRIC_RECORD_COLUMNS)


# --------------------------------------------------------------------------
# Task 5 baseline reproduction
# --------------------------------------------------------------------------


def task5_baseline_from_frame(task5_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select committed Task 5 known LR/LightGBM rows in Task 17 naming."""
    required = {"population", "seed", "model", "n_train", "n_val", "n_test", *METRIC_NAMES}
    missing = sorted(required - set(task5_metrics.columns))
    if missing:
        raise ValueError(f"Task 5 metrics are missing columns: {missing}")
    selected = task5_metrics.loc[
        (task5_metrics["population"] == POPULATION)
        & task5_metrics["model"].isin(TASK5_MODEL_NAMES)
    ].copy()
    selected["model"] = selected["model"].map(TASK5_MODEL_NAMES)
    selected["feature_set"] = "original"
    selected = selected.rename(columns={"n_val": "n_validation"})
    if selected.duplicated(["model", "seed"]).any():
        raise ValueError("Task 5 baseline has duplicate model/seed rows")
    expected = {(model, seed) for model in MODEL_NAMES for seed in SEEDS}
    found = set(zip(selected["model"], selected["seed"]))
    if found != expected:
        raise ValueError(
            f"Task 5 baseline must contain {sorted(expected)}; found {sorted(found)}"
        )
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    selected["_order"] = selected["model"].map(model_order)
    selected = selected.sort_values(["_order", "seed"]).drop(columns="_order")
    return selected.loc[:, BASELINE_COLUMNS].reset_index(drop=True)


def load_task5_baseline(path: Path = TASK5_METRICS_PATH) -> pd.DataFrame:
    """Load the committed Task 5 known-population LR/LightGBM metrics (read-only)."""
    return task5_baseline_from_frame(pd.read_csv(path))


def compare_with_task5_baseline(
    rerun: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    tolerance: float = BASELINE_TOLERANCE,
) -> None:
    """Raise if any rerun original-feature metric differs from committed Task 5."""
    if rerun.empty:
        raise ValueError("rerun metrics must not be empty")
    if set(rerun["feature_set"]) != {"original"}:
        raise ValueError("only original-feature reruns can be compared with Task 5")
    indexed = baseline.set_index(["model", "seed"])
    problems: list[str] = []
    for row in rerun.itertuples(index=False):
        key = (row.model, row.seed)
        if key not in indexed.index:
            problems.append(f"{row.model} seed {row.seed}: no committed Task 5 row")
            continue
        reference = indexed.loc[key]
        for column in ("n_train", "n_validation", "n_test"):
            if int(getattr(row, column)) != int(reference[column]):
                problems.append(
                    f"{row.model} seed {row.seed} {column}: "
                    f"rerun={getattr(row, column)} task5={reference[column]}"
                )
        for metric in METRIC_NAMES:
            value, expected = float(getattr(row, metric)), float(reference[metric])
            if not np.isclose(value, expected, rtol=0.0, atol=tolerance):
                problems.append(
                    f"{row.model} seed {row.seed} {metric}: rerun={value!r} "
                    f"task5={expected!r} (|diff|={abs(value - expected):.3e})"
                )
    if problems:
        raise ValueError("rerun does not reproduce Task 5:\n  " + "\n  ".join(problems))
