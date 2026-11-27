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


# --------------------------------------------------------------------------
# Commit 2: feature importance
# --------------------------------------------------------------------------

TASK17_FIGURE_OUTPUT = ROOT / "results" / "task17_feature_importance.png"
PERMUTATION_REPEATS = 50
PERMUTATION_SEED_OFFSET = 17000
NONZERO_TOLERANCE = 1e-12

IMPORTANCE_TYPES = ("lr_coefficient", "lgbm_gain_normalized", "permutation_auprc")
IMPORTANCE_COLUMNS = (
    "model",
    "importance_type",
    "level",
    "feature",
    "seed",
    "importance",
    "importance_std_repeats",
    "abs_importance",
)
IMPORTANCE_SUMMARY_COLUMNS = (
    "model",
    "importance_type",
    "level",
    "feature",
    "n_seeds",
    "mean_importance",
    "std_importance",
    "mean_abs_importance",
    "sign_consistency",
    "n_seeds_nonzero",
    "rank",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "subject_popularity": ("subject_pop_log",),
    "object_popularity": ("object_pop_log",),
    "relation": tuple(RELATION_FEATURES),
    "entity_type": ENTITY_TYPE_FEATURES,
    "question_length": ("question_length", "question_chars"),
    "answer_length": ("answer_length", "answer_chars"),
}
GROUP_LABELS = {
    "subject_popularity": "Subject popularity",
    "object_popularity": "Object popularity",
    "relation": "Relation (16 one-hot)",
    "entity_type": "Entity type (4 one-hot)",
    "question_length": "Question length (words, chars)",
    "answer_length": "Answer length (words, chars)",
}
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "lightgbm": "LightGBM"}


def permutation_random_state(seed: int) -> int:
    return PERMUTATION_SEED_OFFSET + int(seed)


def _importance_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=IMPORTANCE_COLUMNS)
    if not np.isfinite(frame["importance"].to_numpy(dtype=float)).all():
        raise ValueError("importance values must be finite")
    return frame


def _check_feature_names(fitted_names, predictor_names: tuple[str, ...], what: str) -> None:
    if fitted_names is not None and tuple(fitted_names) != predictor_names:
        raise ValueError(
            f"{what} feature names do not match predictor names: "
            f"fitted={list(fitted_names)} expected={list(predictor_names)}"
        )


def extract_lr_coefficients(pipeline, predictor_names, seed: int) -> pd.DataFrame:
    """Signed standardized-scale coefficients of a fitted Task 5 LR pipeline.

    Positive coefficient = higher forgetting risk. Coefficients are per 1 SD of each
    predictor; rare one-hot indicators are not directly comparable to continuous ones.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    predictor_names = tuple(predictor_names)
    if not isinstance(pipeline, Pipeline) or "classifier" not in pipeline.named_steps:
        raise TypeError("expected a Task 5 Pipeline with a 'classifier' step")
    classifier = pipeline.named_steps["classifier"]
    try:
        check_is_fitted(pipeline)
    except NotFittedError as error:
        raise ValueError("logistic regression pipeline is not fitted") from error
    if not hasattr(classifier, "coef_"):
        raise ValueError("classifier has no coef_")
    if not np.array_equal(np.asarray(classifier.classes_), [0, 1]):
        raise ValueError(f"classifier classes must be [0, 1]; got {list(classifier.classes_)}")
    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.shape != (1, len(predictor_names)):
        raise ValueError(
            f"expected {len(predictor_names)} coefficients; got shape {coefficients.shape}"
        )
    _check_feature_names(
        getattr(pipeline, "feature_names_in_", None), predictor_names, "logistic regression"
    )
    rows = [
        {
            "model": "logistic_regression",
            "importance_type": "lr_coefficient",
            "level": "feature",
            "feature": name,
            "seed": seed,
            "importance": float(value),
            "importance_std_repeats": np.nan,
            "abs_importance": abs(float(value)),
        }
        for name, value in zip(predictor_names, coefficients[0])
    ]
    return _importance_frame(rows)


def extract_lgbm_gain(model, predictor_names, seed: int) -> pd.DataFrame:
    """Gain importance (not split count) of a fitted LightGBM model, normalized to sum 1."""
    predictor_names = tuple(predictor_names)
    try:
        booster = model.booster_
    except Exception as error:  # LightGBM raises LGBMNotFittedError
        raise ValueError("LightGBM model is not fitted") from error
    _check_feature_names(booster.feature_name(), predictor_names, "LightGBM")
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if gain.shape != (len(predictor_names),):
        raise ValueError(f"expected {len(predictor_names)} gain values; got {gain.shape}")
    if not np.isfinite(gain).all() or np.any(gain < 0):
        raise ValueError("LightGBM gain values must be finite and non-negative")
    total = float(gain.sum())
    if total <= 0:
        raise ValueError("LightGBM total gain is zero: the model made no splits")
    rows = [
        {
            "model": "lightgbm",
            "importance_type": "lgbm_gain_normalized",
            "level": "feature",
            "feature": name,
            "seed": seed,
            "importance": float(value),
            "importance_std_repeats": np.nan,
            "abs_importance": np.nan,
        }
        for name, value in zip(predictor_names, gain / total)
    ]
    return _importance_frame(rows)


def _validate_heldout(features: pd.DataFrame, labels) -> np.ndarray:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("held-out features must be a DataFrame")
    labels = np.asarray(labels)
    if len(features) == 0:
        raise ValueError("held-out rows must not be empty")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("held-out features and labels must have matching lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("held-out labels must be binary 0/1")
    if len(np.unique(labels)) != 2:
        raise ValueError("held-out labels must contain both classes")
    return labels.astype(int)


def _positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("model does not expose a unique forgotten class 1")
    return probabilities[:, positive[0]]


def feature_permutation_importance(
    model,
    model_name: str,
    test_features: pd.DataFrame,
    test_labels,
    seed: int,
    *,
    n_repeats: int = PERMUTATION_REPEATS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Per-feature held-out AUPRC decrease via sklearn permutation_importance (descriptive only)."""
    from sklearn.inspection import permutation_importance

    labels = _validate_heldout(test_features, test_labels)
    state = permutation_random_state(seed) if random_state is None else random_state
    result = permutation_importance(
        model,
        test_features,
        labels,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=state,
        n_jobs=1,
    )
    rows = [
        {
            "model": model_name,
            "importance_type": "permutation_auprc",
            "level": "feature",
            "feature": name,
            "seed": seed,
            "importance": float(mean),
            "importance_std_repeats": float(std),
            "abs_importance": np.nan,
        }
        for name, mean, std in zip(
            test_features.columns, result.importances_mean, result.importances_std
        )
    ]
    return _importance_frame(rows)


def validate_feature_groups(
    groups: dict[str, tuple[str, ...]] = FEATURE_GROUPS,
    predictors=ENRICHED_PREDICTOR_COLUMNS,
) -> dict[str, tuple[str, ...]]:
    """Every predictor must belong to exactly one group, with no unknown columns."""
    predictors = tuple(predictors)
    members = [column for columns in groups.values() for column in columns]
    if any(len(columns) == 0 for columns in groups.values()):
        raise ValueError("feature groups must not be empty")
    duplicates = sorted({c for c in members if members.count(c) > 1})
    if duplicates:
        raise ValueError(f"columns assigned to more than one group: {duplicates}")
    unknown = sorted(set(members) - set(predictors))
    if unknown:
        raise ValueError(f"group columns are not approved predictors: {unknown}")
    missing = [c for c in predictors if c not in set(members)]
    if missing:
        raise ValueError(f"predictors not covered by any group: {missing}")
    return groups


def grouped_permutation_importance(
    model,
    model_name: str,
    test_features: pd.DataFrame,
    test_labels,
    seed: int,
    *,
    groups: dict[str, tuple[str, ...]] = FEATURE_GROUPS,
    n_repeats: int = PERMUTATION_REPEATS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Held-out AUPRC decrease when ONE row permutation is applied to a whole group."""
    from sklearn.metrics import average_precision_score

    labels = _validate_heldout(test_features, test_labels)
    validate_feature_groups(groups, tuple(test_features.columns))
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    state = permutation_random_state(seed) if random_state is None else random_state
    rng = np.random.default_rng(state)
    baseline = average_precision_score(labels, _positive_scores(model, test_features))
    rows = []
    for group, columns in groups.items():
        decreases = np.empty(n_repeats, dtype=float)
        for repeat in range(n_repeats):
            order = rng.permutation(len(test_features))
            permuted = test_features.copy()
            for column in columns:  # the same row order for every column in the group
                permuted[column] = test_features[column].to_numpy()[order]
            score = average_precision_score(labels, _positive_scores(model, permuted))
            decreases[repeat] = baseline - score
        rows.append(
            {
                "model": model_name,
                "importance_type": "permutation_auprc",
                "level": "group",
                "feature": group,
                "seed": seed,
                "importance": float(decreases.mean()),
                "importance_std_repeats": float(decreases.std()),
                "abs_importance": np.nan,
            }
        )
    return _importance_frame(rows)


def aggregate_importance(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample SD, coefficient sign consistency and within-type rank across seeds."""
    missing = sorted(set(IMPORTANCE_COLUMNS) - set(by_seed.columns))
    if missing:
        raise ValueError(f"importance rows are missing columns: {missing}")
    if by_seed.empty:
        raise ValueError("importance rows must not be empty")
    keys = ["model", "importance_type", "level", "feature"]
    if by_seed.duplicated([*keys, "seed"]).any():
        raise ValueError("importance rows contain duplicate model/type/level/feature/seed")
    unknown_types = set(by_seed["importance_type"]) - set(IMPORTANCE_TYPES)
    if unknown_types:
        raise ValueError(f"unknown importance types: {sorted(unknown_types)}")

    rows = []
    for key, group in by_seed.groupby(keys, sort=False):
        values = group["importance"].to_numpy(dtype=float)
        mean = float(values.mean())
        row = dict(zip(keys, key))
        row.update(
            n_seeds=int(group["seed"].nunique()),
            mean_importance=mean,
            std_importance=float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            mean_abs_importance=np.nan,
            sign_consistency=np.nan,
            n_seeds_nonzero=int(np.sum(np.abs(values) > NONZERO_TOLERANCE)),
        )
        if row["importance_type"] == "lr_coefficient":
            row["mean_abs_importance"] = float(np.abs(values).mean())
            if abs(mean) <= NONZERO_TOLERANCE:
                row["sign_consistency"] = 0
            else:
                nonzero = np.abs(values) > NONZERO_TOLERANCE
                row["sign_consistency"] = int(np.sum(nonzero & (np.sign(values) == np.sign(mean))))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_rank_key"] = np.where(
        summary["importance_type"] == "lr_coefficient",
        summary["mean_abs_importance"],
        summary["mean_importance"],
    )
    summary["rank"] = (
        summary.groupby(["model", "importance_type", "level"])["_rank_key"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary = summary.sort_values(
        ["model", "importance_type", "level", "rank", "feature"], kind="stable"
    )
    return summary.loc[:, IMPORTANCE_SUMMARY_COLUMNS].reset_index(drop=True)


def plot_grouped_permutation_importance(by_seed: pd.DataFrame, output_path: Path):
    """Two-panel horizontal bar chart of grouped held-out AUPRC decrease, saved as PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("figure output must be a .png path")
    data = by_seed.loc[
        (by_seed["importance_type"] == "permutation_auprc") & (by_seed["level"] == "group")
    ]
    models = list(MODEL_LABELS)
    group_order = list(FEATURE_GROUPS)
    for model in models:
        if set(data.loc[data["model"] == model, "feature"]) != set(group_order):
            raise ValueError(
                f"{model} needs grouped permutation rows for exactly these groups: {group_order}"
            )

    figure = Figure(figsize=(11, 4.6), dpi=200, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    positions = np.arange(len(group_order))[::-1]  # first group drawn at the top
    for axis, model in zip(axes, models):
        model_rows = data.loc[data["model"] == model]
        means = model_rows.groupby("feature")["importance"].mean().reindex(group_order)
        stds = model_rows.groupby("feature")["importance"].std(ddof=1).reindex(group_order)
        axis.barh(positions, means.to_numpy(), xerr=stds.fillna(0.0).to_numpy(),
                  color="#4C72B0", alpha=0.8, capsize=3, height=0.6,
                  label="Mean across seeds (± sample SD)")
        seeds = sorted(model_rows["seed"].unique())
        for seed_index, seed in enumerate(seeds):
            seed_values = (
                model_rows.loc[model_rows["seed"] == seed]
                .set_index("feature")["importance"].reindex(group_order)
            )
            offset = (seed_index - (len(seeds) - 1) / 2) * 0.12
            axis.scatter(seed_values.to_numpy(), positions + offset, s=14, color="black",
                         zorder=3, label="Individual seeds" if seed_index == 0 else None)
        axis.axvline(0.0, color="grey", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Held-out AUPRC decrease when group is permuted")
        axis.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([GROUP_LABELS[group] for group in group_order])
    axes[0].legend(loc="lower right", fontsize=7)
    figure.suptitle("Grouped permutation importance: enriched models on frozen test sets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png")
    return figure


# --------------------------------------------------------------------------
# Commit 3: final experiment and outputs
# --------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
TASK5_SUMMARY_PATH = RESULTS_DIR / "task5_summary.csv"
TASK5_CV_FOLDS_PATH = RESULTS_DIR / "task5_cv_metrics_by_fold.csv"
TASK5_CV_SUMMARY_PATH = RESULTS_DIR / "task5_cv_summary.csv"
TASK8_SUMMARY_PATH = RESULTS_DIR / "task8_graphsage_summary.csv"
TASK13_PAIRED_PATH = RESULTS_DIR / "task13_graph_ablation_paired_differences.csv"
TASK14_RUNS_PATH = RESULTS_DIR / "compare_seeds_runs.csv"

TASK17_OUTPUT_NAMES = {
    "metrics_by_seed": "task17_metrics_by_seed.csv",
    "cv_summary": "task17_cv_summary.csv",
    "importance_by_seed": "task17_importance_by_seed.csv",
    "importance_summary": "task17_importance_summary.csv",
    "final_table": "task17_final_predictor_table.csv",
    "figure": "task17_feature_importance.png",
    "summary_markdown": "task17_summary.md",
}
CONDITIONS = tuple(
    (model, feature_set) for model in MODEL_NAMES for feature_set in ("original", "enriched")
)
PAIRED_COLUMNS = (
    "model",
    "metric",
    "seed0_difference",
    "seed1_difference",
    "seed2_difference",
    "mean_difference",
    "std_difference",
    "n_seeds_improved",
)
CV_FOLD_COLUMNS = (
    "population",
    "model",
    "feature_set",
    "seed",
    "fold",
    "n_train",
    "n_validation",
    *METRIC_NAMES,
)
T17_CV_SUMMARY_COLUMNS = (
    "population",
    "model",
    "feature_set",
    "source",
    "n_evaluations",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std")),
    "paired_delta_auprc_mean",
    "paired_delta_auprc_std",
    "n_folds_auprc_improved",
)
FINAL_TABLE_ROWS = (
    "prevalence_baseline",
    "logistic_regression_original",
    "logistic_regression_enriched",
    "lightgbm_original",
    "lightgbm_enriched",
    "graphsage",
)
FINAL_TABLE_COLUMNS = (
    "model",
    "source",
    "n_features",
    "auprc_mean",
    "auprc_std",
    "f1_mean",
    "f1_std",
    "recall_at_10_mean",
    "recall_at_10_std",
    "paired_delta_auprc_mean",
    "paired_delta_auprc_std",
    "n_seeds_auprc_improved",
    "cv_paired_delta_auprc_mean",
    "clear_improvement",
)
CONFIDENCE_GAP_STATEMENT = (
    "Pre-update model confidence was not present in the frozen Task 7 enriched artifact "
    "(`results/frozen/real_dataset_enriched.csv`), so it was not included in Task 17."
)


@dataclass
class Task17Outputs:
    metrics_by_seed: pd.DataFrame
    paired: pd.DataFrame
    cv_summary: pd.DataFrame
    importance_by_seed: pd.DataFrame
    importance_summary: pd.DataFrame
    final_table: pd.DataFrame
    clear_improvement: dict[str, bool]


# ---- frozen-split conditions ---------------------------------------------


def run_frozen_conditions(
    dataset: pd.DataFrame,
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    seeds: tuple[int, ...] = SEEDS,
    seed_runner: Callable[..., SeedRunResult] = run_frozen_seed,
) -> tuple[dict[tuple[str, str, int], SeedRunResult], pd.DataFrame]:
    """Run the 4 model/feature-set conditions on every frozen seed."""
    if tuple(seeds) != SEEDS:
        raise ValueError(f"Task 17 requires frozen seeds {SEEDS}")
    results: dict[tuple[str, str, int], SeedRunResult] = {}
    for model_name, feature_set in CONDITIONS:
        for seed in seeds:
            result = seed_runner(dataset, seed, feature_set, model_name, splits_dir=splits_dir)
            if (result.model_name, result.feature_set, result.seed) != (
                model_name, feature_set, seed
            ):
                raise ValueError("seed runner returned a result for the wrong condition")
            results[(model_name, feature_set, seed)] = result
    metrics = metric_records_frame(results.values())
    if len(metrics) != len(CONDITIONS) * len(SEEDS):
        raise ValueError("expected 12 condition/seed metric rows")
    return results, metrics


def paired_seed_differences(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Enriched minus original, paired by frozen seed, for every model and metric."""
    rows = []
    for model_name in MODEL_NAMES:
        model_rows = metrics_by_seed.loc[metrics_by_seed["model"] == model_name]
        original = model_rows.loc[model_rows["feature_set"] == "original"].set_index("seed")
        enriched = model_rows.loc[model_rows["feature_set"] == "enriched"].set_index("seed")
        for frame, name in ((original, "original"), (enriched, "enriched")):
            if frame.index.duplicated().any() or sorted(frame.index) != list(SEEDS):
                raise ValueError(f"{model_name} {name} rows must cover seeds {SEEDS} once")
        for metric in METRIC_NAMES:
            deltas = np.array(
                [float(enriched.loc[s, metric]) - float(original.loc[s, metric]) for s in SEEDS]
            )
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    **{f"seed{s}_difference": float(d) for s, d in zip(SEEDS, deltas)},
                    "mean_difference": float(deltas.mean()),
                    "std_difference": float(np.std(deltas, ddof=1)),
                    "n_seeds_improved": int(np.sum(deltas > NONZERO_TOLERANCE)),
                }
            )
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def is_clear_improvement(
    seed_deltas, cv_delta_mean: float, *, n_seeds: int = len(SEEDS)
) -> bool:
    """Predeclared rule: positive on every seed, mean > SD, and CV delta agrees (> 0)."""
    deltas = np.asarray(seed_deltas, dtype=float)
    if deltas.shape != (n_seeds,) or not np.isfinite(deltas).all():
        raise ValueError(f"expected {n_seeds} finite seed deltas")
    if not np.isfinite(cv_delta_mean):
        return False
    all_positive = bool(np.all(deltas > NONZERO_TOLERANCE))
    mean, std = float(deltas.mean()), float(np.std(deltas, ddof=1))
    return all_positive and mean > std and cv_delta_mean > NONZERO_TOLERANCE


# ---- repeated CV supplement (Task 5 procedure) ---------------------------


def run_cv(
    dataset: pd.DataFrame,
    feature_set: str,
    *,
    seeds: tuple[int, ...] | None = None,
    model_factories: dict[str, Callable[[int], object]] | None = None,
) -> pd.DataFrame:
    """Task 5 repeated CV (3 seeds x 5 folds) on the known population for one feature set."""
    from src.evaluation_metrics import evaluate_predictions
    from src.real_label_classifiers import CV_FOLDS, CV_SEEDS, fit_predict_indices, make_cv_folds

    seeds = CV_SEEDS if seeds is None else tuple(seeds)
    if seeds != CV_SEEDS:
        raise ValueError(f"Task 5 CV requires seeds {CV_SEEDS}")
    factories = MODEL_FACTORIES if model_factories is None else model_factories
    known = select_known_population(dataset)
    labels = known[TARGET_COLUMN].to_numpy(dtype=int)
    relations = known["relation"].to_numpy(dtype=object)
    features = select_feature_set(known, feature_set)
    rows = []
    for seed in seeds:
        folds = make_cv_folds(labels, relations, seed=seed, n_splits=CV_FOLDS)
        for fold_index, fold in enumerate(folds):
            for model_name in MODEL_NAMES:
                scores = fit_predict_indices(
                    factories[model_name](seed), features, labels, fold.train, fold.validation
                )
                metrics = evaluate_predictions(
                    labels[fold.validation], scores, threshold=0.5, k_values=(1, 5, 10)
                )
                rows.append(
                    {
                        "population": POPULATION,
                        "model": model_name,
                        "feature_set": feature_set,
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
    frame = pd.DataFrame(rows, columns=CV_FOLD_COLUMNS)
    if len(frame) != len(MODEL_NAMES) * len(seeds) * CV_FOLDS:
        raise ValueError("CV must produce 3 seeds x 5 folds per model")
    return frame


def task5_cv_folds_from_frame(task5_folds: pd.DataFrame) -> pd.DataFrame:
    """Committed Task 5 known CV fold metrics in Task 17 naming."""
    selected = task5_folds.loc[
        (task5_folds["population"] == POPULATION) & task5_folds["model"].isin(TASK5_MODEL_NAMES)
    ].copy()
    selected["model"] = selected["model"].map(TASK5_MODEL_NAMES)
    selected["feature_set"] = "original"
    if selected.duplicated(["model", "seed", "fold"]).any() or len(selected) != 30:
        raise ValueError("Task 5 CV folds must have 30 unique known LR/LightGBM rows")
    return selected.loc[:, CV_FOLD_COLUMNS].reset_index(drop=True)


def compare_cv_with_task5(
    rerun: pd.DataFrame, task5_folds: pd.DataFrame, *, tolerance: float = BASELINE_TOLERANCE
) -> None:
    """Raise unless the original-feature CV rerun reproduces committed Task 5 folds."""
    keys = ["model", "seed", "fold"]
    merged = rerun.merge(task5_folds, on=keys, suffixes=("", "_task5"), how="outer",
                         indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        raise ValueError("CV rerun and Task 5 folds cover different model/seed/fold rows")
    problems = []
    for column in ("n_train", "n_validation", *METRIC_NAMES):
        bad = ~np.isclose(merged[column].astype(float), merged[f"{column}_task5"].astype(float),
                          rtol=0.0, atol=tolerance)
        for row in merged.loc[bad].itertuples(index=False):
            problems.append(f"{row.model} seed {row.seed} fold {row.fold} {column}")
    if problems:
        raise ValueError("CV rerun does not reproduce Task 5:\n  " + "\n  ".join(problems))


def build_cv_summary(enriched_folds: pd.DataFrame, task5_folds: pd.DataFrame) -> pd.DataFrame:
    """Original (committed Task 5) vs enriched CV summary with fold-paired AUPRC deltas."""
    rows = []
    for model_name in MODEL_NAMES:
        original = task5_folds.loc[task5_folds["model"] == model_name]
        enriched = enriched_folds.loc[enriched_folds["model"] == model_name]
        paired = enriched.merge(original, on=["seed", "fold"], suffixes=("", "_orig"),
                                validate="one_to_one")
        if len(paired) != len(original) or len(paired) != len(enriched):
            raise ValueError(f"{model_name}: enriched and original CV folds do not pair up")
        deltas = (paired["auprc"] - paired["auprc_orig"]).to_numpy(dtype=float)
        for feature_set, source, frame in (
            ("original", "task5_committed", original),
            ("enriched", "task17_rerun", enriched),
        ):
            row = {
                "population": POPULATION,
                "model": model_name,
                "feature_set": feature_set,
                "source": source,
                "n_evaluations": len(frame),
            }
            for metric in METRIC_NAMES:
                row[f"{metric}_mean"] = float(frame[metric].mean())
                row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            if feature_set == "enriched":
                row.update(
                    paired_delta_auprc_mean=float(deltas.mean()),
                    paired_delta_auprc_std=float(np.std(deltas, ddof=1)),
                    n_folds_auprc_improved=int(np.sum(deltas > NONZERO_TOLERANCE)),
                )
            else:
                row.update(paired_delta_auprc_mean=np.nan, paired_delta_auprc_std=np.nan,
                           n_folds_auprc_improved=np.nan)
            rows.append(row)
    return pd.DataFrame(rows, columns=T17_CV_SUMMARY_COLUMNS)


# ---- importance ----------------------------------------------------------


def compute_importance(
    results: dict[tuple[str, str, int], SeedRunResult],
    dataset: pd.DataFrame,
    *,
    n_repeats: int = PERMUTATION_REPEATS,
) -> pd.DataFrame:
    """Coefficients, gain, and per-feature + grouped test-set permutation for enriched models."""
    known = select_known_population(dataset)
    features = select_enriched_predictors(known)
    frames = []
    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            result = results[(model_name, "enriched", seed)]
            test_features = features.iloc[result.split.test]
            test_ids = known["id"].to_numpy()[result.split.test]
            if not np.array_equal(test_ids, result.test_ids):
                raise ValueError("importance test rows do not match the frozen test IDs")
            if model_name == "logistic_regression":
                frames.append(extract_lr_coefficients(result.model, result.predictor_names, seed))
            else:
                frames.append(extract_lgbm_gain(result.model, result.predictor_names, seed))
            frames.append(feature_permutation_importance(
                result.model, model_name, test_features, result.test_labels, seed,
                n_repeats=n_repeats))
            frames.append(grouped_permutation_importance(
                result.model, model_name, test_features, result.test_labels, seed,
                n_repeats=n_repeats))
    by_seed = pd.concat(frames, ignore_index=True)
    validate_importance_by_seed(by_seed)
    return by_seed


EXPECTED_IMPORTANCE_BLOCKS = {
    ("logistic_regression", "lr_coefficient", "feature"): len(ENRICHED_PREDICTOR_COLUMNS),
    ("logistic_regression", "permutation_auprc", "feature"): len(ENRICHED_PREDICTOR_COLUMNS),
    ("logistic_regression", "permutation_auprc", "group"): len(FEATURE_GROUPS),
    ("lightgbm", "lgbm_gain_normalized", "feature"): len(ENRICHED_PREDICTOR_COLUMNS),
    ("lightgbm", "permutation_auprc", "feature"): len(ENRICHED_PREDICTOR_COLUMNS),
    ("lightgbm", "permutation_auprc", "group"): len(FEATURE_GROUPS),
}


def validate_importance_by_seed(by_seed: pd.DataFrame) -> None:
    if tuple(by_seed.columns) != IMPORTANCE_COLUMNS:
        raise ValueError("importance rows do not match the approved schema")
    if set(by_seed["seed"]) != set(SEEDS):
        raise ValueError(f"importance rows must cover seeds {SEEDS}")
    keys = ["model", "importance_type", "level", "feature", "seed"]
    if by_seed.duplicated(keys).any():
        raise ValueError("importance rows contain duplicates")
    if not np.isfinite(by_seed["importance"].to_numpy(dtype=float)).all():
        raise ValueError("importance values must be finite")
    blocks = by_seed.groupby(["model", "importance_type", "level", "seed"]).size()
    for (model, kind, level), count in EXPECTED_IMPORTANCE_BLOCKS.items():
        for seed in SEEDS:
            if blocks.get((model, kind, level, seed), 0) != count:
                raise ValueError(f"expected {count} rows for {model}/{kind}/{level}/seed {seed}")
    if len(blocks) != len(EXPECTED_IMPORTANCE_BLOCKS) * len(SEEDS):
        raise ValueError("unexpected model/importance-type/level combinations")
    feature_rows = by_seed.loc[by_seed["level"] == "feature", "feature"]
    validate_predictor_columns(tuple(dict.fromkeys(feature_rows)))
    if set(by_seed.loc[by_seed["level"] == "group", "feature"]) != set(FEATURE_GROUPS):
        raise ValueError("group rows must use exactly the six approved groups")
    gain = by_seed.loc[by_seed["importance_type"] == "lgbm_gain_normalized"]
    if not np.allclose(gain.groupby("seed")["importance"].sum(), 1.0, atol=1e-9):
        raise ValueError("normalized LightGBM gain must sum to 1 per seed")


# ---- final table ---------------------------------------------------------


def _summary_stats(frame: pd.DataFrame) -> dict[str, float]:
    return {
        f"{metric}_{stat}": float(frame[metric].mean() if stat == "mean"
                                  else frame[metric].std(ddof=1))
        for metric in ("auprc", "f1", "recall_at_10")
        for stat in ("mean", "std")
    }


def build_final_predictor_table(
    metrics_by_seed: pd.DataFrame,
    paired: pd.DataFrame,
    cv_summary: pd.DataFrame,
    task5_summary: pd.DataFrame,
    task8_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Six report rows; reference rows reuse committed Task 5 / Task 8 summaries."""
    t5 = task5_summary.loc[task5_summary["population"] == POPULATION].set_index("model")
    t8 = task8_summary.loc[task8_summary["population"] == POPULATION]
    if len(t8) != 1 or t8["model"].iloc[0] != "graphsage":
        raise ValueError("Task 8 summary must have one known graphsage row")
    t8 = t8.iloc[0]

    def committed(row, keys=("auprc", "f1", "recall_at_10")):
        return {f"{k}_{s}": float(row[f"{k}_{s}"]) for k in keys for s in ("mean", "std")}

    blank = {"paired_delta_auprc_mean": np.nan, "paired_delta_auprc_std": np.nan,
             "n_seeds_auprc_improved": np.nan, "cv_paired_delta_auprc_mean": np.nan,
             "clear_improvement": np.nan}
    rows = [{"model": "prevalence_baseline", "source": "task5_committed", "n_features": 0,
             **committed(t5.loc["prevalence"]), **blank}]
    clear: dict[str, bool] = {}
    for model_name in MODEL_NAMES:
        task5_name = {v: k for k, v in TASK5_MODEL_NAMES.items()}[model_name]
        rows.append({"model": f"{model_name}_original", "source": "task5_committed",
                     "n_features": len(ORIGINAL_PREDICTOR_COLUMNS),
                     **committed(t5.loc[task5_name]), **blank})
        enriched = metrics_by_seed.loc[(metrics_by_seed["model"] == model_name)
                                       & (metrics_by_seed["feature_set"] == "enriched")]
        delta = paired.loc[(paired["model"] == model_name) & (paired["metric"] == "auprc")].iloc[0]
        cv_row = cv_summary.loc[(cv_summary["model"] == model_name)
                                & (cv_summary["feature_set"] == "enriched")].iloc[0]
        cv_delta = float(cv_row["paired_delta_auprc_mean"])
        clear[model_name] = is_clear_improvement(
            [delta[f"seed{s}_difference"] for s in SEEDS], cv_delta)
        rows.append({"model": f"{model_name}_enriched", "source": "task17_rerun",
                     "n_features": len(ENRICHED_PREDICTOR_COLUMNS),
                     **_summary_stats(enriched),
                     "paired_delta_auprc_mean": float(delta["mean_difference"]),
                     "paired_delta_auprc_std": float(delta["std_difference"]),
                     "n_seeds_auprc_improved": int(delta["n_seeds_improved"]),
                     "cv_paired_delta_auprc_mean": cv_delta,
                     "clear_improvement": clear[model_name]})
    rows.append({"model": "graphsage", "source": "task8_committed",
                 "n_features": len(ORIGINAL_PREDICTOR_COLUMNS), **committed(t8), **blank})
    table = pd.DataFrame(rows, columns=FINAL_TABLE_COLUMNS)
    if tuple(table["model"]) != FINAL_TABLE_ROWS:
        raise ValueError("final table must contain exactly the six approved rows")
    return table, clear


# ---- orchestration -------------------------------------------------------


def run_task17_experiment(
    dataset: pd.DataFrame | None = None,
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    task5_metrics: pd.DataFrame | None = None,
    task5_summary: pd.DataFrame | None = None,
    task5_cv_folds: pd.DataFrame | None = None,
    task8_summary: pd.DataFrame | None = None,
    n_repeats: int = PERMUTATION_REPEATS,
    seed_runner: Callable[..., SeedRunResult] = run_frozen_seed,
    cv_runner: Callable[[pd.DataFrame, str], pd.DataFrame] = run_cv,
) -> Task17Outputs:
    """Run every Task 17 condition, the CV supplement and importance; return all tables."""
    dataset = load_enriched_dataset() if dataset is None else validate_enriched_dataset(dataset)
    task5_metrics = pd.read_csv(TASK5_METRICS_PATH) if task5_metrics is None else task5_metrics
    task5_summary = pd.read_csv(TASK5_SUMMARY_PATH) if task5_summary is None else task5_summary
    task5_cv_folds = (pd.read_csv(TASK5_CV_FOLDS_PATH) if task5_cv_folds is None
                      else task5_cv_folds)
    task8_summary = pd.read_csv(TASK8_SUMMARY_PATH) if task8_summary is None else task8_summary

    results, metrics_by_seed = run_frozen_conditions(
        dataset, splits_dir=splits_dir, seed_runner=seed_runner)
    compare_with_task5_baseline(
        metrics_by_seed.loc[metrics_by_seed["feature_set"] == "original"],
        task5_baseline_from_frame(task5_metrics),
    )
    paired = paired_seed_differences(metrics_by_seed)

    committed_folds = task5_cv_folds_from_frame(task5_cv_folds)
    compare_cv_with_task5(cv_runner(dataset, "original"), committed_folds)
    cv_summary = build_cv_summary(cv_runner(dataset, "enriched"), committed_folds)

    importance_by_seed = compute_importance(results, dataset, n_repeats=n_repeats)
    importance_summary = aggregate_importance(importance_by_seed)
    final_table, clear = build_final_predictor_table(
        metrics_by_seed, paired, cv_summary, task5_summary, task8_summary)
    return Task17Outputs(metrics_by_seed, paired, cv_summary, importance_by_seed,
                         importance_summary, final_table, clear)


# ---- summary markdown ----------------------------------------------------


def load_report_context(
    task13_paired_path: Path = TASK13_PAIRED_PATH,
    task14_runs_path: Path = TASK14_RUNS_PATH,
) -> dict[str, object]:
    """Read committed Task 13 / Task 14 facts quoted in the summary (read-only)."""
    paired = pd.read_csv(task13_paired_path)
    auprc = paired.loc[paired["metric"] == "auprc"].set_index("comparison")["mean_difference"]
    runs = pd.read_csv(task14_runs_path)
    return {
        "task13_no_edge": float(auprc["no_edge_minus_real_graph"]),
        "task13_shuffled": float(auprc["shuffled_edges_minus_real_graph"]),
        "task13_neighbor": float(auprc["neighbor_lightgbm_minus_task5_lightgbm_full"]),
        "task14_n_runs": int(len(runs)),
        "task14_forget_min": float(runs["forget_rate"].min()),
        "task14_forget_max": float(runs["forget_rate"].max()),
        "task14_s_pop_nonsig": int((runs["p_s_pop"] >= 0.05).sum()),
        "task14_degree_sig": int((runs["p_degree"] < 0.05).sum()),
        "task14_degree_rho_min": float(runs["rho_degree"].min()),
        "task14_degree_rho_max": float(runs["rho_degree"].max()),
    }


def _pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


MODEL_TITLES = {
    "prevalence_baseline": "Prevalence baseline",
    "logistic_regression_original": "Logistic Regression, original (18)",
    "logistic_regression_enriched": "Logistic Regression, enriched (26)",
    "lightgbm_original": "LightGBM, original (18)",
    "lightgbm_enriched": "LightGBM, enriched (26)",
    "graphsage": "GraphSAGE (Task 8)",
}


def build_summary_markdown(outputs: Task17Outputs, context: dict[str, object]) -> str:
    table = outputs.final_table.set_index("model")
    cv = outputs.cv_summary.set_index(["model", "feature_set"])
    groups = outputs.importance_by_seed.loc[
        (outputs.importance_by_seed["importance_type"] == "permutation_auprc")
        & (outputs.importance_by_seed["level"] == "group")]
    group_stats = groups.groupby(["model", "feature"])["importance"].agg(
        mean="mean", std=lambda s: s.std(ddof=1), positive=lambda s: int((s > 0).sum()))
    lines = [
        "# Task 17: final predictor results",
        "",
        "Known (non-guessable) population: 285 facts, target `forgotten_lr1e5`, frozen "
        "splits `results/splits/known/seed{0,1,2}.csv` (199 train / 43 validation / 43 test). "
        "Models reuse the Task 5 Logistic Regression and LightGBM settings unchanged; "
        "metrics are mean ± sample SD over the 3 frozen seeds.",
        "",
        "## Final predictor table",
        "",
        "| Model | Features | AUPRC | F1 | Recall@10 | Δ AUPRC vs original (seeds improved) |",
        "|---|---|---|---|---|---|",
    ]
    for name in FINAL_TABLE_ROWS:
        row = table.loc[name]
        delta = ("—" if pd.isna(row["paired_delta_auprc_mean"]) else
                 f"{_pm(row['paired_delta_auprc_mean'], row['paired_delta_auprc_std'])} "
                 f"({int(row['n_seeds_auprc_improved'])}/3)")
        lines.append(
            f"| {MODEL_TITLES[name]} | {int(row['n_features'])} | "
            f"{_pm(row['auprc_mean'], row['auprc_std'])} | {_pm(row['f1_mean'], row['f1_std'])} | "
            f"{_pm(row['recall_at_10_mean'], row['recall_at_10_std'])} | {delta} |")
    lines += [
        "",
        "## Enriched vs original (paired)",
        "",
        "| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | CV AUPRC original → enriched "
        "(fold-paired Δ, folds improved) | Clear improvement? |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name in MODEL_NAMES:
        delta = outputs.paired.loc[(outputs.paired["model"] == model_name)
                                   & (outputs.paired["metric"] == "auprc")].iloc[0]
        original_cv, enriched_cv = cv.loc[(model_name, "original")], cv.loc[(model_name, "enriched")]
        lines.append(
            f"| {MODEL_LABELS[model_name]} | "
            + " | ".join(_signed(delta[f"seed{s}_difference"]) for s in SEEDS)
            + f" | {_signed(delta['mean_difference'])} ± {delta['std_difference']:.3f} | "
            f"{original_cv['auprc_mean']:.3f} → {enriched_cv['auprc_mean']:.3f} "
            f"({_signed(enriched_cv['paired_delta_auprc_mean'])}, "
            f"{int(enriched_cv['n_folds_auprc_improved'])}/15) | "
            f"{'yes' if outputs.clear_improvement[model_name] else 'no clear improvement'} |")
    lines += [
        "",
        "Predeclared rule for a clear improvement: positive AUPRC delta on all 3 frozen seeds, "
        "mean delta larger than its SD, and a positive fold-paired delta in the Task 5 "
        "repeated CV (3 seeds × 5 folds).",
        "",
        "## Grouped permutation importance (held-out test AUPRC decrease)",
        "",
        "Figure: `results/task17_feature_importance.png`. 50 repeats per seed, "
        "random_state = 17000 + seed; mean ± sample SD over the 3 seeds "
        "(seeds with a positive decrease in brackets).",
        "",
        "| Group | Logistic Regression | LightGBM |",
        "|---|---|---|",
    ]
    for group in FEATURE_GROUPS:
        cells = []
        for model_name in MODEL_NAMES:
            stats = group_stats.loc[(model_name, group)]
            cells.append(f"{_pm(stats['mean'], stats['std'])} [{int(stats['positive'])}/3]")
        lines.append(f"| {GROUP_LABELS[group]} | {cells[0]} | {cells[1]} |")

    best = table.drop(index="prevalence_baseline")["auprc_mean"].idxmax()
    top_groups = {m: group_stats.loc[m]["mean"].idxmax() for m in MODEL_NAMES}

    def consistent(group: str) -> list[str]:
        return [MODEL_LABELS[m] for m in MODEL_NAMES
                if group_stats.loc[(m, group), "positive"] == 3
                and group_stats.loc[(m, group), "mean"] > group_stats.loc[(m, group), "std"]]

    def improvement_sentence(model_name: str) -> str:
        delta = table.loc[f"{model_name}_enriched"]
        verdict = ("a clear improvement" if outputs.clear_improvement[model_name]
                   else "no clear improvement")
        return (f"{MODEL_LABELS[model_name]} {_signed(delta['paired_delta_auprc_mean'])} AUPRC "
                f"({int(delta['n_seeds_auprc_improved'])}/3 seeds improved; CV fold-paired "
                f"{_signed(delta['cv_paired_delta_auprc_mean'])}), {verdict}")

    length_models = {g: consistent(g) for g in ("question_length", "answer_length",
                                                "entity_type")}

    def short_group(group: str) -> str:
        return GROUP_LABELS[group].split(" (")[0].lower()

    if top_groups["logistic_regression"] == top_groups["lightgbm"]:
        top_sentence = (f"the largest group was {short_group(top_groups['lightgbm'])} "
                        f"for both models")
    else:
        top_sentence = (f"the largest group was {short_group(top_groups['logistic_regression'])} "
                        f"for Logistic Regression and {short_group(top_groups['lightgbm'])} "
                        f"for LightGBM")
    harmful = [
        (m, g) for m in MODEL_NAMES for g in FEATURE_GROUPS
        if group_stats.loc[(m, g), "positive"] == 0 and group_stats.loc[(m, g), "mean"] < 0
    ]
    harmful_sentence = "".join(
        f" For {MODEL_LABELS[m]}, permuting {short_group(g)} increased held-out AUPRC on all "
        f"3 seeds (mean decrease {_signed(group_stats.loc[(m, g), 'mean'])}), so the model "
        f"relied on it in a way that did not generalise to the test facts."
        for m, g in harmful
    )

    def group_claim(group: str, label: str) -> str:
        models = length_models[group]
        return (f"{label} gave a consistent positive held-out signal for {' and '.join(models)}"
                if models else f"{label} gave no consistent held-out signal in either model")

    paragraph = (
        f"Adding the Task 7 features (question and answer length in words and characters, and "
        f"a four-way entity type) to the 18 Task 5 predictors gave "
        f"{improvement_sentence('logistic_regression')}, and "
        f"{improvement_sentence('lightgbm')}. "
        f"The strongest model on the frozen test sets was "
        f"{MODEL_TITLES[best]} (AUPRC {_pm(table.loc[best, 'auprc_mean'], table.loc[best, 'auprc_std'])}), "
        f"against a prevalence baseline of {table.loc['prevalence_baseline', 'auprc_mean']:.3f}; "
        f"GraphSAGE reached {_pm(table.loc['graphsage', 'auprc_mean'], table.loc['graphsage', 'auprc_std'])}. "
        f"In grouped held-out permutation importance, {top_sentence}. "
        f"{group_claim('question_length', 'Question length')}; "
        f"{group_claim('answer_length', 'answer length')}; and "
        f"{group_claim('entity_type', 'entity type')}.{harmful_sentence} Entity type is a fixed "
        f"regrouping of relation, so it cannot add information beyond the relation indicators; at "
        f"most it pools rare relations. Answer length should be read with care, because a fact is labelled "
        f"forgotten when the generated text stops matching the gold answer, and longer answers are "
        f"mechanically harder to match. "
        f"Graph structure did not help either: in Task 13, removing all edges changed GraphSAGE "
        f"AUPRC by {_signed(context['task13_no_edge'])} relative to the real graph, shuffling "
        f"edges changed it by {_signed(context['task13_shuffled'])}, and adding neighbour "
        f"features to LightGBM changed it by {_signed(context['task13_neighbor'])}, so graph "
        f"topology showed no measurable benefit. Task 14 shows the labels themselves move "
        f"with the update seed: across {context['task14_n_runs']} update seeds the forgetting rate "
        f"ranged from {context['task14_forget_min']:.2f} to {context['task14_forget_max']:.2f}, "
        f"subject popularity was not significant in {context['task14_s_pop_nonsig']} of "
        f"{context['task14_n_runs']} runs, and degree was significant in only "
        f"{context['task14_degree_sig']} of {context['task14_n_runs']} (rho "
        f"{context['task14_degree_rho_min']:+.2f} to {context['task14_degree_rho_max']:+.2f}). "
        f"{CONFIDENCE_GAP_STATEMENT}"
    )
    lines += [
        "",
        "## Summary",
        "",
        paragraph,
        "",
        "## Limitations",
        "",
        "- Only 3 frozen split seeds, with 43 test facts per seed; seed-to-seed swings are as "
        "large as most model differences.",
        "- No statistical-significance claim is made; the CV supplement is a stability check, "
        "not a test.",
        "- Permutation importance is descriptive and measured on the same 43 test facts; it was "
        "not used for training, tuning or feature selection.",
        f"- {CONFIDENCE_GAP_STATEMENT}",
        "",
    ]
    return "\n".join(lines)


def write_task17_outputs(
    outputs: Task17Outputs,
    context: dict[str, object],
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Write exactly the seven approved Task 17 artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in TASK17_OUTPUT_NAMES.items()}
    outputs.metrics_by_seed.to_csv(paths["metrics_by_seed"], index=False)
    outputs.cv_summary.to_csv(paths["cv_summary"], index=False)
    outputs.importance_by_seed.to_csv(paths["importance_by_seed"], index=False)
    outputs.importance_summary.to_csv(paths["importance_summary"], index=False)
    outputs.final_table.to_csv(paths["final_table"], index=False)
    plot_grouped_permutation_importance(outputs.importance_by_seed, paths["figure"])
    paths["summary_markdown"].write_text(
        build_summary_markdown(outputs, context), encoding="utf-8")
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Task 17 final predictor experiment.")
    parser.add_argument("--run-experiment", action="store_true")
    args = parser.parse_args()
    if not args.run_experiment:
        parser.print_help()
        return
    outputs = run_task17_experiment()
    paths = write_task17_outputs(outputs, load_report_context())
    print(outputs.final_table.to_string(index=False))
    for path in paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
