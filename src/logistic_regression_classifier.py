from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "popqa_features.csv"
DEFAULT_SEED = 572
BASE_FEATURES = ("s_pop_log", "o_pop_log")
FORGOTTEN_RATE = 0.25


@dataclass(frozen=True)
class StratifiedSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, n_rows: int) -> None:
        partitions = (self.train, self.validation, self.test)
        if any(indices.ndim != 1 for indices in partitions):
            raise ValueError("split indices must be one-dimensional")
        if any(
            not np.issubdtype(indices.dtype, np.integer) for indices in partitions
        ):
            raise ValueError("split indices must contain integers")
        combined = np.concatenate(partitions)
        if np.any(combined < 0) or np.any(combined >= n_rows):
            raise ValueError("split indices must refer to input rows")
        if len(np.unique(combined)) != len(combined):
            raise ValueError("train, validation, and test indices must not overlap")
        if len(combined) != n_rows or not np.array_equal(
            np.sort(combined), np.arange(n_rows)
        ):
            raise ValueError("split indices must cover every input row exactly once")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only log-popularity and relation-indicator predictor columns."""
    missing = set(BASE_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    relation_features = [column for column in df.columns if column.startswith("rel_")]
    return df[[*BASE_FEATURES, *relation_features]].copy()


def generate_dummy_labels(n_rows: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Generate reproducible synthetic forgotten (1) / retained (0) labels."""
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    rng = np.random.default_rng(seed)
    return rng.binomial(1, FORGOTTEN_RATE, size=n_rows)


def create_stratified_split(
    labels: np.ndarray,
    relations: pd.Series | np.ndarray,
    seed: int = DEFAULT_SEED,
) -> StratifiedSplit:
    """Create a deterministic 70/15/15 split stratified by label and relation."""
    labels = np.asarray(labels)
    relations = np.asarray(relations, dtype=object)
    if labels.ndim != 1 or relations.ndim != 1:
        raise ValueError("labels and relations must be one-dimensional")
    if len(labels) != len(relations):
        raise ValueError("labels and relations must have the same number of rows")
    if len(labels) == 0:
        raise ValueError("cannot split an empty dataset")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("labels must contain both forgotten and retained examples")
    if pd.isna(relations).any():
        raise ValueError("relations must not contain missing values")

    strata = list(zip(labels.tolist(), relations.tolist()))
    strata_codes, unique_strata = pd.factorize(pd.Series(strata, dtype=object), sort=False)
    strata_counts = pd.Series(strata).value_counts(sort=False)
    undersized = [
        (label, relation, int(count))
        for (label, relation), count in strata_counts.items()
        if count < 2
    ]
    if undersized:
        raise ValueError(
            "joint (label, relation) strata need at least 2 rows for a "
            f"70/15/15 split; undersized strata (label, relation, count): {undersized}"
        )

    n_rows = len(labels)
    holdout_size = int(np.ceil(n_rows * 0.30))
    train_size = n_rows - holdout_size
    n_strata = len(unique_strata)
    if min(train_size, holdout_size // 2) < n_strata:
        raise ValueError(
            "dataset is too small for 70/15/15 stratification: each resulting "
            f"partition must hold all {n_strata} joint (label, relation) strata"
        )

    row_indices = np.arange(n_rows)
    try:
        train_indices, holdout_indices = train_test_split(
            row_indices,
            test_size=0.30,
            random_state=seed,
            stratify=strata_codes,
        )
    except ValueError as error:
        raise ValueError(
            "cannot create the first stratified split for joint "
            f"(label, relation) strata: {error}"
        ) from error

    holdout_codes = strata_codes[holdout_indices]
    holdout_counts = np.bincount(holdout_codes, minlength=n_strata)
    if np.any(holdout_counts < 2):
        sparse_codes = np.flatnonzero(holdout_counts < 2)
        sparse_strata = [strata[int(code)] for code in sparse_codes]
        raise ValueError(
            "temporary holdout has fewer than 2 rows in joint (label, relation) "
            f"strata needed to split validation and test: {sparse_strata}"
        )
    try:
        validation_indices, test_indices = train_test_split(
            holdout_indices,
            test_size=0.5,
            random_state=seed + 1,
            stratify=holdout_codes,
        )
    except ValueError as error:
        raise ValueError(
            "cannot split the temporary holdout into stratified validation and "
            f"test sets for joint (label, relation) strata: {error}"
        ) from error

    split = StratifiedSplit(
        train=np.asarray(train_indices, dtype=int),
        validation=np.asarray(validation_indices, dtype=int),
        test=np.asarray(test_indices, dtype=int),
    )
    split.validate(n_rows)
    return split


def _fit_and_evaluate(
    model: Pipeline | LGBMClassifier,
    features: pd.DataFrame,
    labels: np.ndarray,
    split: StratifiedSplit,
) -> dict[str, object]:
    labels = np.asarray(labels)
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    split.validate(len(features))

    train_features = features.iloc[split.train]
    test_features = features.iloc[split.test]
    model.fit(train_features, labels[split.train])
    predictions = model.predict(test_features)
    return {
        "model": model,
        "train_size": len(split.train),
        "validation_size": len(split.validation),
        "test_size": len(split.test),
        "test_indices": split.test.copy(),
        "accuracy": accuracy_score(labels[split.test], predictions),
        "f1": f1_score(labels[split.test], predictions),
    }


def train_and_evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    split: StratifiedSplit,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Fit Logistic Regression on the shared split and return test metrics."""
    model = Pipeline(
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
    return _fit_and_evaluate(model, features, labels, split)


def train_lightgbm_and_evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    split: StratifiedSplit,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Fit the fixed LightGBM baseline on the shared split and return test metrics."""
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    return _fit_and_evaluate(model, features, labels, split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline classifiers on synthetic forgetting labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "id" not in df.columns:
        raise ValueError(f"{args.input} is missing required column: id")
    if "prop" not in df.columns:
        raise ValueError(f"{args.input} is missing required column: prop")
    features = select_features(df)
    labels = generate_dummy_labels(len(df), seed=args.seed)
    split = create_stratified_split(labels, df["prop"], seed=args.seed)
    logistic_results = train_and_evaluate(features, labels, split, seed=args.seed)
    lightgbm_results = train_lightgbm_and_evaluate(
        features, labels, split, seed=args.seed
    )
    forgotten_count = int(labels.sum())
    retained_count = len(labels) - forgotten_count

    print("DUMMY/SYNTHETIC classifier smoke-test results")
    print(f"DUMMY/SYNTHETIC total rows: {len(df)}")
    print(f"DUMMY/SYNTHETIC predictor features: {features.shape[1]}")
    print(
        "DUMMY/SYNTHETIC forgotten/retained counts: "
        f"{forgotten_count} / {retained_count}"
    )
    print(
        "DUMMY/SYNTHETIC train/validation/test counts: "
        f"{len(split.train)} / {len(split.validation)} / {len(split.test)}"
    )
    print(f"DUMMY/SYNTHETIC Logistic Regression accuracy: {logistic_results['accuracy']:.4f}")
    print(f"DUMMY/SYNTHETIC Logistic Regression F1: {logistic_results['f1']:.4f}")
    print(f"DUMMY/SYNTHETIC LightGBM accuracy: {lightgbm_results['accuracy']:.4f}")
    print(f"DUMMY/SYNTHETIC LightGBM F1: {lightgbm_results['f1']:.4f}")


if __name__ == "__main__":
    main()