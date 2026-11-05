from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_TEST_SIZE = 0.2
BASE_FEATURES = ("s_pop_log", "o_pop_log")
FORGOTTEN_RATE = 0.25


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


def train_and_evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, float | int | Pipeline]:
    """Fit the scaled Logistic Regression pipeline and return held-out metrics."""
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("labels must contain both forgotten and retained examples")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
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
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "model": model,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression classifier on synthetic forgetting labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    df = pd.read_csv(args.input)
    if "id" not in df.columns:
        raise ValueError(f"{args.input} is missing required column: id")
    features = select_features(df)
    labels = generate_dummy_labels(len(df), seed=args.seed)
    results = train_and_evaluate(features, labels, seed=args.seed, test_size=args.test_size)
    forgotten_count = int(labels.sum())
    retained_count = len(labels) - forgotten_count

    print("DUMMY/SYNTHETIC Logistic Regression results")
    print(f"Rows: {len(df)}")
    print(f"Features: {features.shape[1]}")
    print(f"Dummy forgotten/retained counts: {forgotten_count} / {retained_count}")
    print(f"Train/test sizes: {results['train_size']} / {results['test_size']}")
    print(f"Accuracy (DUMMY/SYNTHETIC): {results['accuracy']:.4f}")
    print(f"F1 score (DUMMY/SYNTHETIC): {results['f1']:.4f}")


if __name__ == "__main__":
    main()