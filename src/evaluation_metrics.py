from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _is_real_numeric_array(values: np.ndarray, *, allow_bool: bool = False) -> bool:
    if np.issubdtype(values.dtype, np.number):
        return not np.issubdtype(values.dtype, np.complexfloating)
    if allow_bool and np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return all(
            isinstance(value, Real)
            or (allow_bool and isinstance(value, (bool, np.bool_)))
            for value in values
        )
    return False


def _validated_inputs(
    y_true,
    y_score,
    *,
    require_positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        labels = np.asarray(y_true)
        scores = np.asarray(y_score)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true and y_score must contain numeric values") from error

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional")
    if labels.size == 0 or scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")
    if labels.size != scores.size:
        raise ValueError("y_true and y_score must have the same length")
    if not _is_real_numeric_array(labels, allow_bool=True):
        raise ValueError("y_true must contain only binary 0/1 labels")
    if not _is_real_numeric_array(scores):
        raise ValueError("y_score must contain only finite numeric values")
    try:
        numeric_labels = labels.astype(float)
        numeric_scores = scores.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true and y_score must contain numeric values") from error
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 labels")
    if not np.isfinite(numeric_scores).all():
        raise ValueError("y_score must contain only finite numeric values")
    if np.any((numeric_scores < 0) | (numeric_scores > 1)):
        raise ValueError("y_score values must be in [0, 1]")

    labels = numeric_labels.astype(int)
    scores = numeric_scores
    if require_positive and not np.any(labels == 1):
        raise ValueError("y_true must contain at least one positive/forgotten label")
    return labels, scores


def _validated_threshold(threshold: float) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite number in [0, 1]")
    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a finite number in [0, 1]") from error
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be a finite number in [0, 1]")
    return value


def compute_auprc(y_true, y_score) -> float:
    """Return AUPRC as sklearn average precision (non-interpolated PR summary)."""
    labels, scores = _validated_inputs(y_true, y_score, require_positive=True)
    return float(average_precision_score(labels, scores))


def compute_f1(y_true, y_score, threshold: float = 0.5) -> float:
    """Return F1 after predicting forgotten when risk is >= threshold."""
    labels, scores = _validated_inputs(y_true, y_score)
    threshold_value = _validated_threshold(threshold)
    predictions = (scores >= threshold_value).astype(int)
    return float(f1_score(labels, predictions, zero_division=0))


def recall_at_k(y_true, y_score, k: int) -> float:
    """Return forgotten-label recall among the stable top-k descending scores."""
    labels, scores = _validated_inputs(y_true, y_score, require_positive=True)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    selected_count = min(int(k), labels.size)
    ranked_indices = np.argsort(-scores, kind="stable")
    selected_labels = labels[ranked_indices[:selected_count]]
    return float(np.sum(selected_labels == 1) / np.sum(labels == 1))


def evaluate_predictions(
    y_true,
    y_score,
    threshold: float = 0.5,
    k_values: Iterable[int] = (1, 5, 10),
) -> dict:
    """Compute AP, thresholded F1, and recall@k from forgotten-risk scores."""
    labels, scores = _validated_inputs(y_true, y_score, require_positive=True)
    threshold_value = _validated_threshold(threshold)
    if isinstance(k_values, (str, bytes)):
        raise ValueError("k_values must be an iterable of positive integers")
    try:
        requested_k = tuple(k_values)
    except TypeError as error:
        raise ValueError("k_values must be an iterable of positive integers") from error

    recall_scores = {
        k: recall_at_k(labels, scores, k)
        for k in requested_k
    }
    predictions = (scores >= threshold_value).astype(int)
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": threshold_value,
        "recall_at_k": recall_scores,
    }