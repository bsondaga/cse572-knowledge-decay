import json
import pandas as pd
import pytest

from src.false_fact_experiments import (
    prepare_injections,
    score_predictions,
    summarize,
)


def sample_data():
    facts = pd.DataFrame([
        {
            "id": "1",
            "prop": "country",
            "obj": "India",
            "question": "Which country?",
            "possible_answers": json.dumps(["India"]),
        },
        {
            "id": "2",
            "prop": "country",
            "obj": "Japan",
            "question": "Which other country?",
            "possible_answers": json.dumps(["Japan"]),
        },
    ])

    known = pd.DataFrame({
        "id": ["1", "2"],
        "pop_bucket": [0, 1],
    })

    return facts, known


def test_prepare_injections():
    facts, known = sample_data()
    result = prepare_injections(facts, known, 1, 0)

    assert len(result) == 2
    assert result["id"].is_unique
    assert result["pop_bucket"].value_counts().eq(1).all()
    assert (
        result["original_answer"] != result["false_answer"]
    ).all()


def test_reproducibility():
    facts, known = sample_data()

    pd.testing.assert_frame_equal(
        prepare_injections(facts, known, 1, 0),
        prepare_injections(facts, known, 1, 0),
    )


def test_attack_success():
    facts, known = sample_data()
    injections = prepare_injections(facts, known, 1, 0)

    predictions = [
        {
            "id": row["id"],
            "prediction": row["false_answer"],
        }
        for row in injections.to_dict("records")
    ]

    result = score_predictions(injections, predictions)

    assert result["attack_success"].eq(1).all()
    assert result["true_answer_retained"].eq(0).all()
    assert summarize(result)["attack_success_rate"].eq(1).all()


def test_true_answer_retention():
    facts, known = sample_data()
    injections = prepare_injections(facts, known, 1, 0)

    predictions = [
        {
            "id": row["id"],
            "prediction": row["original_answer"],
        }
        for row in injections.to_dict("records")
    ]

    result = score_predictions(injections, predictions)

    assert result["attack_success"].eq(0).all()
    assert result["true_answer_retained"].eq(1).all()


def test_missing_predictions():
    facts, known = sample_data()
    injections = prepare_injections(facts, known, 1, 0)

    with pytest.raises(ValueError):
        score_predictions(injections, [])
