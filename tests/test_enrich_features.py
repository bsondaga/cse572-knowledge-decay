import json
from pathlib import Path

import pandas as pd
import pytest

from src.enrich_features import ENTITY_TYPES, TYPE_NAMES, enrich

FROZEN = Path(__file__).resolve().parents[1] / "results" / "frozen"
RELATIONS = list(ENTITY_TYPES) + ["brand new relation"]


def tables():
    n = len(RELATIONS)
    ids = list(range(500, 500 + n))[::-1]
    base = pd.DataFrame({"id": ids, "relation": RELATIONS, "subject_pop": range(n),
                         "forgotten_lr1e5": [i % 2 for i in range(n)]})
    clean = pd.DataFrame({
        "id": ids + [9999],
        "question": [f"What is thing number {i}?" for i in ids] + ["extra"],
        "possible_answers": [json.dumps([f"answer {i} x", "alt"]) for i in ids] + ["[]"],
    })
    return base, clean


def test_ids_order_and_existing_columns_unchanged():
    base, clean = tables()
    out = enrich(base, clean)
    assert out["id"].tolist() == base["id"].tolist()
    pd.testing.assert_frame_equal(out[base.columns], base)


def test_new_columns_present_without_missing_values():
    base, clean = tables()
    out = enrich(base, clean)
    for col in ["question_length", "question_chars", "answer_length", "answer_chars", "entity_type"]:
        assert col in out.columns
    assert not out.isna().any().any()
    assert (out["question_length"] == 5).all()
    assert (out["answer_length"] == 3).all()


def test_entity_type_covers_relations_and_unknown_is_other():
    base, clean = tables()
    out = enrich(base, clean)
    assert set(out["entity_type"]) <= set(TYPE_NAMES)
    assert out.loc[out["relation"] == "brand new relation", "entity_type"].iloc[0] == "other"
    assert set(ENTITY_TYPES.values()) <= set(TYPE_NAMES)


def test_one_hot_sums_to_one():
    base, clean = tables()
    out = enrich(base, clean)
    assert (out[[f"et_{t}" for t in TYPE_NAMES]].sum(axis=1) == 1).all()


def test_missing_confidence_column_is_ignored():
    base, clean = tables()
    pre = clean[["id", "question"]].assign(prediction="x", baseline_exact_match=1)
    out = enrich(base, clean, pre)
    assert "confidence" not in out.columns


def test_confidence_merged_when_present():
    base, clean = tables()
    pre = pd.DataFrame({"id": base["id"], "confidence": 0.5})
    out = enrich(base, clean, pre)
    assert (out["confidence"] == 0.5).all()
    assert out["id"].tolist() == base["id"].tolist()


def test_missing_question_raises():
    base, clean = tables()
    with pytest.raises(ValueError):
        enrich(base, clean.iloc[1:])


@pytest.mark.skipif(not (FROZEN / "real_dataset_enriched.csv").exists(),
                    reason="enriched file not present")
def test_committed_enriched_file():
    enriched = pd.read_csv(FROZEN / "real_dataset_enriched.csv")
    labels = pd.read_csv(FROZEN / "labels.csv")
    assert len(enriched) == 929
    assert enriched["id"].tolist() == labels["id"].tolist()
    base = pd.read_csv(FROZEN / "real_dataset.csv")
    pd.testing.assert_frame_equal(enriched[base.columns], base)
