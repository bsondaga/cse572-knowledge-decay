import pandas as pd
import pytest

from src import rehearsal_experiments as exp


def test_run_names():
    assert exp.run_name("none", 5, 0) == "none_seed0"
    assert exp.run_name("none", 20, 0) == "none_seed0"
    assert exp.run_name("random", 10, 2) == (
        "random_budget10_seed2"
    )


def test_load_matches(tmp_path):
    path = tmp_path / "matches.csv"

    pd.DataFrame({
        "id": range(285),
        "post_update_exact_match": [1] * 285
    }).to_csv(path, index=False)

    matches = exp.load_matches(
        path, "post_update_exact_match"
    )

    assert len(matches) == 285
    assert matches.eq(1).all()


def test_reject_duplicate_ids(tmp_path):
    path = tmp_path / "invalid.csv"

    pd.DataFrame({
        "id": [1] * 285,
        "post_update_exact_match": [1] * 285
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        exp.load_matches(path, "post_update_exact_match")


def test_reject_invalid_matches(tmp_path):
    path = tmp_path / "invalid.csv"

    pd.DataFrame({
        "id": range(285),
        "post_update_exact_match": [2] * 285
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        exp.load_matches(path, "post_update_exact_match")


def test_forgetting_reduction(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(exp, "SELECTIONS", tmp_path)

    ids = list(range(285))

    pd.DataFrame({
        "id": ids,
        "post_update_exact_match": [0] * 125 + [1] * 160
    }).to_csv(tmp_path / "none_seed0.csv", index=False)

    pd.DataFrame({
        "id": ids,
        "post_update_exact_match": [0] * 94 + [1] * 191
    }).to_csv(
        tmp_path / "rarest_budget10_seed0.csv",
        index=False
    )

    pd.DataFrame({
        "fact_id": ids[:28]
    }).to_csv(
        tmp_path / "rarest_budget10_seed0.csv.selection",
        index=False
    )

    selection_dir = tmp_path / "selections"
    selection_dir.mkdir()

    pd.DataFrame({
        "fact_id": ids[:28]
    }).to_csv(
        selection_dir / "rarest_budget10_seed0.csv",
        index=False
    )

    monkeypatch.setattr(exp, "SELECTIONS", selection_dir)

    row = exp.calculate_metrics("rarest", 10, 0)

    assert row["forgotten_all"] == 94
    assert row["rehearsed_facts"] == 28
    assert row["non_rehearsed_facts"] == 257
    assert row["forgotten_non_rehearsed"] == 66
    assert row["reduction_all_pp"] == pytest.approx(
        31 / 285 * 100
    )
    assert row["reduction_non_rehearsed_pp"] == pytest.approx(
        31 / 257 * 100
    )
