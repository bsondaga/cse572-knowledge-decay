from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.make_real_dataset import (
    SEEDS,
    build_real_dataset,
    load_real_dataset,
    load_split,
    make_split,
    merge_rare_relations,
    write_outputs,
)

RELATIONS = {"country": 120, "sport": 100, "capital": 60, "tiny_a": 8, "tiny_b": 6}
FROZEN = Path(__file__).resolve().parents[1] / "results" / "frozen" / "labels.csv"


def synthetic_tables(seed=0):
    rng = np.random.default_rng(seed)
    rel = np.concatenate([[r] * n for r, n in RELATIONS.items()])
    n = len(rel)
    ids = np.arange(1000, 1000 + n)
    features = pd.DataFrame({
        "id": ids, "prop": rel,
        "s_pop": rng.integers(1, 10**5, n), "o_pop": rng.integers(1, 10**5, n),
        "s_pop_log": rng.random(n), "o_pop_log": rng.random(n),
        "rel_country": (rel == "country").astype(int),
    })
    edges = pd.DataFrame({"src_id": ids[:50], "dst_id": ids[1:51], "edge_type": "tail"})
    degree = pd.concat([edges.src_id, edges.dst_id]).value_counts()
    lr = rng.binomial(1, 0.4, n)
    lr[-14:] = [0, 1] * 7
    guessable = (rel == "sport").astype(int)
    analysis = pd.DataFrame({
        "fact_id": ids, "popularity": features.s_pop, "o_pop": features.o_pop,
        "prop": rel, "degree": pd.Series(ids).map(degree).fillna(0).astype(int),
        "forgotten": lr, "balanced_subset": 0, "guess_count": guessable * 20,
        "guessable": guessable,
    })
    def labels(y):
        return pd.DataFrame({"id": ids, "label": np.where(y == 1, "forgotten", "retained")})
    default = np.where(rng.random(n) < 0.9, 1, 0)
    return features, edges, analysis, labels(lr), labels(default)


@pytest.fixture(scope="module")
def dataset():
    return build_real_dataset(*synthetic_tables())


@pytest.fixture(scope="module")
def outputs(tmp_path_factory, dataset):
    root = tmp_path_factory.mktemp("out")
    write_outputs(dataset, root / "real.csv", root / "frozen" / "labels.csv", root / "splits")
    return root


def populations(dataset):
    return {"all": dataset, "known": dataset[dataset["guessable"] == 0]}


def test_dataset_columns_and_unique_ids(dataset):
    for col in ["id", "relation", "subject_pop", "subject_pop_log", "object_pop",
                "object_pop_log", "degree", "guessable", "forgotten_lr1e5",
                "forgotten_default"]:
        assert col in dataset.columns
    assert dataset["id"].is_unique
    assert len(dataset) == sum(RELATIONS.values())


def test_load_real_dataset_roundtrip(outputs, dataset):
    loaded = load_real_dataset(outputs / "real.csv")
    assert list(loaded.columns) == list(dataset.columns)
    assert len(loaded) == len(dataset)


def test_frozen_labels_columns(outputs, dataset):
    frozen = pd.read_csv(outputs / "frozen" / "labels.csv")
    assert list(frozen.columns) == ["id", "guessable", "forgotten_lr1e5", "forgotten_default"]
    assert len(frozen) == len(dataset)


def test_mismatched_analysis_table_is_rejected():
    features, edges, analysis, lr, default = synthetic_tables()
    analysis.loc[3, "popularity"] += 1
    with pytest.raises(ValueError):
        build_real_dataset(features, edges, analysis, lr, default)


@pytest.mark.parametrize("population", ["all", "known"])
@pytest.mark.parametrize("seed", SEEDS)
def test_split_covers_every_id_once(outputs, dataset, population, seed):
    split = load_split(seed, population, outputs / "splits")
    pop = populations(dataset)[population]
    assert split["id"].is_unique
    assert set(split["id"]) == set(pop["id"])
    assert set(split["split"]) == {"train", "val", "test"}


@pytest.mark.parametrize("population", ["all", "known"])
def test_split_sizes_and_label_rate(outputs, dataset, population):
    split = load_split(0, population, outputs / "splits")
    pop = populations(dataset)[population]
    merged = split.merge(pop[["id", "forgotten_lr1e5"]], on="id")
    n = len(merged)
    sizes = merged["split"].value_counts()
    assert abs(sizes["train"] / n - 0.70) < 0.03
    assert abs(sizes["val"] / n - 0.15) < 0.03
    assert abs(sizes["test"] / n - 0.15) < 0.03
    rates = merged.groupby("split")["forgotten_lr1e5"].mean()
    assert rates.max() - rates.min() < 0.08


def test_same_seed_identical_different_seed_differs(dataset):
    pop = dataset.reset_index(drop=True)
    a, b = make_split(pop, 0), make_split(pop, 0)
    c = make_split(pop, 1)
    assert a.equals(b)
    assert not a["split"].equals(c["split"])


def test_rare_relations_merged():
    labels = np.array([0, 1] * 30 + [0, 1, 1])
    rel = np.array(["big"] * 60 + ["rare"] * 3, dtype=object)
    merged = merge_rare_relations(labels, rel)
    assert set(merged[60:]) == {"other"}
    assert set(merged[:60]) == {"big"}


@pytest.mark.skipif(not FROZEN.exists(), reason="results/frozen/labels.csv not present")
def test_frozen_labels_numbers():
    f = pd.read_csv(FROZEN)
    assert len(f) == 929
    assert f["forgotten_lr1e5"].sum() == 362
    assert f["forgotten_default"].sum() == 864
    assert f["guessable"].sum() == 644
    known = f[f["guessable"] == 0]
    assert len(known) == 285
    assert known["forgotten_lr1e5"].sum() == 153
    assert known["forgotten_default"].sum() == 279
